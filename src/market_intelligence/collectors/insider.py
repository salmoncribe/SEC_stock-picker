"""Insider-transaction collector: SEC Form 3/4/5 quarterly datasets -> events.

``sync`` iterates SEC's quarterly pre-parsed insider-transaction ZIPs, persists
each quarter's raw bytes verbatim, parses and joins the three TSVs that matter
(``SUBMISSION``, ``NONDERIV_TRANS``, ``REPORTINGOWNER`` -- see
``clients/insider.py``), filters to issuers in the collected universe, builds
one :class:`~market_intelligence.schemas.events.EventRecord` per non-derivative
transaction, validates, and upserts into DuckDB + Parquet. All bookkeeping (the
``pipeline_runs`` row) is handled by ``pipeline_run``.

**Two clocks, and why this collector exists at all.** ``event_time`` is the
transaction date (``TRANS_DATE``) -- when the insider actually traded.
``available_time`` is the filing date (``FILING_DATE``) -- when the public
could first have known about it. SEC gives ``FILING_DATE`` as a bare calendar
date with no time of day, so it is never knowable from this data alone whether
a given filing landed before or after that day's market close. **Stage 2 (the
dataset builder) must therefore treat the first *following* trading day as
t=0** for any feature or label keyed on one of these events -- never the
filing date itself, which could still be an intraday leak.

**Resumability.** Each quarter's ZIP is one static, ~12 MB archive covering
every Form 3/4/5 filed that quarter; unlike the price collector there is no
"what's new since last time" -- a quarter is either fully processed or not
processed at all. Durability comes from flushing to DuckDB after each quarter
(never buffering the ~80-quarter backfill in memory) and skipping a quarter
whose events are already stored, so a run interrupted partway through costs at
most the one quarter in flight, not the whole backfill. Resumption is decided
from the ``events`` table itself (``payload.source_quarter``, queried back via
DuckDB's JSON functions) rather than a new tracking table, because
``database.py`` / ``storage/duckdb.py`` are frozen for this change -- a
quarter whose only in-universe transactions were later rejected as invalid
will therefore report zero events and be re-fetched on the next run. That is
strictly safe (re-fetching a network-cheap, content-hashed archive costs time,
never correctness) and is the only cost of not touching the schema.

**A 404 for a recent quarter is normal.** SEC publishes each quarter's ZIP
with a lag of days to a few weeks after quarter close; a 404 there means "not
published yet," not a broken endpoint. It is caught per quarter, counted and
named in ``summary.notes``, and collection continues.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.clients.insider import (
    InsiderClient,
    InsiderTransactionRow,
    QuarterNotPublished,
    parse_quarter_zip,
    quarter_for_date,
    quarters_between,
)
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.schemas.common import Source, utcnow
from market_intelligence.schemas.events import (
    DIRECTION_NEGATIVE,
    DIRECTION_NEUTRAL,
    DIRECTION_POSITIVE,
    EventRecord,
    EventType,
)
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet, raw
from market_intelligence.validators.events import validate_event

if TYPE_CHECKING:
    import duckdb
    import httpx

    from market_intelligence.config import Config

EXTRACTION_METHOD = "sec_form345_dataset_v1"

# Earliest quarter SEC's structured insider-transaction dataset covers (see
# docs/specs/2026-07-22-signal-graph-design.md §9: "Coverage confirmed 2006q1
# through 2026q1"). Used only as the default lower bound for a full backfill.
EARLIEST_QUARTER = "2006q1"


@dataclass(frozen=True)
class _Universe:
    """The collected universe, loaded once per run.

    ``cik_set`` is the union of ``companies.cik`` and ``filings.cik`` -- an
    issuer counts as "in universe" if either table has ever recorded it.
    ``ticker_set`` is every symbol with stored price history, the fallback
    when an issuer's CIK is not on file (e.g. a company observed only via
    index membership + prices, not yet filing-collected).
    """

    cik_set: frozenset[str]
    ticker_set: frozenset[str]
    company_id_by_cik: dict[str, str]
    ticker_by_cik: dict[str, str]


def _load_universe(con: duckdb.DuckDBPyConnection) -> _Universe:
    company_rows = con.execute("SELECT cik, company_id, ticker FROM companies").fetchall()
    filing_rows = con.execute("SELECT DISTINCT cik FROM filings WHERE cik IS NOT NULL").fetchall()
    price_rows = con.execute(
        "SELECT DISTINCT symbol FROM daily_prices WHERE symbol IS NOT NULL"
    ).fetchall()

    company_id_by_cik: dict[str, str] = {}
    ticker_by_cik: dict[str, str] = {}
    cik_set: set[str] = set()
    for cik, company_id, ticker in company_rows:
        if not cik:
            continue
        cik_set.add(str(cik))
        if company_id:
            company_id_by_cik[str(cik)] = str(company_id)
        if ticker:
            ticker_by_cik[str(cik)] = str(ticker).upper()

    cik_set.update(str(row[0]) for row in filing_rows if row[0])
    ticker_set = {str(row[0]).upper() for row in price_rows if row[0]}

    return _Universe(frozenset(cik_set), frozenset(ticker_set), company_id_by_cik, ticker_by_cik)


def _resolve_ticker(row: InsiderTransactionRow, universe: _Universe) -> str | None:
    """Best-known ticker for a transaction: the filing's own symbol field,
    falling back to whatever ticker the platform has on file for this CIK.
    """
    if row.issuer_trading_symbol:
        return row.issuer_trading_symbol
    if row.issuer_cik is not None:
        return universe.ticker_by_cik.get(row.issuer_cik)
    return None


def _in_universe(row: InsiderTransactionRow, ticker: str | None, universe: _Universe) -> bool:
    if row.issuer_cik is not None and row.issuer_cik in universe.cik_set:
        return True
    return ticker is not None and ticker in universe.ticker_set


def _as_utc_datetime(value: date | None) -> datetime | None:
    """A bare date at midnight UTC -- the platform's internal clock is UTC
    throughout, and SEC gives these dates with no time of day at all.
    """
    if value is None:
        return None
    return datetime.combine(value, time.min, tzinfo=UTC)


def _quarter_already_collected(con: duckdb.DuckDBPyConnection, quarter: str) -> bool:
    """Whether events from ``quarter`` are already durably stored.

    See the module docstring's "Resumability" section: the quarter id is
    carried in ``payload.source_quarter`` (JSON) and queried back with
    DuckDB's built-in ``json_extract_string`` rather than a dedicated column,
    because the ``events`` table schema is frozen for this change.
    """
    result = con.execute(
        """
        SELECT count(*) FROM events
        WHERE event_type = ?
          AND json_extract_string(payload, '$.source_quarter') = ?
        """,
        [EventType.INSIDER_TRANSACTION, quarter],
    ).fetchone()
    return bool(result and result[0])


def _build_event(
    row: InsiderTransactionRow,
    *,
    quarter: str,
    universe: _Universe,
    schema_version: str,
    source_url: str,
) -> EventRecord:
    event_key = row.trans_sk or ""
    event_id = hashing.content_hash("event", EventType.INSIDER_TRANSACTION, event_key)

    magnitude: float | None = None
    if row.trans_shares is not None and row.trans_price_per_share:
        magnitude = row.trans_shares * row.trans_price_per_share

    if row.trans_acquired_disp_cd == "A":
        direction = DIRECTION_POSITIVE
    elif row.trans_acquired_disp_cd == "D":
        direction = DIRECTION_NEGATIVE
    else:
        direction = DIRECTION_NEUTRAL

    company_id = universe.company_id_by_cik.get(row.issuer_cik) if row.issuer_cik else None
    ticker = _resolve_ticker(row, universe)

    payload: dict[str, Any] = {
        "owner_name": row.owner_name,
        "owner_cik": row.owner_cik,
        "owner_relationship": row.owner_relationship,
        "owner_title": row.owner_title,
        "security_title": row.security_title,
        "shares": row.trans_shares,
        "price_per_share": row.trans_price_per_share,
        "shares_owned_following_transaction": row.shares_owned_following_transaction,
        "direct_indirect_ownership": row.direct_indirect_ownership,
        "magnitude_unit": "usd",
        "source_quarter": quarter,
    }

    return EventRecord(
        event_id=event_id,
        event_type=EventType.INSIDER_TRANSACTION,
        event_key=event_key,
        company_id=company_id,
        cik=row.issuer_cik,
        ticker=ticker,
        event_subtype=row.trans_code,
        accession_number=row.accession_number,
        filing_id=None,
        event_time=_as_utc_datetime(row.trans_date),
        available_time=_as_utc_datetime(row.filing_date),
        magnitude=magnitude,
        direction=direction,
        payload=payload,
        extraction_method=EXTRACTION_METHOD,
        source=Source.SEC,
        source_url=source_url,
        content_hash=hashing.content_hash(
            event_key,
            row.trans_code,
            row.trans_shares,
            row.trans_price_per_share,
            row.shares_owned_following_transaction,
            row.direct_indirect_ownership,
        ),
        schema_version=schema_version,
        collected_time=utcnow(),
    )


def sync(
    config: Config,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
    tickers: list[str] | None = None,
    force: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> RunSummary:
    """Collect insider transactions for ``[start_quarter, end_quarter]``.

    Defaults to the full confirmed backfill window (``EARLIEST_QUARTER``
    through the current calendar quarter). ``tickers``, when given, further
    restricts emitted events to issuers whose resolved ticker (see
    :func:`_resolve_ticker`) is in the list -- events for out-of-list issuers
    are neither counted as collected nor as out-of-universe, since they were
    never candidates in the first place. ``force`` re-fetches and re-processes
    a quarter even if it already has stored events.
    """
    with pipeline_run(config, "insider.sync") as (con, summary):
        schema_version = config.settings.app.schema_version
        start = start_quarter or EARLIEST_QUARTER
        end = end_quarter or quarter_for_date(date.today())
        quarters = quarters_between(start, end)
        wanted_tickers = {t.strip().upper() for t in tickers} if tickers else None

        universe = _load_universe(con)

        collected = 0
        rejected = 0
        inserted = 0
        updated = 0
        deduped = 0

        with InsiderClient.from_config(config, transport=transport) as client:
            for quarter in quarters:
                if not force and _quarter_already_collected(con, quarter):
                    summary.note(f"already_collected:{quarter}")
                    continue

                try:
                    fetched = client.fetch_quarter(quarter)
                except QuarterNotPublished as exc:
                    summary.bump("quarter_not_published")
                    summary.note(f"not_published:{quarter}:{exc}")
                    continue

                summary.downloaded += 1
                saved = raw.save_raw(
                    config.paths.raw_dir, "sec", "form345", quarter, fetched.raw, ext="zip"
                )
                if saved.was_new:
                    summary.stored += 1
                else:
                    summary.skipped += 1

                try:
                    txn_rows = parse_quarter_zip(fetched.raw)
                except zipfile.BadZipFile as exc:
                    summary.bump("quarter_parse_failed")
                    summary.note(f"parse_failed:{quarter}:{exc}")
                    continue

                quarter_rows: list[dict[str, Any]] = []
                quarter_out_of_universe = 0
                quarter_out_of_scope = 0
                for txn in txn_rows:
                    ticker = _resolve_ticker(txn, universe)
                    if not _in_universe(txn, ticker, universe):
                        quarter_out_of_universe += 1
                        continue
                    if wanted_tickers is not None and (
                        ticker is None or ticker not in wanted_tickers
                    ):
                        quarter_out_of_scope += 1
                        continue

                    record = _build_event(
                        txn,
                        quarter=quarter,
                        universe=universe,
                        schema_version=schema_version,
                        source_url=fetched.url,
                    )
                    validate_event(record)
                    collected += 1
                    if record.is_rejected:
                        rejected += 1
                    else:
                        quarter_rows.append(record.to_row())

                summary.bump("out_of_universe_rows", quarter_out_of_universe)
                if wanted_tickers is not None:
                    summary.bump("out_of_ticker_scope_rows", quarter_out_of_scope)

                if quarter_rows:
                    # Flush per quarter, not once at the end of the backfill --
                    # the same reasoning as collectors/prices.py: this run
                    # spans ~80 quarters, and a crash near the end must not
                    # cost quarters already durably written. See
                    # test_earlier_quarters_survive_a_later_quarter_crashing.
                    result = duckdb_store.upsert_events(con, quarter_rows)
                    inserted += result.inserted
                    updated += result.updated
                    deduped += result.deduped
                    parquet.write_records(
                        config.paths.parquet_dir,
                        "events",
                        quarter_rows,
                        ["event_type", "event_key"],
                        partition_col="event_type",
                    )

                summary.note(
                    f"{quarter}: {len(txn_rows)} transactions, {len(quarter_rows)} events "
                    f"stored ({quarter_out_of_universe} out of universe)"
                )

        summary.collected = collected
        summary.inserted = inserted
        summary.updated = updated
        summary.deduped = deduped
        summary.rejected = rejected

    return summary


__all__ = ["EARLIEST_QUARTER", "EXTRACTION_METHOD", "sync"]
