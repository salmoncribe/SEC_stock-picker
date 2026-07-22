"""S&P 500 constituents collector: Wikipedia -> reconstruct -> validate -> raw/Parquet/DuckDB.

``sync`` fetches the Wikipedia page once, reconstructs point-in-time membership
windows (see ``clients/constituents.py`` for the algorithm and its documented
coverage limits), persists the raw HTML, validates the normalized records, and
upserts the non-rejected rows into DuckDB and the Parquet mirror. All bookkeeping
(the ``pipeline_runs`` row) is handled by ``pipeline_run``.

``company_id``/``cik`` resolution joins each window to the existing ``companies``
table **on ticker**, per the platform's ticker-based universe convention. This is
imperfect for a ticker symbol that has been reused by different companies over time
(e.g. "Q": Qwest Communications, then QuintilesIMS, then Qnity Electronics) -- such a
join can only ever match whichever company the ``companies`` table currently
associates with that ticker, which is correct for the open (current) window but may
mis-attribute an older, closed window to the wrong company. Where no ticker match
exists at all, ``company_id``/``cik`` are left NULL rather than guessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.clients.constituents import (
    ConstituentsClient,
    parse_changes,
    parse_current_constituents,
    reconstruct_membership,
)
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.schemas.common import utcnow
from market_intelligence.schemas.market import ConstituentRecord
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet, raw
from market_intelligence.validators.market import validate_constituent

if TYPE_CHECKING:
    import duckdb
    import httpx

    from market_intelligence.config import Config

INDEX_ID = "SP500"


def _company_lookup(con: duckdb.DuckDBPyConnection) -> dict[str, tuple[str | None, str | None]]:
    """Upper-cased ticker -> ``(company_id, cik)`` from the existing ``companies`` table."""
    rows = con.execute(
        "SELECT ticker, company_id, cik FROM companies WHERE ticker IS NOT NULL"
    ).fetchall()
    lookup: dict[str, tuple[str | None, str | None]] = {}
    for ticker, company_id, cik in rows:
        lookup[str(ticker).upper()] = (company_id, cik)
    return lookup


def sync(config: Config, *, transport: httpx.BaseTransport | None = None) -> RunSummary:
    """Collect point-in-time S&P 500 membership from Wikipedia."""
    with pipeline_run(config, "constituents.sync") as (con, summary):
        with ConstituentsClient.from_config(config, transport=transport) as client:
            fetched = client.fetch_page()

        raw.save_raw(
            config.paths.raw_dir,
            "reference",
            "sp500_constituents",
            "sp500",
            fetched.raw,
            ext="html",
        )

        windows = reconstruct_membership(
            parse_current_constituents(fetched.raw), parse_changes(fetched.raw)
        )

        schema_version = config.settings.app.schema_version
        collected_time = utcnow()
        company_lookup = _company_lookup(con)

        rows: list[dict[str, Any]] = []
        collected = 0
        rejected = 0
        flagged = 0

        for window in windows:
            collected += 1
            company_id, resolved_cik = company_lookup.get(window.ticker.upper(), (None, None))
            removed_key = window.removed_date.isoformat() if window.removed_date else ""
            record = ConstituentRecord(
                constituent_id=hashing.content_hash(
                    "constituent", INDEX_ID, window.ticker, window.added_date.isoformat()
                ),
                index_id=INDEX_ID,
                ticker=window.ticker,
                company_id=company_id,
                cik=window.cik or resolved_cik,
                company_name=window.company_name,
                added_date=window.added_date,
                removed_date=window.removed_date,
                source_url=fetched.url,
                content_hash=hashing.content_hash(
                    INDEX_ID,
                    window.ticker,
                    window.added_date.isoformat(),
                    removed_key,
                    window.company_name,
                    window.cik,
                ),
                schema_version=schema_version,
                collected_time=collected_time,
            )
            if window.note:
                # Recorded directly (not via validate_constituent, which we must not
                # modify): a warning, never a rejection -- the window is real, only
                # its precision or "still open" status is uncertain. See the client
                # module docstring for what each note means.
                record.add_error(window.note, reject=False)
                flagged += 1

            validate_constituent(record)
            if record.is_rejected:
                rejected += 1
            else:
                rows.append(record.to_row())

        if flagged:
            summary.note(
                f"{flagged} membership window(s) carry an approximated added_date or an "
                "unresolved departure; see each record's validation_errors"
            )

        summary.collected = collected
        summary.rejected = rejected

        result = duckdb_store.upsert_constituents(con, rows)
        summary.inserted = result.inserted
        summary.updated = result.updated
        # Without this the count identity silently fails: rows collapsed by the
        # upsert for sharing a natural key would vanish with no counter
        # explaining why inserted + updated fell short of collected. A non-zero
        # value here means the reconstruction emitted two windows with the same
        # (index_id, ticker, added_date), which is worth noticing rather than
        # absorbing.
        summary.deduped = result.deduped
        if result.deduped:
            summary.note(
                f"{result.deduped} window(s) collapsed on a shared "
                "(index_id, ticker, added_date); the reconstruction produced duplicates"
            )

        parquet.write_records(
            config.paths.parquet_dir, "index_constituents", rows, ["constituent_id"]
        )

    return summary


__all__ = ["sync"]
