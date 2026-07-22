"""8-K item-code events: what a company announced, and when the market knew.

An 8-K is the SEC's "something happened" form, and every one of them carries a
list of item codes naming *what* happened -- ``2.02`` results of operations,
``5.02`` an officer departure, ``4.02`` a restatement. Those codes are already
present in the submissions documents preserved under ``raw/sec/submissions``,
so this collector reads bytes we already hold rather than re-fetching from SEC.

**One event per (filing, item code), never one per filing.** An 8-K reporting
``1.01,2.03,8.01,9.01`` announces four different things, and folding them into a
single "8-K happened" event would mix a financing agreement with an exhibit
list and measure the average of unrelated announcements. The validation gate
scores each item code as its own cell, exactly as it scores each Form 4
transaction code separately.

**Direction is neutral for every filing item, and that is deliberate.** An
insider transaction has an intrinsic sign -- a purchase is a purchase -- so its
producer sets +1/-1 from the transaction code. An item code is a *topic*, not a
signed action: ``4.02`` (non-reliance on prior financials) is plausibly bad and
``1.01`` (a new material agreement) could be either, but "plausibly" is a
hypothesis, not a fact, and hard-coding it would smuggle an unmeasured prior
into the labels. The sign is left for the gate to discover from returns.

**Item 9.01 is boilerplate, and its cell is a co-occurrence diagnostic.**
``9.01`` merely says exhibits are attached; it is the most common code by far
and rides along with genuine news. If the gate admits it with a large effect,
that is the co-occurrence showing through, not information in 9.01 itself.
It is measured rather than dropped precisely because it reads as a check on
how much of any item's effect is really its companions'.

Item numbering changed in August 2004, so pre-2004 filings use bare integers
(``5``) where modern ones use decimals (``5.02``). The two never collide as
strings, so legacy codes stay distinct subtypes and are measured separately --
they are far too rare here to reach the gate's minimum sample size, which is
the correct outcome rather than a silent conflation with a modern code.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.schemas.common import Source, utcnow
from market_intelligence.schemas.events import DIRECTION_NEUTRAL, EventRecord, EventType
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet
from market_intelligence.validators.events import validate_event

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config

EXTRACTION_METHOD = "sec_submissions_items"

#: Forms carrying an item-code list. Amendments are excluded by default: an
#: 8-K/A restates an earlier filing, so counting both would score one
#: announcement twice under the same item code.
DEFAULT_FORMS: tuple[str, ...] = ("8-K",)

#: Events built between writes, matching the document collector's durability
#: pattern so an interrupted run keeps what it finished.
DEFAULT_FLUSH_EVERY = 20_000

#: Codes that announce no news of their own (see the module docstring).
BOILERPLATE_ITEMS = frozenset({"9.01"})

#: Official 8-K item taxonomy (2004 onward). Used for human-readable labels in
#: alerts and the Obsidian projection; an unknown code is kept, not dropped.
ITEM_TITLES: dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety — Reporting of Shutdowns and Patterns of Violations",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure or Election of Directors or Certain Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.04": "Temporary Suspension of Trading Under Employee Benefit Plans",
    "5.05": "Amendment to Registrant's Code of Ethics",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


def parse_item_codes(raw: str | None) -> list[str]:
    """Split SEC's comma-separated item string into distinct, ordered codes.

    Duplicates within one filing are collapsed: a filing listing an item twice
    still announced it once, and emitting two events would let a formatting
    quirk double that filing's weight in its cell.
    """
    if not raw:
        return []
    seen: list[str] = []
    for chunk in raw.split(","):
        code = chunk.strip()
        if code and code not in seen:
            seen.append(code)
    return seen


def load_item_index(raw_dir: Path | str, forms: tuple[str, ...]) -> dict[str, str]:
    """Map ``accession_number -> item string`` from preserved submissions files.

    Reads the raw store rather than the network: these bytes were already
    fetched and kept, and re-deriving from them costs nothing and cannot drift
    from what was originally collected.
    """
    index: dict[str, str] = {}
    wanted = set(forms)
    root = Path(raw_dir) / "sec" / "submissions"
    if not root.exists():
        return index

    for path in sorted(root.glob("*/*/*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                recent = (json.load(handle).get("filings") or {}).get("recent") or {}
        except (OSError, json.JSONDecodeError):
            continue

        accessions = recent.get("accessionNumber") or []
        form_list = recent.get("form") or []
        items = recent.get("items") or []
        for i, form in enumerate(form_list):
            if form not in wanted or i >= len(accessions):
                continue
            index[str(accessions[i])] = str(items[i]) if i < len(items) else ""
    return index


def _select_filings(
    con: duckdb.DuckDBPyConnection,
    *,
    forms: tuple[str, ...],
    tickers: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Read candidate filings joined to their issuer's ticker.

    The ticker join is required, not cosmetic: the dataset builder selects
    events by ticker, so an event without one can never be measured.
    """
    params: list[Any] = list(forms)
    sql = f"""
        SELECT f.filing_id, f.company_id, f.cik, f.accession_number, f.form,
               f.filing_date, f.report_date, f.acceptance_time, c.ticker
        FROM filings f
        JOIN companies c ON c.cik = f.cik
        WHERE f.form IN ({", ".join(["?"] * len(forms))})
          AND f.validation_status <> 'rejected'
          AND c.ticker IS NOT NULL AND c.ticker <> ''
    """
    if tickers:
        uppered = [t.upper() for t in tickers]
        sql += f" AND upper(c.ticker) IN ({', '.join(['?'] * len(uppered))})"
        params.extend(uppered)
    sql += " ORDER BY f.filing_date"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    columns = (
        "filing_id", "company_id", "cik", "accession_number", "form",
        "filing_date", "report_date", "acceptance_time", "ticker",
    )  # fmt: skip
    return [dict(zip(columns, row, strict=True)) for row in con.execute(sql, params).fetchall()]


def _as_utc(value: Any) -> datetime | None:
    """Coerce a date or datetime to an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    return None


def _available_time(row: dict[str, Any]) -> datetime | None:
    """When the public could first have known -- acceptance, else filing date.

    ``acceptance_time`` is the instant EDGAR disseminated the filing, which is
    the honest publication moment. It is preferred over ``filing_date`` because
    SEC dates a filing accepted after 5:30pm ET to the *next* business day: an
    8-K accepted Friday evening carries Monday's filing date even though it was
    readable Friday. Keying on the filing date would place the event a day late
    and quietly measure returns from after the reaction.

    The reverse error is impossible here regardless of which is used: the
    dataset builder starts its window strictly after the available *date*, so
    an announcement is never scored against the session it landed in.
    """
    return _as_utc(row.get("acceptance_time")) or _as_utc(row.get("filing_date"))


def build_events(
    row: dict[str, Any],
    item_codes: list[str],
    *,
    schema_version: str,
) -> list[EventRecord]:
    """Turn one filing and its item codes into one event per code."""
    accession = str(row["accession_number"])
    available = _available_time(row)
    ticker = str(row["ticker"]).upper()

    # The date of the earliest event the filing reports, which is what the 8-K
    # cover page states; it precedes publication and must never be used as t0.
    event_time = _as_utc(row.get("report_date")) or available

    records: list[EventRecord] = []
    for code in item_codes:
        event_key = f"{accession}:{code}"
        payload: dict[str, Any] = {
            "form": row.get("form"),
            "item_code": code,
            "item_title": ITEM_TITLES.get(code),
            "is_boilerplate": code in BOILERPLATE_ITEMS,
            # Kept so a later analysis can ask what else was announced that
            # day without re-reading the raw store.
            "co_items": [other for other in item_codes if other != code],
            "item_count": len(item_codes),
        }
        records.append(
            EventRecord(
                event_id=hashing.content_hash("filing_item", accession, code),
                event_type=EventType.FILING_ITEM,
                event_key=event_key,
                company_id=row.get("company_id"),
                cik=row.get("cik"),
                ticker=ticker,
                event_subtype=code,
                accession_number=accession,
                filing_id=row.get("filing_id"),
                event_time=event_time,
                available_time=available,
                magnitude=None,
                direction=DIRECTION_NEUTRAL,
                payload=payload,
                extraction_method=EXTRACTION_METHOD,
                source=Source.SEC,
                content_hash=hashing.content_hash(event_key, code, str(available)),
                schema_version=schema_version,
                collected_time=utcnow(),
            )
        )
    return records


def _flush(
    config: Config,
    con: duckdb.DuckDBPyConnection,
    summary: RunSummary,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    result = duckdb_store.upsert_events(con, rows)
    summary.inserted += result.inserted
    summary.updated += result.updated
    summary.deduped += result.deduped
    parquet.write_records(config.paths.parquet_dir, "events", rows, ["event_type", "event_key"])


def sync(
    config: Config,
    *,
    forms: tuple[str, ...] | None = None,
    tickers: list[str] | None = None,
    limit: int | None = None,
    flush_every: int = DEFAULT_FLUSH_EVERY,
) -> RunSummary:
    """Emit one event per (8-K, item code) from preserved submissions data.

    Purely local: no network calls. Re-running is idempotent because events are
    keyed on ``(event_type, event_key)`` and ``event_key`` is derived from the
    accession and item code.
    """
    wanted = tuple(forms) if forms else DEFAULT_FORMS

    with pipeline_run(config, "sec.sync-filing-events") as (con, summary):
        schema_version = config.settings.app.schema_version

        index = load_item_index(config.paths.raw_dir, wanted)
        summary.bump("filings_in_item_index", len(index))

        candidates = _select_filings(con, forms=wanted, tickers=tickers, limit=limit)
        summary.bump("candidate_filings", len(candidates))

        pending: list[dict[str, Any]] = []
        for row in candidates:
            accession = str(row["accession_number"])
            if accession not in index:
                # The filing predates the submissions snapshot's 1,000-filing
                # recent window, so its item codes were never collected.
                summary.bump("no_item_data")
                continue

            codes = parse_item_codes(index[accession])
            if not codes:
                summary.bump("filings_without_items")
                continue

            summary.bump("filings_with_items")
            for record in build_events(row, codes, schema_version=schema_version):
                validate_event(record)
                if record.is_rejected:
                    summary.rejected += 1
                    summary.note(
                        f"event_rejected:{record.event_key}:{','.join(record.validation_errors)}"
                    )
                    continue
                if record.event_subtype in BOILERPLATE_ITEMS:
                    summary.bump("boilerplate_events")
                summary.collected += 1
                pending.append(record.to_row())

            if len(pending) >= flush_every:
                _flush(config, con, summary, pending)
                pending = []

        _flush(config, con, summary, pending)
    return summary


__all__ = [
    "BOILERPLATE_ITEMS",
    "DEFAULT_FORMS",
    "EXTRACTION_METHOD",
    "ITEM_TITLES",
    "build_events",
    "load_item_index",
    "parse_item_codes",
    "sync",
]
