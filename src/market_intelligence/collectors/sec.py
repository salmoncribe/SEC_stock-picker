"""SEC EDGAR collectors: client -> validate -> raw + Parquet + DuckDB.

Three entry points the CLI drives:

* :func:`sync_companies` — refresh the ``companies`` table from the SEC ticker map;
* :func:`collect_filings` — pull recent filings for a set of tickers;
* :func:`collect_ipos` — scan the configured universe for registration forms.

IPO limitation
--------------
SEC EDGAR provides no single global "all recent IPOs" endpoint in the form this
platform needs. :func:`collect_ipos` therefore scans the *configured* companies'
submission histories for registration forms (S-1 / F-1 by default). A clean
future extension point is the SEC EDGAR full-text search API
(``https://efts.sec.gov/LATEST/search-index``), which can surface registration
filings across the whole universe; it is intentionally NOT called here, so no
undocumented / global endpoint is relied upon.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.clients.sec import SECClient
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.schemas.common import utcnow
from market_intelligence.schemas.sec import (
    CompanyRecord,
    FilingRecord,
    build_filing_url,
    filter_by_forms,
    iter_submission_filings,
    normalize_cik,
    parse_ticker_map,
)
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet, raw
from market_intelligence.validators.sec import validate_company, validate_filing

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config


def _company_id(cik10: str) -> str:
    return hashing.content_hash("company", cik10)


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# sync-companies                                                               #
# --------------------------------------------------------------------------- #
def sync_companies(config: Config, *, transport: Any = None) -> RunSummary:
    """Refresh the ``companies`` table from SEC's ``company_tickers.json``."""
    with pipeline_run(config, "sec.sync-companies") as (con, summary):
        with SECClient.from_config(config, transport=transport) as client:
            fetched = client.fetch_company_tickers()

        saved = raw.save_raw(config.paths.raw_dir, "sec", "company_tickers", "all", fetched.raw)
        collected_time = utcnow()
        schema_version = config.settings.app.schema_version

        records: list[CompanyRecord] = []
        for entry in parse_ticker_map(fetched.data):
            cik = entry["cik"]
            record = CompanyRecord(
                company_id=_company_id(cik),
                ticker=entry.get("ticker"),
                company_name=entry.get("title"),
                cik=cik,
                content_hash=saved.content_hash,
                source_url=fetched.url,
                first_seen_time=collected_time,
                last_seen_time=collected_time,
                collected_time=collected_time,
                schema_version=schema_version,
            )
            validate_company(record)
            records.append(record)

        rows = [r.to_row() for r in records if not r.is_rejected]
        summary.collected = len(records)
        summary.rejected = sum(1 for r in records if r.is_rejected)

        result = duckdb_store.upsert_companies(con, rows)
        summary.inserted = result.inserted
        summary.updated = result.updated
        parquet.write_records(config.paths.parquet_dir, "companies", rows, ["cik"])
    return summary


# --------------------------------------------------------------------------- #
# shared per-ticker machinery                                                  #
# --------------------------------------------------------------------------- #
def _resolve_cik_map(config: Config, client: SECClient, tickers: list[str]) -> dict[str, str]:
    """Map upper-cased ticker -> 10-digit CIK, fetching SEC's map once if needed."""
    entry_by_ticker = {c.ticker.upper(): c for c in config.companies.companies}
    resolved: dict[str, str] = {}
    unresolved: set[str] = set()

    for ticker in tickers:
        upper = ticker.upper()
        entry = entry_by_ticker.get(upper)
        if entry is not None and entry.cik:
            try:
                resolved[upper] = normalize_cik(entry.cik)
                continue
            except ValueError:
                pass
        unresolved.add(upper)

    if unresolved:
        fetched = client.fetch_company_tickers()
        for row in parse_ticker_map(fetched.data):
            upper = row["ticker"].upper()
            if upper in unresolved and upper not in resolved:
                resolved[upper] = row["cik"]

    return resolved


def _collect_ticker_filings(
    config: Config,
    client: SECClient,
    con: duckdb.DuckDBPyConnection,
    summary: RunSummary,
    *,
    ticker: str,
    cik10: str,
    forms: list[str] | None,
    limit: int | None,
) -> None:
    """Fetch, validate, and persist one ticker's filings (used by both entry points)."""
    fetched = client.fetch_submissions(cik10)
    saved = raw.save_raw(config.paths.raw_dir, "sec", "submissions", cik10, fetched.raw)
    collected_time = utcnow()
    schema_version = config.settings.app.schema_version
    company_id = _company_id(cik10)

    # Upsert the owning company so filings always have a parent row to link to.
    company = CompanyRecord(
        company_id=company_id,
        ticker=ticker,
        company_name=(fetched.data or {}).get("name") if isinstance(fetched.data, dict) else None,
        cik=cik10,
        content_hash=saved.content_hash,
        source_url=fetched.url,
        first_seen_time=collected_time,
        last_seen_time=collected_time,
        collected_time=collected_time,
        schema_version=schema_version,
    )
    validate_company(company)
    if not company.is_rejected:
        duckdb_store.upsert_companies(con, [company.to_row()])

    filings = filter_by_forms(iter_submission_filings(fetched.data), forms)
    if limit is not None:
        filings = filings[:limit]

    supported_forms = set(config.forms.supported)
    rows: list[dict[str, Any]] = []
    built = 0
    for filing in filings:
        accession = filing.get("accession_number")
        form = filing.get("form")
        if accession is None or form is None:
            continue
        built += 1
        filing_url = build_filing_url(cik10, accession, filing.get("primary_document"))
        record = FilingRecord(
            filing_id=hashing.content_hash("filing", accession),
            company_id=company_id,
            cik=cik10,
            accession_number=accession,
            form=form,
            filing_date=_parse_date(filing.get("filing_date")),
            report_date=_parse_date(filing.get("report_date")),
            acceptance_time=_parse_datetime(filing.get("acceptance_time")),
            primary_document=filing.get("primary_document"),
            filing_url=filing_url,
            source_url=filing_url,
            content_hash=hashing.content_hash(accession, form),
            collected_time=collected_time,
            schema_version=schema_version,
        )
        validate_filing(record, supported_forms)
        if record.is_rejected:
            summary.rejected += 1
            continue
        row = record.to_row()
        # raw_file_path is a storage column, not a record field (extra="forbid").
        row["raw_file_path"] = saved.path
        rows.append(row)

    summary.collected += built
    result = duckdb_store.upsert_filings(con, rows)
    summary.inserted += result.inserted
    summary.updated += result.updated
    parquet.write_records(config.paths.parquet_dir, "filings", rows, ["accession_number"])


def _default_tickers(config: Config, tickers: list[str] | None) -> list[str]:
    if tickers is not None:
        return tickers
    return [c.ticker for c in config.companies.companies]


def _run_ticker_collection(
    config: Config,
    con: duckdb.DuckDBPyConnection,
    summary: RunSummary,
    *,
    tickers: list[str],
    forms: list[str] | None,
    limit: int | None,
    transport: Any,
) -> None:
    with SECClient.from_config(config, transport=transport) as client:
        cik_map = _resolve_cik_map(config, client, tickers)
        for ticker in tickers:
            cik10 = cik_map.get(ticker.upper())
            if not cik10:
                summary.note(f"unresolved_cik:{ticker}")
                continue
            _collect_ticker_filings(
                config,
                client,
                con,
                summary,
                ticker=ticker.upper(),
                cik10=cik10,
                forms=forms,
                limit=limit,
            )


# --------------------------------------------------------------------------- #
# collect-filings / collect-ipos                                               #
# --------------------------------------------------------------------------- #
def collect_filings(
    config: Config,
    *,
    tickers: list[str] | None = None,
    forms: list[str] | None = None,
    limit: int | None = None,
    transport: Any = None,
) -> RunSummary:
    """Collect recent filings for ``tickers`` (default: the configured universe)."""
    with pipeline_run(config, "sec.collect-filings") as (con, summary):
        _run_ticker_collection(
            config,
            con,
            summary,
            tickers=_default_tickers(config, tickers),
            forms=forms,
            limit=limit,
            transport=transport,
        )
    return summary


def collect_ipos(
    config: Config,
    *,
    forms: list[str] | None = None,
    tickers: list[str] | None = None,
    limit: int | None = None,
    transport: Any = None,
) -> RunSummary:
    """Scan the configured universe for registration (IPO) filings.

    SEC provides no global "all recent IPOs" endpoint in the required form; this
    scans configured companies' submissions for registration forms (S-1 / F-1 by
    default). See the module docstring for the full-text-search extension point.
    """
    effective_forms = forms if forms is not None else list(config.forms.ipo)
    with pipeline_run(config, "sec.collect-ipos") as (con, summary):
        summary.note(
            "SEC provides no global all-recent-IPOs endpoint in the required form; "
            "scanning configured companies' submissions for registration forms "
            f"({', '.join(effective_forms) or 'none configured'}). Future extension: "
            "SEC EDGAR full-text search API (https://efts.sec.gov/LATEST/search-index)."
        )
        _run_ticker_collection(
            config,
            con,
            summary,
            tickers=_default_tickers(config, tickers),
            forms=effective_forms,
            limit=limit,
            transport=transport,
        )
    return summary


__all__ = ["collect_filings", "collect_ipos", "sync_companies"]
