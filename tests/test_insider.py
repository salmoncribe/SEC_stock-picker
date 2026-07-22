"""Offline tests for the Form 4 insider-transaction client and collector.

Every network-facing test injects an ``httpx.MockTransport`` (see
``clients.insider.InsiderClient``); nothing here touches the network or the
real ``data/`` directory. Small in-memory ZIPs built by ``build_quarter_zip``
stand in for SEC's quarterly archives.
"""

from __future__ import annotations

import io
import locale
import zipfile
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

from market_intelligence import database
from market_intelligence.clients.insider import (
    InsiderClient,
    QuarterNotPublished,
    parse_quarter_zip,
    parse_sec_date,
    parse_sec_float,
    quarter_for_date,
    quarters_between,
)
from market_intelligence.collectors import insider as insider_collector
from market_intelligence.config import Config
from market_intelligence.schemas.sec import normalize_cik
from market_intelligence.storage import duckdb as duckdb_store

# --------------------------------------------------------------------------- #
# fixture builders                                                            #
# --------------------------------------------------------------------------- #
SUBMISSION_COLUMNS = [
    "ACCESSION_NUMBER",
    "FILING_DATE",
    "PERIOD_OF_REPORT",
    "DOCUMENT_TYPE",
    "ISSUERCIK",
    "ISSUERNAME",
    "ISSUERTRADINGSYMBOL",
]
TRANS_COLUMNS = [
    "ACCESSION_NUMBER",
    "NONDERIV_TRANS_SK",
    "SECURITY_TITLE",
    "TRANS_DATE",
    "TRANS_CODE",
    "TRANS_SHARES",
    "TRANS_PRICEPERSHARE",
    "TRANS_ACQUIRED_DISP_CD",
    "SHRS_OWND_FOLWNG_TRANS",
    "DIRECT_INDIRECT_OWNERSHIP",
]
OWNER_COLUMNS = [
    "ACCESSION_NUMBER",
    "RPTOWNERCIK",
    "RPTOWNERNAME",
    "RPTOWNER_RELATIONSHIP",
    "RPTOWNER_TITLE",
]


def _tsv(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(col, "")) for col in columns))
    return "\n".join(lines) + "\n"


def build_quarter_zip(
    *,
    submissions: list[dict[str, str]],
    transactions: list[dict[str, str]],
    owners: list[dict[str, str]],
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("SUBMISSION.tsv", _tsv(submissions, SUBMISSION_COLUMNS))
        zf.writestr("NONDERIV_TRANS.tsv", _tsv(transactions, TRANS_COLUMNS))
        zf.writestr("REPORTINGOWNER.tsv", _tsv(owners, OWNER_COLUMNS))
    return buffer.getvalue()


def submission(
    accession: str,
    *,
    filing_date: str = "17-JAN-2024",
    period: str = "15-JAN-2024",
    doc_type: str = "4",
    cik: str = "1000001",
    name: str = "Acme Corp",
    symbol: str = "AAA",
) -> dict[str, str]:
    return {
        "ACCESSION_NUMBER": accession,
        "FILING_DATE": filing_date,
        "PERIOD_OF_REPORT": period,
        "DOCUMENT_TYPE": doc_type,
        "ISSUERCIK": cik,
        "ISSUERNAME": name,
        "ISSUERTRADINGSYMBOL": symbol,
    }


def transaction(
    accession: str,
    sk: str,
    *,
    trans_date: str = "15-JAN-2024",
    code: str = "P",
    shares: str = "1000",
    price: str = "10.50",
    disp: str = "A",
    shares_after: str = "5000",
    ownership: str = "D",
    security_title: str = "Common Stock",
) -> dict[str, str]:
    return {
        "ACCESSION_NUMBER": accession,
        "NONDERIV_TRANS_SK": sk,
        "SECURITY_TITLE": security_title,
        "TRANS_DATE": trans_date,
        "TRANS_CODE": code,
        "TRANS_SHARES": shares,
        "TRANS_PRICEPERSHARE": price,
        "TRANS_ACQUIRED_DISP_CD": disp,
        "SHRS_OWND_FOLWNG_TRANS": shares_after,
        "DIRECT_INDIRECT_OWNERSHIP": ownership,
    }


def owner(
    accession: str,
    *,
    cik: str = "2000002",
    name: str = "Doe Jane",
    relationship: str = "Officer",
    title: str = "CFO",
) -> dict[str, str]:
    return {
        "ACCESSION_NUMBER": accession,
        "RPTOWNERCIK": cik,
        "RPTOWNERNAME": name,
        "RPTOWNER_RELATIONSHIP": relationship,
        "RPTOWNER_TITLE": title,
    }


def _quarter_handler(zips: dict[str, bytes]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for quarter, content in zips.items():
            if f"{quarter}_form345.zip" in url:
                return httpx.Response(200, content=content)
        return httpx.Response(404)

    return handler


def _seed_universe(
    config: Config,
    *,
    cik: str | None = None,
    ticker: str | None = None,
    via_filings_only: bool = False,
    via_price_only: bool = False,
) -> None:
    """Register one issuer as "in universe" via companies/filings/prices."""
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        if via_price_only:
            duckdb_store.upsert_daily_prices(
                con,
                [
                    {
                        "price_id": f"seed-{ticker}",
                        "symbol": ticker,
                        "price_date": date(2024, 1, 2),
                        "close": 10.0,
                        "adj_close": 10.0,
                        "open": 10.0,
                        "high": 10.0,
                        "low": 10.0,
                        "volume": 100,
                        "provider": "yfinance",
                        "validation_status": "valid",
                        "source": "market",
                    }
                ],
            )
            return

        company_id = f"company-{cik}"
        if via_filings_only:
            duckdb_store.upsert_filings(
                con,
                [
                    {
                        "filing_id": f"filing-{cik}",
                        "company_id": company_id,
                        "cik": cik,
                        "accession_number": "0000000000-24-000001",
                        "form": "10-K",
                        "validation_status": "valid",
                        "source": "sec",
                    }
                ],
            )
            return

        duckdb_store.upsert_companies(
            con,
            [
                {
                    "company_id": company_id,
                    "cik": cik,
                    "ticker": ticker,
                    "company_name": ticker,
                    "validation_status": "valid",
                    "source": "sec",
                }
            ],
        )


def _events(config: Config) -> list[dict[str, Any]]:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        rows = con.execute(
            """
            SELECT event_key, event_subtype, cik, ticker, direction, magnitude,
                   event_time, available_time, payload, company_id
            FROM events ORDER BY event_key
            """
        ).fetchall()
    columns = [
        "event_key", "event_subtype", "cik", "ticker", "direction", "magnitude",
        "event_time", "available_time", "payload", "company_id",
    ]  # fmt: skip
    return [dict(zip(columns, row, strict=True)) for row in rows]


CIK_A = normalize_cik("1000001")


def _standard_zip(accession: str = "0001000001-24-000001") -> bytes:
    return build_quarter_zip(
        submissions=[submission(accession, cik="1000001", symbol="AAA")],
        transactions=[transaction(accession, "SK-1")],
        owners=[owner(accession)],
    )


# --------------------------------------------------------------------------- #
# pure: quarters_between / quarter_for_date                                   #
# --------------------------------------------------------------------------- #
def test_quarters_between_spans_a_year_boundary() -> None:
    assert quarters_between("2016q3", "2017q2") == ["2016q3", "2016q4", "2017q1", "2017q2"]


def test_quarters_between_single_quarter() -> None:
    assert quarters_between("2020q1", "2020q1") == ["2020q1"]


def test_quarters_between_inverted_range_is_empty() -> None:
    assert quarters_between("2020q4", "2020q1") == []


def test_quarters_between_rejects_malformed_id() -> None:
    with pytest.raises(ValueError):
        quarters_between("not-a-quarter", "2020q1")


def test_quarter_for_date() -> None:
    assert quarter_for_date(date(2026, 7, 22)) == "2026q3"
    assert quarter_for_date(date(2026, 1, 1)) == "2026q1"
    assert quarter_for_date(date(2026, 12, 31)) == "2026q4"


# --------------------------------------------------------------------------- #
# pure: DD-MON-YYYY date parsing                                              #
# --------------------------------------------------------------------------- #
def test_parse_sec_date_valid() -> None:
    assert parse_sec_date("31-MAR-2025") == date(2025, 3, 31)
    assert parse_sec_date("01-JAN-2020") == date(2020, 1, 1)


def test_parse_sec_date_covers_every_month() -> None:
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    for index, name in enumerate(months, start=1):
        assert parse_sec_date(f"15-{name}-2024") == date(2024, index, 15)


def test_parse_sec_date_is_locale_independent() -> None:
    """SEC always publishes English months; the process locale must not matter.

    strptime's %b resolves month names through LC_TIME, so under a non-English
    locale every date in the dataset would return None, every record would be
    rejected for a missing clock, and the run would report success having
    stored nothing.
    """
    original = locale.setlocale(locale.LC_TIME)
    installed = None
    for candidate in ("de_DE.UTF-8", "fr_FR.UTF-8", "es_ES.UTF-8"):
        try:
            locale.setlocale(locale.LC_TIME, candidate)
            installed = candidate
            break
        except locale.Error:
            continue

    if installed is None:
        pytest.skip("no non-English locale available on this host")

    try:
        assert parse_sec_date("31-MAR-2025") == date(2025, 3, 31)
    finally:
        locale.setlocale(locale.LC_TIME, original)


def test_parse_sec_date_rejects_an_impossible_day() -> None:
    """A day out of range is unparseable, never silently clamped."""
    assert parse_sec_date("31-FEB-2025") is None
    assert parse_sec_date("00-JAN-2025") is None


@pytest.mark.parametrize("bad", ["", None, "2025-03-31", "31/03/2025", "not-a-date", "32-JAN-2025"])
def test_parse_sec_date_unparseable_returns_none(bad: str | None) -> None:
    assert parse_sec_date(bad) is None


def test_parse_sec_float() -> None:
    assert parse_sec_float("1000.5") == 1000.5
    assert parse_sec_float("") is None
    assert parse_sec_float(None) is None
    assert parse_sec_float("garbage") is None


# --------------------------------------------------------------------------- #
# pure: three-way TSV join                                                    #
# --------------------------------------------------------------------------- #
def test_parse_quarter_zip_joins_all_three_tsvs() -> None:
    accession = "0001000001-24-000001"
    zip_bytes = build_quarter_zip(
        submissions=[submission(accession, cik="1000001", name="Acme Corp", symbol="AAA")],
        transactions=[transaction(accession, "SK-1", security_title="Common Stock")],
        owners=[owner(accession, name="Doe Jane", relationship="Officer", title="CFO")],
    )

    rows = parse_quarter_zip(zip_bytes)
    assert len(rows) == 1
    row = rows[0]
    assert row.accession_number == accession
    assert row.issuer_cik == normalize_cik("1000001")
    assert row.issuer_name == "Acme Corp"
    assert row.issuer_trading_symbol == "AAA"
    assert row.filing_date == date(2024, 1, 17)
    assert row.trans_sk == "SK-1"
    assert row.security_title == "Common Stock"
    assert row.trans_date == date(2024, 1, 15)
    assert row.owner_name == "Doe Jane"
    assert row.owner_relationship == "Officer"
    assert row.owner_title == "CFO"
    assert row.owner_cik == normalize_cik("2000002")


def test_parse_quarter_zip_multiple_transactions_same_filing() -> None:
    accession = "0001000001-24-000001"
    zip_bytes = build_quarter_zip(
        submissions=[submission(accession)],
        transactions=[
            transaction(accession, "SK-1", code="P"),
            transaction(accession, "SK-2", code="A"),
        ],
        owners=[owner(accession)],
    )

    rows = parse_quarter_zip(zip_bytes)
    assert len(rows) == 2
    codes = {row.trans_sk: row.trans_code for row in rows}
    assert codes == {"SK-1": "P", "SK-2": "A"}


def test_parse_quarter_zip_drops_transaction_with_no_matching_submission() -> None:
    zip_bytes = build_quarter_zip(
        submissions=[submission("0001000001-24-000001")],
        transactions=[transaction("0001000001-24-999999", "SK-1")],  # different accession
        owners=[],
    )
    assert parse_quarter_zip(zip_bytes) == []


def test_parse_quarter_zip_missing_owner_row_yields_none_owner_fields() -> None:
    accession = "0001000001-24-000001"
    zip_bytes = build_quarter_zip(
        submissions=[submission(accession)],
        transactions=[transaction(accession, "SK-1")],
        owners=[],
    )
    rows = parse_quarter_zip(zip_bytes)
    assert len(rows) == 1
    assert rows[0].owner_name is None
    assert rows[0].owner_cik is None


def test_parse_quarter_zip_unparseable_filing_date_is_none_not_defaulted() -> None:
    accession = "0001000001-24-000001"
    zip_bytes = build_quarter_zip(
        submissions=[submission(accession, filing_date="NOT-A-DATE")],
        transactions=[transaction(accession, "SK-1")],
        owners=[owner(accession)],
    )
    rows = parse_quarter_zip(zip_bytes)
    assert len(rows) == 1
    assert rows[0].filing_date is None


def test_parse_quarter_zip_joint_filing_uses_first_owner() -> None:
    accession = "0001000001-24-000001"
    zip_bytes = build_quarter_zip(
        submissions=[submission(accession)],
        transactions=[transaction(accession, "SK-1")],
        owners=[
            owner(accession, name="First Owner"),
            owner(accession, name="Second Owner"),
        ],
    )
    rows = parse_quarter_zip(zip_bytes)
    assert rows[0].owner_name == "First Owner"


# --------------------------------------------------------------------------- #
# client: fetch + 404 handling                                                #
# --------------------------------------------------------------------------- #
def test_fetch_quarter_returns_raw_bytes(tmp_config: Config) -> None:
    zip_bytes = _standard_zip()

    def handler(request: httpx.Request) -> httpx.Response:
        assert "2024q1_form345.zip" in str(request.url)
        return httpx.Response(200, content=zip_bytes)

    with InsiderClient.from_config(tmp_config, transport=httpx.MockTransport(handler)) as client:
        result = client.fetch_quarter("2024q1")

    assert result.raw == zip_bytes
    assert result.quarter == "2024q1"


def test_fetch_quarter_404_raises_quarter_not_published(tmp_config: Config) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with (
        InsiderClient.from_config(tmp_config, transport=httpx.MockTransport(handler)) as client,
        pytest.raises(QuarterNotPublished),
    ):
        client.fetch_quarter("2026q4")


def test_fetch_quarter_non_404_error_propagates(tmp_config: Config) -> None:
    """A 500 is retried (it's in RETRYABLE_STATUS) and, once exhausted,
    surfaces as RetryableHTTPError -- never mistaken for QuarterNotPublished.
    """
    from market_intelligence.clients.base import RetryableHTTPError
    from market_intelligence.config import RetryConfig

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    retry_config = RetryConfig(
        max_attempts=2, initial_backoff_seconds=0.0, max_backoff_seconds=0.0, jitter_seconds=0.0
    )
    client = InsiderClient(
        user_agent="Test/1.0 (test@example.com)",
        base_url=tmp_config.settings.sec.base_url,
        retry_config=retry_config,
        transport=httpx.MockTransport(handler),
    )
    with client, pytest.raises(RetryableHTTPError):
        client.fetch_quarter("2024q1")


# --------------------------------------------------------------------------- #
# collector: happy path                                                       #
# --------------------------------------------------------------------------- #
def test_sync_persists_events_and_preserves_raw(tmp_config: Config) -> None:
    _seed_universe(tmp_config, cik=CIK_A, ticker="AAA")
    zip_bytes = _standard_zip()
    handler = _quarter_handler({"2024q1": zip_bytes})

    summary = insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q1",
        transport=httpx.MockTransport(handler),
    )

    assert summary.status == "success"
    assert summary.inserted == 1
    assert summary.rejected == 0

    rows = _events(tmp_config)
    assert len(rows) == 1
    assert rows[0]["event_key"] == "SK-1"
    assert rows[0]["event_subtype"] == "P"
    assert rows[0]["cik"] == CIK_A
    assert rows[0]["ticker"] == "AAA"

    raw_dir = tmp_config.paths.raw_dir / "sec" / "form345" / "2024q1"
    assert any(raw_dir.rglob("*.zip"))


def test_two_clocks_are_not_conflated(tmp_config: Config) -> None:
    """event_time is the trade date; available_time is the filing date -- distinct."""
    _seed_universe(tmp_config, cik=CIK_A, ticker="AAA")
    accession = "0001000001-24-000001"
    zip_bytes = build_quarter_zip(
        submissions=[submission(accession, filing_date="17-JAN-2024", cik="1000001")],
        transactions=[transaction(accession, "SK-1", trans_date="15-JAN-2024")],
        owners=[owner(accession)],
    )
    handler = _quarter_handler({"2024q1": zip_bytes})

    insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q1",
        transport=httpx.MockTransport(handler),
    )

    row = _events(tmp_config)[0]
    assert row["event_time"] == datetime(2024, 1, 15, tzinfo=UTC)
    assert row["available_time"] == datetime(2024, 1, 17, tzinfo=UTC)
    assert row["event_time"] != row["available_time"]


# --------------------------------------------------------------------------- #
# collector: transaction codes preserved distinctly                           #
# --------------------------------------------------------------------------- #
def test_purchase_and_grant_in_same_filing_produce_two_distinct_events(tmp_config: Config) -> None:
    _seed_universe(tmp_config, cik=CIK_A, ticker="AAA")
    accession = "0001000001-24-000001"
    zip_bytes = build_quarter_zip(
        submissions=[submission(accession, cik="1000001")],
        transactions=[
            transaction(accession, "SK-1", code="P", disp="A"),
            transaction(accession, "SK-2", code="A", disp="A"),
        ],
        owners=[owner(accession)],
    )
    handler = _quarter_handler({"2024q1": zip_bytes})

    summary = insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q1",
        transport=httpx.MockTransport(handler),
    )

    assert summary.inserted == 2
    rows = {r["event_key"]: r["event_subtype"] for r in _events(tmp_config)}
    assert rows == {"SK-1": "P", "SK-2": "A"}


# --------------------------------------------------------------------------- #
# collector: direction mapping                                                #
# --------------------------------------------------------------------------- #
def test_direction_mapping(tmp_config: Config) -> None:
    _seed_universe(tmp_config, cik=CIK_A, ticker="AAA")
    accession = "0001000001-24-000001"
    zip_bytes = build_quarter_zip(
        submissions=[submission(accession, cik="1000001")],
        transactions=[
            transaction(accession, "SK-A", disp="A"),
            transaction(accession, "SK-D", disp="D"),
            transaction(accession, "SK-OTHER", disp=""),
        ],
        owners=[owner(accession)],
    )
    handler = _quarter_handler({"2024q1": zip_bytes})

    insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q1",
        transport=httpx.MockTransport(handler),
    )

    direction_by_key = {r["event_key"]: r["direction"] for r in _events(tmp_config)}
    assert direction_by_key == {"SK-A": 1, "SK-D": -1, "SK-OTHER": 0}


# --------------------------------------------------------------------------- #
# collector: magnitude                                                        #
# --------------------------------------------------------------------------- #
def test_magnitude_is_none_when_price_absent(tmp_config: Config) -> None:
    _seed_universe(tmp_config, cik=CIK_A, ticker="AAA")
    accession = "0001000001-24-000001"
    zip_bytes = build_quarter_zip(
        submissions=[submission(accession, cik="1000001")],
        transactions=[
            transaction(accession, "SK-GRANT", price="", code="A"),  # grant, no price
            transaction(accession, "SK-BUY", price="10.00", shares="100", code="P"),
        ],
        owners=[owner(accession)],
    )
    handler = _quarter_handler({"2024q1": zip_bytes})

    insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q1",
        transport=httpx.MockTransport(handler),
    )

    magnitude_by_key = {r["event_key"]: r["magnitude"] for r in _events(tmp_config)}
    assert magnitude_by_key["SK-GRANT"] is None
    assert magnitude_by_key["SK-BUY"] == pytest.approx(1000.0)


# --------------------------------------------------------------------------- #
# collector: universe filtering                                               #
# --------------------------------------------------------------------------- #
def test_out_of_universe_rows_are_dropped_and_counted(tmp_config: Config) -> None:
    cik_in = normalize_cik("1000001")
    cik_filings_only = normalize_cik("1000002")

    _seed_universe(tmp_config, cik=cik_in, ticker="AAA")
    _seed_universe(tmp_config, cik=cik_filings_only, via_filings_only=True)
    _seed_universe(tmp_config, ticker="ZZZ", via_price_only=True)  # ticker-only fallback

    zip_bytes = build_quarter_zip(
        submissions=[
            submission("0001-24-000001", cik="1000001", symbol="AAA"),
            submission("0001-24-000002", cik="1000002", symbol="BBB"),
            submission(
                "0001-24-000003", cik="1000009", symbol="ZZZ"
            ),  # cik unknown, ticker matches
            submission("0001-24-000004", cik="9999999", symbol="NOPE"),  # not in universe at all
        ],
        transactions=[
            transaction("0001-24-000001", "SK-1"),
            transaction("0001-24-000002", "SK-2"),
            transaction("0001-24-000003", "SK-3"),
            transaction("0001-24-000004", "SK-4"),
        ],
        owners=[
            owner("0001-24-000001"),
            owner("0001-24-000002"),
            owner("0001-24-000003"),
            owner("0001-24-000004"),
        ],
    )
    handler = _quarter_handler({"2024q1": zip_bytes})

    summary = insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q1",
        transport=httpx.MockTransport(handler),
    )

    stored_keys = {r["event_key"] for r in _events(tmp_config)}
    assert stored_keys == {"SK-1", "SK-2", "SK-3"}  # SK-4 dropped
    assert summary.stage["out_of_universe_rows"] == 1


def test_tickers_argument_further_restricts_emitted_events(tmp_config: Config) -> None:
    cik_a = normalize_cik("1000001")
    cik_b = normalize_cik("1000002")
    _seed_universe(tmp_config, cik=cik_a, ticker="AAA")
    _seed_universe(tmp_config, cik=cik_b, ticker="BBB")

    zip_bytes = build_quarter_zip(
        submissions=[
            submission("0001-24-000001", cik="1000001", symbol="AAA"),
            submission("0001-24-000002", cik="1000002", symbol="BBB"),
        ],
        transactions=[
            transaction("0001-24-000001", "SK-1"),
            transaction("0001-24-000002", "SK-2"),
        ],
        owners=[owner("0001-24-000001"), owner("0001-24-000002")],
    )
    handler = _quarter_handler({"2024q1": zip_bytes})

    insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q1",
        tickers=["AAA"],
        transport=httpx.MockTransport(handler),
    )

    stored_keys = {r["event_key"] for r in _events(tmp_config)}
    assert stored_keys == {"SK-1"}


# --------------------------------------------------------------------------- #
# collector: unparseable date -> rejected, not stored                         #
# --------------------------------------------------------------------------- #
def test_unparseable_filing_date_is_rejected_not_stored(tmp_config: Config) -> None:
    _seed_universe(tmp_config, cik=CIK_A, ticker="AAA")
    accession = "0001000001-24-000001"
    zip_bytes = build_quarter_zip(
        submissions=[submission(accession, cik="1000001", filing_date="GARBAGE")],
        transactions=[transaction(accession, "SK-1")],
        owners=[owner(accession)],
    )
    handler = _quarter_handler({"2024q1": zip_bytes})

    summary = insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q1",
        transport=httpx.MockTransport(handler),
    )

    assert summary.collected == 1
    assert summary.rejected == 1
    assert summary.inserted == 0
    assert _events(tmp_config) == []


# --------------------------------------------------------------------------- #
# collector: quarter not yet published                                       #
# --------------------------------------------------------------------------- #
def test_quarter_not_published_is_logged_and_run_continues(tmp_config: Config) -> None:
    _seed_universe(tmp_config, cik=CIK_A, ticker="AAA")
    zip_bytes = _standard_zip()
    # Only 2024q1 is "published"; 2024q2 404s.
    handler = _quarter_handler({"2024q1": zip_bytes})

    summary = insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q2",
        transport=httpx.MockTransport(handler),
    )

    assert summary.status == "success"
    assert summary.inserted == 1
    assert any("not_published:2024q2" in note for note in summary.notes)
    assert summary.stage["quarter_not_published"] == 1


# --------------------------------------------------------------------------- #
# collector: resumability (skip an already-collected quarter)                 #
# --------------------------------------------------------------------------- #
def test_already_collected_quarter_is_skipped_without_a_second_fetch(tmp_config: Config) -> None:
    _seed_universe(tmp_config, cik=CIK_A, ticker="AAA")
    zip_bytes = _standard_zip()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=zip_bytes)

    insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q1",
        transport=httpx.MockTransport(handler),
    )
    assert len(calls) == 1

    second = insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q1",
        transport=httpx.MockTransport(handler),
    )
    assert len(calls) == 1  # no new network call
    assert any("already_collected:2024q1" in note for note in second.notes)
    assert len(_events(tmp_config)) == 1


# --------------------------------------------------------------------------- #
# collector: idempotency (force re-collection -> updates, not inserts)        #
# --------------------------------------------------------------------------- #
def test_forced_recollection_updates_rather_than_duplicates(tmp_config: Config) -> None:
    _seed_universe(tmp_config, cik=CIK_A, ticker="AAA")
    zip_bytes = _standard_zip()
    handler = _quarter_handler({"2024q1": zip_bytes})

    first = insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q1",
        transport=httpx.MockTransport(handler),
    )
    assert (first.inserted, first.updated) == (1, 0)

    second = insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q1",
        force=True,
        transport=httpx.MockTransport(handler),
    )
    assert (second.inserted, second.updated) == (0, 1)

    assert len(_events(tmp_config)) == 1  # row count unchanged


# --------------------------------------------------------------------------- #
# collector: durability across a crash                                        #
# --------------------------------------------------------------------------- #
def test_earlier_quarters_survive_a_later_quarter_crashing(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run killed while processing quarter 2 must not lose quarter 1's events.

    Modeled on test_earlier_symbols_survive_a_later_symbol_crashing in
    test_prices.py: the failure is injected at the DuckDB write, after
    quarter 1 has already been handed to storage, so it exercises per-quarter
    flushing rather than a whole-run buffer.
    """

    class Boom(RuntimeError):
        pass

    _seed_universe(tmp_config, cik=CIK_A, ticker="AAA")
    zips = {
        "2024q1": build_quarter_zip(
            submissions=[submission("0001-24-000001", cik="1000001")],
            transactions=[transaction("0001-24-000001", "SK-Q1")],
            owners=[owner("0001-24-000001")],
        ),
        "2024q2": build_quarter_zip(
            submissions=[submission("0001-24-000002", cik="1000001")],
            transactions=[transaction("0001-24-000002", "SK-Q2")],
            owners=[owner("0001-24-000002")],
        ),
    }
    handler = _quarter_handler(zips)

    real_upsert = duckdb_store.upsert_events
    calls = {"n": 0}

    def exploding_upsert(con: Any, rows: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise Boom("killed mid-run")
        return real_upsert(con, rows)

    monkeypatch.setattr(insider_collector.duckdb_store, "upsert_events", exploding_upsert)

    with pytest.raises(Boom):
        insider_collector.sync(
            tmp_config,
            start_quarter="2024q1",
            end_quarter="2024q2",
            transport=httpx.MockTransport(handler),
        )

    rows = _events(tmp_config)
    assert len(rows) == 1
    assert rows[0]["event_key"] == "SK-Q1", "quarter 1's event collected before the crash was lost"


# --------------------------------------------------------------------------- #
# collector: corrupt ZIP for one quarter does not sink the run                #
# --------------------------------------------------------------------------- #
def test_corrupt_zip_is_counted_and_run_continues(tmp_config: Config) -> None:
    _seed_universe(tmp_config, cik=CIK_A, ticker="AAA")
    good_zip = _standard_zip("0001000001-24-000002")
    zips_raw = {"2024q1": b"not a zip file", "2024q2": good_zip}
    handler = _quarter_handler(zips_raw)

    summary = insider_collector.sync(
        tmp_config,
        start_quarter="2024q1",
        end_quarter="2024q2",
        transport=httpx.MockTransport(handler),
    )

    assert summary.status == "success"
    assert summary.stage["quarter_parse_failed"] == 1
    assert summary.inserted == 1
    assert len(_events(tmp_config)) == 1
