"""Offline tests for the SEC EDGAR vertical: parsing, client, and validation."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import duckdb
import httpx
import pytest

from market_intelligence import database
from market_intelligence.clients.sec import FetchResult, SECClient
from market_intelligence.config import RetryConfig, SecConfig
from market_intelligence.schemas.sec import (
    FilingRecord,
    build_filing_url,
    filter_by_forms,
    is_valid_accession,
    iter_submission_filings,
    normalize_cik,
    parse_ticker_map,
)
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.validators.sec import validate_filing

FIXTURES = Path(__file__).parent / "fixtures"

SEC_CONFIG = SecConfig(
    base_url="https://www.sec.gov",
    data_base_url="https://data.sec.gov",
    company_tickers_path="/files/company_tickers.json",
    submissions_path_template="/submissions/CIK{cik10}.json",
    company_facts_path_template="/api/xbrl/companyfacts/CIK{cik10}.json",
)

USER_AGENT = "MarketIntelligenceTest/1.0 (test@example.com)"


def _load_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _load_json(name: str) -> dict:
    return json.loads(_load_bytes(name))


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    retry_config: RetryConfig | None = None,
) -> SECClient:
    return SECClient(
        user_agent=USER_AGENT,
        sec_config=SEC_CONFIG,
        retry_config=retry_config,
        transport=httpx.MockTransport(handler),
    )


def _fixture_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "company_tickers.json" in url:
        return httpx.Response(200, content=_load_bytes("sec_company_tickers.json"))
    if "/submissions/CIK" in url:
        return httpx.Response(200, content=_load_bytes("sec_submissions_nvda.json"))
    return httpx.Response(404)


# --------------------------------------------------------------------------- #
# 1. normalize_cik                                                             #
# --------------------------------------------------------------------------- #
def test_normalize_cik_valid_inputs() -> None:
    assert normalize_cik("320193") == "0000320193"
    assert normalize_cik(320193) == "0000320193"
    assert normalize_cik("0000320193") == "0000320193"
    assert normalize_cik("CIK0000320193") == "0000320193"


@pytest.mark.parametrize("bad", ["", "   ", "abc", "12345678901", "CIK", "12a45"])
def test_normalize_cik_invalid_inputs(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_cik(bad)


# --------------------------------------------------------------------------- #
# 2. parse_ticker_map                                                          #
# --------------------------------------------------------------------------- #
def test_parse_ticker_map() -> None:
    rows = parse_ticker_map(_load_json("sec_company_tickers.json"))
    by_ticker = {row["ticker"]: row for row in rows}
    assert by_ticker["NVDA"]["cik"] == "0001045810"
    assert by_ticker["AAPL"]["cik"] == "0000320193"
    assert by_ticker["NVDA"]["title"] == "NVIDIA CORP"
    assert all(len(row["cik"]) == 10 for row in rows)


def test_parse_ticker_map_accepts_list_shape() -> None:
    rows = parse_ticker_map(list(_load_json("sec_company_tickers.json").values()))
    assert {row["ticker"] for row in rows} == {"NVDA", "AAPL", "MSFT"}


# --------------------------------------------------------------------------- #
# 3. iter_submission_filings                                                   #
# --------------------------------------------------------------------------- #
def test_iter_submission_filings() -> None:
    filings = iter_submission_filings(_load_json("sec_submissions_nvda.json"))
    assert len(filings) == 4
    first = filings[0]
    assert first["accession_number"] == "0001045810-24-000029"
    assert first["form"] == "10-K"
    assert first["filing_date"] == "2024-02-21"
    assert first["report_date"] == "2024-01-28"
    assert first["primary_document"] == "nvda-20240128.htm"
    # Missing reportDate stays as the source's empty string (parsed later).
    assert filings[2]["report_date"] == ""


def test_iter_submission_filings_robust_to_missing() -> None:
    assert iter_submission_filings({}) == []
    assert iter_submission_filings({"filings": {"recent": {}}}) == []


# --------------------------------------------------------------------------- #
# 4. filter_by_forms                                                           #
# --------------------------------------------------------------------------- #
def test_filter_by_forms() -> None:
    filings = iter_submission_filings(_load_json("sec_submissions_nvda.json"))
    tens = filter_by_forms(filings, ["10-K", "10-Q"])
    assert {f["form"] for f in tens} == {"10-K", "10-Q"}
    assert len(tens) == 2
    assert filter_by_forms(filings, None) == filings
    assert filter_by_forms(filings, []) == []


# --------------------------------------------------------------------------- #
# 5. duplicate accession handling (upsert idempotency)                         #
# --------------------------------------------------------------------------- #
def _sample_filing_rows() -> list[dict]:
    filings = iter_submission_filings(_load_json("sec_submissions_nvda.json"))
    rows: list[dict] = []
    for filing in filings:
        accession = filing["accession_number"]
        if not is_valid_accession(accession):
            continue
        record = FilingRecord(
            filing_id=accession,
            company_id="company-1",
            cik="0001045810",
            accession_number=accession,
            form=filing["form"],
            filing_date=date.fromisoformat(filing["filing_date"]),
            filing_url=build_filing_url("0001045810", accession, filing["primary_document"]),
            source_url="https://www.sec.gov/",
        )
        row = record.to_row()
        row["raw_file_path"] = "/tmp/raw.json"
        rows.append(row)
    return rows


def test_upsert_filings_is_idempotent() -> None:
    con = duckdb.connect(":memory:")
    database.init_db(con)
    rows = _sample_filing_rows()
    assert len(rows) == 3

    first = duckdb_store.upsert_filings(con, rows)
    assert (first.inserted, first.updated) == (len(rows), 0)

    second = duckdb_store.upsert_filings(con, rows)
    assert (second.inserted, second.updated) == (0, len(rows))

    count = con.execute("SELECT count(*) FROM filings").fetchone()[0]
    assert count == len(rows)


# --------------------------------------------------------------------------- #
# 6. SECClient fetches + User-Agent header                                     #
# --------------------------------------------------------------------------- #
def test_fetch_company_tickers_and_submissions() -> None:
    seen_agents: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_agents.append(request.headers.get("User-Agent"))
        return _fixture_handler(request)

    with _make_client(handler) as client:
        tickers = client.fetch_company_tickers()
        submissions = client.fetch_submissions("1045810")

    assert isinstance(tickers, FetchResult)
    assert {row["ticker"] for row in parse_ticker_map(tickers.data)} == {"NVDA", "AAPL", "MSFT"}
    assert submissions.data["name"] == "NVIDIA CORP"
    assert "CIK0001045810.json" in submissions.url
    assert seen_agents == [USER_AGENT, USER_AGENT]


# --------------------------------------------------------------------------- #
# 7. retry on 503                                                              #
# --------------------------------------------------------------------------- #
def test_request_retries_on_503() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=_load_bytes("sec_company_tickers.json"))

    retry_config = RetryConfig(
        max_attempts=3,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
        jitter_seconds=0.0,
    )
    with _make_client(handler, retry_config=retry_config) as client:
        result = client.fetch_company_tickers()

    assert calls["n"] == 3
    assert parse_ticker_map(result.data)


# --------------------------------------------------------------------------- #
# 8. validate_filing flags bad accession + unsupported form                    #
# --------------------------------------------------------------------------- #
def test_validate_filing_rejects_bad_accession() -> None:
    supported = {"10-K", "10-Q", "8-K"}
    record = FilingRecord(
        filing_id="f-bad",
        company_id="company-1",
        cik="0001045810",
        accession_number="BAD-ACC",
        form="8-K",
        filing_date=date(2024, 9, 3),
        filing_url="https://www.sec.gov/x",
        source_url="https://www.sec.gov/x",
    )
    validate_filing(record, supported)
    assert record.is_rejected
    assert any("invalid_accession" in err for err in record.validation_errors)


def test_validate_filing_warns_on_unsupported_form() -> None:
    supported = {"10-K", "10-Q", "8-K"}
    record = FilingRecord(
        filing_id="f-warn",
        company_id="company-1",
        cik="0001045810",
        accession_number="0001045810-24-000029",
        form="SC 13D",
        filing_date=date(2024, 2, 21),
        filing_url="https://www.sec.gov/x",
        source_url="https://www.sec.gov/x",
    )
    validate_filing(record, supported)
    assert not record.is_rejected
    assert record.validation_status == "warning"
    assert any("unsupported_form:SC 13D" in err for err in record.validation_errors)
