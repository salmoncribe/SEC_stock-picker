"""Offline tests for the FRED vertical: parsing, validation, client, upserts."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import duckdb
import httpx

from market_intelligence import database, hashing
from market_intelligence.clients.fred import FREDClient
from market_intelligence.config import FredConfig, RetryConfig
from market_intelligence.schemas.fred import (
    ObservationRecord,
    parse_observations,
    parse_series_metadata,
    parse_value,
)
from market_intelligence.storage.duckdb import upsert_observations
from market_intelligence.validators.fred import validate_observation

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _fred_config() -> FredConfig:
    return FredConfig(
        base_url="https://api.stlouisfed.org",
        series_observations_path="/fred/series/observations",
        series_path="/fred/series",
        file_type="json",
    )


def test_parse_value_handles_missing_and_numeric() -> None:
    assert parse_value(".") is None
    assert parse_value("3.14") == 3.14
    assert parse_value(" 2.5 ") == 2.5
    assert parse_value("") is None
    assert parse_value(None) is None


def test_parse_observations_flattens_payload() -> None:
    rows = parse_observations(_load("fred_observations_dgs10.json"))

    assert len(rows) == 2
    assert rows[0]["value"] == 4.45
    assert rows[1]["value"] is None  # FRED "." gap marker
    assert isinstance(rows[0]["observation_date"], date)
    assert rows[0]["observation_date"] == date(2026, 7, 14)
    assert isinstance(rows[0]["realtime_start"], date)


def test_parse_series_metadata_returns_first_series() -> None:
    meta = parse_series_metadata(_load("fred_series_dgs10.json"))

    assert isinstance(meta, dict)
    assert meta["id"] == "DGS10"


def test_parse_series_metadata_empty_payload() -> None:
    assert parse_series_metadata({}) == {}
    assert parse_series_metadata({"seriess": []}) == {}


def test_validate_observation_missing_value_is_warning() -> None:
    record = ObservationRecord(
        observation_id="obs-1",
        series_id="DGS10",
        observation_date=date(2026, 7, 15),
        value=None,
    )

    validate_observation(record)

    assert record.validation_status == "warning"
    assert "missing_value" in record.validation_errors
    assert not record.is_rejected


def test_upsert_observations_is_idempotent() -> None:
    con = duckdb.connect(":memory:")
    database.init_db(con)
    try:
        rows = []
        for obs in parse_observations(_load("fred_observations_dgs10.json")):
            observation_date = obs["observation_date"]
            record = ObservationRecord(
                observation_id=hashing.content_hash(
                    "obs",
                    "DGS10",
                    observation_date.isoformat(),
                    obs["realtime_start"],
                    obs["realtime_end"],
                ),
                series_id="DGS10",
                observation_date=observation_date,
                value=obs["value"],
                realtime_start=obs["realtime_start"],
                realtime_end=obs["realtime_end"],
                content_hash=hashing.content_hash(
                    "DGS10", observation_date.isoformat(), obs["value"]
                ),
            )
            validate_observation(record)
            rows.append(record.to_row())

        first = upsert_observations(con, rows)
        assert first.inserted == 2
        assert first.updated == 0

        second = upsert_observations(con, rows)
        assert second.inserted == 0
        assert second.updated == 2

        result = con.execute("SELECT count(*) FROM economic_observations").fetchone()
        assert result is not None
        assert result[0] == 2
    finally:
        con.close()


def test_fetch_observations_parses_and_sends_auth_params() -> None:
    payload = _load("fred_observations_dgs10.json")
    captured: dict[str, httpx.URL] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = request.url
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = FREDClient(api_key="test_key", fred_config=_fred_config(), transport=transport)
    with client:
        result = client.fetch_observations("DGS10", observation_start="2015-01-01")
        rows = parse_observations(result.data)

    assert len(rows) == 2
    params = captured["url"].params
    assert params["api_key"] == "test_key"
    assert params["file_type"] == "json"
    assert params["series_id"] == "DGS10"
    assert params["observation_start"] == "2015-01-01"


def test_request_retries_on_5xx_then_succeeds() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"seriess": []})

    transport = httpx.MockTransport(handler)
    retry_config = RetryConfig(
        max_attempts=3,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
        jitter_seconds=0.0,
    )
    client = FREDClient(
        api_key="test_key",
        fred_config=_fred_config(),
        retry_config=retry_config,
        transport=transport,
    )
    with client:
        result = client.fetch_series("DGS10")

    assert calls["count"] == 2
    assert result.data == {"seriess": []}
