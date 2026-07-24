"""Morning gap scan: offline end-to-end and failure-mode coverage.

Everything here runs against ``tmp_config`` (a file-backed throwaway DuckDB)
with a purpose-built stub ``MarketDataProvider`` -- nothing touches the
network, the real yfinance provider, or the real database. Config defaults
(``trading``/``gap_scanner``) come straight from ``config/settings.yaml``:
``min_gap_pct=3.0``, ``atr_period=14``, ``atr_stop_multiple=2.0``,
``gap_target_r_multiple=2.0``, ``min_confidence=60``.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import duckdb
import pytest

from market_intelligence import database
from market_intelligence.autopilot import notify
from market_intelligence.clients.market import LatestPrice, MarketDataProvider
from market_intelligence.collectors import gaps
from market_intelligence.config import Config
from market_intelligence.signals.trade_plan import Bar, wilder_atr
from market_intelligence.storage import duckdb as duckdb_store

AS_OF = date(2026, 7, 20)


# --------------------------------------------------------------------------- #
# stub provider                                                                #
# --------------------------------------------------------------------------- #
class _StubProvider(MarketDataProvider):
    """Canned quotes per symbol; optionally raises for a chosen set of symbols."""

    def __init__(
        self, quotes: dict[str, LatestPrice], *, raise_for: set[str] | None = None
    ) -> None:
        self._quotes = quotes
        self._raise_for = raise_for or set()

    def get_daily_prices(self, symbol: str, start: date, end: date) -> list[Any]:
        raise NotImplementedError("gap scan never calls get_daily_prices")

    def get_latest_price(self, symbol: str) -> LatestPrice:
        if symbol in self._raise_for:
            raise RuntimeError(f"stub provider error for {symbol}")
        return self._quotes[symbol]

    def get_corporate_actions(self, symbol: str, start: date, end: date) -> list[Any]:
        raise NotImplementedError("gap scan never calls get_corporate_actions")


# --------------------------------------------------------------------------- #
# seeding helpers                                                              #
# --------------------------------------------------------------------------- #
def _seed_constituents(config: Config, tickers: list[str]) -> None:
    rows = [
        {
            "constituent_id": f"idx-{t}",
            "index_id": "SP500",
            "ticker": t,
            "added_date": date(2020, 1, 1),
            "removed_date": None,
        }
        for t in tickers
    ]
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_constituents(con, rows)


def _price_row(
    symbol: str, price_date: date, close: float, volume: int = 5_000_000
) -> dict[str, Any]:
    return {
        "price_id": f"{symbol}-{price_date.isoformat()}",
        "symbol": symbol,
        "price_date": price_date,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "adj_close": close,
        "volume": volume,
        "provider": "yfinance",
        "validation_status": "valid",
        "source": "market",
    }


def _seed_prices(
    config: Config,
    symbol: str,
    *,
    end: date,
    n: int,
    start_close: float = 100.0,
    volume: int = 5_000_000,
) -> list[dict[str, Any]]:
    """``n`` bars for ``symbol`` ending on ``end`` (oldest first). Returns the rows."""
    rows = []
    close = start_close
    first_day = end - timedelta(days=n - 1)
    for i in range(n):
        close += 0.4 if i % 2 == 0 else -0.3
        rows.append(_price_row(symbol, first_day + timedelta(days=i), close, volume=volume))
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_daily_prices(con, rows)
    return rows


def _seed_event(
    config: Config,
    *,
    ticker: str,
    event_id: str,
    available: datetime,
    event_type: str = "insider_transaction",
    subtype: str = "P",
) -> None:
    row = {
        "event_id": event_id,
        "event_type": event_type,
        "event_key": f"{ticker}:{subtype}:{available.date()}:{event_id}",
        "event_subtype": subtype,
        "ticker": ticker,
        "available_time": available,
    }
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_events(con, [row])


def _trade_alert_rows(config: Config) -> list[dict[str, Any]]:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        cols = [d[0] for d in con.execute("SELECT * FROM trade_alerts LIMIT 0").description]
        rows = con.execute("SELECT * FROM trade_alerts").fetchall()
    return [dict(zip(cols, row, strict=True)) for row in rows]


def _bars_from_rows(rows: list[dict[str, Any]]) -> list[Bar]:
    return [
        Bar(
            date=r["price_date"],
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            adj_close=r["adj_close"],
        )
        for r in rows
    ]


# --------------------------------------------------------------------------- #
# tests                                                                        #
# --------------------------------------------------------------------------- #
def test_end_to_end_one_gapper_persists_and_dedups_on_rerun(tmp_config: Config) -> None:
    trading = tmp_config.settings.trading
    gap_cfg = tmp_config.settings.gap_scanner
    n = trading.atr_period * 4
    end = AS_OF - timedelta(days=1)

    _seed_constituents(tmp_config, ["GAPX", "FLAT"])
    gap_rows = _seed_prices(tmp_config, "GAPX", end=end, n=n)
    flat_rows = _seed_prices(tmp_config, "FLAT", end=end, n=n, start_close=50.0)

    gap_prior_close = gap_rows[-1]["close"]
    flat_prior_close = flat_rows[-1]["close"]
    gap_quote = gap_prior_close * 1.05  # +5% gap, well past the 3% floor

    provider = _StubProvider(
        {
            "GAPX": LatestPrice(symbol="GAPX", price=gap_quote, as_of=AS_OF),
            "FLAT": LatestPrice(symbol="FLAT", price=flat_prior_close, as_of=AS_OF),
        }
    )

    summary = gaps.scan(tmp_config, provider=provider, as_of=AS_OF, notify=False)
    assert summary.status == "success"
    assert summary.pipeline_name == "gaps.scan"

    rows = _trade_alert_rows(tmp_config)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "price_gap"
    assert row["ticker"] == "GAPX"
    assert row["trigger_key"] == AS_OF.isoformat()
    assert row["confidence"] <= 50

    atr = wilder_atr(_bars_from_rows(gap_rows), trading.atr_period)
    assert atr is not None
    stop_distance = trading.atr_stop_multiple * atr
    expected_target = gap_quote + gap_cfg.gap_target_r_multiple * stop_distance
    assert row["target"] == pytest.approx(expected_target)
    assert row["entry_ref"] == pytest.approx(gap_quote)
    assert row["direction"] == 1

    evidence_text = json.dumps(json.loads(row["evidence"]))
    assert "no track record yet" in evidence_text

    # Re-running the scan for the same day must not double-fire the alert.
    summary_again = gaps.scan(tmp_config, provider=provider, as_of=AS_OF, notify=False)
    assert summary_again.status == "success"
    rows_again = _trade_alert_rows(tmp_config)
    assert len(rows_again) == 1
    assert rows_again[0]["alert_id"] == row["alert_id"]


def test_quote_equal_to_prior_close_is_the_weekend_holiday_noop(tmp_config: Config) -> None:
    n = tmp_config.settings.trading.atr_period * 4
    end = AS_OF - timedelta(days=1)
    _seed_constituents(tmp_config, ["FLAT"])
    rows = _seed_prices(tmp_config, "FLAT", end=end, n=n)
    prior_close = rows[-1]["close"]

    provider = _StubProvider(
        {"FLAT": LatestPrice(symbol="FLAT", price=prior_close, as_of=AS_OF)}
    )

    summary = gaps.scan(tmp_config, provider=provider, as_of=AS_OF, notify=False)

    assert _trade_alert_rows(tmp_config) == []
    assert summary.stage.get("gap_scan_below_threshold", 0) == 1


def test_illiquid_gapper_is_dropped(tmp_config: Config) -> None:
    n = tmp_config.settings.trading.atr_period * 4
    end = AS_OF - timedelta(days=1)
    _seed_constituents(tmp_config, ["THIN"])
    rows = _seed_prices(tmp_config, "THIN", end=end, n=n, volume=100)
    prior_close = rows[-1]["close"]
    quote = prior_close * 1.10  # a big gap, but volume is far below the floor

    provider = _StubProvider({"THIN": LatestPrice(symbol="THIN", price=quote, as_of=AS_OF)})

    summary = gaps.scan(tmp_config, provider=provider, as_of=AS_OF, notify=False)

    assert _trade_alert_rows(tmp_config) == []
    assert summary.stage.get("gap_scan_illiquid", 0) == 1


def test_require_catalyst_gates_until_an_event_is_seeded(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_config.settings.gap_scanner.require_catalyst = True
    n = tmp_config.settings.trading.atr_period * 4
    end = AS_OF - timedelta(days=1)
    _seed_constituents(tmp_config, ["CATX"])
    rows = _seed_prices(tmp_config, "CATX", end=end, n=n)
    prior_close = rows[-1]["close"]
    quote = prior_close * 1.05
    provider = _StubProvider({"CATX": LatestPrice(symbol="CATX", price=quote, as_of=AS_OF)})

    # No catalyst yet: the gap is dropped outright, never persisted.
    summary = gaps.scan(tmp_config, provider=provider, as_of=AS_OF, notify=False)
    assert _trade_alert_rows(tmp_config) == []
    assert summary.stage.get("gap_scan_no_catalyst", 0) == 1

    # Spy on confidence.score to confirm times_asserted reflects the catalyst.
    captured: dict[str, Any] = {}
    original_score = gaps.score

    def spy_score(inputs: Any) -> int:
        captured["inputs"] = inputs
        return original_score(inputs)

    monkeypatch.setattr(gaps, "score", spy_score)

    _seed_event(
        tmp_config,
        ticker="CATX",
        event_id="ev-catx",
        available=datetime.combine(AS_OF - timedelta(days=2), datetime.min.time(), tzinfo=UTC),
    )

    summary2 = gaps.scan(tmp_config, provider=provider, as_of=AS_OF, notify=False)
    assert summary2.status == "success"

    rows_out = _trade_alert_rows(tmp_config)
    assert len(rows_out) == 1
    assert rows_out[0]["event_id"] == "ev-catx"

    evidence = json.loads(rows_out[0]["evidence"])
    assert evidence["catalyst"]["event_id"] == "ev-catx"

    assert "inputs" in captured
    assert captured["inputs"].times_asserted == 1
    assert captured["inputs"].has_track_record is False


def test_provider_error_on_one_ticker_does_not_block_the_rest(tmp_config: Config) -> None:
    n = tmp_config.settings.trading.atr_period * 4
    end = AS_OF - timedelta(days=1)
    _seed_constituents(tmp_config, ["BADX", "GOODX"])
    _seed_prices(tmp_config, "BADX", end=end, n=n)
    good_rows = _seed_prices(tmp_config, "GOODX", end=end, n=n, start_close=80.0)
    good_quote = good_rows[-1]["close"] * 1.05

    provider = _StubProvider(
        {"GOODX": LatestPrice(symbol="GOODX", price=good_quote, as_of=AS_OF)},
        raise_for={"BADX"},
    )

    summary = gaps.scan(tmp_config, provider=provider, as_of=AS_OF, notify=False)

    assert summary.status == "success"
    rows = _trade_alert_rows(tmp_config)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "GOODX"
    assert summary.stage.get("gap_scan_provider_error", 0) == 1


def test_db_busy_retries_three_times_then_skips_without_raising(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []

    def boom(*_args: object, **_kwargs: object) -> object:
        attempts.append(1)
        raise duckdb.IOException("could not set lock on file")

    monkeypatch.setattr(gaps, "pipeline_run", boom)

    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    summary = gaps.scan(
        tmp_config,
        provider=_StubProvider({}),
        as_of=AS_OF,
        notify=False,
        sleep=fake_sleep,
    )

    assert summary.status == "skipped"
    assert summary.pipeline_name == "gaps.scan"
    assert len(attempts) == 3
    assert sleep_calls == [30.0, 30.0]


def test_notify_false_never_calls_send(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the record to be sendable regardless of the no-track-record cap,
    # so this test actually exercises the notify=False guard rather than
    # relying on gating to make send() unreachable anyway.
    tmp_config.settings.trading.min_confidence = 0
    n = tmp_config.settings.trading.atr_period * 4
    end = AS_OF - timedelta(days=1)
    _seed_constituents(tmp_config, ["GAPX"])
    rows = _seed_prices(tmp_config, "GAPX", end=end, n=n)
    quote = rows[-1]["close"] * 1.05
    provider = _StubProvider({"GAPX": LatestPrice(symbol="GAPX", price=quote, as_of=AS_OF)})

    def fail_if_called(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("send_trade_alerts must not be called when notify=False")

    monkeypatch.setattr(notify, "send_trade_alerts", fail_if_called)

    summary = gaps.scan(tmp_config, provider=provider, as_of=AS_OF, notify=False)

    assert summary.status == "success"
    rows_out = _trade_alert_rows(tmp_config)
    assert len(rows_out) == 1
    assert rows_out[0]["delivered"] is False
    assert rows_out[0]["delivery_note"] is None
