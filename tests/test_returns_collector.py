"""Offline tests for the returns collector.

This collector reads only from DuckDB -- no network, no yfinance. Prices are
seeded directly via ``storage.duckdb.upsert_daily_prices``.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from market_intelligence import database
from market_intelligence.analytics.returns import AbnormalReturnMethod, ReturnPoint
from market_intelligence.collectors import returns as returns_collector
from market_intelligence.config import Config
from market_intelligence.storage import duckdb as duckdb_store

START = date(2024, 1, 2)


def _price_row(symbol: str, price_date: date, close: float) -> dict[str, Any]:
    return {
        "price_id": f"{symbol}-{price_date.isoformat()}",
        "symbol": symbol,
        "price_date": price_date,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "adj_close": close,
        "volume": 1000,
        "provider": "yfinance",
        "validation_status": "valid",
        "source": "market",
    }


def _seed_series(config: Config, symbol: str, closes: list[float]) -> None:
    rows = [_price_row(symbol, START + timedelta(days=i), close) for i, close in enumerate(closes)]
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_daily_prices(con, rows)


def _returns(config: Config) -> list[tuple]:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        return con.execute(
            "SELECT symbol, price_date, abnormal_return, validation_status "
            "FROM daily_returns ORDER BY symbol, price_date"
        ).fetchall()


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_compute_persists_abnormal_returns(tmp_config: Config) -> None:
    _seed_series(tmp_config, "AAA", [100, 102, 101, 105, 104])
    _seed_series(tmp_config, "SPY", [100, 101, 100, 103, 102])

    summary = returns_collector.compute(
        tmp_config, symbols=["AAA"], method=AbnormalReturnMethod.MARKET_ADJUSTED
    )

    assert summary.status == "success"
    assert summary.rejected == 0

    rows = _returns(tmp_config)
    aaa_rows = [r for r in rows if r[0] == "AAA"]
    assert len(aaa_rows) == 4  # 5 prices -> 4 daily returns
    assert all(r[2] is not None for r in aaa_rows)  # abnormal_return populated
    assert all(r[3] == "valid" for r in aaa_rows)


def test_benchmark_symbol_is_always_included(tmp_config: Config) -> None:
    _seed_series(tmp_config, "AAA", [100, 102])
    _seed_series(tmp_config, "SPY", [100, 101])

    returns_collector.compute(
        tmp_config, symbols=["AAA"], method=AbnormalReturnMethod.MARKET_ADJUSTED
    )

    rows = _returns(tmp_config)
    assert {r[0] for r in rows} == {"AAA", "SPY"}


def test_symbol_with_no_price_history_is_skipped_not_crashed(tmp_config: Config) -> None:
    _seed_series(tmp_config, "SPY", [100, 101, 102])

    summary = returns_collector.compute(
        tmp_config, symbols=["GHOST"], method=AbnormalReturnMethod.MARKET_ADJUSTED
    )

    assert summary.status == "success"
    assert any("GHOST" in note for note in summary.notes)
    rows = _returns(tmp_config)
    assert "GHOST" not in {r[0] for r in rows}


# --------------------------------------------------------------------------- #
# idempotency
# --------------------------------------------------------------------------- #
def test_rerun_is_idempotent(tmp_config: Config) -> None:
    _seed_series(tmp_config, "AAA", [100, 102, 101])
    _seed_series(tmp_config, "SPY", [100, 101, 100])

    first = returns_collector.compute(
        tmp_config, symbols=["AAA"], method=AbnormalReturnMethod.MARKET_ADJUSTED
    )
    second = returns_collector.compute(
        tmp_config, symbols=["AAA"], method=AbnormalReturnMethod.MARKET_ADJUSTED
    )

    assert first.inserted > 0
    assert second.inserted == 0
    assert second.updated == first.inserted
    assert len(_returns(tmp_config)) == first.inserted


# --------------------------------------------------------------------------- #
# lookahead
# --------------------------------------------------------------------------- #
class _RecordingLogger:
    """Stands in for the module's structlog logger so a test can assert on
    what got logged without depending on structlog's own configuration.
    """

    def __init__(self) -> None:
        self.errors: list[tuple[str, dict[str, Any]]] = []

    def error(self, event: str, **kwargs: Any) -> None:
        self.errors.append((event, kwargs))


def test_lookahead_row_is_rejected_and_logged_loudly(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``compute_abnormal_returns`` is contracted to never produce this; if it
    ever does, that is a bug in this collector's wiring, not bad input data.
    The row must be loudly rejected, not silently dropped. Simulated by
    monkeypatching the analytics call to hand back a poisoned point.
    """
    _seed_series(tmp_config, "AAA", [100, 101])
    _seed_series(tmp_config, "SPY", [100, 101])

    poisoned_day = START + timedelta(days=1)
    poisoned = ReturnPoint(
        symbol="AAA",
        price_date=poisoned_day,
        total_return=0.01,
        estimation_window_start=poisoned_day,  # on-or-after price_date: lookahead
        method="market_adjusted",
    )

    def fake_compute_abnormal_returns(
        symbol: str, asset_returns: Any, **kwargs: Any
    ) -> list[ReturnPoint]:
        return [poisoned]

    # Patched on the module object, not the imported name, so the collector's
    # own (bare-name, module-global) call site picks up the stub.
    monkeypatch.setattr(
        returns_collector, "compute_abnormal_returns", fake_compute_abnormal_returns
    )
    recorder = _RecordingLogger()
    monkeypatch.setattr(returns_collector, "_log", recorder)

    summary = returns_collector.compute(tmp_config, symbols=["AAA"])

    assert summary.rejected >= 1
    assert summary.stage.get("lookahead_rejected", 0) >= 1
    assert len(recorder.errors) >= 1
    event, fields = recorder.errors[0]
    assert event == "lookahead_detected"
    assert fields["price_date"] == poisoned_day.isoformat()

    rows = _returns(tmp_config)
    assert not any(r[1] == poisoned_day for r in rows)
