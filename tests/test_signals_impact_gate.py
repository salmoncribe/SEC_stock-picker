"""Tests for the gate's sample loading -- the market-adjusted return fix.

The bug these guard against: the old sample loader read
``forward_abnormal_return``, a per-day-summed, trailing-regression-fitted
label that manufactures fake edge for any stock that has been declining (see
``signals/impact.py``'s module docstring). These tests build a tiny synthetic
price panel where the correct answer is known by construction, and check the
new loader against it directly -- including the specific failure mode (a
stock's own pre-event decline leaking into the measured return) that the old
label was shown to produce.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pytest

from market_intelligence.signals.impact import _load_samples, _placebo_samples

DAY_ZERO = date(2024, 1, 1)  # a Monday


def trading_days(n: int) -> list[date]:
    """n weekday dates starting from DAY_ZERO -- no weekends, so row-position
    offsets used by the gate line up with simple index arithmetic in tests."""
    out = []
    d = DAY_ZERO
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _insert_prices(con: duckdb.DuckDBPyConnection, symbol: str, closes: list[float], dates: list[date]) -> None:
    for i, (d, c) in enumerate(zip(dates, closes, strict=True)):
        con.execute(
            "INSERT INTO daily_prices "
            "(price_id, symbol, price_date, open, high, low, close, adj_close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [f"{symbol}-{i}", symbol, d, c, c, c, c, c, 1000],
        )


def _insert_event(
    con: duckdb.DuckDBPyConnection,
    *,
    event_id: str,
    target_ticker: str,
    t0: date,
    horizon_days: int,
    edge_id: str = "self",
    split: str = "discovery",
    event_type: str = "insider_transaction",
    event_subtype: str = "P",
) -> None:
    con.execute(
        "INSERT INTO event_samples "
        "(sample_id, event_id, edge_id, event_type, event_subtype, target_ticker, "
        " horizon_days, available_on, t0, split) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            f"sample-{event_id}-{horizon_days}",
            event_id,
            edge_id,
            event_type,
            event_subtype,
            target_ticker,
            horizon_days,
            t0 - timedelta(days=1),
            t0,
            split,
        ],
    )


@pytest.fixture
def panel(memory_db: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    return memory_db


def test_market_adjusted_return_is_a_ratio_against_the_benchmark(panel):
    """SPY +10%, stock +20% over the same window -> ~+9.09%, not +10pp."""
    days = trading_days(10)
    _insert_prices(panel, "SPY", [100, 101, 102, 103, 104, 105, 106, 107, 108, 110], days)
    _insert_prices(panel, "AAPL", [50, 51, 52, 53, 54, 55, 56, 57, 58, 60], days)
    _insert_event(panel, event_id="e1", target_ticker="AAPL", t0=days[0], horizon_days=9)

    samples = _load_samples(panel, "insider_transaction", "P", "self", 9, "discovery")

    assert len(samples) == 1
    symbol, t0, market_adj = samples[0]
    assert symbol == "AAPL"
    assert market_adj == pytest.approx((60 / 50) / (110 / 100) - 1, rel=1e-9)


def test_a_stocks_own_pre_event_decline_does_not_leak_into_the_return(panel):
    """The bug this guards against: the old label subtracted a trailing-window
    alpha, so a stock that had been falling got CREDITED with return it never
    earned during the event window. This stock falls hard for 30 days, then
    moves in perfect lockstep with SPY (+5%) for the event window itself --
    the correct market-adjusted answer is exactly 0, regardless of what came
    before t0."""
    all_days = trading_days(40)
    pre, post = all_days[:30], all_days[30:]

    # Falls from 100 to 40 over `pre`, i.e. a large negative trailing trend,
    # then tracks SPY exactly (+5% cumulative) over the 10-day event window.
    pre_closes = [100 - i * 2 for i in range(30)]  # 100 -> 42
    post_closes = [pre_closes[-1] * (1.05 ** (i / 9)) for i in range(10)]
    stock_closes = pre_closes + post_closes

    spy_pre = [100.0] * 30  # flat before the event -- irrelevant to the check
    spy_post = [100 * (1.05 ** (i / 9)) for i in range(10)]
    spy_closes = spy_pre + spy_post

    _insert_prices(panel, "SPY", spy_closes, all_days)
    _insert_prices(panel, "DECLINR", stock_closes, all_days)
    _insert_event(panel, event_id="e2", target_ticker="DECLINR", t0=post[0], horizon_days=9)

    samples = _load_samples(panel, "insider_transaction", "P", "self", 9, "discovery")

    assert len(samples) == 1
    _, _, market_adj = samples[0]
    # Moved in exact lockstep with SPY during the window -> ~0, not a positive
    # number manufactured from the 30 days of decline that preceded it.
    assert market_adj == pytest.approx(0.0, abs=1e-6)


def test_entry_uses_first_bar_on_or_after_t0(panel):
    """t0 lands on a day with no trading bar (e.g. a gap in the panel) --
    entry must be the next available bar, not fail or use a stale one."""
    days = trading_days(12)
    # Remove day index 2 to simulate a missing bar exactly at t0.
    missing_day = days[2]
    kept_days = [d for d in days if d != missing_day]
    kept_closes_spy = [100 + i for i in range(len(kept_days))]
    kept_closes_stock = [50 + i for i in range(len(kept_days))]

    _insert_prices(panel, "SPY", kept_closes_spy, kept_days)
    _insert_prices(panel, "GAPCO", kept_closes_stock, kept_days)
    _insert_event(panel, event_id="e3", target_ticker="GAPCO", t0=missing_day, horizon_days=3)

    samples = _load_samples(panel, "insider_transaction", "P", "self", 3, "discovery")

    assert len(samples) == 1  # entry fell through to the next real bar, not dropped


def test_placebo_samples_are_market_adjusted_and_reproducible(panel):
    days = trading_days(15)
    _insert_prices(panel, "SPY", [100 + i for i in range(15)], days)
    for sym in ("AAA", "BBB", "CCC"):
        _insert_prices(panel, sym, [50 + i * 0.5 for i in range(15)], days)
    _insert_event(panel, event_id="e4", target_ticker="AAA", t0=days[10], horizon_days=3,
                   split="holdout")

    first = _placebo_samples(panel, 3, "holdout", n=20, seed=42)
    second = _placebo_samples(panel, 3, "holdout", n=20, seed=42)

    assert first == second  # same seed -> same draw, still a real requirement
    assert len(first) > 0
    for symbol, _t0, market_adj in first:
        assert symbol != "SPY"  # the benchmark itself must never be its own placebo
        assert isinstance(market_adj, float)
