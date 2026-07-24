"""Trade-plan math: pure functions over bars. No I/O, no network."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from market_intelligence.signals.trade_plan import Bar, build_trade_plan, wilder_atr


def _bars(n=30, close=100.0, spread=2.0, start=date(2026, 6, 1)):
    """n flat-ish bars with a known true range of `spread` each day."""
    out = []
    for i in range(n):
        d = start + timedelta(days=i)
        out.append(Bar(date=d, open=close, high=close + spread / 2,
                       low=close - spread / 2, close=close, adj_close=close))
    return out


class TestWilderAtr:
    def test_constant_range_converges_to_that_range(self):
        atr = wilder_atr(_bars(40, spread=2.0), period=14)
        assert atr == pytest.approx(2.0, rel=0.05)

    def test_insufficient_history_returns_none(self):
        assert wilder_atr(_bars(10), period=14) is None

    def test_split_adjusted_bars_ignore_raw_split_cliff(self):
        # 20 bars at raw 100 (adj 50 — pre 2:1 split), then 20 at raw 50 (adj 50).
        pre = [Bar(date=date(2026, 5, 1) + timedelta(days=i), open=100, high=101,
                   low=99, close=100, adj_close=50.0) for i in range(20)]
        post = [Bar(date=date(2026, 5, 21) + timedelta(days=i), open=50, high=50.5,
                    low=49.5, close=50, adj_close=50.0) for i in range(20)]
        atr = wilder_atr(pre + post, period=14)
        assert atr is not None
        # In adjusted space the split cliff does not exist: ATR stays ~ the
        # true daily range (~1 pre-split-adjusted / ~1 post), never ~50.
        assert atr < 3.0

    def test_bar_with_zero_adj_close_is_dropped_not_poisoning(self):
        # A corrupt bar (adj_close=0 would collapse its scaled OHLC to 0 and
        # blow ATR up toward the full price) is excluded, not smoothed in.
        clean = _bars(40, spread=2.0)
        corrupt = Bar(date=date(2099, 1, 1), open=100.0, high=101.0, low=99.0,
                      close=100.0, adj_close=0.0)
        atr = wilder_atr([*clean, corrupt], period=14)
        assert atr is not None
        assert atr == pytest.approx(2.0, rel=0.05)

    def test_all_flat_bars_have_zero_atr(self):
        bars = [Bar(date=date(2026, 6, 1) + timedelta(days=i), open=100.0, high=100.0,
                    low=100.0, close=100.0, adj_close=100.0) for i in range(40)]
        assert wilder_atr(bars, period=14) == 0.0


class TestBuildTradePlan:
    def _plan(self, direction=1, predicted_move=0.03, equity=10_000.0, **kw):
        return build_trade_plan(
            bars=_bars(40, close=100.0, spread=2.0),
            direction=direction,
            predicted_move=predicted_move,
            horizon_days=20,
            as_of=date(2026, 7, 23),
            account_equity=equity,
            risk_pct_per_trade=1.0,
            atr_period=14,
            atr_stop_multiple=2.0,
            max_position_pct=20.0,
            **kw,
        )

    def test_long_plan_geometry(self):
        plan = self._plan()
        assert plan.entry_ref == pytest.approx(100.0)
        assert plan.stop == pytest.approx(100.0 - 2.0 * 2.0, rel=0.05)   # k*ATR below
        assert plan.target == pytest.approx(103.0)                        # 1 + predicted
        assert plan.time_exit_date == date(2026, 7, 23) + timedelta(days=20)
        # risk-based size would be floor((10_000 * 1%) / 4) = 25 shares, but the
        # 20% position cap (2_000 notional / 100) binds first → 20 shares.
        assert plan.shares == 20
        assert plan.risk_amount == pytest.approx(plan.shares * (plan.entry_ref - plan.stop))
        assert not plan.unsizeable

    def test_short_plan_flips_stop_and_target(self):
        plan = self._plan(direction=-1, predicted_move=-0.03)
        assert plan.stop > plan.entry_ref
        assert plan.target < plan.entry_ref

    def test_position_cap_binds(self):
        # equity 1M: risk budget 10k / 4 = 2500 shares uncapped (250k notional);
        # the 20% cap (200k) binds → exactly floor(200_000 / 100) = 2000 shares.
        plan = self._plan(equity=1_000_000.0)
        assert plan.shares == 2000

    def test_entry_override_anchors_the_plan_at_the_quote(self):
        # The gap path plans off the live morning quote, not the last bar's
        # close — the whole point is that those differ by the gap.
        plan = self._plan(entry_override=105.0)
        assert plan.entry_ref == pytest.approx(105.0)
        assert plan.stop == pytest.approx(105.0 - 4.0, rel=0.05)

    def test_tiny_account_is_unsizeable_but_still_planned(self):
        plan = self._plan(equity=100.0)
        assert plan.shares == 0
        assert plan.unsizeable
        assert plan.stop < plan.entry_ref  # the plan geometry still renders

    def test_nan_close_on_latest_bar_never_becomes_the_entry(self):
        # The freshest bar is exactly the one a halted/no-trade day corrupts.
        # ATR drops the bar, but entry must not read the unfiltered list and
        # carry a NaN through every <= 0 guard into a rendered plan.
        bars = _bars(40, close=100.0, spread=2.0)
        last = bars[-1]
        bars[-1] = Bar(
            date=last.date,
            open=last.open,
            high=last.high,
            low=last.low,
            close=float("nan"),
            adj_close=last.adj_close,
        )
        plan = build_trade_plan(
            bars=bars, direction=1, predicted_move=0.03, horizon_days=20,
            as_of=date(2026, 7, 23), account_equity=10_000.0,
            risk_pct_per_trade=1.0, atr_period=14, atr_stop_multiple=2.0,
            max_position_pct=20.0,
        )
        assert plan is not None
        assert math.isfinite(plan.entry_ref)
        assert plan.entry_ref == pytest.approx(100.0)

    def test_insufficient_history_returns_none(self):
        plan = build_trade_plan(
            bars=_bars(5), direction=1, predicted_move=0.02, horizon_days=5,
            as_of=date(2026, 7, 23), account_equity=10_000.0,
            risk_pct_per_trade=1.0, atr_period=14, atr_stop_multiple=2.0,
            max_position_pct=20.0,
        )
        assert plan is None

    def test_degenerate_long_stop_returns_none(self):
        # entry $5, ATR $3 (spread=3 on flat bars), k=2 -> stop = 5 - 6 = -1:
        # a stop at or below zero means the geometry doesn't fit this
        # instrument, so no plan is returned rather than a clamped one.
        plan = build_trade_plan(
            bars=_bars(40, close=5.0, spread=3.0),
            direction=1,
            predicted_move=0.03,
            horizon_days=20,
            as_of=date(2026, 7, 23),
            account_equity=10_000.0,
            risk_pct_per_trade=1.0,
            atr_period=14,
            atr_stop_multiple=2.0,
            max_position_pct=20.0,
        )
        assert plan is None

    def test_nan_bar_is_dropped_and_plan_still_builds(self):
        # A NaN high in one bar must not propagate into a NaN ATR (which
        # would slip past a plain `atr <= 0` guard and later blow up in
        # math.floor) — the bad bar is dropped before it ever reaches ATR.
        bars = _bars(40, close=100.0, spread=2.0)
        nan_bar = Bar(date=date(2099, 1, 1), open=100.0, high=float("nan"),
                     low=99.0, close=100.0, adj_close=100.0)
        plan = build_trade_plan(
            bars=[*bars, nan_bar],
            direction=1,
            predicted_move=0.03,
            horizon_days=20,
            as_of=date(2026, 7, 23),
            account_equity=10_000.0,
            risk_pct_per_trade=1.0,
            atr_period=14,
            atr_stop_multiple=2.0,
            max_position_pct=20.0,
        )
        assert plan is not None
        assert math.isfinite(plan.atr)

    def test_zero_range_atr_makes_plan_unbuildable(self):
        bars = [Bar(date=date(2026, 6, 1) + timedelta(days=i), open=100.0, high=100.0,
                    low=100.0, close=100.0, adj_close=100.0) for i in range(40)]
        plan = build_trade_plan(
            bars=bars,
            direction=1,
            predicted_move=0.03,
            horizon_days=20,
            as_of=date(2026, 7, 23),
            account_equity=10_000.0,
            risk_pct_per_trade=1.0,
            atr_period=14,
            atr_stop_multiple=2.0,
            max_position_pct=20.0,
        )
        assert plan is None
