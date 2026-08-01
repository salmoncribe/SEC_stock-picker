"""Transaction-cost tests: hand-computed fills and bit-exact backtest parity."""

from market_intelligence.analytics.backtest import _fill
from market_intelligence.portfolio.costs import (
    CostModel,
    commission,
    fill_price,
    round_trip_cost_bps,
)


def test_buy_fills_above_the_raw_price():
    # 100.0 * (1 + 5/10_000) = 100.05, exactly representable in the mid-range.
    assert fill_price(100.0, is_buy=True, model=CostModel()) == 100.05


def test_sell_fills_below_the_raw_price():
    # 100.0 * (1 - 5/10_000) = 99.95.
    assert fill_price(100.0, is_buy=False, model=CostModel()) == 99.95


def test_fill_price_matches_backtest_fill_bit_for_bit():
    # ``_fill`` is keyed on (direction, entry); this module is keyed on is_buy.
    # Opening a long buys, closing a long sells; a short is the mirror image,
    # so is_buy is true exactly when (direction > 0) equals entry.
    raws = [1.0, 9.99, 42.42, 100.0, 137.035, 1_234.5678, 0.0001]
    bps_values = [0.0, 0.5, 1.0, 5.0, 7.3, 25.0, 100.0]
    for raw in raws:
        for bps in bps_values:
            model = CostModel(slippage_bps=bps)
            for direction in (1, -1):
                for entry in (True, False):
                    is_buy = (direction > 0) == entry
                    assert fill_price(raw, is_buy=is_buy, model=model) == _fill(
                        raw, direction, bps, entry
                    )


def test_commission_is_absolute_and_linear_with_zero_default():
    assert commission(100.0, CostModel()) == 0.0
    model = CostModel(commission_per_share=0.005)
    assert commission(100.0, model) == commission(-100.0, model)
    assert commission(100.0, model) == 0.5
    assert commission(200.0, model) == 2.0 * commission(100.0, model)
    assert commission(0.0, model) == 0.0


def test_round_trip_cost_is_both_legs():
    assert round_trip_cost_bps(CostModel()) == 10.0
    assert round_trip_cost_bps(CostModel(slippage_bps=7.5)) == 15.0
    assert round_trip_cost_bps(CostModel(slippage_bps=0.0)) == 0.0
