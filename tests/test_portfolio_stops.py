"""Stop-loss math tests: hand-computed prices, a hand-computed Wilder ATR,
and every "cannot evaluate today" edge case that must read as not-triggered
rather than as an error.
"""

from __future__ import annotations

import math

import pytest

from market_intelligence.portfolio.stops import (
    StopCheck,
    atr_stop_price,
    average_cost,
    evaluate_stop,
    fixed_pct_stop_price,
    trailing_atr,
)


# --------------------------------------------------------------------------- #
# average_cost                                                                 #
# --------------------------------------------------------------------------- #
def test_average_cost_is_total_paid_over_shares_held():
    assert average_cost(10_000.0, 100.0) == pytest.approx(100.0)


def test_average_cost_of_a_non_positive_share_count_is_zero_not_a_crash():
    assert average_cost(10_000.0, 0.0) == 0.0
    assert average_cost(10_000.0, -5.0) == 0.0


# --------------------------------------------------------------------------- #
# fixed_pct_stop_price                                                         #
# --------------------------------------------------------------------------- #
def test_fixed_pct_stop_price_is_entry_times_one_minus_pct():
    assert fixed_pct_stop_price(100.0, 0.20) == pytest.approx(80.0)
    assert fixed_pct_stop_price(50.0, 0.10) == pytest.approx(45.0)


@pytest.mark.parametrize("entry", [0.0, -10.0])
def test_fixed_pct_stop_price_rejects_a_non_positive_entry(entry):
    with pytest.raises(ValueError, match="entry_price"):
        fixed_pct_stop_price(entry, 0.2)


@pytest.mark.parametrize("pct", [0.0, 1.0, -0.1, 1.5])
def test_fixed_pct_stop_price_rejects_a_pct_outside_the_open_unit_interval(pct):
    with pytest.raises(ValueError, match="stop_pct"):
        fixed_pct_stop_price(100.0, pct)


# --------------------------------------------------------------------------- #
# atr_stop_price                                                               #
# --------------------------------------------------------------------------- #
def test_atr_stop_price_mirrors_trade_plans_entry_minus_multiple_times_atr():
    # Same geometry as signals.trade_plan.build_trade_plan's long-side stop.
    assert atr_stop_price(100.0, atr=2.0, atr_multiple=3.0) == pytest.approx(94.0)


def test_atr_stop_price_rejects_a_non_positive_entry():
    with pytest.raises(ValueError, match="entry_price"):
        atr_stop_price(0.0, atr=1.0, atr_multiple=2.0)


def test_atr_stop_price_rejects_a_negative_atr():
    with pytest.raises(ValueError, match="atr"):
        atr_stop_price(100.0, atr=-1.0, atr_multiple=2.0)


def test_atr_stop_price_rejects_a_non_positive_multiple():
    with pytest.raises(ValueError, match="atr_multiple"):
        atr_stop_price(100.0, atr=1.0, atr_multiple=0.0)


# --------------------------------------------------------------------------- #
# trailing_atr -- one implementation, reused via signals.trade_plan.wilder_atr #
# --------------------------------------------------------------------------- #
def test_trailing_atr_matches_a_hand_computed_wilder_average():
    """Four bars, period=3: exactly enough for one un-smoothed average --
    ranges[:period] with nothing left over to smooth, so the answer is a
    plain mean of three true ranges, computed by hand below.

    bar0 H=10 L=8  C=9
    bar1 H=11 L=9  C=10  TR1 = max(2, |11-9|=2,  |9-9|=0)  = 2
    bar2 H=12 L=10 C=11  TR2 = max(2, |12-10|=2, |10-10|=0) = 2
    bar3 H=9  L=7  C=8   TR3 = max(2, |9-11|=2,  |7-11|=4)  = 4
    atr = mean(2, 2, 4) = 8/3
    """
    high = [10.0, 11.0, 12.0, 9.0]
    low = [8.0, 9.0, 10.0, 7.0]
    close = [9.0, 10.0, 11.0, 8.0]
    atr = trailing_atr(high, low, close, period=3)
    assert atr == pytest.approx(8.0 / 3.0, rel=1e-12)


def test_trailing_atr_returns_none_below_period_plus_one_valid_rows():
    high = [10.0, 11.0, 12.0]
    low = [8.0, 9.0, 10.0]
    close = [9.0, 10.0, 11.0]
    assert trailing_atr(high, low, close, period=3) is None


def test_trailing_atr_drops_non_finite_rows_rather_than_poisoning_the_window():
    # A NaN row is dropped by wilder_atr itself; four *valid* rows remain,
    # so the answer must equal the hand-computed case above with a gap
    # spliced in.
    high = [10.0, math.nan, 11.0, 12.0, 9.0]
    low = [8.0, math.nan, 9.0, 10.0, 7.0]
    close = [9.0, math.nan, 10.0, 11.0, 8.0]
    atr = trailing_atr(high, low, close, period=3)
    assert atr == pytest.approx(8.0 / 3.0, rel=1e-12)


def test_trailing_atr_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        trailing_atr([1.0, 2.0], [1.0], [1.0, 2.0], period=1)


# --------------------------------------------------------------------------- #
# evaluate_stop                                                                #
# --------------------------------------------------------------------------- #
def test_evaluate_stop_kind_none_never_triggers():
    check = evaluate_stop(
        symbol="AAA",
        kind="none",
        entry_price=100.0,
        reference_price=1.0,
        atr=None,
        stop_pct=0.2,
        atr_multiple=3.0,
    )
    assert check == StopCheck("AAA", "none", 100.0, 1.0, None, False)


def test_evaluate_stop_fixed_pct_triggers_at_and_below_the_threshold_inclusive():
    kwargs = {
        "symbol": "AAA",
        "kind": "fixed_pct",
        "entry_price": 100.0,
        "atr": None,
        "stop_pct": 0.2,
        "atr_multiple": 3.0,
    }
    at_threshold = evaluate_stop(reference_price=80.0, **kwargs)
    assert at_threshold.stop_price == pytest.approx(80.0)
    assert at_threshold.triggered is True

    just_above = evaluate_stop(reference_price=80.01, **kwargs)
    assert just_above.triggered is False

    well_below = evaluate_stop(reference_price=50.0, **kwargs)
    assert well_below.triggered is True


def test_evaluate_stop_atr_multiple_triggers_below_entry_minus_multiple_atr():
    kwargs = {
        "symbol": "AAA",
        "kind": "atr_multiple",
        "entry_price": 100.0,
        "stop_pct": 0.2,
        "atr_multiple": 3.0,
    }
    # stop = 100 - 3*2 = 94
    below = evaluate_stop(reference_price=93.0, atr=2.0, **kwargs)
    assert below.stop_price == pytest.approx(94.0)
    assert below.triggered is True

    above = evaluate_stop(reference_price=95.0, atr=2.0, **kwargs)
    assert above.triggered is False


@pytest.mark.parametrize("atr", [None, 0.0, -1.0, math.nan])
def test_evaluate_stop_atr_multiple_without_a_usable_atr_is_not_triggered_not_an_error(atr):
    check = evaluate_stop(
        symbol="AAA",
        kind="atr_multiple",
        entry_price=100.0,
        reference_price=1.0,  # would trigger under any real stop
        atr=atr,
        stop_pct=0.2,
        atr_multiple=3.0,
    )
    assert check.triggered is False
    assert check.stop_price is None


@pytest.mark.parametrize(
    "entry,reference", [(0.0, 50.0), (-1.0, 50.0), (100.0, 0.0), (100.0, -1.0)]
)
def test_evaluate_stop_non_positive_prices_read_as_cannot_evaluate(entry, reference):
    check = evaluate_stop(
        symbol="AAA",
        kind="fixed_pct",
        entry_price=entry,
        reference_price=reference,
        atr=None,
        stop_pct=0.2,
        atr_multiple=3.0,
    )
    assert check.triggered is False


def test_evaluate_stop_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown stop kind"):
        evaluate_stop(
            symbol="AAA",
            kind="bogus",  # type: ignore[arg-type]
            entry_price=100.0,
            reference_price=50.0,
            atr=None,
            stop_pct=0.2,
            atr_multiple=3.0,
        )
