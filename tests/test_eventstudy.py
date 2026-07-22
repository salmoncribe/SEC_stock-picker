"""Tests for event-study sample construction.

The t=0 and purging tests are the ones that matter. Both encode rules whose
violation produces a *better-looking* backtest, which is why they are asserted
directly rather than trusted to review.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from market_intelligence.analytics.eventstudy import (
    ReturnSeries,
    SampleSplit,
    assign_split,
    first_trading_day_after,
    forward_window,
)
from market_intelligence.analytics.returns import ReturnPoint

DAY_ZERO = date(2024, 1, 1)


def days(n: int) -> date:
    return DAY_ZERO + timedelta(days=n)


def series(values: list[float | None], *, start: int = 1) -> list[ReturnPoint]:
    return [
        ReturnPoint(
            symbol="AAA",
            price_date=days(start + i),
            total_return=0.0,
            abnormal_return=v,
        )
        for i, v in enumerate(values)
    ]


# ---------------------------------------------------------------------------
# t=0
# ---------------------------------------------------------------------------


def test_first_trading_day_is_strictly_after_publication():
    """The publication date itself is never tradeable."""
    trading = [days(1), days(2), days(3)]

    assert first_trading_day_after(trading, days(1)) == days(2)


def test_first_trading_day_skips_non_trading_days():
    trading = [days(1), days(5), days(6)]

    assert first_trading_day_after(trading, days(2)) == days(5)


def test_first_trading_day_is_none_past_the_series():
    assert first_trading_day_after([days(1), days(2)], days(9)) is None


def test_window_excludes_the_publication_day_itself():
    """A huge move on the filing day must not be credited to the event.

    If t=0 were the publication date, an after-hours filing would collect the
    day's move that happened before anyone could read it.
    """
    points = series([9.99, 0.01, 0.02])

    window = forward_window(points, available_on=days(1), horizon_days=2)

    assert window is not None
    assert window.t0 == days(2)
    assert window.cumulative_abnormal_return == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# window formation
# ---------------------------------------------------------------------------


def test_window_sums_the_requested_trading_days():
    points = series([0.01, 0.02, 0.03, 0.04])

    window = forward_window(points, available_on=days(0), horizon_days=3)

    assert window is not None
    assert window.trading_days == 3
    assert window.t0 == days(1)
    assert window.window_end == days(3)
    assert window.cumulative_abnormal_return == pytest.approx(0.06)


def test_window_counts_trading_days_not_calendar_days():
    """Holidays and halts need no special handling: the series defines the calendar."""
    points = [
        ReturnPoint("AAA", days(1), 0.0, abnormal_return=0.01),
        ReturnPoint("AAA", days(8), 0.0, abnormal_return=0.02),
        ReturnPoint("AAA", days(9), 0.0, abnormal_return=0.03),
    ]

    window = forward_window(points, available_on=days(0), horizon_days=3)

    assert window is not None
    assert window.window_end == days(9)


def test_window_is_none_when_too_few_days_remain():
    """A partial window is not a shorter window; it is no observation."""
    points = series([0.01, 0.02])

    assert forward_window(points, available_on=days(0), horizon_days=5) is None


def test_window_is_none_when_a_day_is_unmeasured():
    points = series([0.01, None, 0.03])

    assert forward_window(points, available_on=days(0), horizon_days=3) is None


def test_window_rejects_a_non_positive_horizon():
    assert forward_window(series([0.01]), available_on=days(0), horizon_days=0) is None


def test_indexed_lookups_match_the_one_shot_path_exactly():
    """The bisect index and the reference wrapper must never diverge.

    They implement the same leakage boundary, so a divergence would not be a
    performance difference -- it would be a wrong t=0 that still looked
    plausible.
    """
    points = series([0.01 * (i % 7) - 0.02 for i in range(60)])
    indexed = ReturnSeries(points)

    for offset in range(0, 60):
        for horizon in (1, 5, 20):
            assert indexed.forward(days(offset), horizon) == forward_window(
                points, days(offset), horizon
            )


def test_index_handles_a_date_present_in_the_series():
    """bisect must land past an exact match, keeping t=0 strictly after."""
    points = series([0.01, 0.02, 0.03])
    indexed = ReturnSeries(points)

    window = indexed.forward(days(1), 1)

    assert window is not None
    assert window.t0 == days(2)


def test_window_tolerates_unsorted_input():
    points = list(reversed(series([0.01, 0.02, 0.03])))

    window = forward_window(points, available_on=days(0), horizon_days=2)

    assert window is not None
    assert window.t0 == days(1)


# ---------------------------------------------------------------------------
# purged chronological split
# ---------------------------------------------------------------------------


def _window(t0_offset: int, end_offset: int):
    points = series([0.01] * (end_offset + 1))
    window = forward_window(points, available_on=days(t0_offset - 1), horizon_days=1)
    assert window is not None
    return window


def test_window_entirely_before_the_boundary_is_discovery():
    window = _window(1, 1)

    assert assign_split(window, split_date=days(50)) is SampleSplit.DISCOVERY


def test_window_opening_on_the_boundary_is_holdout():
    points = series([0.01] * 60)
    window = forward_window(points, available_on=days(49), horizon_days=1)
    assert window is not None
    assert window.t0 == days(50)

    assert assign_split(window, split_date=days(50)) is SampleSplit.HOLDOUT


def test_window_straddling_the_boundary_is_dropped():
    """Its label is computed from prices on both sides, so it belongs to neither.

    Counting a straddling sample in either half lets discovery-period prices
    determine part of a holdout label -- the leak that purging exists to stop.
    """
    points = series([0.01] * 60)
    window = forward_window(points, available_on=days(47), horizon_days=5)
    assert window is not None
    assert window.t0 < days(50) < window.window_end

    assert assign_split(window, split_date=days(50)) is None


def test_a_longer_horizon_purges_more_samples():
    """The boundary cost scales with the horizon, and that is the correct behaviour."""
    points = series([0.01] * 80)

    dropped_1d = dropped_20d = 0
    for offset in range(40, 55):
        short = forward_window(points, available_on=days(offset), horizon_days=1)
        long = forward_window(points, available_on=days(offset), horizon_days=20)
        if short and assign_split(short, days(50)) is None:
            dropped_1d += 1
        if long and assign_split(long, days(50)) is None:
            dropped_20d += 1

    assert dropped_20d > dropped_1d
