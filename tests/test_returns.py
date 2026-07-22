"""Tests for the pure return / abnormal-return analytics.

The lookahead tests here are the important ones. Everything the signal layer
eventually claims rests on abnormal returns being computable using only
information that existed at the time, so that property is asserted directly
rather than assumed.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from market_intelligence.analytics.returns import (
    AbnormalReturnMethod,
    ReturnPoint,
    compute_abnormal_returns,
    cumulative_abnormal_return,
    fit_ols,
    simple_returns,
)

DAY_ZERO = date(2024, 1, 1)


def days(n: int) -> date:
    return DAY_ZERO + timedelta(days=n)


# ---------------------------------------------------------------------------
# simple_returns
# ---------------------------------------------------------------------------


def test_simple_returns_computes_consecutive_moves():
    bars = [(days(0), 100.0), (days(1), 110.0), (days(2), 99.0)]

    result = simple_returns(bars)

    assert result[0][0] == days(1)
    assert result[0][1] == pytest.approx(0.10)
    assert result[1][1] == pytest.approx(-0.10)


def test_simple_returns_sorts_unordered_input():
    bars = [(days(2), 121.0), (days(0), 100.0), (days(1), 110.0)]

    result = simple_returns(bars)

    assert [d for d, _ in result] == [days(1), days(2)]


def test_simple_returns_drops_moves_spanning_a_gap():
    """A hole in the series is a gap, not a one-day move."""
    bars = [(days(0), 100.0), (days(1), 110.0), (days(30), 200.0)]

    result = simple_returns(bars, max_gap_days=7)

    assert [d for d, _ in result] == [days(1)]


def test_simple_returns_drops_non_positive_prices():
    bars = [(days(0), 100.0), (days(1), 0.0), (days(2), 120.0)]

    result = simple_returns(bars)

    # The zero bar is unusable, so the only return is 100 -> 120 across it.
    assert [d for d, _ in result] == [days(2)]
    assert result[0][1] == pytest.approx(0.20)


def test_simple_returns_needs_a_prior_price():
    assert simple_returns([(days(0), 100.0)]) == []
    assert simple_returns([]) == []


# ---------------------------------------------------------------------------
# fit_ols
# ---------------------------------------------------------------------------


def test_fit_ols_recovers_known_alpha_and_beta():
    # y = 0.5 + 2x, exactly.
    points = [(0.5 + 2 * x, x) for x in (0.01, 0.02, 0.03, -0.01)]

    fit = fit_ols(points)

    assert fit is not None
    assert fit.beta == pytest.approx(2.0)
    assert fit.alpha == pytest.approx(0.5)
    assert fit.observations == 4


def test_fit_ols_undefined_without_benchmark_variance():
    """A flat benchmark says nothing about beta; that is None, not zero."""
    assert fit_ols([(0.01, 0.05), (0.02, 0.05), (0.03, 0.05)]) is None


def test_fit_ols_undefined_for_an_all_zero_benchmark():
    assert fit_ols([(0.01, 0.0), (0.02, 0.0)]) is None


def test_fit_ols_still_fits_a_genuinely_small_variance():
    """The degeneracy guard must not reject real but quiet benchmarks."""
    points = [(2 * x, x) for x in (1e-6, 2e-6, 3e-6, -1e-6)]

    fit = fit_ols(points)

    assert fit is not None
    assert fit.beta == pytest.approx(2.0)


def test_fit_ols_needs_two_points():
    assert fit_ols([(0.01, 0.02)]) is None
    assert fit_ols([]) is None


# ---------------------------------------------------------------------------
# compute_abnormal_returns
# ---------------------------------------------------------------------------


def test_market_adjusted_subtracts_the_benchmark():
    asset = [(days(1), 0.05), (days(2), -0.02)]
    market = [(days(1), 0.02), (days(2), -0.01)]

    points = compute_abnormal_returns(
        "AAA", asset, market_returns=market, method=AbnormalReturnMethod.MARKET_ADJUSTED
    )

    assert points[0].abnormal_return == pytest.approx(0.03)
    assert points[1].abnormal_return == pytest.approx(-0.01)


def test_sector_adjusted_uses_the_sector_benchmark():
    """A sector-wide shock should net out, not read as company-specific news."""
    asset = [(days(1), 0.06)]
    market = [(days(1), 0.01)]
    sector = [(days(1), 0.06)]

    points = compute_abnormal_returns(
        "AAA",
        asset,
        market_returns=market,
        sector_returns=sector,
        method=AbnormalReturnMethod.SECTOR_ADJUSTED,
    )

    assert points[0].abnormal_return == pytest.approx(0.0)
    # Both benchmarks are still recorded even though only one was subtracted.
    assert points[0].market_return == pytest.approx(0.01)
    assert points[0].sector_return == pytest.approx(0.06)


def test_abnormal_return_is_none_without_a_benchmark_value():
    """A missing expectation is not a zero surprise."""
    asset = [(days(1), 0.05)]

    points = compute_abnormal_returns(
        "AAA", asset, market_returns=[], method=AbnormalReturnMethod.MARKET_ADJUSTED
    )

    assert points[0].abnormal_return is None
    assert points[0].total_return == pytest.approx(0.05)


def test_market_model_needs_minimum_history_before_it_fits():
    asset = [(days(i), 0.01) for i in range(1, 11)]
    market = [(days(i), 0.005) for i in range(1, 11)]

    points = compute_abnormal_returns(
        "AAA",
        asset,
        market_returns=market,
        method=AbnormalReturnMethod.MARKET_MODEL,
        min_observations=5,
    )

    # Nothing can be fitted until 5 prior observations exist.
    assert all(p.abnormal_return is None for p in points[:5])
    assert all(p.beta is None for p in points[:5])


def test_market_model_recovers_beta_and_flags_no_abnormal_move():
    """A name that moves exactly 2x its benchmark has no abnormal return."""
    asset, market = [], []
    for i in range(1, 81):
        market_move = 0.01 if i % 2 else -0.005
        market.append((days(i), market_move))
        asset.append((days(i), 2.0 * market_move))

    points = compute_abnormal_returns(
        "AAA",
        asset,
        market_returns=market,
        method=AbnormalReturnMethod.MARKET_MODEL,
        min_observations=60,
    )

    fitted = [p for p in points if p.beta is not None]
    assert fitted, "expected at least one fitted day"
    assert fitted[-1].beta == pytest.approx(2.0)
    assert fitted[-1].abnormal_return == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Lookahead
# ---------------------------------------------------------------------------


def _series(n: int) -> tuple[list[tuple[date, float]], list[tuple[date, float]]]:
    asset, market = [], []
    for i in range(1, n + 1):
        market_move = 0.01 if i % 3 else -0.008
        market.append((days(i), market_move))
        asset.append((days(i), 1.5 * market_move + (0.002 if i % 5 else -0.001)))
    return asset, market


def test_future_data_cannot_change_past_estimates():
    """Perturbing the last day must leave every earlier day byte-identical.

    This is the direct test for lookahead: if a beta were fitted on a window
    that included its own day -- or on the whole series -- a shock at the end
    would ripple backwards and change earlier abnormal returns.
    """
    asset, market = _series(90)

    baseline = compute_abnormal_returns("AAA", asset, market_returns=market, min_observations=60)

    shocked_asset = [*asset[:-1], (asset[-1][0], 5.0)]  # a violent final day
    shocked = compute_abnormal_returns(
        "AAA", shocked_asset, market_returns=market, min_observations=60
    )

    assert baseline[:-1] == shocked[:-1]


def test_estimation_window_ends_before_the_day_it_prices():
    asset, market = _series(90)

    points = compute_abnormal_returns("AAA", asset, market_returns=market, min_observations=60)

    for point in points:
        if point.estimation_window_start is not None:
            assert point.estimation_window_start < point.price_date


# ---------------------------------------------------------------------------
# cumulative_abnormal_return
# ---------------------------------------------------------------------------


def _points(values: list[float | None]) -> list[ReturnPoint]:
    return [
        ReturnPoint(
            symbol="AAA",
            price_date=days(i + 1),
            total_return=0.0,
            abnormal_return=v,
        )
        for i, v in enumerate(values)
    ]


def test_car_sums_the_forward_window_only():
    points = _points([0.01, 0.02, 0.03, 0.04])

    # Starting at day 1 excludes day 1 itself: 0.02 + 0.03 = 0.05
    assert cumulative_abnormal_return(points, days(1), 2) == pytest.approx(0.05)


def test_car_excludes_the_event_day_itself():
    """An event must not get credit for a move that already happened."""
    points = _points([9.99, 0.01, 0.02])

    assert cumulative_abnormal_return(points, days(1), 2) == pytest.approx(0.03)


def test_car_is_none_on_a_short_window():
    points = _points([0.01, 0.02])

    assert cumulative_abnormal_return(points, days(1), 5) is None


def test_car_is_none_when_a_day_in_the_window_is_unmeasured():
    points = _points([0.01, None, 0.03])

    assert cumulative_abnormal_return(points, days(1), 2) is None


def test_car_rejects_a_non_positive_horizon():
    assert cumulative_abnormal_return(_points([0.01, 0.02]), days(1), 0) is None
