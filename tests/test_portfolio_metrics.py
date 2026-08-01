"""Descriptive metrics, and the statistics that decide whether the holdout is read.

The PSR/DSR/PBO tests are the anti-overfitting police policing themselves. PSR
is anchored to the published worked example in Bailey & Lopez de Prado (2014),
"The Deflated Sharpe Ratio", pp. 9-10, and cross-checked against a closed form
evaluated with ``math.erf`` so the check does not run through the same
``scipy.stats.norm`` call the implementation uses.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from market_intelligence.portfolio.metrics import (
    compute_metrics,
    conditional_value_at_risk,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_drawdown,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)

ROOT_252 = math.sqrt(252)


def _normal_cdf(z: float) -> float:
    """Independent standard normal CDF -- not scipy, so the check is not circular."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _curve(returns: list[float], *, start: float = 100.0) -> np.ndarray:
    """Equity curve of ``len(returns) + 1`` marks, compounded from ``start``."""
    return start * np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(returns))])


def _rescaled(values: np.ndarray, *, std: float) -> np.ndarray:
    """Centre and rescale so two arrays of different length share one variance."""
    centred = values - values.mean()
    return centred / centred.std(ddof=1) * std


def test_sharpe_and_annualization_are_applied_to_both_return_and_volatility() -> None:
    returns = [0.01, -0.005, 0.02, 0.0, -0.01]
    # Hand arithmetic: sum = 0.015 over 5 days, so the daily mean is 0.003.
    # Deviations 0.007, -0.008, 0.017, -0.003, -0.013 square to a sum of
    # 0.000580, so the sample variance is 0.000580 / 4 = 0.000145.
    daily_mean = 0.003
    daily_vol = math.sqrt(0.000145)
    # Compounded, never summed: 1.01 * 0.995 * 1.02 * 1.00 * 0.99.
    growth = 1.01 * 0.995 * 1.02 * 1.00 * 0.99

    metrics = compute_metrics(np.asarray(returns), _curve(returns))

    assert metrics.n_days == 5
    assert metrics.total_return == pytest.approx(growth - 1.0)
    assert metrics.total_return == pytest.approx(0.01479851)
    assert metrics.total_return != pytest.approx(sum(returns))  # summed would be 0.015
    # Volatility annualizes by sqrt(252); the return annualizes geometrically.
    assert metrics.annualized_vol == pytest.approx(daily_vol * ROOT_252)
    assert metrics.annualized_vol == pytest.approx(0.19115439, abs=1e-8)
    assert metrics.annualized_return == pytest.approx(growth ** (252 / 5) - 1.0)
    assert metrics.annualized_return == pytest.approx(1.09673224, abs=1e-8)
    assert metrics.annualized_return != pytest.approx(daily_mean * 252)
    # Sharpe: arithmetic mean annualized by 252 over a vol annualized by sqrt(252),
    # which is the same as the daily ratio times sqrt(252).
    assert metrics.sharpe == pytest.approx(daily_mean * 252 / (daily_vol * ROOT_252))
    assert metrics.sharpe == pytest.approx(daily_mean / daily_vol * ROOT_252)
    assert metrics.sharpe == pytest.approx(3.95491837, abs=1e-8)
    assert (metrics.alpha, metrics.beta, metrics.information_ratio) == (None, None, None)

    # A risk-free rate is annual and is charged at 1/252 per day.
    charged = compute_metrics(np.asarray(returns), _curve(returns), risk_free_rate=0.252)
    assert charged.sharpe == pytest.approx((daily_mean - 0.001) / daily_vol * ROOT_252)

    # Turnover annualizes linearly: 2% of the book traded per day is 5.04x a year.
    traded = compute_metrics(np.asarray(returns), _curve(returns), turnover=np.full(5, 0.02))
    assert traded.turnover_annual == pytest.approx(0.02 * 252)

    # Benchmark-relative fields appear only with a benchmark, and alpha is the
    # daily intercept annualized by 252.
    benchmark = np.asarray(returns)
    tracking = compute_metrics(np.asarray(returns), _curve(returns), benchmark_returns=benchmark)
    assert tracking.beta == pytest.approx(1.0)
    assert tracking.alpha == pytest.approx(0.0, abs=1e-12)
    assert tracking.information_ratio == pytest.approx(0.0)
    levered = compute_metrics(benchmark * 2, _curve(returns), benchmark_returns=benchmark)
    assert levered.beta == pytest.approx(2.0)
    assert levered.alpha == pytest.approx(0.0, abs=1e-12)
    outperformer = compute_metrics(benchmark + 0.001, _curve(returns), benchmark_returns=benchmark)
    assert outperformer.beta == pytest.approx(1.0)
    assert outperformer.alpha == pytest.approx(0.001 * 252)
    with pytest.raises(ValueError, match="one entry per day"):
        compute_metrics(np.asarray(returns), _curve(returns), benchmark_returns=benchmark[:3])


def test_max_drawdown_is_the_positive_peak_to_trough_fraction() -> None:
    # Peaks run 100, 120, 120, 130, 130; the troughs give 30/120 and 26/130.
    assert max_drawdown(np.array([100.0, 120.0, 90.0, 130.0, 104.0])) == pytest.approx(0.25)
    # A curve that only rises has no drawdown at all.
    assert max_drawdown(np.array([100.0, 101.0, 102.0, 103.0])) == pytest.approx(0.0)
    assert max_drawdown(np.array([100.0, 100.0, 100.0])) == pytest.approx(0.0)
    # Recovery does not erase the drawdown a live account had to sit through.
    assert max_drawdown(np.array([100.0, 50.0, 200.0])) == pytest.approx(0.5)

    rising = [0.01, 0.02, 0.03]
    assert compute_metrics(np.asarray(rising), _curve(rising)).max_drawdown == pytest.approx(0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        max_drawdown(np.array([0.0, 100.0]))


def test_conditional_value_at_risk_is_a_positive_loss_on_the_worst_tail() -> None:
    returns = np.concatenate([[-0.10, -0.08, -0.06, -0.05, -0.04], np.full(95, 0.01)])
    # 5% of 100 observations is the worst 5: (-0.10 - 0.08 - 0.06 - 0.05 - 0.04) / 5.
    assert conditional_value_at_risk(returns, alpha=0.95) == pytest.approx(0.066)
    assert conditional_value_at_risk(returns, alpha=0.95) > 0.0
    # 1% of 100 is the single worst day.
    assert conditional_value_at_risk(returns, alpha=0.99) == pytest.approx(0.10)
    # A record that never lost money reports a negative loss rather than a clamped zero.
    assert conditional_value_at_risk(np.full(100, 0.01), alpha=0.95) == pytest.approx(-0.01)

    metrics = compute_metrics(returns, _curve(list(returns)))
    assert metrics.cvar_95 == pytest.approx(0.066)


def test_sortino_beats_sharpe_when_the_volatility_is_upside_and_calmar_uses_drawdown() -> None:
    returns = [0.05, 0.04, -0.01, 0.03, -0.01, 0.06]
    # Only two losing days, each -0.01, so the second lower partial moment over
    # the FULL six observations is (0.0001 + 0.0001) / 6.
    downside = math.sqrt(0.0002 / 6)
    assert downside == pytest.approx(0.005773502691896258)
    # Total volatility is far larger because the upside moves dominate.
    daily_mean = 0.16 / 6
    daily_vol = float(np.std(returns, ddof=1))
    assert daily_vol == pytest.approx(0.03011090610836324)

    metrics = compute_metrics(np.asarray(returns), _curve(returns))

    assert metrics.sortino == pytest.approx(daily_mean / downside * ROOT_252)
    assert metrics.sortino == pytest.approx(73.32121112, abs=1e-6)
    assert metrics.sharpe == pytest.approx(daily_mean / daily_vol * ROOT_252)
    assert metrics.sharpe == pytest.approx(14.05870047, abs=1e-6)
    assert metrics.sortino > metrics.sharpe

    # Each of the two down days is exactly -1% straight off a running peak.
    assert metrics.max_drawdown == pytest.approx(0.01)
    growth = 1.05 * 1.04 * 0.99 * 1.03 * 0.99 * 1.06
    assert metrics.calmar == pytest.approx((growth ** (252 / 6) - 1.0) / 0.01)
    # Calmar is undefined without a drawdown, and reports 0.0 rather than infinity.
    rising = [0.01, 0.02, 0.03]
    assert compute_metrics(np.asarray(rising), _curve(rising)).calmar == pytest.approx(0.0)


def test_probabilistic_sharpe_ratio_matches_the_published_worked_example() -> None:
    # Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", pp. 9-10.
    # A treasury-seasonality strategist reports an annualized SR of 2.5 over a
    # daily sample of 5 years at 250 observations per year, with N = 100 trials,
    # V[{SR_n}] = 1/2, T = 1250, skewness = -3 and kurtosis = 10. The paper
    # de-annualizes to SR = 2.5 / sqrt(250) and publishes SR_0 ~= 0.1132 and
    # DSR ~= 0.9004, which is below the 95% bar, so the investor declines.
    observed = 2.5 / math.sqrt(250)
    published = probabilistic_sharpe_ratio(
        observed, n_observations=1250, skewness=-3.0, kurtosis=10.0, benchmark_sharpe=0.1132
    )
    assert published == pytest.approx(0.9004, abs=1e-3)

    # The same number reached through the unrounded eq. (1) threshold.
    threshold = expected_max_sharpe(100, variance_of_trial_sharpes=1.0 / (2 * 250))
    assert threshold == pytest.approx(0.1132, abs=1e-4)
    assert probabilistic_sharpe_ratio(
        observed, n_observations=1250, skewness=-3.0, kurtosis=10.0, benchmark_sharpe=threshold
    ) == pytest.approx(0.9004, abs=1e-3)

    # The paper's second checkpoint: had only N = 46 trials been run, DSR = 0.9505.
    assert probabilistic_sharpe_ratio(
        observed,
        n_observations=1250,
        skewness=-3.0,
        kurtosis=10.0,
        benchmark_sharpe=expected_max_sharpe(46, variance_of_trial_sharpes=1.0 / (2 * 250)),
    ) == pytest.approx(0.9505, abs=1e-3)

    # Closed form when skew = 0 and kurtosis = 3.0 (RAW, so the (g4-1)/4 term is
    # exactly 1/2), evaluated with math.erf rather than scipy.
    sharpe, n = 0.12, 500
    analytic = _normal_cdf(sharpe * math.sqrt(n - 1) / math.sqrt(1.0 + 0.5 * sharpe**2))
    assert probabilistic_sharpe_ratio(
        sharpe, n_observations=n, skewness=0.0, kurtosis=3.0
    ) == pytest.approx(analytic, abs=1e-12)

    # Excess kurtosis for a normal is 0.0; passing it here is the classic error.
    with pytest.raises(ValueError, match="RAW fourth moment"):
        probabilistic_sharpe_ratio(sharpe, n_observations=n, skewness=0.0, kurtosis=0.0)


def test_probabilistic_sharpe_ratio_rises_with_track_record_length() -> None:
    # A short record and a long one making the same claim are not the same claim.
    lengths = [60, 125, 252, 504, 1260, 2520]
    values = [
        probabilistic_sharpe_ratio(0.05, n_observations=n, skewness=-0.5, kurtosis=4.0)
        for n in lengths
    ]
    assert values == sorted(values)
    assert all(later > earlier for earlier, later in pairwise(values))
    assert values[0] < 0.75 < values[-1]

    # 7 months of a 0.30 Sharpe is not the same evidence as 7 years of it.
    months_7 = probabilistic_sharpe_ratio(0.03, n_observations=147, skewness=0.0, kurtosis=3.0)
    years_7 = probabilistic_sharpe_ratio(0.03, n_observations=1764, skewness=0.0, kurtosis=3.0)
    assert years_7 > months_7


def test_deflated_sharpe_falls_as_the_trial_count_rises() -> None:
    variance = 0.05**2
    # The bar itself rises with N -- which is why N must be preregistered.
    bars = [expected_max_sharpe(n, variance_of_trial_sharpes=variance) for n in (5, 50, 500)]
    assert bars == sorted(bars)
    assert all(later > earlier for earlier, later in pairwise(bars))
    assert expected_max_sharpe(1, variance_of_trial_sharpes=variance) == 0.0

    # Same observed record, same trial-Sharpe variance, only the count differs.
    rng = np.random.default_rng(7)
    five = _rescaled(rng.standard_normal(5), std=0.05)
    fifty = _rescaled(rng.standard_normal(50), std=0.05)
    assert float(np.var(five, ddof=1)) == pytest.approx(float(np.var(fifty, ddof=1)))

    moments = {"n_observations": 1250, "skewness": -1.0, "kurtosis": 6.0}
    dsr_5 = deflated_sharpe_ratio(0.15, five, **moments)
    dsr_50 = deflated_sharpe_ratio(0.15, fifty, **moments)
    assert dsr_5 > dsr_50
    # Deflation only ever costs; it can never score above the undeflated PSR.
    assert dsr_5 < probabilistic_sharpe_ratio(0.15, benchmark_sharpe=0.0, **moments)


def test_pbo_is_near_one_half_on_noise_and_low_for_a_genuine_winner() -> None:
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.01, size=(520, 12))

    # Pure noise: the in-sample winner is a coin flip out of sample, so the
    # selection carried no information and PBO must land near 0.5.
    pbo_noise = probability_of_backtest_overfitting(noise)
    assert 0.35 <= pbo_noise <= 0.65

    # One noise matrix is a weak version of that claim: every partition is drawn
    # from a single realization, so the per-matrix spread is wide (a standard
    # deviation of roughly 0.22 across draws) even though nothing is predictable.
    # Averaged over independent draws the estimator is centred, which is the
    # property that actually says the statistic is unbiased rather than lucky.
    repeated = [pbo_noise] + [
        probability_of_backtest_overfitting(rng.normal(0.0, 0.01, size=(520, 12)))
        for _ in range(15)
    ]
    assert abs(sum(repeated) / len(repeated) - 0.5) <= 0.10

    # One trial genuinely dominates: a real edge, not a selection artefact.
    planted = noise.copy()
    planted[:, 0] += 0.004
    pbo_planted = probability_of_backtest_overfitting(planted)
    assert pbo_planted < 0.25
    assert pbo_planted < pbo_noise

    # Deterministic: every partition is enumerated, none is sampled.
    assert probability_of_backtest_overfitting(noise) == pbo_noise
    with pytest.raises(ValueError, match="even number"):
        probability_of_backtest_overfitting(noise, n_blocks=15)
    with pytest.raises(ValueError, match="2-D"):
        probability_of_backtest_overfitting(noise[:, 0])
