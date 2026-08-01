"""Performance metrics, and the anti-overfitting statistics.

The ordinary metrics (Sharpe, Sortino, Calmar, CVaR, turnover) describe what
happened. The last three describe whether it means anything:

  * **PSR** -- Probability of Sharpe Ratio: given this track record's length,
    skew, and kurtosis, what is the chance the true Sharpe exceeds a benchmark?
    A Sharpe of 0.30 over 7 years is a different claim than the same number
    over 7 months, and a raw Sharpe cannot tell them apart.

  * **DSR** -- Deflated Sharpe Ratio: PSR against a benchmark raised to account
    for how many configurations were tried. Search 18 configs and the best one
    looks good by construction; DSR asks whether it looks good *anyway*.

  * **PBO** -- Probability of Backtest Overfitting (CSCV): split the trial
    return matrix into 16 blocks, and across every train/test partition ask how
    often the in-sample winner lands in the bottom half out-of-sample. Near 0.5
    means the selection carried no information at all.

scipy.stats.norm only. These are ~40 lines each and inspectable; that matters
more here than anywhere else, because these numbers decide whether the sealed
holdout gets read.

References
----------
Bailey, D. and Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient
Frontier." *Journal of Risk* 15(2). -- PSR.

Bailey, D. and Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
*Journal of Portfolio Management*. -- eq. (1) expected max Sharpe, eq. (2) DSR.

Bailey, D., Borwein, J., Lopez de Prado, M. and Zhu, Q. (2017). "The Probability
of Backtest Overfitting." *Journal of Computational Finance* 20(4). -- CSCV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

# This module is the platform's only scipy importer, and it uses exactly two
# functions: the normal CDF and its inverse. scipy ships no py.typed marker, so
# it is declared in pyproject's mypy ignore_missing_imports override alongside
# duckdb/pyarrow/yfinance rather than silenced inline at each import site.
from scipy.stats import norm

TRADING_DAYS_PER_YEAR = 252

# Euler-Mascheroni, the gamma of eq. (1) in Bailey & Lopez de Prado (2014).
_EULER_MASCHERONI = 0.5772156649015329

# CSCV enumerates C(S, S/2) partitions. S=16 gives 12,870, which is the whole
# point of the paper's choice; the cap stops a caller from silently asking for
# a combinatorial explosion instead of an answer.
_MAX_CSCV_BLOCKS = 20


@dataclass(frozen=True)
class PortfolioMetrics:
    """Descriptive performance. Benchmark-relative fields are None without SPY."""

    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    cvar_95: float
    turnover_annual: float
    n_days: int
    alpha: float | None = None
    beta: float | None = None
    information_ratio: float | None = None


def compute_metrics(
    daily_returns: np.ndarray,
    equity_curve: np.ndarray,
    *,
    turnover: np.ndarray | None = None,
    benchmark_returns: np.ndarray | None = None,
    risk_free_rate: float = 0.0,
) -> PortfolioMetrics:
    """Descriptive metrics. Compounded, not summed -- see the drag note in views.

    ``total_return`` is ``prod(1 + r) - 1``, never ``sum(r)``. The two agree only
    to first order, and on the volatile names this platform trades the gap is
    worth roughly the entire measured edge.

    ``annualized_return`` is geometric: ``(1 + total)^(252/n) - 1``. Sharpe and
    Sortino instead annualize the *arithmetic* mean excess return by 252 against
    a volatility scaled by ``sqrt(252)``, which is the standard convention and
    the one PSR/DSR below assume.

    ``risk_free_rate`` is an annual rate, divided by 252 to a daily rate.
    Benchmark-relative fields are ``None`` when ``benchmark_returns`` is None;
    a length mismatch is an error rather than a silent truncation.
    """
    returns = _as_series(daily_returns, name="daily_returns", minimum=2)
    equity = _as_series(equity_curve, name="equity_curve", minimum=1)
    n_days = int(returns.size)

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    growth = float(np.prod(1.0 + returns))
    total_return = growth - 1.0
    years = n_days / TRADING_DAYS_PER_YEAR

    vol = float(np.std(returns, ddof=1))
    annualized_vol = vol * math.sqrt(TRADING_DAYS_PER_YEAR)
    annualized_return = growth ** (1.0 / years) - 1.0 if growth > 0.0 else -1.0
    drawdown = max_drawdown(equity)

    turnover_annual = 0.0
    if turnover is not None:
        daily_turnover = _as_series(turnover, name="turnover", minimum=1)
        if daily_turnover.size != n_days:
            raise ValueError("turnover must have one entry per day of daily_returns")
        turnover_annual = float(np.mean(daily_turnover)) * TRADING_DAYS_PER_YEAR

    alpha: float | None = None
    beta: float | None = None
    information_ratio: float | None = None
    if benchmark_returns is not None:
        benchmark = _as_series(benchmark_returns, name="benchmark_returns", minimum=2)
        if benchmark.size != n_days:
            raise ValueError("benchmark_returns must have one entry per day of daily_returns")
        alpha, beta, information_ratio = _benchmark_relative(excess, benchmark - daily_rf)

    return PortfolioMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        sharpe=_annualized_ratio(excess, vol),
        sortino=_annualized_ratio(excess, _downside_deviation(excess)),
        calmar=annualized_return / drawdown if drawdown > 0.0 else 0.0,
        max_drawdown=drawdown,
        cvar_95=conditional_value_at_risk(returns, alpha=0.95),
        turnover_annual=turnover_annual,
        n_days=n_days,
        alpha=alpha,
        beta=beta,
        information_ratio=information_ratio,
    )


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Largest peak-to-trough decline, as a positive fraction.

    Measured against the running peak, so a curve that only rises returns 0.0.
    A curve that halves and recovers still reports 0.5 -- the recovery does not
    erase the drawdown that a live account would have had to sit through.
    """
    equity = _as_series(equity_curve, name="equity_curve", minimum=1)
    if equity[0] <= 0.0:
        raise ValueError("equity_curve must start strictly positive to have a peak")
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def conditional_value_at_risk(daily_returns: np.ndarray, *, alpha: float = 0.95) -> float:
    """Mean of the worst ``1-alpha`` tail, as a positive loss magnitude.

    The tail size is ``ceil((1 - alpha) * n)`` observations, at least one, so a
    short record still reports its single worst day rather than nothing. The
    sign is flipped on return: a positive number is a loss. A negative result is
    legitimate and means even the worst tail made money -- it is not clamped,
    because clamping would hide that the sample never saw a bad day.
    """
    returns = _as_series(daily_returns, name="daily_returns", minimum=1)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    # Rounded before the ceiling: ``(1 - 0.95) * 100`` is 5.000000000000004 in
    # binary floating point, and a naive ceiling would quietly widen the 5%
    # tail to six observations and understate the loss.
    tail_size = max(1, math.ceil(round((1.0 - alpha) * returns.size, 9)))
    worst = np.sort(returns)[:tail_size]
    return float(-np.mean(worst))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_observations: int,
    skewness: float,
    kurtosis: float,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Bailey & Lopez de Prado (2014). Tested against their published example.

    ``kurtosis`` is the raw fourth moment (3.0 for a normal), not excess.

    Passing excess kurtosis is the single most common implementation error in
    PSR: for a normal record it would supply 0.0 where the formula wants 3.0,
    shrinking the denominator and inflating PSR. Values below 1.0 are rejected
    outright, because 1.0 is the theoretical minimum of a raw fourth moment and
    anything under it is excess kurtosis wearing the wrong label.

        PSR(SR*) = Z[ (SR - SR*) sqrt(n - 1)
                      / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) ]

    All Sharpe ratios must share one frequency -- if ``n_observations`` counts
    days then the Sharpes are per-day, not annualized. The published example
    de-annualizes a Sharpe of 2.5 to ``2.5 / sqrt(250)`` for exactly this
    reason.
    """
    if n_observations < 2:
        raise ValueError("n_observations must be at least 2 for sqrt(n - 1) to be defined")
    if kurtosis < 1.0:
        raise ValueError(
            f"kurtosis is the RAW fourth moment (3.0 for a normal), not excess; got {kurtosis}"
        )
    variance = (
        1.0
        - skewness * observed_sharpe
        + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    )
    if variance <= 0.0:
        raise ValueError(
            "the estimated Sharpe variance is non-positive; check that skewness and "
            "kurtosis come from the same return series as observed_sharpe"
        )
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_observations - 1) / math.sqrt(variance)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, *, variance_of_trial_sharpes: float) -> float:
    """Expected maximum Sharpe under the null that every trial has zero skill.

    This is the bar DSR deflates against, and it rises with the number of
    trials -- which is exactly why the trial count must be preregistered rather
    than counted after the fact.

    Equation (1) of Bailey & Lopez de Prado (2014), with ``E[{SR_n}] = 0``:

        E[max SR] ~= sqrt(V) ( (1 - g) Z^-1[1 - 1/N] + g Z^-1[1 - 1/(N e)] )

    where ``g`` is Euler-Mascheroni. One trial is not a maximum over anything,
    so N <= 1 returns 0.0 -- the null mean itself -- rather than the negative
    infinity that ``Z^-1[0]`` would produce.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if variance_of_trial_sharpes < 0.0:
        raise ValueError("variance_of_trial_sharpes cannot be negative")
    if n_trials == 1 or variance_of_trial_sharpes == 0.0:
        return 0.0
    upper = float(norm.ppf(1.0 - 1.0 / n_trials))
    tail = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    quantile_blend = (1.0 - _EULER_MASCHERONI) * upper + _EULER_MASCHERONI * tail
    return math.sqrt(variance_of_trial_sharpes) * quantile_blend


def deflated_sharpe_ratio(
    observed_sharpe: float,
    trial_sharpes: np.ndarray,
    *,
    n_observations: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """PSR against the expected-max-Sharpe null. Must be monotone decreasing in trials.

    ``trial_sharpes`` is every configuration that was tried, including the
    winner. Its length is N and its sample variance is ``V[{SR_n}]``; both push
    the rejection threshold up, so a wider or a longer search has to clear a
    higher bar. Dropping the losers from this array is the precise act of
    self-deception DSR exists to prevent.

    Sample variance uses ``ddof=1``. Every Sharpe here -- observed and trial --
    must be in the same per-observation units as ``n_observations``.
    """
    trials = _as_series(trial_sharpes, name="trial_sharpes", minimum=1)
    variance = float(np.var(trials, ddof=1)) if trials.size > 1 else 0.0
    threshold = expected_max_sharpe(int(trials.size), variance_of_trial_sharpes=variance)
    return probabilistic_sharpe_ratio(
        observed_sharpe,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
        benchmark_sharpe=threshold,
    )


def probability_of_backtest_overfitting(
    trial_returns: np.ndarray, *, n_blocks: int = 16
) -> float:
    """CSCV. ``trial_returns`` is ``[T, n_trials]`` of daily returns.

    Approximately 0.5 on pure noise; low when one trial genuinely dominates.
    Both cases are tested -- a PBO implementation that only reports low numbers
    is indistinguishable from one that is broken.

    The procedure, from Bailey, Borwein, Lopez de Prado & Zhu (2017): cut the
    rows into ``n_blocks`` contiguous blocks (contiguous, so serial structure
    survives), then for every way of choosing half the blocks as training, pick
    the trial with the best in-sample Sharpe and find its rank among the
    out-of-sample Sharpes. The relative rank ``w = rank / (N + 1)`` becomes a
    logit ``log(w / (1 - w))``, and PBO is the share of partitions whose logit
    is non-positive -- the share where the in-sample winner landed in the bottom
    half out of sample.

    Every partition is enumerated, not sampled, so the result is deterministic.
    """
    returns = np.asarray(trial_returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError("trial_returns must be a 2-D [T, n_trials] array")
    n_periods, n_trials = returns.shape
    if n_trials < 2:
        raise ValueError("PBO compares trials against each other; at least 2 are required")
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError("n_blocks must be an even number of at least 2")
    if n_blocks > _MAX_CSCV_BLOCKS:
        raise ValueError(f"n_blocks above {_MAX_CSCV_BLOCKS} enumerates too many partitions")
    if n_periods < n_blocks * 2:
        raise ValueError("trial_returns needs at least 2 rows per block for a Sharpe to exist")

    # Centering per trial before accumulating leaves every subset mean and
    # variance unchanged while keeping E[x^2] - E[x]^2 far from cancellation.
    offset = returns.mean(axis=0)
    blocks = np.array_split(returns - offset, n_blocks, axis=0)
    block_len = np.array([float(block.shape[0]) for block in blocks])
    block_sum = np.array([block.sum(axis=0) for block in blocks])
    block_sq = np.array([np.square(block).sum(axis=0) for block in blocks])

    partitions = list(combinations(range(n_blocks), n_blocks // 2))
    train_mask = np.zeros((len(partitions), n_blocks))
    for row, chosen in enumerate(partitions):
        train_mask[row, list(chosen)] = 1.0

    args = (block_len, block_sum, block_sq, offset)
    in_sample = _subset_sharpe(train_mask, *args)
    out_of_sample = _subset_sharpe(1.0 - train_mask, *args)

    winners = np.argmax(in_sample, axis=1)
    winner_oos = out_of_sample[np.arange(len(partitions)), winners]
    rank = np.count_nonzero(out_of_sample <= winner_oos[:, None], axis=1)
    relative_rank = rank / (n_trials + 1.0)
    logits = np.log(relative_rank / (1.0 - relative_rank))
    return float(np.mean(logits <= 0.0))


def _as_series(values: np.ndarray, *, name: str, minimum: int) -> np.ndarray:
    """One-dimensional float view, or a named error naming what was wrong."""
    series = np.asarray(values, dtype=float).ravel()
    if series.size < minimum:
        raise ValueError(f"{name} needs at least {minimum} observation(s), got {series.size}")
    if not np.all(np.isfinite(series)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return series


def _annualized_ratio(excess: np.ndarray, deviation: float) -> float:
    """``mean * 252 / (deviation * sqrt(252))``; 0.0 when there is no deviation."""
    if deviation <= 0.0:
        return 0.0
    return float(np.mean(excess)) / deviation * math.sqrt(TRADING_DAYS_PER_YEAR)


def _downside_deviation(excess: np.ndarray) -> float:
    """Root second lower partial moment about zero, divided by the FULL sample.

    Upside days enter as zeros rather than being dropped. Dividing only by the
    count of losing days would make a strategy look better the fewer losses it
    happened to record, which is backwards.
    """
    shortfall = np.minimum(excess, 0.0)
    return float(math.sqrt(np.mean(np.square(shortfall))))


def _benchmark_relative(
    excess: np.ndarray, benchmark_excess: np.ndarray
) -> tuple[float, float, float]:
    """Annualized Jensen alpha, beta, and information ratio.

    Beta and alpha come from excess returns on both legs; the information ratio
    is measured against the raw active return ``portfolio - benchmark``, which
    is what a tracking mandate actually cares about.
    """
    benchmark_variance = float(np.var(benchmark_excess, ddof=1))
    if benchmark_variance <= 0.0:
        raise ValueError("benchmark_returns have zero variance; beta is undefined")
    covariance = float(np.cov(excess, benchmark_excess, ddof=1)[0, 1])
    beta = covariance / benchmark_variance
    alpha = (
        float(np.mean(excess)) - beta * float(np.mean(benchmark_excess))
    ) * TRADING_DAYS_PER_YEAR
    active = excess - benchmark_excess
    information_ratio = _annualized_ratio(active, float(np.std(active, ddof=1)))
    return alpha, beta, information_ratio


def _subset_sharpe(
    mask: np.ndarray,
    block_len: np.ndarray,
    block_sum: np.ndarray,
    block_sq: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Sharpe of every trial on every masked block subset, as ``[partitions, trials]``.

    The mask contracts pre-computed block sums, so all C(S, S/2) subsets cost
    two small matrix products instead of one pass over the data each.
    """
    counts = (mask @ block_len)[:, None]
    centered_mean = (mask @ block_sum) / counts
    variance = (mask @ block_sq) / counts - np.square(centered_mean)
    deviation = np.sqrt(np.maximum(variance, 0.0))
    mean = centered_mean + offset
    return np.divide(mean, deviation, out=np.zeros_like(mean), where=deviation > 0.0)


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "PortfolioMetrics",
    "compute_metrics",
    "conditional_value_at_risk",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "max_drawdown",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
]
