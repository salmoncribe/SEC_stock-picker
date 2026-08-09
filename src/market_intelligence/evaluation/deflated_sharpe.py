"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

Answers the question this project never asked: *given that N strategies were
tried, what is the probability this Sharpe is real?*

Testing ~50 variants at a 5% threshold manufactures ~2.5 "discoveries" by
construction — which is roughly how many this project produced before anyone
counted the trials. ``n_trials`` MUST come from
``docs/preregistration/2026-08-09-factor-slate.md``, not from recollection;
that is the entire point of writing the slate down first.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(*, sr_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent null strategies.

    This is the bar a result must clear. With one trial the bar is zero; with
    fifty, the best of fifty coin-flips is expected to look good on its own.
    """
    if n_trials <= 1:
        return 0.0

    gamma = EULER_MASCHERONI
    term = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials) + gamma * norm.ppf(
        1.0 - 1.0 / (n_trials * np.e)
    )
    return float(np.sqrt(sr_variance) * term)


def deflated_sharpe(
    *,
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    sr_variance: float,
    n_trials: int,
) -> float:
    """Probability the true Sharpe exceeds the trial-adjusted benchmark.

    Negative skew and fat tails reduce the score: a strategy that wins small
    often and loses big rarely is less trustworthy than a symmetric one with the
    same headline Sharpe.
    """
    benchmark = expected_max_sharpe(sr_variance=sr_variance, n_trials=n_trials)
    denominator = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2

    if denominator <= 0 or n_obs < 2:
        return 0.0

    statistic = (sharpe - benchmark) * np.sqrt(n_obs - 1) / np.sqrt(denominator)
    return float(norm.cdf(statistic))
