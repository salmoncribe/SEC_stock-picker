"""Canaries: tests of the harness itself, not of any factor.

Every other test in this suite checks that a component does what it claims.
These check that the harness as a whole **cannot manufacture an edge that isn't
there** — the exact failure that produced three months of unreplicable results.

If either canary fails, every number the harness produces is void. Do not tune
a threshold to make one pass; find the bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_intelligence.evaluation.fama_macbeth import fama_macbeth
from market_intelligence.evaluation.portfolio_sort import MIN_NAMES_PER_BUCKET, monthly_spread

N_NAMES = MIN_NAMES_PER_BUCKET * 10


def _ciks(n: int) -> list[str]:
    return [str(i).zfill(10) for i in range(n)]


def test_lookahead_canary_screams_when_given_the_future() -> None:
    """A factor built FROM future returns must post an enormous spread.

    This proves the harness can detect an edge at all. A harness silent here
    would also have been silent for a genuine factor, and nobody would have
    known why.
    """
    rng = np.random.default_rng(0)
    index = _ciks(N_NAMES)
    forward = pd.Series(rng.normal(scale=0.08, size=N_NAMES), index=index)
    cheating_factor = forward.copy()  # perfect foreknowledge

    result = monthly_spread(factor=cheating_factor, forward_returns=forward, n_buckets=10)

    assert result.spread > 0.10, (
        "the harness failed to detect perfect foreknowledge — it cannot detect "
        "anything, and no result from it means anything"
    )


def test_lookahead_canary_goes_quiet_when_the_future_is_removed() -> None:
    """The same shape of factor, but independent of the returns it is scored on."""
    rng = np.random.default_rng(0)
    index = _ciks(N_NAMES)
    forward = pd.Series(rng.normal(scale=0.08, size=N_NAMES), index=index)
    independent_factor = pd.Series(rng.normal(size=N_NAMES), index=index)

    result = monthly_spread(factor=independent_factor, forward_returns=forward, n_buckets=10)

    assert abs(result.spread) < 0.05, (
        "the harness reported an edge from a factor independent of returns"
    )


def test_null_canary_t_stats_are_standard_normal() -> None:
    """100 random factors must produce t-stats distributed ~N(0,1).

    If pure noise scores well here, the harness has a bug and would score the
    six real factors the same way. This is the test that would have caught the
    t=8.85 concentration result before anyone believed it.
    """
    rng = np.random.default_rng(42)
    t_stats = []

    for _ in range(100):
        months = [
            (pd.Series(rng.normal(size=200)), pd.Series(rng.normal(scale=0.05, size=200)))
            for _ in range(120)
        ]
        t_stats.append(fama_macbeth(months, lags=3).t_stat)

    values = np.asarray(t_stats)

    assert abs(values.mean()) < 0.3, f"t-stats are biased: mean={values.mean():.3f}"
    assert 0.7 < values.std() < 1.4, f"t-stats are misscaled: sd={values.std():.3f}"

    false_positive_rate = float(np.mean(np.abs(values) > 1.96))
    assert false_positive_rate < 0.12, (
        f"{false_positive_rate:.0%} of PURE NOISE factors cleared p<0.05 "
        f"(expected ~5%). The harness manufactures edges."
    )


def test_null_canary_spreads_are_centred_on_zero() -> None:
    """The sort itself must not drift positive on noise."""
    rng = np.random.default_rng(7)
    index = _ciks(N_NAMES)
    spreads = []

    for _ in range(200):
        factor = pd.Series(rng.normal(size=N_NAMES), index=index)
        forward = pd.Series(rng.normal(scale=0.05, size=N_NAMES), index=index)
        spreads.append(monthly_spread(factor=factor, forward_returns=forward, n_buckets=10).spread)

    mean_spread = float(np.mean(spreads))
    standard_error = float(np.std(spreads) / np.sqrt(len(spreads)))

    assert abs(mean_spread) < 3 * standard_error, (
        f"noise sorts drift to {mean_spread:+.5f} ({mean_spread / standard_error:.1f} SE from 0)"
    )
