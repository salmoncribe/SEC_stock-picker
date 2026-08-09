"""Cross-sectional portfolio sort.

This is the statistic the event-study harness should have been computing.

An event study averages forward returns across events, so whichever ticker
emits the most events dominates the answer — 76% of the measured insider edge
came from one name. A portfolio sort ranks the universe each month, so one
company contributes at most one name to one bucket in one month, and event
concentration cannot inflate the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_NAMES_PER_BUCKET = 20
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99


@dataclass(frozen=True)
class SortResult:
    """One month of a sort. ``spread`` is NaN when the month was refused."""

    spread: float
    long_return: float
    short_return: float
    n_buckets: int
    n_names: int
    reason: str = field(default="")


def compound(returns: pd.Series) -> float:
    """Geometric compounding. NEVER sum returns.

    ``forward_abnormal_return`` in the existing schema is a cumulative *sum* of
    daily returns. A real position compounds, and the gap is roughly
    ``0.5 * variance * horizon`` — for 20 days at 5% daily vol that is ~2.5%,
    the same order of magnitude as the entire edge it was used to measure.
    """
    if returns.empty:
        return 0.0
    return float(np.prod(1.0 + returns.to_numpy(dtype=float)) - 1.0)


def _winsorize(values: pd.Series) -> pd.Series:
    """Clip to the 1st/99th percentile so one absurd value cannot own a bucket."""
    return values.clip(lower=values.quantile(WINSOR_LOWER), upper=values.quantile(WINSOR_UPPER))


def monthly_spread(
    *,
    factor: pd.Series,
    forward_returns: pd.Series,
    n_buckets: int,
) -> SortResult:
    """One month: rank by factor, bucket, return long-top minus short-bottom.

    Names missing either the factor or the forward return are **dropped**, never
    zero-filled: a zero-filled missing return turns a delisted position into a
    flat one instead of a loss, which is precisely the survivorship error this
    project is trying to stop making.
    """
    paired = pd.DataFrame({"factor": factor, "fwd": forward_returns}).dropna()
    n_names = len(paired)

    if n_names < MIN_NAMES_PER_BUCKET * n_buckets:
        return SortResult(np.nan, np.nan, np.nan, n_buckets, n_names, "insufficient_breadth")

    winsorized = _winsorize(paired["factor"])
    # rank(method="first") breaks ties deterministically, so qcut cannot fail
    # on duplicate factor values (common for sparse fundamental data).
    buckets = pd.qcut(winsorized.rank(method="first"), n_buckets, labels=False)

    returns = _winsorize(paired["fwd"])
    long_return = float(returns[buckets == n_buckets - 1].mean())
    short_return = float(returns[buckets == 0].mean())

    return SortResult(
        spread=long_return - short_return,
        long_return=long_return,
        short_return=short_return,
        n_buckets=n_buckets,
        n_names=n_names,
    )
