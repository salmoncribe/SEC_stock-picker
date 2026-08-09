"""Fama-MacBeth cross-sectional regression with Newey-West standard errors.

This replaces the per-event t-statistics used previously. On this corpus a
per-event t-stat is invalid: events cluster in a handful of tickers, so the
effective sample size is the number of *companies*, not the number of events.
That is how a ``t=8.85, p<0.0001`` result later collapsed inside a
ticker-clustered confidence interval containing zero.

Here the regression runs cross-sectionally *within* each month, and the t-stat
is computed on the time series of monthly coefficients — so each month
contributes exactly one observation no matter how many events it held.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FamaMacBethResult:
    mean_coefficient: float
    t_stat: float
    n_months: int


def _newey_west_se(coefficients: np.ndarray, lags: int) -> float:
    """Standard error robust to autocorrelation up to ``lags`` periods.

    Monthly factor premia are serially correlated; an ordinary standard error
    understates the true uncertainty and inflates the t-stat.
    """
    demeaned = coefficients - coefficients.mean()
    n = len(demeaned)
    if n < 2:
        return 0.0

    variance = float(demeaned @ demeaned / n)
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)  # Bartlett kernel
        variance += 2.0 * weight * float(demeaned[lag:] @ demeaned[:-lag] / n)

    return float(np.sqrt(max(variance, 0.0) / n))


def fama_macbeth(
    monthly: list[tuple[pd.Series, pd.Series]],
    *,
    lags: int = 3,
) -> FamaMacBethResult:
    """``monthly`` holds one ``(factor, forward_return)`` pair per month."""
    coefficients: list[float] = []

    for factor, forward in monthly:
        paired = pd.DataFrame({"x": factor, "y": forward}).dropna()
        if len(paired) < 2:
            continue  # a month too thin to regress contributes nothing
        design = np.column_stack([np.ones(len(paired)), paired["x"].to_numpy(dtype=float)])
        beta, *_ = np.linalg.lstsq(design, paired["y"].to_numpy(dtype=float), rcond=None)
        coefficients.append(float(beta[1]))

    if not coefficients:
        return FamaMacBethResult(float("nan"), float("nan"), 0)

    values = np.asarray(coefficients, dtype=float)
    standard_error = _newey_west_se(values, lags)
    t_stat = float(values.mean() / standard_error) if standard_error > 0 else 0.0

    return FamaMacBethResult(float(values.mean()), t_stat, len(values))
