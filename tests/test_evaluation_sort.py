"""Portfolio sort: bucketing, compounding, breadth rules.

Each test here corresponds to a documented past failure in this project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_intelligence.evaluation.portfolio_sort import (
    MIN_NAMES_PER_BUCKET,
    SortResult,
    compound,
    monthly_spread,
)

N_NAMES = MIN_NAMES_PER_BUCKET * 10


def _ciks(n: int) -> list[str]:
    return [str(i).zfill(10) for i in range(n)]


def test_returns_compound_not_sum() -> None:
    # +10% then -10% is -1%, not 0%. The existing forward_abnormal_return
    # column sums daily returns and would report 0%.
    assert compound(pd.Series([0.10, -0.10])) == pytest.approx(-0.01)


def test_compound_of_empty_is_zero() -> None:
    assert compound(pd.Series([], dtype=float)) == 0.0


def test_spread_is_long_top_minus_short_bottom() -> None:
    # factor value == next-month return, so the spread must be strongly positive
    index = _ciks(N_NAMES)
    factor = pd.Series(np.linspace(0.0, 1.0, N_NAMES), index=index)

    result = monthly_spread(factor=factor, forward_returns=factor, n_buckets=10)

    assert isinstance(result, SortResult)
    assert result.spread > 0.5
    assert result.n_buckets == 10
    assert result.n_names == N_NAMES


def test_thin_month_yields_nan_not_partial_sort() -> None:
    # 10 names cannot fill 10 buckets at 20/bucket. Refuse, never average.
    index = _ciks(10)
    factor = pd.Series(np.arange(10.0), index=index)

    result = monthly_spread(factor=factor, forward_returns=factor, n_buckets=10)

    assert np.isnan(result.spread)
    assert result.reason == "insufficient_breadth"


def test_winsorization_caps_outliers() -> None:
    index = _ciks(N_NAMES)
    factor = pd.Series(np.linspace(0.0, 1.0, N_NAMES), index=index)
    forward = factor.copy()
    forward.iloc[-1] = 1e9  # one absurd return

    result = monthly_spread(factor=factor, forward_returns=forward, n_buckets=10)

    # A single 1e9 return must not drag the long bucket's mean to ~5e7.
    assert result.long_return < 10


def test_names_missing_a_forward_return_are_dropped_not_zero_filled() -> None:
    # Treating a missing return as 0 is how a delisted name silently becomes
    # a flat position instead of a loss.
    index = _ciks(N_NAMES)
    factor = pd.Series(np.linspace(0.0, 1.0, N_NAMES), index=index)
    forward = factor.copy()
    forward.iloc[:50] = np.nan

    result = monthly_spread(factor=factor, forward_returns=forward, n_buckets=10)

    assert result.n_names == N_NAMES - 50
