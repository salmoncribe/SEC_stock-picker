"""Fama-MacBeth, deflated Sharpe, and purged CPCV."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from market_intelligence.evaluation.cpcv import cpcv_splits, n_backtest_paths
from market_intelligence.evaluation.deflated_sharpe import deflated_sharpe
from market_intelligence.evaluation.fama_macbeth import fama_macbeth


def _noise_months(rng: np.random.Generator, n: int = 120) -> list:
    return [
        (pd.Series(rng.normal(size=200)), pd.Series(rng.normal(scale=0.05, size=200)))
        for _ in range(n)
    ]


# --------------------------------------------------------------------------
# Fama-MacBeth
# --------------------------------------------------------------------------


def test_fama_macbeth_recovers_a_known_slope() -> None:
    rng = np.random.default_rng(0)
    months = []
    for _ in range(120):
        x = pd.Series(rng.normal(size=200))
        y = 0.02 * x + pd.Series(rng.normal(scale=0.05, size=200))
        months.append((x, y))

    result = fama_macbeth(months, lags=3)

    assert result.mean_coefficient == pytest.approx(0.02, abs=0.005)
    assert result.t_stat > 3
    assert result.n_months == 120


def test_fama_macbeth_finds_nothing_in_noise() -> None:
    """Noise t-stats must CENTRE on zero.

    Deliberately not asserted on a single seed. One draw from an N(0,1) variable
    exceeds 2.5 about 1.2% of the time by construction, so a single-seed
    assertion is a coin that lands tails once every eighty runs regardless of
    whether the code is correct — and the temptation when it fails is to raise
    the threshold, which silently weakens the very check that matters.

    The distribution is the property worth testing. Verified separately over 300
    draws: mean -0.014, sd 1.024, |t|>1.96 at 6.3%, |t|>2.50 at 1.0%.
    """
    rng = np.random.default_rng(1)
    t_stats = [fama_macbeth(_noise_months(rng, n=60), lags=3).t_stat for _ in range(20)]

    mean_t = float(np.mean(t_stats))
    standard_error = float(np.std(t_stats) / np.sqrt(len(t_stats)))

    assert abs(mean_t) < 3 * standard_error, (
        f"noise t-stats centre on {mean_t:+.3f}, {mean_t / standard_error:.1f} SE from zero"
    )


def test_fama_macbeth_skips_months_too_thin_to_regress() -> None:
    rng = np.random.default_rng(2)
    months = _noise_months(rng, n=10)
    months.append((pd.Series([1.0]), pd.Series([1.0])))  # single observation

    assert fama_macbeth(months, lags=3).n_months == 10


# --------------------------------------------------------------------------
# Deflated Sharpe
# --------------------------------------------------------------------------


def test_more_trials_lowers_the_deflated_sharpe() -> None:
    kwargs = {"sharpe": 1.0, "n_obs": 120, "skew": 0.0, "kurtosis": 3.0, "sr_variance": 0.25}
    assert deflated_sharpe(n_trials=1, **kwargs) > deflated_sharpe(n_trials=50, **kwargs), (
        "testing 50 strategies must raise the bar, not lower it"
    )


def test_a_strong_single_trial_result_survives() -> None:
    assert (
        deflated_sharpe(
            sharpe=2.0, n_obs=240, skew=0.0, kurtosis=3.0, sr_variance=0.1, n_trials=1
        )
        > 0.95
    )


def test_dsr_is_a_probability() -> None:
    value = deflated_sharpe(
        sharpe=0.5, n_obs=60, skew=-1.0, kurtosis=6.0, sr_variance=0.3, n_trials=20
    )
    assert 0.0 <= value <= 1.0


def test_negative_skew_and_fat_tails_are_penalised() -> None:
    # Strategies that lose big rarely and win small often should score worse
    # than a symmetric one with the same Sharpe.
    base = {"sharpe": 1.0, "n_obs": 120, "sr_variance": 0.2, "n_trials": 6}
    symmetric = deflated_sharpe(skew=0.0, kurtosis=3.0, **base)
    crash_prone = deflated_sharpe(skew=-2.0, kurtosis=12.0, **base)
    assert crash_prone < symmetric


# --------------------------------------------------------------------------
# CPCV
# --------------------------------------------------------------------------


def test_path_count_matches_the_formula() -> None:
    # phi = k/N * C(N,k); N=6, k=2 -> 2/6 * 15 = 5
    splits = list(cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=0))
    assert len(splits) == 15
    assert n_backtest_paths(n_groups=6, n_test_groups=2) == 5


def test_train_and_test_never_overlap() -> None:
    for train, test in cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=0):
        assert not (set(train.tolist()) & set(test.tolist()))


def test_embargo_removes_observations_after_each_test_block() -> None:
    no_embargo = list(cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=0))
    embargoed = list(cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=5))

    for (train_a, _), (train_b, _) in zip(no_embargo, embargoed, strict=True):
        assert len(train_b) <= len(train_a), "embargo may only shrink the training set"

    assert sum(len(t) for t, _ in embargoed) < sum(len(t) for t, _ in no_embargo)


def test_every_observation_is_tested_exactly_once_per_path_set() -> None:
    splits = list(cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=0))
    tested = np.concatenate([test for _, test in splits])
    counts = np.bincount(tested, minlength=120)
    # Each observation appears in C(N-1, k-1) = C(5,1) = 5 test sets.
    assert set(counts.tolist()) == {5}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _passing_report():
    from market_intelligence.evaluation.report import FactorReport

    return FactorReport(
        factor="x",
        sharpe=1.2,
        mean_monthly_spread=0.01,
        value_weighted_spread=0.008,
        fama_macbeth_t=3.1,
        cpcv_sharpes=[0.9, 1.1, 1.0, 1.3, 0.8],
        deflated_sharpe=0.97,
        survivorship="clean",
        n_trials=6,
        bucket_counts={10: 120},
    )


def test_report_requires_every_statistic() -> None:
    from market_intelligence.evaluation.report import FactorReport

    required = {
        "sharpe",
        "mean_monthly_spread",
        "fama_macbeth_t",
        "cpcv_sharpes",
        "deflated_sharpe",
        "survivorship",
        "n_trials",
        "bucket_counts",
    }
    fields = {f.name for f in dataclasses.fields(FactorReport)}
    assert required <= fields, f"missing: {required - fields}"


def test_promotion_requires_all_four_conditions() -> None:
    passing = _passing_report()
    assert passing.promotes()

    # Each pre-registered condition must independently block promotion.
    assert not dataclasses.replace(passing, deflated_sharpe=0.80).promotes()
    assert not dataclasses.replace(passing, value_weighted_spread=-0.004).promotes()
    assert not dataclasses.replace(passing, cpcv_sharpes=[0.9, -0.1, -0.2, 1.3, 0.8]).promotes()
    assert not dataclasses.replace(passing, survivorship="biased").promotes()


def test_report_rejects_an_unknown_survivorship_label() -> None:
    with pytest.raises(ValueError, match="survivorship"):
        dataclasses.replace(_passing_report(), survivorship="probably fine")
