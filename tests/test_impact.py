"""Tests for impact statistics and the admission verdict.

Two of these are the controls the whole design rests on. The null control
proves the gate can reject noise; the positive control proves it can still
detect something. A gate with only the first passes by rejecting everything,
which looks like rigour and is indistinguishable from being broken.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from market_intelligence.analytics.impact import (
    AdmissionThresholds,
    Verdict,
    cluster_samples,
    judge,
    summarize,
)

DAY_ZERO = date(2020, 1, 1)


def days(n: int) -> date:
    return DAY_ZERO + timedelta(days=n)


def cell(samples, *, split="discovery", subtype="P", horizon=5):
    return summarize(
        samples,
        event_type="insider_transaction",
        event_subtype=subtype,
        edge_type="self",
        horizon_days=horizon,
        split=split,
    )


def spread(mean: float, n: int, *, sd: float = 0.05, seed: int = 7):
    """n distinct company-days drawn around `mean` — no duplicate clusters."""
    rng = random.Random(seed)
    return [(f"T{i}", days(i % 900), rng.gauss(mean, sd)) for i in range(n)]


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


def test_same_company_day_collapses_to_one_observation():
    """327 line items from one vesting event are one fact, not 327."""
    samples = [("APP", days(1), 0.17)] * 327

    assert cluster_samples(samples) == [pytest.approx(0.17)]


def test_clusters_average_rather_than_discard():
    samples = [("AAA", days(1), 0.10), ("AAA", days(1), 0.20), ("BBB", days(1), 0.30)]

    assert sorted(cluster_samples(samples)) == [pytest.approx(0.15), pytest.approx(0.30)]


def test_same_company_different_days_stay_separate():
    samples = [("AAA", days(1), 0.10), ("AAA", days(2), 0.20)]

    assert len(cluster_samples(samples)) == 2


def test_clustering_deflates_the_t_statistic():
    """The bug this exists to prevent: duplicates masquerading as evidence."""
    honest = cell([("T", days(i), 0.01) for i in range(50)])
    duplicated = cell([("T", days(i), 0.01) for i in range(50)] + [("T", days(0), 0.01)] * 500)

    assert honest is not None and duplicated is not None
    assert duplicated.n_samples > honest.n_samples
    assert duplicated.n_clusters == honest.n_clusters
    assert duplicated.t_stat == pytest.approx(honest.t_stat)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summary_reports_cluster_and_sample_counts_separately():
    samples = [("AAA", days(1), 0.05)] * 10 + [("BBB", days(2), -0.01)] * 4

    result = cell(samples)

    assert result is not None
    assert result.n_samples == 14
    assert result.n_clusters == 2


def test_summary_is_none_below_two_clusters():
    assert cell([("AAA", days(1), 0.05)] * 99) is None
    assert cell([]) is None


def test_zero_variance_reports_no_confidence_not_infinite():
    """Identical values across clusters must not divide to infinity."""
    result = cell([("A", days(1), 0.01), ("B", days(2), 0.01)])

    assert result is not None
    assert result.t_stat == 0.0


def test_hit_rate_counts_positive_clusters():
    samples = [("A", days(1), 0.01), ("B", days(2), 0.02), ("C", days(3), -0.01)]

    result = cell(samples)

    assert result is not None
    assert result.hit_rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# the two controls
# ---------------------------------------------------------------------------


def test_null_control_is_rejected():
    """Pure noise must never be admitted, at any sample size.

    If this fails the gate is broken and every result downstream is suspect.
    """
    rng = random.Random(11)
    noise = [(f"T{i}", days(i % 900), rng.gauss(0.0, 0.05)) for i in range(20_000)]

    verdict, reason = judge(cell(noise), cell(noise, split="holdout"))

    assert verdict is Verdict.REJECTED, reason


def test_positive_control_is_admitted():
    """A real effect must still get through.

    A gate that rejects noise *and* rejects a genuine +0.9% edge is not strict,
    it is inert -- and the null control alone cannot tell those apart.
    """
    verdict, reason = judge(
        cell(spread(0.009, 4000, seed=3)),
        cell(spread(0.009, 2000, seed=4), split="holdout"),
    )

    assert verdict is Verdict.ADMITTED, reason


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def test_too_few_observations_is_insufficient_not_rejected():
    """'We could not look' is a different fact from 'we looked and found nothing'."""
    verdict, reason = judge(cell(spread(0.05, 20)), cell(spread(0.05, 20), split="holdout"))

    assert verdict is Verdict.INSUFFICIENT
    assert "independent observations" in reason


def test_statistically_certain_but_economically_trivial_is_rejected():
    """A 2bp edge that is certain is still not tradeable."""
    verdict, reason = judge(
        cell(spread(0.0002, 20_000, sd=0.002, seed=5)),
        cell(spread(0.0002, 8000, sd=0.002, seed=6), split="holdout"),
    )

    assert verdict is Verdict.REJECTED
    assert "economic floor" in reason


def test_holdout_sign_reversal_demotes():
    """Admission comes from discovery; holdout can only take it away."""
    verdict, reason = judge(
        cell(spread(0.009, 4000, seed=3)),
        cell(spread(-0.009, 2000, seed=4), split="holdout"),
    )

    assert verdict is Verdict.DEMOTED
    assert "reversed sign" in reason


def test_holdout_collapse_demotes():
    verdict, reason = judge(
        cell(spread(0.009, 4000, seed=3)),
        cell(spread(0.0001, 2000, sd=0.002, seed=4), split="holdout"),
    )

    assert verdict is Verdict.DEMOTED
    assert "collapsed" in reason


def test_missing_holdout_demotes_rather_than_admits():
    """An unconfirmed cell must not fire merely because nothing contradicted it."""
    verdict, reason = judge(cell(spread(0.009, 4000, seed=3)), None)

    assert verdict is Verdict.DEMOTED
    assert "no holdout" in reason


def test_thresholds_are_configurable():
    strict = AdmissionThresholds(min_abs_mean_car=0.05)

    verdict, _ = judge(
        cell(spread(0.009, 4000, seed=3)),
        cell(spread(0.009, 2000, seed=4), split="holdout"),
        strict,
    )

    assert verdict is Verdict.REJECTED


def test_verdict_reason_is_always_populated():
    for discovery, holdout in [
        (cell(spread(0.009, 4000, seed=3)), cell(spread(0.009, 2000, seed=4), split="holdout")),
        (cell(spread(0.0, 4000, seed=8)), cell(spread(0.0, 2000, seed=9), split="holdout")),
        (None, None),
    ]:
        _, reason = judge(discovery, holdout)
        assert reason and len(reason) > 10
