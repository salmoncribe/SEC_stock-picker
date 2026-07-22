"""Impact statistics and the admission verdict. Pure and deterministic.

Turns a pile of (event, target, horizon) samples into the one thing an alert is
allowed to claim: how often this kind of event moved this kind of target the
predicted way, and by how much.

**Clustering is not optional here.** The unit of observation is the (target,
t0) pair, never the individual sample. One vesting event makes dozens of
officers each file a Form 4 line item on the same day -- 327 of them for a
single company-day in the collected data -- and every one of those samples
carries an *identical* forward return. Counted as independent draws they
inflate confidence by about the square root of the cluster size; measured
across the real dataset the naive t-statistic ran roughly three times the
clustered one. A gate fed unclustered statistics admits cells at three times
its intended confidence, which is the exact failure it exists to prevent,
arriving through the arithmetic rather than through the data.

**The reported t-statistic is an upper bound on confidence, not a p-value.**
Clustering removes same-day duplication but not the rest of the dependence:
multi-day windows still overlap in time for one company, and insiders across
different companies transact in the same post-earnings weeks, so observations
share a common factor. Thresholds are therefore set well above textbook
significance, and the number is documented as optimistic rather than presented
as exact.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class Verdict(StrEnum):
    """What a cell is allowed to do.

    ``ADMITTED``  -- cleared the discovery thresholds and held up on holdout.
                     May fire alerts.
    ``DEMOTED``   -- cleared discovery but collapsed or reversed on holdout.
                     Recorded with its history, never fires. This is the state
                     that keeps an overfit cell from reaching a user.
    ``REJECTED``  -- failed the discovery thresholds.
    ``INSUFFICIENT`` -- too few independent observations to judge either way.
                     Distinct from REJECTED on purpose: "we looked and found
                     nothing" and "we could not look" are different facts, and
                     collapsing them hides where more data would help.
    """

    ADMITTED = "admitted"
    DEMOTED = "demoted"
    REJECTED = "rejected"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class AdmissionThresholds:
    """Bars a cell must clear on the discovery split to be admitted.

    Deliberately stricter than textbook significance. ``min_abs_t`` of 3.0
    against a nominal 1.96 buys margin for the residual dependence clustering
    does not remove; ``min_abs_mean_car`` demands the effect be economically
    real and not merely detectable, since a 2bp move that is statistically
    certain is still not tradeable after costs.
    """

    min_clusters: int = 200
    min_abs_t: float = 3.0
    min_abs_mean_car: float = 0.002  # 20bp
    min_hit_rate_edge: float = 0.02  # 2pp away from a coin flip


@dataclass(frozen=True)
class ImpactCell:
    """Measured behaviour of one (event kind, edge kind, horizon) combination."""

    event_type: str
    event_subtype: str | None
    edge_type: str
    horizon_days: int
    split: str
    n_samples: int
    n_clusters: int
    mean_car: float
    median_car: float
    std_car: float
    hit_rate: float
    t_stat: float

    @property
    def direction(self) -> int:
        """Sign of the measured effect: +1, -1, or 0."""
        if self.mean_car > 0:
            return 1
        if self.mean_car < 0:
            return -1
        return 0


def cluster_samples(
    samples: list[tuple[str, date, float]],
) -> list[float]:
    """Collapse ``(target, t0, car)`` rows to one observation per company-day.

    Averaging within a cluster rather than picking one member keeps every
    sample's information while counting the company-day once, which is what the
    observation actually is.
    """
    grouped: dict[tuple[str, date], list[float]] = {}
    for target, t0, car in samples:
        grouped.setdefault((target, t0), []).append(car)
    return [statistics.fmean(values) for values in grouped.values()]


def summarize(
    samples: list[tuple[str, date, float]],
    *,
    event_type: str,
    event_subtype: str | None,
    edge_type: str,
    horizon_days: int,
    split: str,
) -> ImpactCell | None:
    """Compute one cell's statistics from its samples.

    Returns ``None`` for fewer than two clusters, where a standard deviation is
    undefined -- not a zero-variance cell, which would report infinite
    confidence.
    """
    clusters = cluster_samples(samples)
    if len(clusters) < 2:
        return None

    mean = statistics.fmean(clusters)
    std = statistics.stdev(clusters)
    hits = sum(1 for value in clusters if value > 0)

    # A degenerate zero-variance cluster set would divide to infinity; report
    # no confidence instead of infinite confidence.
    t_stat = 0.0 if std == 0 else mean / (std / (len(clusters) ** 0.5))

    return ImpactCell(
        event_type=event_type,
        event_subtype=event_subtype,
        edge_type=edge_type,
        horizon_days=horizon_days,
        split=split,
        n_samples=len(samples),
        n_clusters=len(clusters),
        mean_car=mean,
        median_car=statistics.median(clusters),
        std_car=std,
        hit_rate=hits / len(clusters),
        t_stat=t_stat,
    )


def judge(
    discovery: ImpactCell | None,
    holdout: ImpactCell | None,
    thresholds: AdmissionThresholds | None = None,
) -> tuple[Verdict, str]:
    """Decide what a cell may do, and say why in words.

    Admission is decided on ``discovery`` alone; ``holdout`` can only take the
    verdict away, never grant it. Deciding on the same data the cell was
    selected from would make the reported hit rate a description of the sample
    it won on rather than a prediction about new data.

    The returned reason is stored alongside the verdict so a demoted cell
    explains itself later, when nobody remembers which threshold it missed.
    """
    limits = thresholds or AdmissionThresholds()

    if discovery is None or discovery.n_clusters < limits.min_clusters:
        found = discovery.n_clusters if discovery else 0
        return (
            Verdict.INSUFFICIENT,
            f"{found} independent observations on discovery, need {limits.min_clusters}",
        )

    if abs(discovery.t_stat) < limits.min_abs_t:
        return (
            Verdict.REJECTED,
            f"discovery |t| {abs(discovery.t_stat):.2f} below {limits.min_abs_t}",
        )

    if abs(discovery.mean_car) < limits.min_abs_mean_car:
        return (
            Verdict.REJECTED,
            f"discovery mean CAR {discovery.mean_car:.4%} is below the "
            f"{limits.min_abs_mean_car:.2%} economic floor",
        )

    if abs(discovery.hit_rate - 0.5) < limits.min_hit_rate_edge:
        return (
            Verdict.REJECTED,
            f"discovery hit rate {discovery.hit_rate:.1%} is within "
            f"{limits.min_hit_rate_edge:.0%} of a coin flip",
        )

    if holdout is None or holdout.n_clusters < 2:
        return Verdict.DEMOTED, "no holdout observations to confirm the discovery result"

    if holdout.direction != discovery.direction:
        return (
            Verdict.DEMOTED,
            f"holdout reversed sign: discovery {discovery.mean_car:+.4%}, "
            f"holdout {holdout.mean_car:+.4%}",
        )

    if abs(holdout.mean_car) < limits.min_abs_mean_car:
        return (
            Verdict.DEMOTED,
            f"holdout mean CAR {holdout.mean_car:+.4%} collapsed below the economic floor",
        )

    return (
        Verdict.ADMITTED,
        f"discovery {discovery.mean_car:+.4%} (t={discovery.t_stat:.2f}, "
        f"n={discovery.n_clusters}), holdout {holdout.mean_car:+.4%} "
        f"(hit {holdout.hit_rate:.1%}, n={holdout.n_clusters})",
    )


__all__ = [
    "AdmissionThresholds",
    "ImpactCell",
    "Verdict",
    "cluster_samples",
    "judge",
    "summarize",
]
