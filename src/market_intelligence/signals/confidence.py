"""Confidence: compress an alert's evidence into one auditable 0-100 number.

The score is derived from *measured* quantities — the cell's holdout hit rate
(shrunk toward 50 when the sample is thin), how often the underlying
relationship was independently asserted, and the extractor's own confidence.
The rendered message always shows the components next to the score, so the
number can be audited rather than believed.

The blend weights are deliberately a judgment call, not a discovery:
# TODO(michael): these weights are yours to tune. The contract tests in
# tests/test_confidence.py pin bounds/monotonicity/shrinkage only — any
# weighting that keeps those properties is fair game.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Pseudo-observations pulling a thin cell's hit rate toward a coin flip.
SHRINKAGE_N = 20


@dataclass(frozen=True)
class ConfidenceInputs:
    hit_rate: float | None          # holdout hit rate of the cell, 0..1
    n_clusters: int                 # independent holdout observations behind it
    times_asserted: int             # independent filings asserting the edge (0 for self)
    extraction_confidence: float | None  # extractor's own 0..1 confidence
    has_track_record: bool          # False for gap alerts until the ledger matures
    corroborating_people: int = 1   # distinct insiders transacting this ticker's way
                                     # within the trailing corroboration window, incl. this one


def shrunk_hit_rate(hit_rate: float | None, n: int) -> float:
    """Hit rate pulled toward 0.5 as evidence thins; exactly 0.5 with none."""
    if hit_rate is None or n <= 0:
        return 0.5
    return 0.5 + (hit_rate - 0.5) * (n / (n + SHRINKAGE_N))


def score(inputs: ConfidenceInputs) -> int:
    """Blend the components into 0-100. Weights: see module TODO."""
    base = shrunk_hit_rate(inputs.hit_rate, inputs.n_clusters) * 100.0

    # Corroboration bonuses are small on purpose: they refine a measured base,
    # they must never manufacture confidence a track record didn't earn. The
    # inputs are sanitized to their natural domains first -- these arrive from
    # nullable DB columns, and a stray NULL or out-of-scale value must distort
    # nothing (a negative count is not "anti-evidence"; confidence is 0..1).
    edge_bonus = min(max(inputs.times_asserted or 0, 0), 5) * 1.0
    extract_bonus = min(1.0, max(0.0, inputs.extraction_confidence or 0.0)) * 5.0

    # A second independent insider transacting the same qualifying way on the
    # same ticker within the trailing window is at least as strong a signal
    # as one alone -- measured 120d edge 11.5%/21.2% (search/confirm) for
    # exactly 2, vs 7.9%/21.0% for exactly 1. A third or more is NOT
    # stronger: this is precisely where the old cluster-buy detector's flaw
    # lived -- it looked strong in-sample (search 9.2%) but collapsed
    # out-of-sample (confirm 2.1%, failing year-robustness). So the bonus
    # applies only at exactly 2 corroborators; 1 and 3+ get none -- a crowd
    # is a caution, not confirmation, and does not get penalized either,
    # since the underlying signal can still be real on its own.
    corroboration_bonus = 3.0 if inputs.corroborating_people == 2 else 0.0

    value = base + edge_bonus + extract_bonus + corroboration_bonus
    if not inputs.has_track_record:
        value = min(value, 50.0)
    return int(max(0.0, min(100.0, round(value))))


__all__ = ["SHRINKAGE_N", "ConfidenceInputs", "score", "shrunk_hit_rate"]
