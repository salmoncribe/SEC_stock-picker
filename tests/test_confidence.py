"""Confidence contract tests: bounds, monotonicity, small-n humility.

Deliberately weight-agnostic — the blend weights are Michael's to tune; these
tests only pin properties any sane blend must satisfy.
"""

from __future__ import annotations

from typing import Any

from market_intelligence.signals.confidence import ConfidenceInputs, score, shrunk_hit_rate


def _inputs(**overrides: Any) -> ConfidenceInputs:
    base: dict[str, Any] = {
        "hit_rate": 0.65,
        "n_clusters": 40,
        "times_asserted": 0,
        "extraction_confidence": 0.9,
        "has_track_record": True,
    }
    base.update(overrides)
    return ConfidenceInputs(**base)


class TestShrinkage:
    def test_small_n_pulls_toward_half(self):
        assert abs(shrunk_hit_rate(0.9, 2) - 0.5) < abs(shrunk_hit_rate(0.9, 200) - 0.5)

    def test_zero_n_is_half(self):
        assert shrunk_hit_rate(0.9, 0) == 0.5


class TestScore:
    def test_bounded(self):
        for hr in (0.0, 0.5, 1.0):
            for n in (0, 5, 500):
                s = score(_inputs(hit_rate=hr, n_clusters=n))
                assert 0 <= s <= 100

    def test_monotonic_in_hit_rate(self):
        assert score(_inputs(hit_rate=0.8)) >= score(_inputs(hit_rate=0.55))

    def test_more_evidence_never_hurts(self):
        assert score(_inputs(times_asserted=5)) >= score(_inputs(times_asserted=0))
        assert score(_inputs(extraction_confidence=0.95)) >= score(
            _inputs(extraction_confidence=0.30)
        )

    def test_score_respects_small_n_shrinkage(self):
        # The one regression the whole contract exists to catch: a reweight
        # that reads the raw hit rate and forgets to shrink. At hit_rate=0.9,
        # thin evidence must score below deep evidence, and no evidence must
        # collapse toward the 50 midpoint.
        thin = score(_inputs(hit_rate=0.9, n_clusters=2))
        deep = score(_inputs(hit_rate=0.9, n_clusters=200))
        none_at_all = score(_inputs(hit_rate=0.9, n_clusters=0))
        assert thin < deep
        assert abs(none_at_all - 50) <= 10

    def test_garbage_inputs_cannot_distort_the_bonuses(self):
        # Negative counts are not anti-evidence; out-of-scale confidences are
        # clamped to 0..1 before weighting.
        assert score(_inputs(times_asserted=-100)) == score(_inputs(times_asserted=0))
        assert score(_inputs(extraction_confidence=25.0)) == score(
            _inputs(extraction_confidence=1.0)
        )

    def test_no_track_record_is_capped(self):
        s = score(_inputs(has_track_record=False, hit_rate=None, n_clusters=0))
        assert s <= 50

    def test_none_hit_rate_treated_as_no_signal(self):
        s = score(_inputs(hit_rate=None))
        assert 0 <= s <= 60
