"""The five statistics, reported together, and the promotion rule.

The pre-registration requires all five statistics be reported together; no
subset may be quoted alone. That is enforced here in the type rather than left
to discipline, because "we only mentioned the Sharpe" is how a fragile result
becomes a believed one.

``promotes()`` encodes the four conditions fixed in
``docs/preregistration/2026-08-09-factor-slate.md`` §4 — before any result
existed. A factor failing any condition is reported as failed and is **not**
re-specified and retried; re-specification after seeing a result is a new trial
and increments ``n_trials``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SURVIVORSHIP_LABELS = frozenset({"clean", "bounded", "biased"})
TRUSTWORTHY_SURVIVORSHIP = frozenset({"clean", "bounded"})

MIN_DEFLATED_SHARPE = 0.95
MIN_POSITIVE_CPCV_PATHS = 4


@dataclass(frozen=True)
class FactorReport:
    """Everything required to judge one factor. All fields mandatory."""

    factor: str
    sharpe: float
    mean_monthly_spread: float
    value_weighted_spread: float
    fama_macbeth_t: float
    cpcv_sharpes: list[float]
    deflated_sharpe: float
    survivorship: str
    n_trials: int
    bucket_counts: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.survivorship not in SURVIVORSHIP_LABELS:
            raise ValueError(
                f"survivorship must be one of {sorted(SURVIVORSHIP_LABELS)}, "
                f"got {self.survivorship!r}"
            )

    @property
    def positive_cpcv_paths(self) -> int:
        return sum(1 for s in self.cpcv_sharpes if s > 0)

    @property
    def signs_agree(self) -> bool:
        """Equal- and value-weighted spreads must point the same way.

        A factor that works only equal-weighted is a micro-cap illiquidity
        artifact, not an edge — the smallest names dominate an equal-weighted
        bucket and are the ones you cannot actually trade.
        """
        return bool(np.sign(self.mean_monthly_spread) == np.sign(self.value_weighted_spread))

    def promotes(self) -> bool:
        """The four pre-registered conditions for a VAULT read. All required."""
        return (
            self.deflated_sharpe > MIN_DEFLATED_SHARPE
            and self.positive_cpcv_paths >= MIN_POSITIVE_CPCV_PATHS
            and self.signs_agree
            and self.survivorship in TRUSTWORTHY_SURVIVORSHIP
        )

    def summary(self) -> str:
        verdict = "PROMOTE" if self.promotes() else "fail"
        return (
            f"{self.factor:20} {verdict:8} "
            f"SR={self.sharpe:+.2f} spread={self.mean_monthly_spread:+.4f} "
            f"vw={self.value_weighted_spread:+.4f} t={self.fama_macbeth_t:+.2f} "
            f"DSR={self.deflated_sharpe:.3f} "
            f"cpcv={self.positive_cpcv_paths}/{len(self.cpcv_sharpes)} "
            f"[{self.survivorship}, N={self.n_trials}]"
        )
