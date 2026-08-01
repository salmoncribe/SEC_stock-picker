"""Pure, fail-closed portfolio and operational gate for decision candidates.

This module never places or changes an order.  It merely determines whether a
fully audited research candidate may be *notified* after every earlier gate
has passed.  Persistence of reservations and kill-switch history belongs to
the coordinator so this code remains deterministic and easy to replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PortfolioPolicy:
    max_gross_notional: float
    max_net_notional: float
    max_sector_notional: float
    max_daily_loss: float
    max_catalyst_notional: float


@dataclass(frozen=True)
class RiskReservation:
    """An already-notified or pending candidate consuming portfolio capacity."""

    opportunity_id: str
    ticker: str
    sector: str | None
    catalyst_key: str | None
    signed_notional: float
    max_loss: float


@dataclass(frozen=True)
class CandidateRisk:
    opportunity_id: str
    ticker: str
    sector: str | None
    catalyst_key: str | None
    signed_notional: float
    max_loss: float


@dataclass(frozen=True)
class PortfolioGateResult:
    allowed: bool
    suppression_reasons: tuple[str, ...] = field(default_factory=tuple)


def evaluate(
    candidate: CandidateRisk,
    reservations: list[RiskReservation],
    policy: PortfolioPolicy,
    *,
    daily_realized_loss: float = 0.0,
    kill_switch_active: bool = False,
) -> PortfolioGateResult:
    """Evaluate capacity without mutating it.

    Missing sector/catalyst values are not silently diversified: they block a
    candidate because concentration cannot be measured.  An existing
    reservation for the same opportunity is treated as a duplicate, avoiding a
    second notification with a fresh apparent capacity check.
    """
    reasons: list[str] = []
    if kill_switch_active:
        reasons.append("kill_switch_active")
    if candidate.sector is None:
        reasons.append("unknown_sector")
    if candidate.catalyst_key is None:
        reasons.append("unknown_catalyst")
    if candidate.max_loss <= 0 or candidate.signed_notional == 0:
        reasons.append("invalid_candidate_risk")
    if any(item.opportunity_id == candidate.opportunity_id for item in reservations):
        reasons.append("duplicate_portfolio_reservation")

    all_signed = [item.signed_notional for item in reservations] + [candidate.signed_notional]
    gross = sum(abs(value) for value in all_signed)
    net = abs(sum(all_signed))
    if gross > policy.max_gross_notional:
        reasons.append("gross_exposure_limit")
    if net > policy.max_net_notional:
        reasons.append("net_exposure_limit")

    if candidate.sector is not None:
        sector_notional = sum(
            abs(item.signed_notional)
            for item in reservations
            if item.sector == candidate.sector
        ) + abs(candidate.signed_notional)
        if sector_notional > policy.max_sector_notional:
            reasons.append("sector_exposure_limit")
    if candidate.catalyst_key is not None:
        catalyst_notional = sum(
            abs(item.signed_notional)
            for item in reservations
            if item.catalyst_key == candidate.catalyst_key
        ) + abs(candidate.signed_notional)
        if catalyst_notional > policy.max_catalyst_notional:
            reasons.append("catalyst_overlap_limit")

    loss = daily_realized_loss + sum(item.max_loss for item in reservations) + candidate.max_loss
    if loss > policy.max_daily_loss:
        reasons.append("daily_loss_limit")
    return PortfolioGateResult(allowed=not reasons, suppression_reasons=tuple(reasons))


__all__ = [
    "CandidateRisk",
    "PortfolioGateResult",
    "PortfolioPolicy",
    "RiskReservation",
    "evaluate",
]
