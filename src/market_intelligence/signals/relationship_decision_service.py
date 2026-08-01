"""Explicit, fail-closed entry point for relationship decision research.

Nothing schedules this service from the existing alert paths.  A caller must
provide a frozen opportunity input and a model provider; configuration starts
disabled and missing external-provider decisions leave the result in
research-only state.  It never sends Telegram or routes an order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from market_intelligence.llm.opportunity_decider import decide_opportunity
from market_intelligence.schemas.opportunities import (
    FrozenEvidencePacket,
    OpportunityEvaluation,
    RelationshipOpportunityInput,
    policy_from_config_mapping,
)
from market_intelligence.signals.relationship_decision_workflow import (
    PersistedDecision,
    decide_after_persist,
    persist_evaluation,
)
from market_intelligence.signals.relationship_opportunities import (
    build_frozen_evidence_packet,
    evaluate_relationship_opportunity,
)

if TYPE_CHECKING:
    from market_intelligence.config import Config
    from market_intelligence.llm.base import LLMProvider


@dataclass(frozen=True)
class RelationshipDecisionRun:
    evaluation: OpportunityEvaluation
    packet: FrozenEvidencePacket
    decision: PersistedDecision | None
    research_only_reasons: tuple[str, ...]


def evaluate_configured_opportunity(
    config: Config,
    opportunity: RelationshipOpportunityInput,
) -> tuple[OpportunityEvaluation, FrozenEvidencePacket]:
    """Score against the exact policy derived from human-readable config."""
    policy = policy_from_config_mapping(config.settings.relationship_decision.model_dump())
    evaluation = evaluate_relationship_opportunity(opportunity, policy)
    return evaluation, build_frozen_evidence_packet(opportunity, evaluation)


def run_configured_decision(
    config: Config,
    opportunity: RelationshipOpportunityInput,
    provider: LLMProvider | None = None,
    *,
    now: datetime | None = None,
) -> RelationshipDecisionRun:
    """Persist research evidence and call the LLM only when all prerequisites pass.

    Provider readiness is deliberately checked *before* a model call.  A
    disabled or unconfigured environment still stores the deterministic result
    for later replay, but cannot create a verdict that downstream code might
    mistake for a live candidate.
    """
    decision_time = now or datetime.now(UTC)
    evaluation, packet = evaluate_configured_opportunity(config, opportunity)
    readiness, readiness_reasons = config.relationship_decision_readiness()
    persist_evaluation(config.paths.database_path, packet, now=decision_time)
    if not readiness:
        return RelationshipDecisionRun(evaluation, packet, None, readiness_reasons)
    if not evaluation.llm_eligible:
        return RelationshipDecisionRun(
            evaluation,
            packet,
            None,
            tuple(str(reason) for reason in evaluation.suppression_reasons),
        )
    if provider is None:
        return RelationshipDecisionRun(evaluation, packet, None, ("llm_provider_unconfigured",))

    def decider(frozen_packet: FrozenEvidencePacket) -> PersistedDecision:
        decision = decide_opportunity(frozen_packet, provider, now=decision_time)
        return PersistedDecision(
            decision_id=decision.decision_id,
            verdict=decision.verdict,
            verification_status=decision.verification_status,
            verification_reasons=decision.verification_reasons,
            decision_payload=decision.decision_payload,
        )

    decision = decide_after_persist(
        config.paths.database_path,
        packet,
        decider,
        now=decision_time,
    )
    return RelationshipDecisionRun(evaluation, packet, decision, ())


__all__ = [
    "RelationshipDecisionRun",
    "evaluate_configured_opportunity",
    "run_configured_decision",
]
