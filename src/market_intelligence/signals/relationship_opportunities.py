"""Pure Gate A/Gate B scoring for SEC relationship opportunities.

There is intentionally no database, clock, model, or market-provider import in
this module.  ``candidate_scored_at`` is the only time reference, which makes a
historical replay immune to data that arrived after its original decision.
"""

from __future__ import annotations

import math
from datetime import datetime

from market_intelligence.hashing import sha256_json
from market_intelligence.schemas.opportunities import (
    BorrowAvailability,
    Direction,
    EvidenceComponent,
    EvidenceFacts,
    FirstPublicState,
    FrozenEvidencePacket,
    HistoricalCellMetrics,
    MarketSnapshot,
    OpportunityEvaluation,
    OpportunityStatus,
    RelationshipDecisionPolicy,
    RelationshipOpportunityInput,
    SuppressionReason,
)
from market_intelligence.signals.net_return import calculate_net_economics


def _aware_not_after(value: datetime | None, as_of: datetime) -> bool:
    """Whether a timestamp is aware and point-in-time valid for this decision."""
    if value is None or value.tzinfo is None or as_of.tzinfo is None:
        return False
    try:
        return value <= as_of
    except TypeError:
        return False


def _seconds_between(later: datetime, earlier: datetime) -> float | None:
    if later.tzinfo is None or earlier.tzinfo is None:
        return None
    try:
        return (later - earlier).total_seconds()
    except TypeError:
        return None


def _aware_after(value: datetime | None, reference: datetime) -> bool:
    """Whether ``value`` is a usable strictly-future expiry relative to a decision."""
    if value is None or value.tzinfo is None or reference.tzinfo is None:
        return False
    try:
        return value > reference
    except TypeError:
        return False


def evidence_components(
    facts: EvidenceFacts,
    policy: RelationshipDecisionPolicy,
    *,
    as_of: datetime,
) -> tuple[EvidenceComponent, ...]:
    """Return every transparent score term; each term is applied at most once."""
    valid_fact_times = tuple(
        value for value in facts.related_fact_public_at if _aware_not_after(value, as_of)
    )
    timing_cluster = False
    if len(valid_fact_times) >= 2:
        timing_cluster = (
            max(valid_fact_times) - min(valid_fact_times)
        ).total_seconds() <= policy.insider_cluster_window_days * 24 * 60 * 60
    ambiguous = facts.vague_or_ambiguous_entity or facts.weak_extraction
    return (
        EvidenceComponent(
            name="new_material_fact",
            points=policy.material_fact_points,
            applied=facts.new_material_fact,
            detail="new material relationship fact",
        ),
        EvidenceComponent(
            name="named_public_counterparty",
            points=policy.named_counterparty_points,
            applied=facts.named_public_company_counterparty,
            detail="named public-company counterparty",
        ),
        EvidenceComponent(
            name="qualifying_open_market_insider_purchase",
            points=policy.insider_purchase_points,
            applied=facts.qualifying_open_market_insider_purchase,
            detail="CEO, CFO, or director open-market purchase",
        ),
        EvidenceComponent(
            name="distinct_insider_cluster",
            points=policy.insider_cluster_points,
            applied=facts.distinct_insider_buyers >= 2,
            detail="two or more distinct insiders buying",
        ),
        EvidenceComponent(
            name="validated_relationship",
            points=policy.validated_relationship_points,
            applied=facts.validated_relationship,
            detail="existing validated relationship corroborates the fact",
        ),
        EvidenceComponent(
            name="material_magnitude",
            points=policy.material_magnitude_points,
            applied=facts.material_magnitude,
            detail="contract/value magnitude is material",
        ),
        EvidenceComponent(
            name="timing_cluster",
            points=policy.timing_cluster_points,
            applied=timing_cluster,
            detail="facts were public within the configured window",
        ),
        EvidenceComponent(
            name="shared_board_member",
            points=policy.shared_board_points,
            applied=facts.shared_board_member,
            detail="shared board member corroborates channel",
        ),
        EvidenceComponent(
            name="vague_or_ambiguous_entity",
            points=policy.ambiguous_entity_penalty,
            applied=ambiguous,
            detail="vague, ambiguous, or weakly extracted entity",
        ),
        EvidenceComponent(
            name="non_open_market_transaction",
            points=policy.non_open_market_penalty,
            applied=facts.non_open_market_transaction,
            detail="grant, exercise, gift, withholding, or non-open-market transaction",
        ),
    )


def score_evidence(
    facts: EvidenceFacts,
    policy: RelationshipDecisionPolicy,
    *,
    as_of: datetime,
) -> tuple[int, tuple[EvidenceComponent, ...]]:
    """Score evidence additively, without allowing repeated filings to add points."""
    components = evidence_components(facts, policy, as_of=as_of)
    return sum(component.points for component in components if component.applied), components


def _append(reasons: list[SuppressionReason], reason: SuppressionReason) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _evidence_gates(
    opportunity: RelationshipOpportunityInput,
    policy: RelationshipDecisionPolicy,
    score: int,
    reasons: list[SuppressionReason],
) -> None:
    facts = opportunity.evidence
    if facts.is_amendment or facts.is_correction:
        _append(reasons, SuppressionReason.AMENDMENT_OR_CORRECTION)
    if facts.duplicate_filing:
        _append(reasons, SuppressionReason.DUPLICATE_FILING)
    if facts.duplicate_root_event:
        _append(reasons, SuppressionReason.DUPLICATE_ROOT_EVENT)
    if facts.already_public:
        _append(reasons, SuppressionReason.ALREADY_PUBLIC)
    if facts.conflicting_evidence:
        _append(reasons, SuppressionReason.CONFLICTING_EVIDENCE)
    if score < policy.score_threshold:
        _append(reasons, SuppressionReason.EVIDENCE_BELOW_THRESHOLD)
    if any(
        not _aware_not_after(value, opportunity.candidate_scored_at)
        for value in facts.related_fact_public_at
    ):
        _append(reasons, SuppressionReason.FUTURE_EVIDENCE)
    if opportunity.first_public_state != FirstPublicState.VERIFIED.value:
        _append(reasons, SuppressionReason.FIRST_PUBLIC_UNVERIFIED)
    elif not _aware_not_after(opportunity.first_public_at, opportunity.candidate_scored_at):
        _append(reasons, SuppressionReason.FUTURE_EVIDENCE)
    else:
        assert opportunity.first_public_at is not None
        age = _seconds_between(opportunity.candidate_scored_at, opportunity.first_public_at)
        if age is None:
            _append(reasons, SuppressionReason.FUTURE_EVIDENCE)
        elif age > policy.max_first_public_age_seconds:
            _append(reasons, SuppressionReason.FIRST_PUBLIC_STALE)


def _market_gates(
    snapshot: MarketSnapshot | None,
    opportunity: RelationshipOpportunityInput,
    policy: RelationshipDecisionPolicy,
    reasons: list[SuppressionReason],
) -> None:
    if snapshot is None:
        _append(reasons, SuppressionReason.MISSING_MARKET_SNAPSHOT)
        return
    if not _aware_not_after(snapshot.market_snapshot_at, opportunity.candidate_scored_at):
        _append(reasons, SuppressionReason.FUTURE_MARKET_SNAPSHOT)
        return
    age = _seconds_between(opportunity.candidate_scored_at, snapshot.market_snapshot_at)
    if age is None or age > policy.max_quote_age_seconds:
        _append(reasons, SuppressionReason.STALE_MARKET_SNAPSHOT)
    for timestamp in (snapshot.exchange_event_at, snapshot.local_received_at):
        if timestamp is not None and not _aware_not_after(
            timestamp, opportunity.candidate_scored_at
        ):
            _append(reasons, SuppressionReason.FUTURE_MARKET_SNAPSHOT)
    if snapshot.feed_sequence_gap:
        _append(reasons, SuppressionReason.FEED_SEQUENCE_GAP)
    if snapshot.is_halted or snapshot.is_luld:
        _append(reasons, SuppressionReason.HALT_OR_LULD)
    if snapshot.is_ssr and not policy.allow_ssr:
        _append(reasons, SuppressionReason.SSR_BLOCKED)
    if snapshot.market_status.lower() not in {"regular", "open"} or not snapshot.regular_session:
        _append(reasons, SuppressionReason.MARKET_STATUS_BLOCKED)

    quote_numbers = (snapshot.bid, snapshot.ask)
    if (
        not snapshot.quote_firm
        or any(value is None or not math.isfinite(value) for value in quote_numbers)
        or snapshot.bid is None
        or snapshot.ask is None
        or snapshot.bid <= 0
        or snapshot.ask <= snapshot.bid
        or snapshot.bid_size is None
        or snapshot.ask_size is None
        or snapshot.bid_size <= 0
        or snapshot.ask_size <= 0
    ):
        _append(reasons, SuppressionReason.INVALID_QUOTE)
    else:
        midpoint = (snapshot.bid + snapshot.ask) / 2.0
        spread_bps = (snapshot.ask - snapshot.bid) / midpoint * 10_000.0
        if spread_bps > policy.max_spread_bps:
            _append(reasons, SuppressionReason.SPREAD_LIMIT)

    if (
        snapshot.one_minute_volume is None
        or snapshot.one_minute_volume < policy.min_one_minute_volume
    ):
        _append(reasons, SuppressionReason.LIQUIDITY_LIMIT)
    economics = opportunity.economics
    if economics is not None and (
        snapshot.average_daily_volume is None
        or snapshot.average_daily_volume <= 0
        or economics.quantity / snapshot.average_daily_volume > policy.max_participation_of_adv
    ):
        _append(reasons, SuppressionReason.PARTICIPATION_LIMIT)


def _borrow_gate(
    borrow: BorrowAvailability | None,
    opportunity: RelationshipOpportunityInput,
    reasons: list[SuppressionReason],
) -> None:
    economics = opportunity.economics
    if economics is None or economics.direction != Direction.SHORT.value:
        return
    if (
        borrow is None
        or not borrow.locate_id
        or not borrow.broker_approved
        or borrow.recall_active
        or borrow.broker_restricted
        or borrow.hard_to_borrow
        or borrow.approved_quantity < economics.quantity
        or not _aware_after(borrow.expires_at, opportunity.candidate_scored_at)
    ):
        _append(reasons, SuppressionReason.BORROW_UNAVAILABLE)


def _calibration_gates(
    metrics: HistoricalCellMetrics | None,
    opportunity: RelationshipOpportunityInput,
    policy: RelationshipDecisionPolicy,
    reasons: list[SuppressionReason],
) -> bool:
    """Return whether sealed independent evidence supports a 2%-net claim."""
    if (
        metrics is None
        or metrics.mature_independent_root_clusters < policy.minimum_independent_samples
    ):
        _append(reasons, SuppressionReason.MATURE_INDEPENDENT_SAMPLE_REQUIRED)
        _append(reasons, SuppressionReason.NET_2_CLAIM_UNPROVEN)
        return False
    if (
        not metrics.fully_costed
        or not metrics.sealed_test_passed
        or metrics.p_net_2 is None
        or metrics.mu_net_lcb is None
        or metrics.as_of is None
        or not _aware_not_after(metrics.as_of, opportunity.candidate_scored_at)
    ):
        _append(reasons, SuppressionReason.CALIBRATION_UNAVAILABLE)
        _append(reasons, SuppressionReason.NET_2_CLAIM_UNPROVEN)
        return False
    if metrics.p_net_2 < policy.min_p_net_2 or metrics.mu_net_lcb < policy.net_target_pct:
        _append(reasons, SuppressionReason.NET_2_CLAIM_UNPROVEN)
        return False
    return True


def evaluate_relationship_opportunity(
    opportunity: RelationshipOpportunityInput,
    policy: RelationshipDecisionPolicy,
) -> OpportunityEvaluation:
    """Evaluate an opportunity with fail-closed evidence, market, and economics gates."""
    reasons: list[SuppressionReason] = []
    score, components = score_evidence(
        opportunity.evidence, policy, as_of=opportunity.candidate_scored_at
    )
    _evidence_gates(opportunity, policy, score, reasons)
    _market_gates(opportunity.market_snapshot, opportunity, policy, reasons)
    if opportunity.price_already_absorbed:
        _append(reasons, SuppressionReason.PRICE_ALREADY_ABSORBED)
    if opportunity.earnings_block:
        _append(reasons, SuppressionReason.EARNINGS_BLOCK)

    net_economics = None
    if opportunity.economics is None:
        _append(reasons, SuppressionReason.MISSING_ECONOMICS)
    else:
        net_economics = calculate_net_economics(opportunity.economics)
        if not net_economics.valid:
            _append(reasons, SuppressionReason.INVALID_ECONOMICS)
        else:
            assert net_economics.remaining_net_pct is not None
            assert net_economics.reward_to_r is not None
            if net_economics.remaining_net_pct < policy.net_target_pct:
                _append(reasons, SuppressionReason.NET_TARGET_NOT_MET)
            if net_economics.reward_to_r < policy.min_reward_to_r:
                _append(reasons, SuppressionReason.R_MULTIPLE_NOT_MET)
    _borrow_gate(opportunity.borrow, opportunity, reasons)
    calibration_passes = _calibration_gates(
        opportunity.historical_metrics, opportunity, policy, reasons
    )

    research_only_reasons = {
        SuppressionReason.FIRST_PUBLIC_UNVERIFIED,
        SuppressionReason.FIRST_PUBLIC_STALE,
        SuppressionReason.MATURE_INDEPENDENT_SAMPLE_REQUIRED,
        SuppressionReason.CALIBRATION_UNAVAILABLE,
        SuppressionReason.NET_2_CLAIM_UNPROVEN,
    }
    if not reasons:
        status = OpportunityStatus.ELIGIBLE
    elif all(reason in research_only_reasons for reason in reasons):
        status = OpportunityStatus.RESEARCH_ONLY
    else:
        status = OpportunityStatus.SUPPRESSED

    canonical_input = opportunity.model_dump(mode="json")
    canonical_input["source_span_ids"] = sorted(set(opportunity.source_span_ids))
    canonical_input["source_hashes"] = sorted(set(opportunity.source_hashes))
    evidence_snapshot_hash = sha256_json(
        {"opportunity": canonical_input, "policy_hash": policy.policy_hash}
    )
    return OpportunityEvaluation(
        idempotency_key=opportunity.idempotency_key,
        evidence_snapshot_hash=evidence_snapshot_hash,
        policy_version=policy.policy_version,
        policy_hash=policy.policy_hash or "",
        candidate_scored_at=opportunity.candidate_scored_at,
        evidence_score=score,
        evidence_components=components,
        economics=net_economics,
        suppression_reasons=tuple(reasons),
        status=status,
        llm_eligible=status == OpportunityStatus.ELIGIBLE,
        net_2_claimable=status == OpportunityStatus.ELIGIBLE and calibration_passes,
    )


def build_frozen_evidence_packet(
    opportunity: RelationshipOpportunityInput,
    evaluation: OpportunityEvaluation,
) -> FrozenEvidencePacket:
    """Create a normalized, immutable packet suitable for a DB-free model call."""
    if evaluation.idempotency_key != opportunity.idempotency_key:
        raise ValueError("evaluation does not belong to opportunity natural key")
    normalized_opportunity = opportunity.model_copy(
        update={
            "source_span_ids": tuple(sorted(set(opportunity.source_span_ids))),
            "source_hashes": tuple(sorted(set(opportunity.source_hashes))),
        }
    )
    payload = {
        "idempotency_key": evaluation.idempotency_key,
        "policy_hash": evaluation.policy_hash,
        "candidate_scored_at": evaluation.candidate_scored_at,
        "opportunity_input": normalized_opportunity.model_dump(mode="json"),
        "evaluation": evaluation.model_dump(mode="json"),
    }
    return FrozenEvidencePacket(
        packet_hash=sha256_json(payload),
        idempotency_key=evaluation.idempotency_key,
        policy_version=evaluation.policy_version,
        policy_hash=evaluation.policy_hash,
        candidate_scored_at=evaluation.candidate_scored_at,
        opportunity_input=normalized_opportunity,
        evaluation=evaluation,
        source_span_ids=normalized_opportunity.source_span_ids,
        source_hashes=normalized_opportunity.source_hashes,
    )


# Compact aliases make the pure module pleasant for integration callers.
score_relationship_opportunity = evaluate_relationship_opportunity
freeze_evidence_packet = build_frozen_evidence_packet


__all__ = [
    "build_frozen_evidence_packet",
    "evaluate_relationship_opportunity",
    "evidence_components",
    "freeze_evidence_packet",
    "score_evidence",
    "score_relationship_opportunity",
]
