"""Pure deterministic tests for Phase 3 relationship opportunity gates."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from market_intelligence.schemas.opportunities import (
    BorrowAvailability,
    EvidenceFacts,
    HistoricalCellMetrics,
    MarketSnapshot,
    RelationshipDecisionPolicy,
    RelationshipOpportunityInput,
    SuppressionReason,
    TradeEconomics,
    policy_from_config_mapping,
)
from market_intelligence.signals.relationship_opportunities import (
    build_frozen_evidence_packet,
    evaluate_relationship_opportunity,
    score_evidence,
)

NOW = datetime(2026, 7, 26, 15, 0, tzinfo=UTC)


def _policy(**changes) -> RelationshipDecisionPolicy:
    return RelationshipDecisionPolicy(policy_version="relationship-v1", **changes)


def _facts(**changes) -> EvidenceFacts:
    values = {
        "new_material_fact": True,
        "named_public_company_counterparty": True,
        "qualifying_open_market_insider_purchase": True,
        "distinct_insider_buyers": 2,
        "validated_relationship": True,
        "material_magnitude": True,
        "related_fact_public_at": (NOW - timedelta(hours=2), NOW - timedelta(hours=1)),
        "shared_board_member": True,
    }
    values.update(changes)
    return EvidenceFacts(**values)


def _snapshot(**changes) -> MarketSnapshot:
    values = {
        "snapshot_id": "quote-1",
        "market_snapshot_at": NOW - timedelta(seconds=2),
        "exchange_event_at": NOW - timedelta(seconds=3),
        "local_received_at": NOW - timedelta(seconds=2),
        "feed_sequence": 11,
        "bid": 100.00,
        "ask": 100.08,
        "bid_size": 100,
        "ask_size": 100,
        "quote_firm": True,
        "market_status": "regular",
        "regular_session": True,
        "one_minute_volume": 20_000,
        "average_daily_volume": 1_000_000,
    }
    values.update(changes)
    return MarketSnapshot(**values)


def _economics(**changes) -> TradeEconomics:
    values = {
        "direction": "long",
        "quantity": 1_000,
        "entry_vwap": 100.0,
        "target_exit_vwap": 105.0,
        "stop_exit_vwap": 98.0,
        "fees": 10.0,
        "borrow": 0.0,
        "p95_spread": 20.0,
        "p95_slippage": 20.0,
        "p95_impact": 20.0,
        "p95_exit_slippage": 20.0,
        "gap_r_p95": 1_800.0,
    }
    values.update(changes)
    return TradeEconomics(**values)


def _metrics(**changes) -> HistoricalCellMetrics:
    values = {
        "strategy_family": "relationship-v1",
        "mature_independent_root_clusters": 30,
        "fully_costed": True,
        "sealed_test_passed": True,
        "p_net_2": 0.65,
        "mu_net_lcb": 0.03,
        "as_of": NOW - timedelta(seconds=1),
    }
    values.update(changes)
    return HistoricalCellMetrics(**values)


def _opportunity(**changes) -> RelationshipOpportunityInput:
    values = {
        "event_id": "event-1",
        "edge_id": "edge-1",
        "target_ticker": "TARGET",
        "horizon_days": 10,
        "strategy_version": "relationship-v1",
        "candidate_scored_at": NOW,
        "first_public_state": "verified",
        "first_public_at": NOW - timedelta(seconds=10),
        "evidence": _facts(),
        "market_snapshot": _snapshot(),
        "economics": _economics(),
        "historical_metrics": _metrics(),
        "source_span_ids": ("span-b", "span-a"),
        "source_hashes": ("hash-b", "hash-a"),
    }
    values.update(changes)
    return RelationshipOpportunityInput(**values)


def test_evidence_components_are_additive_and_do_not_count_extra_insiders() -> None:
    policy = _policy()
    score, components = score_evidence(_facts(distinct_insider_buyers=7), policy, as_of=NOW)
    assert score == 19
    assert sum(part.points for part in components if part.applied) == score

    two_buyers_score, _ = score_evidence(_facts(distinct_insider_buyers=2), policy, as_of=NOW)
    assert two_buyers_score == score


def test_hard_duplicate_rejection_overrides_strong_evidence() -> None:
    evaluation = evaluate_relationship_opportunity(
        _opportunity(evidence=_facts(is_amendment=True, duplicate_filing=True)), _policy()
    )
    assert evaluation.evidence_score >= 8
    assert not evaluation.llm_eligible
    assert SuppressionReason.AMENDMENT_OR_CORRECTION in evaluation.suppression_reasons
    assert SuppressionReason.DUPLICATE_FILING in evaluation.suppression_reasons


def test_policy_hash_is_immutable_and_contents_require_a_new_hash() -> None:
    first = _policy()
    second = _policy(score_threshold=9)
    assert first.policy_hash != second.policy_hash
    with pytest.raises(ValidationError, match="policy_hash"):
        RelationshipDecisionPolicy(
            policy_version="relationship-v1", score_threshold=9, policy_hash=first.policy_hash
        )
    with pytest.raises(ValidationError):
        first.score_threshold = 9
    with pytest.raises(ValidationError, match="policy_hash"):
        first.model_copy(update={"score_threshold": 9})


def test_config_adapter_converts_human_percent_values_to_decimal_gate_values() -> None:
    policy = policy_from_config_mapping(
        {
            "strategy_version": "relationship-decision-v1",
            "net_target_pct": 2.0,
            "max_participation_pct": 10.0,
        }
    )
    assert policy.net_target_pct == 0.02
    assert policy.max_participation_of_adv == 0.10


def test_same_input_is_idempotent_and_packet_canonicalizes_source_order() -> None:
    policy = _policy()
    first = _opportunity()
    reordered = _opportunity(
        source_span_ids=("span-a", "span-b", "span-a"),
        source_hashes=("hash-a", "hash-b", "hash-a"),
    )
    first_result = evaluate_relationship_opportunity(first, policy)
    second_result = evaluate_relationship_opportunity(reordered, policy)
    assert first_result.idempotency_key == second_result.idempotency_key
    assert first_result.evidence_snapshot_hash == second_result.evidence_snapshot_hash
    first_packet = build_frozen_evidence_packet(first, first_result)
    second_packet = build_frozen_evidence_packet(reordered, second_result)
    assert first_packet.packet_hash == second_packet.packet_hash


def test_future_timestamp_is_suppressed_not_used_in_a_historical_decision() -> None:
    future_snapshot = _snapshot(market_snapshot_at=NOW + timedelta(seconds=1))
    evaluation = evaluate_relationship_opportunity(
        _opportunity(market_snapshot=future_snapshot), _policy()
    )
    assert not evaluation.llm_eligible
    assert SuppressionReason.FUTURE_MARKET_SNAPSHOT in evaluation.suppression_reasons


def test_duplicate_root_event_cannot_multiply_a_score_or_reach_the_llm() -> None:
    normal = evaluate_relationship_opportunity(_opportunity(), _policy())
    duplicate = evaluate_relationship_opportunity(
        _opportunity(evidence=_facts(duplicate_root_event=True)), _policy()
    )
    assert duplicate.evidence_score == normal.evidence_score
    assert duplicate.idempotency_key == normal.idempotency_key
    assert SuppressionReason.DUPLICATE_ROOT_EVENT in duplicate.suppression_reasons
    assert not duplicate.llm_eligible


def test_2_percent_claim_fails_closed_without_mature_independent_costed_evidence() -> None:
    evaluation = evaluate_relationship_opportunity(
        _opportunity(historical_metrics=_metrics(mature_independent_root_clusters=29)), _policy()
    )
    assert not evaluation.net_2_claimable
    assert not evaluation.llm_eligible
    assert SuppressionReason.MATURE_INDEPENDENT_SAMPLE_REQUIRED in evaluation.suppression_reasons
    assert SuppressionReason.NET_2_CLAIM_UNPROVEN in evaluation.suppression_reasons


def test_short_requires_current_broker_approved_locate_for_exact_quantity() -> None:
    short_economics = _economics(direction="short", target_exit_vwap=95, stop_exit_vwap=102)
    short = _opportunity(economics=short_economics)
    rejected = evaluate_relationship_opportunity(short, _policy())
    assert SuppressionReason.BORROW_UNAVAILABLE in rejected.suppression_reasons

    approved = short.model_copy(
        update={
            "borrow": BorrowAvailability(
                locate_id="locate-1",
                approved_quantity=1_000,
                broker_approved=True,
                expires_at=NOW + timedelta(minutes=5),
            )
        }
    )
    assert evaluate_relationship_opportunity(approved, _policy()).llm_eligible

    hard_to_borrow = approved.model_copy(
        update={"borrow": approved.borrow.model_copy(update={"hard_to_borrow": True})}
    )
    assert SuppressionReason.BORROW_UNAVAILABLE in evaluate_relationship_opportunity(
        hard_to_borrow, _policy()
    ).suppression_reasons
