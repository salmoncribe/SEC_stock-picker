"""Fail-closed tests for the database-free relationship opportunity decider."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from market_intelligence import hashing
from market_intelligence.llm.base import LLMError
from market_intelligence.llm.opportunity_decider import (
    DECISION_SCHEMA,
    PROMPT_VERSION,
    decide_opportunity,
)
from market_intelligence.schemas.opportunities import (
    Direction,
    EvidenceFacts,
    FirstPublicState,
    FrozenEvidencePacket,
    OpportunityEvaluation,
    OpportunityStatus,
    RelationshipOpportunityInput,
    TradeEconomics,
)

NOW = datetime(2026, 7, 26, 15, tzinfo=UTC)


class FakeProvider:
    model = "fake-reviewer-v1"

    def __init__(self, response: dict[str, Any] | Exception) -> None:
        self.response = response
        self.calls = 0
        self.last_schema: dict[str, Any] | None = None

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.calls += 1
        self.last_schema = schema
        assert temperature == 0.0
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _packet(*, candidate_at: datetime = NOW, source_spans: Any = None) -> FrozenEvidencePacket:
    if source_spans is None:
        source_spans = {
            "span-a": {"text": "TARGET signed a material agreement.", "entity": "TARGET"},
            "span-b": {"text": "The agreement may be immaterial.", "entity": "TARGET"},
        }
    opportunity = RelationshipOpportunityInput(
        event_id="event-1",
        edge_id="edge-1",
        target_ticker="TARGET",
        horizon_days=10,
        strategy_version="relationship-v1",
        candidate_scored_at=candidate_at,
        first_public_state=FirstPublicState.VERIFIED,
        first_public_at=candidate_at - timedelta(seconds=1),
        evidence=EvidenceFacts(),
        economics=TradeEconomics(
            direction=Direction.LONG,
            quantity=100,
            entry_vwap=100.0,
            target_exit_vwap=103.0,
            stop_exit_vwap=99.0,
        ),
        source_span_ids=("span-a", "span-b"),
        source_hashes=("source-hash-a", "source-hash-b"),
        extra_evidence={"source_spans": source_spans},
    )
    evaluation = OpportunityEvaluation(
        idempotency_key=opportunity.idempotency_key,
        evidence_snapshot_hash="evidence-hash",
        policy_version="relationship-v1",
        policy_hash="policy-hash",
        candidate_scored_at=candidate_at,
        evidence_score=10,
        evidence_components=(),
        status=OpportunityStatus.ELIGIBLE,
        llm_eligible=True,
        net_2_claimable=True,
    )
    payload = {
        "idempotency_key": evaluation.idempotency_key,
        "policy_hash": evaluation.policy_hash,
        "candidate_scored_at": evaluation.candidate_scored_at,
        "opportunity_input": opportunity.model_dump(mode="json"),
        "evaluation": evaluation.model_dump(mode="json"),
    }
    return FrozenEvidencePacket(
        packet_hash=hashing.sha256_json(payload),
        idempotency_key=evaluation.idempotency_key,
        policy_version=evaluation.policy_version,
        policy_hash=evaluation.policy_hash,
        candidate_scored_at=candidate_at,
        opportunity_input=opportunity,
        evaluation=evaluation,
        source_span_ids=opportunity.source_span_ids,
        source_hashes=opportunity.source_hashes,
    )


def _valid_response(**changes: Any) -> dict[str, Any]:
    response = {
        "verdict": "approve_long",
        "target_ticker": "TARGET",
        "causal_chain": "A material agreement benefits TARGET.",
        "why_now": "The evidence packet is newly public.",
        "evidence_ids": ["span-a"],
        "disconfirming_evidence_ids": ["span-b"],
        "risk_flags": ["execution risk"],
        "missing_information": ["contract value"],
    }
    response.update(changes)
    return response


def test_verified_response_is_structurally_compatible_with_persisted_decision() -> None:
    provider = FakeProvider(_valid_response())

    decision = decide_opportunity(_packet(), provider, now=NOW)

    assert decision.verdict == "approve_long"
    assert decision.verification_status == "verified"
    assert decision.decision_payload is not None
    assert decision.decision_payload["model_version"] == provider.model
    assert decision.decision_payload["prompt_version"] == PROMPT_VERSION
    assert provider.last_schema == DECISION_SCHEMA


def test_hallucinated_or_entity_mismatched_citations_cannot_approve() -> None:
    hallucinated = decide_opportunity(
        _packet(), FakeProvider(_valid_response(evidence_ids=["invented-span"])), now=NOW
    )
    mismatched = decide_opportunity(
        _packet(), FakeProvider(_valid_response(target_ticker="OTHER")), now=NOW
    )

    assert hallucinated.verdict == "hold"
    assert "hallucinated_evidence_span" in hallucinated.verification_reasons
    assert mismatched.verdict == "hold"
    assert "target_entity_mismatch" in mismatched.verification_reasons


def test_missing_counterevidence_or_wrong_direction_cannot_approve() -> None:
    missing_counterevidence = decide_opportunity(
        _packet(), FakeProvider(_valid_response(disconfirming_evidence_ids=[])), now=NOW
    )
    wrong_direction = decide_opportunity(
        _packet(), FakeProvider(_valid_response(verdict="approve_short")), now=NOW
    )

    assert missing_counterevidence.verdict == "hold"
    assert "missing_disconfirming_citations" in missing_counterevidence.verification_reasons
    assert wrong_direction.verdict == "hold"
    assert "verdict_direction_mismatch" in wrong_direction.verification_reasons


def test_stale_packet_or_missing_source_content_never_calls_provider() -> None:
    stale_provider = FakeProvider(_valid_response())
    stale = decide_opportunity(
        _packet(candidate_at=NOW - timedelta(minutes=6)), stale_provider, now=NOW
    )
    missing_source_provider = FakeProvider(_valid_response())
    missing_source = decide_opportunity(_packet(source_spans={}), missing_source_provider, now=NOW)

    assert stale.verdict == "hold"
    assert "packet_stale" in stale.verification_reasons
    assert stale_provider.calls == 0
    assert missing_source.verdict == "hold"
    assert "packet_source_spans_mismatch" in missing_source.verification_reasons
    assert missing_source_provider.calls == 0


def test_provider_failure_and_malformed_json_become_no_approval() -> None:
    failed = decide_opportunity(_packet(), FakeProvider(LLMError("timeout")), now=NOW)
    malformed = decide_opportunity(
        _packet(), FakeProvider(_valid_response(extra_field="not permitted")), now=NOW
    )

    assert failed.verdict == "hold"
    assert failed.verification_status == "model_failure"
    assert malformed.verdict == "hold"
    assert "response_has_unknown_fields" in malformed.verification_reasons
