from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from market_intelligence import database
from market_intelligence.schemas.opportunities import (
    EvidenceComponent,
    EvidenceFacts,
    FirstPublicState,
    FrozenEvidencePacket,
    OpportunityEvaluation,
    OpportunityStatus,
    RelationshipOpportunityInput,
)
from market_intelligence.signals.relationship_decision_workflow import (
    PersistedDecision,
    decide_after_persist,
)

NOW = datetime(2026, 7, 26, 15, tzinfo=UTC)


def _packet(*, eligible: bool) -> FrozenEvidencePacket:
    opportunity = RelationshipOpportunityInput(
        event_id="event-1",
        edge_id="edge-1",
        target_ticker="ABC",
        horizon_days=5,
        strategy_version="v1",
        candidate_scored_at=NOW,
        first_public_state=FirstPublicState.UNKNOWN,
        evidence=EvidenceFacts(),
    )
    evaluation = OpportunityEvaluation(
        idempotency_key=opportunity.idempotency_key,
        evidence_snapshot_hash="evidence-hash",
        policy_version="v1",
        policy_hash="policy-hash",
        candidate_scored_at=NOW,
        evidence_score=10,
        evidence_components=(EvidenceComponent(name="fact", points=10, applied=True, detail="x"),),
        status=OpportunityStatus.ELIGIBLE if eligible else OpportunityStatus.SUPPRESSED,
        llm_eligible=eligible,
        net_2_claimable=eligible,
    )
    return FrozenEvidencePacket(
        packet_hash=f"packet-{eligible}",
        idempotency_key=opportunity.idempotency_key,
        policy_version="prompt-v1",
        policy_hash="policy-hash",
        candidate_scored_at=NOW,
        opportunity_input=opportunity,
        evaluation=evaluation,
        source_span_ids=("span-1",),
        source_hashes=("source-hash",),
    )


def test_ineligible_packet_never_calls_model_and_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "decision.duckdb"
    called = False

    def decider(_: FrozenEvidencePacket) -> PersistedDecision:
        nonlocal called
        called = True
        raise AssertionError("ineligible packet must not call a model")

    result = decide_after_persist(path, _packet(eligible=False), decider, now=NOW)
    assert not called
    assert result.verification_status == "not_eligible"
    with database.connection(path) as con:
        row = con.execute(
            "SELECT verdict, verification_status FROM opportunity_decisions"
        ).fetchone()
        assert row == ("hold", "not_eligible")


def test_model_callback_runs_without_a_held_database_connection(tmp_path: Path) -> None:
    path = tmp_path / "decision.duckdb"

    def decider(_: FrozenEvidencePacket) -> PersistedDecision:
        # This would conflict with a held single-writer connection. The callback
        # can open and write a short independent connection before the decision
        # workflow opens its append-only result connection.
        with database.connection(path) as con:
            database.init_db(con)
            con.execute(
                "INSERT INTO system_health VALUES (?, ?, ?, ?, ?, ?)",
                ["health-1", "test-model", "ok", NOW, "callback", "hash"],
            )
        return PersistedDecision(
            decision_id="decision-1",
            verdict="approve_long",
            verification_status="verified",
            verification_reasons=(),
            decision_payload={"model_version": "fake-v1", "verdict": "approve_long"},
        )

    result = decide_after_persist(
        path,
        _packet(eligible=True),
        decider,
        now=NOW + timedelta(seconds=1),
    )
    assert result.verdict == "approve_long"
    with database.connection(path) as con:
        assert con.execute("SELECT count(*) FROM system_health").fetchone() == (1,)
        status = con.execute("SELECT status FROM relationship_opportunities").fetchone()
        assert status == ("decided",)
