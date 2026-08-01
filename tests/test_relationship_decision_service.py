from __future__ import annotations

from datetime import UTC, datetime

from market_intelligence import database
from market_intelligence.config import Config
from market_intelligence.schemas.opportunities import EvidenceFacts, RelationshipOpportunityInput
from market_intelligence.signals.relationship_decision_service import run_configured_decision


def test_disabled_config_persists_research_evaluation_without_calling_model(
    tmp_config: Config,
) -> None:
    opportunity = RelationshipOpportunityInput(
        event_id="event-1",
        edge_id="edge-1",
        target_ticker="ABC",
        horizon_days=5,
        strategy_version="relationship-decision-v1",
        candidate_scored_at=datetime(2026, 7, 26, 15, tzinfo=UTC),
        first_public_state="unknown",
        evidence=EvidenceFacts(),
    )

    class MustNotCall:
        model = "must-not-call"

        def complete_json(self, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("disabled configuration must not call an LLM")

    result = run_configured_decision(
        tmp_config,
        opportunity,
        MustNotCall(),
        now=opportunity.candidate_scored_at,
    )
    assert result.decision is None
    assert "relationship_decision_disabled" in result.research_only_reasons
    with database.connection(tmp_config.paths.database_path) as con:
        assert con.execute("SELECT count(*) FROM relationship_opportunities").fetchone() == (1,)
        assert con.execute("SELECT count(*) FROM opportunity_decisions").fetchone() == (0,)
