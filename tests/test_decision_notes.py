from __future__ import annotations

from market_intelligence.autopilot.decision_notes import (
    OpportunityDecisionNote,
    render_opportunity_decision_note,
    write_opportunity_decision_note,
)


def _note() -> OpportunityDecisionNote:
    return OpportunityDecisionNote(
        opportunity_id="opp-1",
        ticker="ABC",
        status="suppressed",
        evidence_score=9,
        strategy_version="v1",
        evidence_snapshot_hash="hash",
        candidate_scored_at="2026-07-26T12:00:00Z",
        first_public_state="unknown",
        suppression_reasons=("first_public_unverified",),
        risk_flags=("counterevidence span-2",),
    )


def test_decision_note_is_explicitly_non_executable_and_has_no_broker_state() -> None:
    rendered = render_opportunity_decision_note(_note())
    assert "not an order instruction" in rendered
    assert "first_public_unverified" in rendered
    assert "borrow" not in rendered.lower()


def test_write_decision_note_is_regenerable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    target = write_opportunity_decision_note(tmp_path, _note())
    assert target == tmp_path / "relationship-opportunities" / "opp-1.md"
    assert target.read_text(encoding="utf-8") == render_opportunity_decision_note(_note())
