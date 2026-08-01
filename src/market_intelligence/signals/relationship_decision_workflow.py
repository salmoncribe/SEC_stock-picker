"""Short-lived persistence around a frozen, database-free LLM decision.

The critical invariant is structural: the potentially slow model callback runs
after the first DuckDB connection has closed and before the append-only result
connection is opened.  This prevents an Ollama timeout from holding DuckDB's
single-writer lock and blocking SEC/market collectors.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_intelligence import database, hashing
from market_intelligence.schemas.opportunities import (
    FrozenEvidencePacket,
    OpportunityStatus,
)
from market_intelligence.storage import duckdb as duckdb_store


@dataclass(frozen=True)
class PersistedDecision:
    decision_id: str
    verdict: str
    verification_status: str
    verification_reasons: tuple[str, ...]
    decision_payload: dict[str, Any] | None


DecisionCallback = Callable[[FrozenEvidencePacket], PersistedDecision]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def persist_evaluation(
    db_path: str | Path,
    packet: FrozenEvidencePacket,
    *,
    now: datetime | None = None,
) -> str:
    """Persist a deterministic evaluation before any model/network work."""
    timestamp = now or _utc_now()
    opportunity = packet.opportunity_input
    evaluation = packet.evaluation
    opportunity_id = evaluation.idempotency_key
    status = (
        OpportunityStatus.ELIGIBLE.value
        if evaluation.llm_eligible
        else str(evaluation.status)
    )
    row = {
        "opportunity_id": opportunity_id,
        "event_id": opportunity.event_id,
        "edge_id": opportunity.edge_id,
        "target_ticker": opportunity.target_ticker,
        "horizon_days": opportunity.horizon_days,
        "strategy_version": opportunity.strategy_version,
        "evidence_score": evaluation.evidence_score,
        "evidence_components": json.dumps(
            [component.model_dump(mode="json") for component in evaluation.evidence_components],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "evidence_snapshot_hash": evaluation.evidence_snapshot_hash,
        "market_snapshot_id": (
            opportunity.market_snapshot.snapshot_id
            if opportunity.market_snapshot is not None
            else None
        ),
        "tradeability_json": json.dumps(
            {
                "economics": (
                    evaluation.economics.model_dump(mode="json")
                    if evaluation.economics is not None
                    else None
                ),
                "net_2_claimable": evaluation.net_2_claimable,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "status": status,
        "suppression_reasons": json.dumps(
            [str(reason) for reason in evaluation.suppression_reasons],
            separators=(",", ":"),
        ),
        "candidate_scored_at": evaluation.candidate_scored_at,
        "decision_at": None,
        "alert_sent_at": None,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    with database.connection(db_path) as con:
        database.init_db(con)
        duckdb_store.upsert_relationship_opportunities(con, [row])
    return opportunity_id


def decide_after_persist(
    db_path: str | Path,
    packet: FrozenEvidencePacket,
    decider: DecisionCallback,
    *,
    attempt_number: int = 1,
    now: datetime | None = None,
) -> PersistedDecision:
    """Persist, call a model with no open DB connection, then append the result.

    A model exception becomes a persisted ``hold`` result.  It never escapes as
    an implicit approval or causes a retry to overwrite an earlier attempt.
    The caller may schedule another attempt with a new explicit number.
    """
    if attempt_number < 1:
        raise ValueError("attempt_number must be positive")
    opportunity_id = persist_evaluation(db_path, packet, now=now)
    started_at = now or _utc_now()
    if not packet.evaluation.llm_eligible:
        result = PersistedDecision(
            decision_id=hashing.content_hash(
                opportunity_id, packet.packet_hash, "not-eligible", attempt_number
            ),
            verdict="hold",
            verification_status="not_eligible",
            verification_reasons=("deterministic_gates_not_eligible",),
            decision_payload=None,
        )
    else:
        try:
            # No DuckDB connection is live in this scope.
            result = decider(packet)
        except Exception as exc:
            result = PersistedDecision(
                decision_id=hashing.content_hash(
                    opportunity_id, packet.packet_hash, "decision-error", attempt_number
                ),
                verdict="hold",
                verification_status="model_failure",
                verification_reasons=(f"model_failure:{type(exc).__name__}",),
                decision_payload=None,
            )
    completed_at = _utc_now()
    row = {
        "decision_id": result.decision_id,
        "opportunity_id": opportunity_id,
        "input_hash": packet.packet_hash,
        "prompt_version": packet.policy_version,
        "model_version": (
            str((result.decision_payload or {}).get("model_version", "unknown-model"))
        ),
        "attempt_number": attempt_number,
        "verdict": result.verdict,
        "decision_json": (
            json.dumps(result.decision_payload, sort_keys=True, separators=(",", ":"))
            if result.decision_payload is not None
            else None
        ),
        "verification_status": result.verification_status,
        "verification_reasons": json.dumps(result.verification_reasons, separators=(",", ":")),
        "started_at": started_at,
        "completed_at": completed_at,
    }
    with database.connection(db_path) as con:
        database.init_db(con)
        duckdb_store.insert_opportunity_decisions(con, [row])
        con.execute(
            """
            UPDATE relationship_opportunities
            SET decision_at = ?, updated_at = ?,
                status = CASE WHEN ? = 'verified' THEN 'decided' ELSE 'suppressed' END
            WHERE opportunity_id = ?
            """,
            [completed_at, completed_at, result.verification_status, opportunity_id],
        )
    return result


__all__ = ["DecisionCallback", "PersistedDecision", "decide_after_persist", "persist_evaluation"]
