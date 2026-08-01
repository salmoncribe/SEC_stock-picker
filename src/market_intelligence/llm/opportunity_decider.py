"""Fail-closed, database-free LLM verdicts for frozen relationship evidence.

The provider proposes a verdict; this module verifies that every referenced
span, entity, and direction is already present in the immutable packet.  Any
provider error or invalid output becomes ``hold`` and can never be promoted to
an approval by an integration caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from market_intelligence import hashing
from market_intelligence.llm.base import LLMError, LLMProvider
from market_intelligence.schemas.opportunities import Direction, FrozenEvidencePacket

PROMPT_VERSION = "relationship-opportunity-decider-v1"
DEFAULT_MAX_PACKET_AGE = timedelta(minutes=5)
VALID_VERDICTS = frozenset({"approve_long", "approve_short", "hold", "reject"})

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "target_ticker",
        "causal_chain",
        "why_now",
        "evidence_ids",
        "disconfirming_evidence_ids",
        "risk_flags",
        "missing_information",
    ],
    "properties": {
        "verdict": {"type": "string", "enum": sorted(VALID_VERDICTS)},
        "target_ticker": {"type": "string", "minLength": 1},
        "causal_chain": {"type": "string", "minLength": 1},
        "why_now": {"type": "string", "minLength": 1},
        "evidence_ids": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "disconfirming_evidence_ids": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "missing_information": {"type": "array", "items": {"type": "string"}},
    },
}

SYSTEM_PROMPT = """You are a conservative reviewer of a frozen SEC relationship evidence packet.
Return only a JSON object matching the supplied schema.  You may cite only source span IDs
provided in the packet.  Cite at least one supporting span and at least one disconfirming span,
even for hold or reject.  Do not invent entities, facts, prices, timing, or source IDs.  Approve
only in the deterministic packet direction and only when the cited source spans support it."""


@dataclass(frozen=True)
class OpportunityDecision:
    """Structural match for the workflow's ``PersistedDecision`` callback value."""

    decision_id: str
    verdict: str
    verification_status: str
    verification_reasons: tuple[str, ...]
    decision_payload: dict[str, Any] | None


@dataclass(frozen=True)
class SourceSpan:
    """The minimum frozen content required for a model citation to be meaningful."""

    span_id: str
    text: str
    entity: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _packet_payload(packet: FrozenEvidencePacket) -> dict[str, Any]:
    """Reconstruct the exact Phase-3 hash input, excluding derived packet fields."""
    return {
        "idempotency_key": packet.evaluation.idempotency_key,
        "policy_hash": packet.evaluation.policy_hash,
        "candidate_scored_at": packet.evaluation.candidate_scored_at,
        "opportunity_input": packet.opportunity_input.model_dump(mode="json"),
        "evaluation": packet.evaluation.model_dump(mode="json"),
    }


def _source_spans(packet: FrozenEvidencePacket) -> tuple[dict[str, SourceSpan] | None, str | None]:
    """Normalize the explicit ``extra_evidence.source_spans`` citation contract.

    Accept either a list of ``{"id", "text", "entity"}`` records or a
    mapping from span ID to ``{"text", "entity"}``.  Both forms must exactly
    cover the packet's declared source IDs, preventing a model from citing an
    ID that has no frozen, reviewable source content.
    """
    raw = packet.opportunity_input.extra_evidence.get("source_spans")
    rows: list[tuple[Any, Any]]
    if isinstance(raw, dict):
        rows = list(raw.items())
        parsed: dict[str, SourceSpan] = {}
        for span_id, content in rows:
            if not _is_nonempty_string(span_id) or not isinstance(content, dict):
                return None, "packet_source_spans_invalid"
            if set(content) != {"text", "entity"}:
                return None, "packet_source_spans_invalid"
            if not _is_nonempty_string(content.get("text")) or not _is_nonempty_string(
                content.get("entity")
            ):
                return None, "packet_source_spans_invalid"
            key = span_id.strip()
            if key in parsed:
                return None, "packet_source_spans_invalid"
            parsed[key] = SourceSpan(key, content["text"].strip(), content["entity"].strip())
    elif isinstance(raw, list):
        parsed = {}
        for content in raw:
            if not isinstance(content, dict) or set(content) != {"id", "text", "entity"}:
                return None, "packet_source_spans_invalid"
            if any(
                not _is_nonempty_string(content.get(field)) for field in ("id", "text", "entity")
            ):
                return None, "packet_source_spans_invalid"
            key = content["id"].strip()
            if key in parsed:
                return None, "packet_source_spans_invalid"
            parsed[key] = SourceSpan(key, content["text"].strip(), content["entity"].strip())
    else:
        return None, "packet_source_spans_missing"
    if set(parsed) != set(packet.source_span_ids):
        return None, "packet_source_spans_mismatch"
    return parsed, None


def validate_packet(
    packet: FrozenEvidencePacket,
    *,
    now: datetime,
    max_packet_age: timedelta = DEFAULT_MAX_PACKET_AGE,
) -> tuple[str, ...]:
    """Return every deterministic reason a packet must not reach a provider."""
    reasons: list[str] = []
    if max_packet_age < timedelta(0):
        raise ValueError("max_packet_age must be non-negative")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(UTC)
    candidate_time = packet.candidate_scored_at
    if candidate_time.tzinfo is None or candidate_time.utcoffset() is None:
        reasons.append("packet_candidate_time_naive")
    else:
        candidate_time = candidate_time.astimezone(UTC)
        if candidate_time > now:
            reasons.append("packet_candidate_time_in_future")
        elif now - candidate_time > max_packet_age:
            reasons.append("packet_stale")
    if packet.idempotency_key != packet.evaluation.idempotency_key:
        reasons.append("packet_idempotency_mismatch")
    if packet.idempotency_key != packet.opportunity_input.idempotency_key:
        reasons.append("opportunity_idempotency_mismatch")
    if packet.policy_version != packet.evaluation.policy_version:
        reasons.append("packet_policy_version_mismatch")
    if packet.policy_hash != packet.evaluation.policy_hash:
        reasons.append("packet_policy_hash_mismatch")
    if packet.candidate_scored_at != packet.evaluation.candidate_scored_at:
        reasons.append("packet_evaluation_time_mismatch")
    if packet.candidate_scored_at != packet.opportunity_input.candidate_scored_at:
        reasons.append("packet_opportunity_time_mismatch")
    if packet.packet_hash != hashing.sha256_json(_packet_payload(packet)):
        reasons.append("packet_hash_mismatch")
    if packet.source_span_ids != packet.opportunity_input.source_span_ids:
        reasons.append("packet_source_spans_mismatch")
    if packet.source_hashes != packet.opportunity_input.source_hashes:
        reasons.append("packet_source_hashes_mismatch")
    if not packet.source_span_ids:
        reasons.append("packet_has_no_source_spans")
    _, source_span_error = _source_spans(packet)
    if source_span_error is not None:
        reasons.append(source_span_error)
    if not packet.evaluation.llm_eligible:
        reasons.append("deterministic_gates_not_eligible")
    if packet.opportunity_input.economics is None:
        reasons.append("packet_missing_economics")
    return tuple(reasons)


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or any(not _is_nonempty_string(item) for item in value):
        return None
    return [item.strip() for item in value]


def validate_verdict(packet: FrozenEvidencePacket, verdict: Any) -> tuple[str, ...]:
    """Verify model output independently of provider schema-constrained decoding."""
    if not isinstance(verdict, dict):
        return ("response_not_object",)
    required = set(DECISION_SCHEMA["required"])
    unknown = set(verdict) - required
    missing = required - set(verdict)
    reasons: list[str] = []
    if unknown:
        reasons.append("response_has_unknown_fields")
    if missing:
        reasons.append("response_missing_required_fields")
    if reasons:
        return tuple(reasons)

    raw_verdict = verdict.get("verdict")
    if raw_verdict not in VALID_VERDICTS:
        reasons.append("unsupported_verdict")
    if not _is_nonempty_string(verdict.get("target_ticker")):
        reasons.append("invalid_target_ticker")
    elif verdict["target_ticker"].strip().upper() != packet.opportunity_input.target_ticker.upper():
        reasons.append("target_entity_mismatch")
    for field in ("causal_chain", "why_now"):
        if not _is_nonempty_string(verdict.get(field)):
            reasons.append(f"invalid_{field}")
    evidence_ids = _string_list(verdict.get("evidence_ids"))
    disconfirming_ids = _string_list(verdict.get("disconfirming_evidence_ids"))
    if not evidence_ids:
        reasons.append("missing_evidence_citations")
    if not disconfirming_ids:
        reasons.append("missing_disconfirming_citations")
    allowed_spans = set(packet.source_span_ids)
    for cited_ids, reason in (
        (evidence_ids, "hallucinated_evidence_span"),
        (disconfirming_ids, "hallucinated_disconfirming_span"),
    ):
        if cited_ids is not None and (
            len(set(cited_ids)) != len(cited_ids) or not set(cited_ids).issubset(allowed_spans)
        ):
            reasons.append(reason)
    spans, source_span_error = _source_spans(packet)
    if source_span_error is not None:
        reasons.append(source_span_error)
    elif evidence_ids is not None and disconfirming_ids is not None:
        assert spans is not None
        # A relationship thesis commonly needs both the affected target and a
        # named issuer/counterparty span.  The frozen packet may explicitly
        # allow those additional normalized entities; without that allow-list,
        # the target alone is permitted.  This remains a verifier rule, never
        # a model-selected expansion of the entity set.
        raw_allowed_entities = packet.opportunity_input.extra_evidence.get("allowed_entities", [])
        extra_entities = (
            {value.upper() for value in raw_allowed_entities if _is_nonempty_string(value)}
            if isinstance(raw_allowed_entities, list)
            else set()
        )
        allowed_entities = {packet.opportunity_input.target_ticker.upper()} | extra_entities
        for cited_ids, reason in (
            (evidence_ids, "evidence_span_entity_mismatch"),
            (disconfirming_ids, "disconfirming_span_entity_mismatch"),
        ):
            if set(cited_ids).issubset(allowed_spans) and any(
                spans[span_id].entity.upper() not in allowed_entities for span_id in cited_ids
            ):
                reasons.append(reason)
    for field in ("risk_flags", "missing_information"):
        if _string_list(verdict.get(field)) is None:
            reasons.append(f"invalid_{field}")

    direction = (
        packet.opportunity_input.economics.direction if packet.opportunity_input.economics else None
    )
    if raw_verdict == "approve_long" and direction != Direction.LONG.value:
        reasons.append("verdict_direction_mismatch")
    if raw_verdict == "approve_short" and direction != Direction.SHORT.value:
        reasons.append("verdict_direction_mismatch")
    return tuple(reasons)


def _failure(
    packet: FrozenEvidencePacket, status: str, reasons: tuple[str, ...]
) -> OpportunityDecision:
    return OpportunityDecision(
        decision_id=hashing.content_hash(packet.packet_hash, status, *reasons),
        verdict="hold",
        verification_status=status,
        verification_reasons=reasons,
        decision_payload=None,
    )


def decide_opportunity(
    packet: FrozenEvidencePacket,
    provider: LLMProvider,
    *,
    now: datetime | None = None,
    max_packet_age: timedelta = DEFAULT_MAX_PACKET_AGE,
) -> OpportunityDecision:
    """Ask a provider once, then return a strict, fail-closed verified decision.

    This function has no database imports or I/O.  It is intentionally safe to
    pass directly as the decision callback after binding ``provider`` in a
    closure for ``relationship_decision_workflow.decide_after_persist``.
    """
    decision_time = now or _utc_now()
    packet_reasons = validate_packet(packet, now=decision_time, max_packet_age=max_packet_age)
    if packet_reasons:
        return _failure(packet, "packet_rejected", packet_reasons)

    economics = packet.opportunity_input.economics
    user = hashing.canonical_json(
        {
            "packet": packet.model_dump(mode="json"),
            "allowed_source_span_ids": packet.source_span_ids,
            "required_target_ticker": packet.opportunity_input.target_ticker,
            "required_approval_direction": economics.direction if economics is not None else None,
        }
    )
    try:
        output = provider.complete_json(
            system=SYSTEM_PROMPT,
            user=user,
            schema=DECISION_SCHEMA,
            temperature=0.0,
        )
    except (LLMError, TimeoutError) as exc:
        return _failure(packet, "model_failure", (f"model_failure:{type(exc).__name__}",))
    except Exception as exc:  # Provider implementations are external boundaries.
        return _failure(packet, "model_failure", (f"model_failure:{type(exc).__name__}",))

    verdict_reasons = validate_verdict(packet, output)
    if verdict_reasons:
        return _failure(packet, "verification_rejected", verdict_reasons)
    assert isinstance(output, dict)  # established by validate_verdict above
    payload = {**output, "model_version": provider.model, "prompt_version": PROMPT_VERSION}
    return OpportunityDecision(
        decision_id=hashing.content_hash(
            packet.packet_hash, provider.model, hashing.sha256_json(output)
        ),
        verdict=str(output["verdict"]),
        verification_status="verified",
        verification_reasons=(),
        decision_payload=payload,
    )


__all__ = [
    "DECISION_SCHEMA",
    "DEFAULT_MAX_PACKET_AGE",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "VALID_VERDICTS",
    "OpportunityDecision",
    "SourceSpan",
    "decide_opportunity",
    "validate_packet",
    "validate_verdict",
]
