"""Immutable, storage-neutral contracts for relationship opportunities.

These records deliberately do not import the provenance or real-time market
implementations.  A collector may adapt its records into these small values,
but scoring must remain replayable from the values stored here alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_intelligence.hashing import sha256_json


class FirstPublicState(StrEnum):
    """How confidently a broad public-distribution time is known."""

    VERIFIED = "verified"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    STALE = "stale"


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"


class OpportunityStatus(StrEnum):
    ELIGIBLE = "eligible"
    RESEARCH_ONLY = "research_only"
    SUPPRESSED = "suppressed"


class SuppressionReason(StrEnum):
    """Stable, human-readable reasons an opportunity cannot reach an LLM."""

    AMENDMENT_OR_CORRECTION = "amendment_or_correction"
    DUPLICATE_FILING = "duplicate_filing"
    DUPLICATE_ROOT_EVENT = "duplicate_root_event"
    ALREADY_PUBLIC = "already_public"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    EVIDENCE_BELOW_THRESHOLD = "evidence_below_threshold"
    FUTURE_EVIDENCE = "future_evidence"
    FIRST_PUBLIC_UNVERIFIED = "first_public_unverified"
    FIRST_PUBLIC_STALE = "first_public_stale"
    MISSING_MARKET_SNAPSHOT = "missing_market_snapshot"
    FUTURE_MARKET_SNAPSHOT = "future_market_snapshot"
    STALE_MARKET_SNAPSHOT = "stale_market_snapshot"
    INVALID_QUOTE = "invalid_quote"
    MARKET_STATUS_BLOCKED = "market_status_blocked"
    HALT_OR_LULD = "halt_or_luld"
    SSR_BLOCKED = "ssr_blocked"
    FEED_SEQUENCE_GAP = "feed_sequence_gap"
    LIQUIDITY_LIMIT = "liquidity_limit"
    SPREAD_LIMIT = "spread_limit"
    PARTICIPATION_LIMIT = "participation_limit"
    PRICE_ALREADY_ABSORBED = "price_already_absorbed"
    EARNINGS_BLOCK = "earnings_block"
    MISSING_ECONOMICS = "missing_economics"
    INVALID_ECONOMICS = "invalid_economics"
    NET_TARGET_NOT_MET = "net_target_not_met"
    R_MULTIPLE_NOT_MET = "r_multiple_not_met"
    MATURE_INDEPENDENT_SAMPLE_REQUIRED = "mature_independent_sample_required"
    CALIBRATION_UNAVAILABLE = "calibration_unavailable"
    NET_2_CLAIM_UNPROVEN = "net_2_claim_unproven"
    BORROW_UNAVAILABLE = "borrow_unavailable"


class ImmutableModel(BaseModel):
    """Strict values which cannot be silently changed after being hashed."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)


class EvidenceFacts(ImmutableModel):
    """Deterministic facts used for Gate A, never LLM confidence."""

    new_material_fact: bool = False
    named_public_company_counterparty: bool = False
    qualifying_open_market_insider_purchase: bool = False
    distinct_insider_buyers: int = Field(default=0, ge=0)
    validated_relationship: bool = False
    material_magnitude: bool = False
    related_fact_public_at: tuple[datetime, ...] = ()
    shared_board_member: bool = False
    vague_or_ambiguous_entity: bool = False
    weak_extraction: bool = False
    non_open_market_transaction: bool = False
    is_amendment: bool = False
    is_correction: bool = False
    duplicate_filing: bool = False
    duplicate_root_event: bool = False
    already_public: bool = False
    conflicting_evidence: bool = False


class MarketSnapshot(ImmutableModel):
    """The immutable quote/status subset necessary for opportunity gating."""

    snapshot_id: str
    market_snapshot_at: datetime
    exchange_event_at: datetime | None = None
    local_received_at: datetime | None = None
    feed_sequence: int | None = Field(default=None, ge=0)
    feed_sequence_gap: bool = False
    bid: float | None = None
    ask: float | None = None
    bid_size: int | None = Field(default=None, ge=0)
    ask_size: int | None = Field(default=None, ge=0)
    quote_firm: bool = False
    market_status: str = "unknown"
    regular_session: bool = False
    is_halted: bool = False
    is_luld: bool = False
    is_ssr: bool = False
    one_minute_volume: float | None = Field(default=None, ge=0)
    average_daily_volume: float | None = Field(default=None, ge=0)


class BorrowAvailability(ImmutableModel):
    """A broker-approved locate.  It is required only for short candidates."""

    locate_id: str | None = None
    approved_quantity: int = Field(default=0, ge=0)
    expires_at: datetime | None = None
    broker_approved: bool = False
    recall_active: bool = False
    broker_restricted: bool = False
    hard_to_borrow: bool = False


class TradeEconomics(ImmutableModel):
    """All economics are total currency amounts at the intended quantity."""

    direction: Direction
    quantity: int = Field(gt=0)
    entry_vwap: float
    target_exit_vwap: float
    stop_exit_vwap: float
    fees: float = Field(default=0.0, ge=0.0)
    borrow: float = Field(default=0.0, ge=0.0)
    p95_spread: float = Field(default=0.0, ge=0.0)
    p95_slippage: float = Field(default=0.0, ge=0.0)
    p95_impact: float = Field(default=0.0, ge=0.0)
    p95_exit_slippage: float = Field(default=0.0, ge=0.0)
    gap_r_p95: float = Field(default=0.0, ge=0.0)


class HistoricalCellMetrics(ImmutableModel):
    """Sealed, fully matured independent-cluster calibration evidence."""

    strategy_family: str
    mature_independent_root_clusters: int = Field(default=0, ge=0)
    fully_costed: bool = False
    sealed_test_passed: bool = False
    p_net_2: float | None = Field(default=None, ge=0.0, le=1.0)
    mu_net_lcb: float | None = None
    as_of: datetime | None = None


class RelationshipDecisionPolicy(ImmutableModel):
    """A self-fingerprinting policy: a changed threshold requires a new version."""

    policy_version: str
    score_threshold: int = Field(default=8)
    material_fact_points: int = Field(default=4)
    named_counterparty_points: int = Field(default=3)
    insider_purchase_points: int = Field(default=3)
    insider_cluster_points: int = Field(default=2)
    validated_relationship_points: int = Field(default=2)
    material_magnitude_points: int = Field(default=2)
    timing_cluster_points: int = Field(default=2)
    shared_board_points: int = Field(default=1)
    ambiguous_entity_penalty: int = Field(default=-3)
    non_open_market_penalty: int = Field(default=-3)
    insider_cluster_window_days: int = Field(default=7, ge=0)
    max_first_public_age_seconds: int = Field(default=7 * 24 * 60 * 60, ge=0)
    max_quote_age_seconds: int = Field(default=30, ge=0)
    minimum_independent_samples: int = Field(default=30, ge=1)
    net_target_pct: float = Field(default=0.02, gt=0.0)
    min_reward_to_r: float = Field(default=2.0, gt=0.0)
    min_p_net_2: float = Field(default=0.5, ge=0.0, le=1.0)
    max_spread_bps: float = Field(default=50.0, gt=0.0)
    min_one_minute_volume: float = Field(default=1.0, gt=0.0)
    max_participation_of_adv: float = Field(default=0.01, gt=0.0, le=1.0)
    allow_ssr: bool = False
    policy_hash: str | None = None

    @model_validator(mode="after")
    def _bind_hash(self) -> RelationshipDecisionPolicy:
        calculated = sha256_json(self.model_dump(mode="json", exclude={"policy_hash"}))
        if self.policy_hash is not None and self.policy_hash != calculated:
            raise ValueError("policy_hash does not match immutable policy contents")
        object.__setattr__(self, "policy_hash", calculated)
        return self

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> RelationshipDecisionPolicy:
        """Revalidate copies so a changed policy cannot retain an old hash.

        Pydantic's default ``model_copy(update=...)`` intentionally skips
        validation. That is unsafe for a policy fingerprint because it permits
        a caller to alter a threshold while retaining its historical hash.
        """
        del deep  # Every policy field is scalar or an immutable container.
        values = self.model_dump()
        values.update(update or {})
        return type(self).model_validate(values)


def policy_from_config_mapping(config: Mapping[str, Any]) -> RelationshipDecisionPolicy:
    """Adapt the human-percent relationship config without importing ``config``.

    The persisted configuration intentionally expresses percentages as people
    do (``2.0`` means two percent, ``10.0`` means ten percent).  The pure Gate
    B contract uses decimal fractions (``0.02`` and ``0.10``) so its arithmetic
    cannot accidentally compare a 2% move against 200%.  Keeping this adapter
    here makes that boundary explicit and independently testable.
    """
    try:
        net_target_pct = float(config["net_target_pct"])
        max_participation_pct = float(config["max_participation_pct"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("relationship decision config has invalid percentage fields") from exc
    if not 0.0 < net_target_pct <= 100.0 or not 0.0 < max_participation_pct <= 100.0:
        raise ValueError("relationship decision percentages must be in (0, 100]")

    return RelationshipDecisionPolicy(
        policy_version=str(config["strategy_version"]),
        score_threshold=int(config.get("evidence_score_threshold", 8)),
        minimum_independent_samples=int(config.get("minimum_independent_clusters", 30)),
        net_target_pct=net_target_pct / 100.0,
        min_reward_to_r=float(config.get("min_reward_to_risk", 2.0)),
        max_quote_age_seconds=int(float(config.get("max_quote_age_seconds", 30.0))),
        max_spread_bps=float(config.get("max_spread_bps", 50.0)),
        max_participation_of_adv=max_participation_pct / 100.0,
    )


class RelationshipOpportunityInput(ImmutableModel):
    """All values needed to reproduce one score at ``candidate_scored_at``."""

    event_id: str
    edge_id: str
    target_ticker: str
    horizon_days: int = Field(gt=0)
    strategy_version: str
    candidate_scored_at: datetime
    first_public_state: FirstPublicState
    first_public_at: datetime | None = None
    evidence: EvidenceFacts
    market_snapshot: MarketSnapshot | None = None
    economics: TradeEconomics | None = None
    historical_metrics: HistoricalCellMetrics | None = None
    borrow: BorrowAvailability | None = None
    price_already_absorbed: bool = False
    earnings_block: bool = False
    source_span_ids: tuple[str, ...] = ()
    source_hashes: tuple[str, ...] = ()
    extra_evidence: dict[str, Any] = Field(default_factory=dict)

    @property
    def natural_key(self) -> tuple[str, str, str, int, str]:
        return (
            self.event_id,
            self.edge_id,
            self.target_ticker,
            self.horizon_days,
            self.strategy_version,
        )

    @property
    def idempotency_key(self) -> str:
        return sha256_json(self.natural_key)


class EvidenceComponent(ImmutableModel):
    name: str
    points: int
    applied: bool
    detail: str


class NetEconomics(ImmutableModel):
    valid: bool
    entry_notional: float | None = None
    net_reward: float | None = None
    risk_r: float | None = None
    gap_r_p95: float | None = None
    reward_to_r: float | None = None
    remaining_net_pct: float | None = None


class OpportunityEvaluation(ImmutableModel):
    """A replayable score and every reason it did or did not pass."""

    idempotency_key: str
    evidence_snapshot_hash: str
    policy_version: str
    policy_hash: str
    candidate_scored_at: datetime
    evidence_score: int
    evidence_components: tuple[EvidenceComponent, ...]
    economics: NetEconomics | None = None
    suppression_reasons: tuple[SuppressionReason, ...] = ()
    status: OpportunityStatus
    llm_eligible: bool
    net_2_claimable: bool


class FrozenEvidencePacket(ImmutableModel):
    """Database-free, hashable payload handed to a later LLM worker."""

    packet_hash: str
    idempotency_key: str
    policy_version: str
    policy_hash: str
    candidate_scored_at: datetime
    opportunity_input: RelationshipOpportunityInput
    evaluation: OpportunityEvaluation
    source_span_ids: tuple[str, ...]
    source_hashes: tuple[str, ...]


__all__ = [
    "BorrowAvailability",
    "Direction",
    "EvidenceComponent",
    "EvidenceFacts",
    "FirstPublicState",
    "FrozenEvidencePacket",
    "HistoricalCellMetrics",
    "MarketSnapshot",
    "NetEconomics",
    "OpportunityEvaluation",
    "OpportunityStatus",
    "RelationshipDecisionPolicy",
    "RelationshipOpportunityInput",
    "SuppressionReason",
    "TradeEconomics",
    "policy_from_config_mapping",
]
