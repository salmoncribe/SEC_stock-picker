"""Pure, fail-closed research validation for executable *paper* scenarios.

Nothing in this module sends an order or talks to a broker.  It makes the
assumptions which a historical claim needs explicit: only information known at
the decision time is used, a fill crosses the quoted spread, stops can gap,
and statistical observations are independent root events rather than rows.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from statistics import mean
from typing import Protocol


class ValidationWindow(StrEnum):
    """Immutable chronological partitions used by a strategy family."""

    DISCOVERY = "discovery"
    CALIBRATION = "calibration"
    SEALED = "sealed"


class TradeDirection(StrEnum):
    LONG = "long"
    SHORT = "short"


class ExitReason(StrEnum):
    TARGET = "target"
    STOP = "stop"
    TIME = "time"
    HALT = "halt"
    UNFILLED = "unfilled"


class ProofFailure(StrEnum):
    """Stable reasons a proposed ``2% net`` label is not allowed."""

    MISSING_SEALED_METRICS = "missing_sealed_metrics"
    SEALED_WINDOW_NOT_LOCKED = "sealed_window_not_locked"
    EXPERIMENT_NOT_PREREGISTERED = "experiment_not_preregistered"
    INSUFFICIENT_INDEPENDENT_ROOT_CLUSTERS = "insufficient_independent_root_clusters"
    MISSING_NET_RETURN_LCB = "missing_net_return_lcb"
    NET_RETURN_LCB_BELOW_TARGET = "net_return_lcb_below_target"
    MISSING_NET_2_PROBABILITY = "missing_net_2_probability"
    NET_2_PROBABILITY_BELOW_TARGET = "net_2_probability_below_target"
    MISSING_CALIBRATION_METRIC = "missing_calibration_metric"
    BRIER_SCORE_TOO_HIGH = "brier_score_too_high"
    SCORE_BANDS_NOT_MONOTONE = "score_bands_not_monotone"
    MISSING_REGIME_STABILITY = "missing_regime_stability"
    REGIME_STABILITY_FAILED = "regime_stability_failed"
    METRICS_NOT_FULLY_COSTED = "metrics_not_fully_costed"


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True)
class ChronologicalSplitPlan:
    """Three fixed windows separated by an embargo at least as long as holding.

    ``*_end`` values are inclusive for already-matured labels.  A candidate is
    assigned by both its decision time and its outcome time; samples crossing a
    boundary (or either embargo) are deliberately discarded.
    """

    discovery_end: datetime
    calibration_start: datetime
    calibration_end: datetime
    sealed_start: datetime
    max_holding_period: timedelta

    def __post_init__(self) -> None:
        for name in (
            "discovery_end",
            "calibration_start",
            "calibration_end",
            "sealed_start",
        ):
            _require_aware(getattr(self, name), name)
        if self.max_holding_period < timedelta(0):
            raise ValueError("max_holding_period cannot be negative")
        if (
            not self.discovery_end
            < self.calibration_start
            <= self.calibration_end
            < self.sealed_start
        ):
            raise ValueError("chronological window boundaries are not ordered")
        if self.calibration_start - self.discovery_end < self.max_holding_period:
            raise ValueError("discovery/calibration embargo is shorter than max_holding_period")
        if self.sealed_start - self.calibration_end < self.max_holding_period:
            raise ValueError("calibration/sealed embargo is shorter than max_holding_period")

    def assign(self, *, decision_at: datetime, outcome_at: datetime) -> ValidationWindow | None:
        """Return a window only when all information used by its label fits it."""
        _require_aware(decision_at, "decision_at")
        _require_aware(outcome_at, "outcome_at")
        if outcome_at < decision_at:
            raise ValueError("outcome_at cannot precede decision_at")
        if outcome_at - decision_at > self.max_holding_period:
            raise ValueError("outcome exceeds configured max_holding_period")
        if decision_at <= self.discovery_end and outcome_at <= self.discovery_end:
            return ValidationWindow.DISCOVERY
        if self.calibration_start <= decision_at and outcome_at <= self.calibration_end:
            return ValidationWindow.CALIBRATION
        if decision_at >= self.sealed_start:
            return ValidationWindow.SEALED
        return None


@dataclass(frozen=True)
class ValidationObservation:
    """One replayed scenario row before duplicate/fan-out collapse."""

    root_event_id: str
    issuer_event_id: str
    target_ticker: str
    decision_at: datetime
    outcome_at: datetime
    net_return_pct: float
    filing_id: str | None = None
    edge_id: str | None = None
    common_shock_id: str | None = None
    predicted_p_net_2: float | None = None
    regime: str | None = None

    def __post_init__(self) -> None:
        if not self.root_event_id or not self.issuer_event_id or not self.target_ticker:
            raise ValueError("root_event_id, issuer_event_id, and target_ticker are required")
        _require_aware(self.decision_at, "decision_at")
        _require_aware(self.outcome_at, "outcome_at")
        if self.outcome_at < self.decision_at:
            raise ValueError("outcome_at cannot precede decision_at")
        if not math.isfinite(self.net_return_pct):
            raise ValueError("net_return_pct must be finite")
        if self.predicted_p_net_2 is not None and not 0 <= self.predicted_p_net_2 <= 1:
            raise ValueError("predicted_p_net_2 must lie in [0, 1]")

    @property
    def target_day(self) -> date:
        return self.decision_at.date()

    @property
    def cluster_key(self) -> tuple[str, str, date]:
        """Repeated filings and fan-out edges share this independent unit.

        ``issuer_event_id`` is the durable root-event identity.  We do not use
        a filing or edge id here: either may be duplicated by an amendment or
        fan-out extraction and neither represents a new independent shock.
        """
        return (
            self.issuer_event_id,
            self.target_ticker.upper(),
            self.target_day,
        )


@dataclass(frozen=True)
class RootEventCluster:
    """Exactly one statistical observation for a root event / target / day."""

    cluster_id: str
    root_event_id: str
    issuer_event_id: str
    target_ticker: str
    decision_at: datetime
    outcome_at: datetime
    net_return_pct: float
    member_count: int
    common_shock_id: str | None = None
    predicted_p_net_2: float | None = None
    regime: str | None = None


def collapse_root_event_clusters(
    observations: Iterable[ValidationObservation],
    *,
    aggregate: Callable[[Sequence[float]], float] = mean,
) -> tuple[RootEventCluster, ...]:
    """Collapse duplicate filings and fan-out edges before any inference.

    One result is emitted per issuer-event/target/day key.  The
    default mean is deterministic and duplicate-invariant for duplicate rows;
    callers can inject a more conservative aggregation policy.  Conflicting
    provenance (shock, regime, or score) is never guessed: it is removed from
    the collapsed metadata, which prevents an over-specific downstream claim.
    """
    grouped: dict[tuple[str, str, date], list[ValidationObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.cluster_key].append(observation)

    collapsed: list[RootEventCluster] = []
    for key in sorted(grouped, key=lambda value: (value[2], value[0], value[1])):
        rows = sorted(
            grouped[key],
            key=lambda value: (value.decision_at, value.outcome_at, value.edge_id or ""),
        )
        returns = [row.net_return_pct for row in rows]
        value = float(aggregate(returns))
        if not math.isfinite(value):
            raise ValueError("aggregate must return a finite value")
        shocks = {row.common_shock_id for row in rows}
        regimes = {row.regime for row in rows}
        predictions = {row.predicted_p_net_2 for row in rows}
        issuer_id, ticker, _ = key
        root_id = min(row.root_event_id for row in rows)
        cluster_id = "|".join((issuer_id, ticker, rows[0].target_day.isoformat()))
        collapsed.append(
            RootEventCluster(
                cluster_id=cluster_id,
                root_event_id=root_id,
                issuer_event_id=issuer_id,
                target_ticker=ticker,
                decision_at=min(row.decision_at for row in rows),
                outcome_at=max(row.outcome_at for row in rows),
                net_return_pct=value,
                member_count=len(rows),
                common_shock_id=shocks.pop() if len(shocks) == 1 else None,
                predicted_p_net_2=predictions.pop() if len(predictions) == 1 else None,
                regime=regimes.pop() if len(regimes) == 1 else None,
            )
        )
    return tuple(collapsed)


@dataclass(frozen=True)
class Quote:
    """A firm NBBO observation used solely for paper-fill replay."""

    observed_at: datetime
    bid: float
    ask: float
    #: Execution replay must use raw quotes.  Kept explicit to prevent a split
    #: adjusted daily series from silently being combined with an NBBO fill.
    price_basis: str = "raw"

    def __post_init__(self) -> None:
        _require_aware(self.observed_at, "observed_at")
        if not all(math.isfinite(value) and value > 0 for value in (self.bid, self.ask)):
            raise ValueError("quote prices must be finite and positive")
        if self.ask <= self.bid:
            raise ValueError("quote must be uncrossed and unlocked")
        if self.price_basis not in {"raw", "adjusted"}:
            raise ValueError("price_basis must be raw or adjusted")


@dataclass(frozen=True)
class QuoteBar:
    """One post-entry bar with bid/ask extrema, including halt evidence."""

    observed_at: datetime
    open_bid: float
    open_ask: float
    high_bid: float
    low_bid: float
    high_ask: float
    low_ask: float
    close_bid: float
    close_ask: float
    halted: bool = False
    price_basis: str = "raw"

    def __post_init__(self) -> None:
        _require_aware(self.observed_at, "observed_at")
        values = (
            self.open_bid,
            self.open_ask,
            self.high_bid,
            self.low_bid,
            self.high_ask,
            self.low_ask,
            self.close_bid,
            self.close_ask,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("bar prices must be finite and positive")
        if self.low_bid > self.high_bid or self.low_ask > self.high_ask:
            raise ValueError("bar low cannot exceed high")
        if self.price_basis not in {"raw", "adjusted"}:
            raise ValueError("price_basis must be raw or adjusted")


@dataclass(frozen=True)
class CostScenario:
    """Conservative costs applied to every paper fill in one named scenario."""

    name: str
    fixed_fees: float = 0.0
    borrow_cost: float = 0.0
    entry_slippage_bps: float = 0.0
    exit_slippage_bps: float = 0.0
    impact_bps: float = 0.0
    halt_exit_penalty_bps: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name is required")
        values = (
            self.fixed_fees,
            self.borrow_cost,
            self.entry_slippage_bps,
            self.exit_slippage_bps,
            self.impact_bps,
            self.halt_exit_penalty_bps,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("cost inputs must be finite and non-negative")


@dataclass(frozen=True)
class PaperScenario:
    """Inputs to a deterministic paper-fill replay after the decision clock."""

    direction: TradeDirection
    quantity: int
    decision_at: datetime
    measured_latency: timedelta
    target_price: float
    stop_price: float
    entry_quotes: tuple[Quote, ...]
    path: tuple[QuoteBar, ...]

    def __post_init__(self) -> None:
        _require_aware(self.decision_at, "decision_at")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.measured_latency < timedelta(0):
            raise ValueError("measured_latency cannot be negative")
        if not all(
            math.isfinite(value) and value > 0 for value in (self.target_price, self.stop_price)
        ):
            raise ValueError("target and stop must be finite and positive")
        if self.direction == TradeDirection.LONG and self.target_price <= self.stop_price:
            raise ValueError("long target must exceed stop")
        if self.direction == TradeDirection.SHORT and self.target_price >= self.stop_price:
            raise ValueError("short target must be below stop")


@dataclass(frozen=True)
class CostedScenarioResult:
    """A replay result, never an execution instruction or broker acknowledgement."""

    filled: bool
    entry_at: datetime | None
    entry_price: float | None
    exit_at: datetime | None
    exit_price: float | None
    exit_reason: ExitReason
    gross_pnl: float | None
    total_cost: float | None
    net_pnl: float | None
    net_return_pct: float | None


def _entry_price(direction: TradeDirection, quote: Quote, costs: CostScenario) -> float:
    bps = (costs.entry_slippage_bps + costs.impact_bps) / 10_000
    return quote.ask * (1 + bps) if direction == TradeDirection.LONG else quote.bid * (1 - bps)


def _exit_price(
    direction: TradeDirection, raw: float, costs: CostScenario, *, halted: bool
) -> float:
    bps = (
        costs.exit_slippage_bps + costs.impact_bps + (costs.halt_exit_penalty_bps if halted else 0)
    ) / 10_000
    return raw * (1 - bps) if direction == TradeDirection.LONG else raw * (1 + bps)


def simulate_costed_scenario(scenario: PaperScenario, costs: CostScenario) -> CostedScenarioResult:
    """Replay a quote-aware fill with latency, spread, stop gaps, and adverse order.

    When one bar reaches both target and stop, the stop is selected.  The order
    inside an aggregate bar is unknowable, and assuming the target occurred
    first would manufacture an optimistic historical result.  A halt also
    exits at the bar's adverse extreme with the configured stress penalty.
    """
    ready_at = scenario.decision_at + scenario.measured_latency
    entries = sorted(
        (quote for quote in scenario.entry_quotes if quote.observed_at >= ready_at),
        key=lambda q: q.observed_at,
    )
    if not entries:
        return CostedScenarioResult(
            False, None, None, None, None, ExitReason.UNFILLED, None, None, None, None
        )
    entry_quote = entries[0]
    if entry_quote.price_basis != "raw" or any(bar.price_basis != "raw" for bar in scenario.path):
        raise ValueError("costed execution replay requires raw, unadjusted quote prices")
    entry = _entry_price(scenario.direction, entry_quote, costs)
    bars = sorted(
        (bar for bar in scenario.path if bar.observed_at >= entry_quote.observed_at),
        key=lambda bar: bar.observed_at,
    )
    if not bars:
        return CostedScenarioResult(
            False, None, None, None, None, ExitReason.UNFILLED, None, None, None, None
        )

    exit_raw: float | None = None
    exit_at: datetime | None = None
    reason = ExitReason.TIME
    halted = False
    for bar in bars:
        if scenario.direction == TradeDirection.LONG:
            target_hit = bar.high_bid >= scenario.target_price
            stop_hit = bar.low_bid <= scenario.stop_price
            if bar.halted:
                exit_raw, reason, halted = min(bar.low_bid, bar.open_bid), ExitReason.HALT, True
            elif stop_hit:  # stop wins ties: unknown intrabar sequencing fails closed
                exit_raw, reason = (
                    min(scenario.stop_price, bar.open_bid, bar.low_bid),
                    ExitReason.STOP,
                )
            elif target_hit:
                exit_raw, reason = max(scenario.target_price, bar.open_bid), ExitReason.TARGET
        else:
            target_hit = bar.low_ask <= scenario.target_price
            stop_hit = bar.high_ask >= scenario.stop_price
            if bar.halted:
                exit_raw, reason, halted = max(bar.high_ask, bar.open_ask), ExitReason.HALT, True
            elif stop_hit:
                exit_raw, reason = (
                    max(scenario.stop_price, bar.open_ask, bar.high_ask),
                    ExitReason.STOP,
                )
            elif target_hit:
                exit_raw, reason = min(scenario.target_price, bar.open_ask), ExitReason.TARGET
        if exit_raw is not None:
            exit_at = bar.observed_at
            break
    if exit_raw is None:
        last = bars[-1]
        exit_raw = last.close_bid if scenario.direction == TradeDirection.LONG else last.close_ask
        exit_at = last.observed_at
    exit_price = _exit_price(scenario.direction, exit_raw, costs, halted=halted)
    quantity = float(scenario.quantity)
    gross = (
        (exit_price - entry) * quantity
        if scenario.direction == TradeDirection.LONG
        else (entry - exit_price) * quantity
    )
    total_cost = costs.fixed_fees + costs.borrow_cost
    net = gross - total_cost
    notional = entry * quantity
    return CostedScenarioResult(
        True,
        entry_quote.observed_at,
        entry,
        exit_at,
        exit_price,
        reason,
        gross,
        total_cost,
        net,
        net / notional,
    )


class RandomSource(Protocol):
    def randrange(self, stop: int) -> int: ...


@dataclass(frozen=True)
class BootstrapSummary:
    """Distribution of cluster-resampled fully costed metrics."""

    resamples: int
    independent_root_clusters: int
    independent_shock_blocks: int
    mean_net_return: float
    net_return_lcb: float
    p_net_2: float
    p_net_2_lcb: float


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of no values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def moving_block_bootstrap(
    clusters: Sequence[RootEventCluster],
    *,
    resamples: int,
    block_size: int = 1,
    target_net_return_pct: float = 0.02,
    rng: RandomSource | None = None,
    lower_quantile: float = 0.05,
) -> BootstrapSummary:
    """Moving/block bootstrap with injected randomness and shock-block resampling.

    ``rng`` is injectable (for example ``random.Random(7)``), making a replay
    exactly reproducible.  Root clusters sharing ``common_shock_id`` are drawn
    together; treating those as independent would make confidence intervals
    too narrow after a sector-wide move.
    """
    if resamples < 1 or block_size < 1:
        raise ValueError("resamples and block_size must be positive")
    if not 0 < lower_quantile < 1:
        raise ValueError("lower_quantile must lie in (0, 1)")
    ordered = sorted(clusters, key=lambda cluster: (cluster.decision_at, cluster.cluster_id))
    if not ordered:
        raise ValueError("at least one root-event cluster is required")
    random_source = rng or random.Random(0)
    by_shock: dict[str, list[RootEventCluster]] = defaultdict(list)
    for cluster in ordered:
        shock_key = cluster.common_shock_id or f"root:{cluster.cluster_id}"
        by_shock[shock_key].append(cluster)
    shock_blocks = sorted(
        by_shock.values(), key=lambda block: min(item.decision_at for item in block)
    )
    # A moving block uses chronological adjacent shock blocks; wrapping keeps
    # every block equally likely at an edge without adding future observations.
    block_count = len(shock_blocks)
    mean_draws: list[float] = []
    probability_draws: list[float] = []
    for _ in range(resamples):
        sampled: list[RootEventCluster] = []
        while len(sampled) < len(ordered):
            start = random_source.randrange(block_count)
            for offset in range(block_size):
                sampled.extend(shock_blocks[(start + offset) % block_count])
                if len(sampled) >= len(ordered):
                    break
        sampled = sampled[: len(ordered)]
        values = [cluster.net_return_pct for cluster in sampled]
        mean_draws.append(mean(values))
        probability_draws.append(
            sum(value >= target_net_return_pct for value in values) / len(values)
        )
    return BootstrapSummary(
        resamples=resamples,
        independent_root_clusters=len(ordered),
        independent_shock_blocks=block_count,
        mean_net_return=mean(cluster.net_return_pct for cluster in ordered),
        net_return_lcb=_quantile(mean_draws, lower_quantile),
        p_net_2=sum(cluster.net_return_pct >= target_net_return_pct for cluster in ordered)
        / len(ordered),
        p_net_2_lcb=_quantile(probability_draws, lower_quantile),
    )


@dataclass(frozen=True)
class RegimeMetric:
    sample_count: int
    net_return_lcb: float | None
    p_net_2_lcb: float | None

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("regime sample_count cannot be negative")


@dataclass(frozen=True)
class SealedValidationMetrics:
    """Metrics emitted after one sealed replay, with absent values preserved."""

    independent_root_clusters: int
    net_return_lcb: float | None
    p_net_2_lcb: float | None
    brier_score: float | None
    score_bands_monotone: bool | None
    regime_metrics: tuple[RegimeMetric, ...] = ()
    fully_costed: bool = False
    sealed_window_locked: bool = False
    experiment_preregistered: bool = False

    def __post_init__(self) -> None:
        if self.independent_root_clusters < 0:
            raise ValueError("independent_root_clusters cannot be negative")


@dataclass(frozen=True)
class TwoPercentProofRequirements:
    minimum_independent_root_clusters: int = 30
    min_net_return_lcb: float = 0.02
    min_p_net_2_lcb: float = 0.5
    max_brier_score: float = 0.25
    minimum_regimes: int = 1
    min_regime_samples: int = 5


@dataclass(frozen=True)
class TwoPercentProof:
    """The sole permission object for describing a family as ``2% net``."""

    claimable: bool
    research_only: bool
    failures: tuple[ProofFailure, ...]


def evaluate_sealed_two_percent_proof(
    metrics: SealedValidationMetrics | None,
    requirements: TwoPercentProofRequirements = TwoPercentProofRequirements(),
) -> TwoPercentProof:
    """Fail closed unless every pre-registered sealed-test gate is supplied."""
    if metrics is None:
        return TwoPercentProof(False, True, (ProofFailure.MISSING_SEALED_METRICS,))
    failures: list[ProofFailure] = []
    if not metrics.sealed_window_locked:
        failures.append(ProofFailure.SEALED_WINDOW_NOT_LOCKED)
    if not metrics.experiment_preregistered:
        failures.append(ProofFailure.EXPERIMENT_NOT_PREREGISTERED)
    if not metrics.fully_costed:
        failures.append(ProofFailure.METRICS_NOT_FULLY_COSTED)
    if metrics.independent_root_clusters < requirements.minimum_independent_root_clusters:
        failures.append(ProofFailure.INSUFFICIENT_INDEPENDENT_ROOT_CLUSTERS)
    if metrics.net_return_lcb is None or not math.isfinite(metrics.net_return_lcb):
        failures.append(ProofFailure.MISSING_NET_RETURN_LCB)
    elif metrics.net_return_lcb < requirements.min_net_return_lcb:
        failures.append(ProofFailure.NET_RETURN_LCB_BELOW_TARGET)
    if (
        metrics.p_net_2_lcb is None
        or not math.isfinite(metrics.p_net_2_lcb)
        or not 0 <= metrics.p_net_2_lcb <= 1
    ):
        failures.append(ProofFailure.MISSING_NET_2_PROBABILITY)
    elif metrics.p_net_2_lcb < requirements.min_p_net_2_lcb:
        failures.append(ProofFailure.NET_2_PROBABILITY_BELOW_TARGET)
    if (
        metrics.brier_score is None
        or not math.isfinite(metrics.brier_score)
        or not 0 <= metrics.brier_score <= 1
    ):
        failures.append(ProofFailure.MISSING_CALIBRATION_METRIC)
    elif metrics.brier_score > requirements.max_brier_score:
        failures.append(ProofFailure.BRIER_SCORE_TOO_HIGH)
    if metrics.score_bands_monotone is not True:
        failures.append(ProofFailure.SCORE_BANDS_NOT_MONOTONE)
    usable_regimes = [
        regime
        for regime in metrics.regime_metrics
        if regime.sample_count >= requirements.min_regime_samples
    ]
    if len(usable_regimes) < requirements.minimum_regimes:
        failures.append(ProofFailure.MISSING_REGIME_STABILITY)
    elif any(
        regime.net_return_lcb is None
        or not math.isfinite(regime.net_return_lcb)
        or regime.p_net_2_lcb is None
        or not math.isfinite(regime.p_net_2_lcb)
        or not 0 <= regime.p_net_2_lcb <= 1
        or regime.net_return_lcb < requirements.min_net_return_lcb
        or regime.p_net_2_lcb < requirements.min_p_net_2_lcb
        for regime in usable_regimes
    ):
        failures.append(ProofFailure.REGIME_STABILITY_FAILED)
    return TwoPercentProof(not failures, bool(failures), tuple(failures))


__all__ = [
    "BootstrapSummary",
    "ChronologicalSplitPlan",
    "CostScenario",
    "CostedScenarioResult",
    "ExitReason",
    "PaperScenario",
    "ProofFailure",
    "Quote",
    "QuoteBar",
    "RegimeMetric",
    "RootEventCluster",
    "SealedValidationMetrics",
    "TradeDirection",
    "TwoPercentProof",
    "TwoPercentProofRequirements",
    "ValidationObservation",
    "ValidationWindow",
    "collapse_root_event_clusters",
    "evaluate_sealed_two_percent_proof",
    "moving_block_bootstrap",
    "simulate_costed_scenario",
]
