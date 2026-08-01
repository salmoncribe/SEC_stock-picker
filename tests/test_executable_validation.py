"""Adversarial tests for Phase 4 executable research validation."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from market_intelligence.analytics.executable_validation import (
    ChronologicalSplitPlan,
    CostScenario,
    ExitReason,
    PaperScenario,
    ProofFailure,
    Quote,
    QuoteBar,
    RegimeMetric,
    SealedValidationMetrics,
    TradeDirection,
    TwoPercentProofRequirements,
    ValidationObservation,
    ValidationWindow,
    collapse_root_event_clusters,
    evaluate_sealed_two_percent_proof,
    moving_block_bootstrap,
    simulate_costed_scenario,
)
from market_intelligence.analytics.experiment_registry import (
    ExperimentRecord,
    ExperimentRegistry,
    ExperimentStage,
)

NOW = datetime(2025, 1, 10, 14, 30, tzinfo=UTC)


def _observation(
    *,
    root: str = "root-1",
    issuer: str = "issuer-1",
    ticker: str = "AAA",
    at: datetime = NOW,
    result: float = 0.03,
    shock: str | None = None,
) -> ValidationObservation:
    return ValidationObservation(
        root_event_id=root,
        issuer_event_id=issuer,
        target_ticker=ticker,
        decision_at=at,
        outcome_at=at + timedelta(days=2),
        net_return_pct=result,
        common_shock_id=shock,
    )


def test_chronological_windows_purge_boundaries_with_a_holding_period_embargo() -> None:
    plan = ChronologicalSplitPlan(
        discovery_end=NOW,
        calibration_start=NOW + timedelta(days=3),
        calibration_end=NOW + timedelta(days=10),
        sealed_start=NOW + timedelta(days=13),
        max_holding_period=timedelta(days=3),
    )
    assert (
        plan.assign(decision_at=NOW - timedelta(days=2), outcome_at=NOW)
        is ValidationWindow.DISCOVERY
    )
    assert (
        plan.assign(decision_at=NOW + timedelta(days=3), outcome_at=NOW + timedelta(days=6))
        is ValidationWindow.CALIBRATION
    )
    assert (
        plan.assign(decision_at=NOW + timedelta(days=13), outcome_at=NOW + timedelta(days=16))
        is ValidationWindow.SEALED
    )
    # This label leaks across the discovery/calibration boundary, so it is not
    # silently put in either window.
    assert (
        plan.assign(decision_at=NOW - timedelta(days=1), outcome_at=NOW + timedelta(days=2)) is None
    )


def test_plan_rejects_an_embargo_shorter_than_maximum_holding_period() -> None:
    with pytest.raises(ValueError, match="embargo"):
        ChronologicalSplitPlan(
            discovery_end=NOW,
            calibration_start=NOW + timedelta(days=1),
            calibration_end=NOW + timedelta(days=10),
            sealed_start=NOW + timedelta(days=13),
            max_holding_period=timedelta(days=2),
        )


def test_one_hundred_duplicate_rows_remain_one_independent_root_cluster() -> None:
    rows = [_observation() for _ in range(100)]
    clusters = collapse_root_event_clusters(rows)
    assert len(clusters) == 1
    assert clusters[0].member_count == 100
    assert clusters[0].net_return_pct == pytest.approx(0.03)


def test_fanout_and_repeated_filing_ids_cannot_create_extra_root_events() -> None:
    rows = [
        _observation(root="original", issuer="issuer", ticker="AAA"),
        _observation(root="amended-filing", issuer="issuer", ticker="AAA"),
    ]
    clusters = collapse_root_event_clusters(rows)
    assert len(clusters) == 1
    assert clusters[0].member_count == 2


def test_common_shock_resampling_counts_one_block_and_widens_effective_uncertainty() -> None:
    rows = [
        _observation(root="a", issuer="a", ticker="A", at=NOW, result=-0.10, shock="sector-day"),
        _observation(
            root="b",
            issuer="b",
            ticker="B",
            at=NOW + timedelta(minutes=1),
            result=-0.10,
            shock="sector-day",
        ),
        _observation(root="c", issuer="c", ticker="C", at=NOW + timedelta(days=1), result=0.10),
    ]
    summary = moving_block_bootstrap(
        collapse_root_event_clusters(rows), resamples=100, rng=random.Random(7)
    )
    assert summary.independent_root_clusters == 3
    assert summary.independent_shock_blocks == 2
    assert summary.net_return_lcb < summary.mean_net_return


def test_bootstrap_is_replayable_with_injected_random_source() -> None:
    clusters = collapse_root_event_clusters(
        [
            _observation(
                root=str(i),
                issuer=str(i),
                ticker=str(i),
                at=NOW + timedelta(days=i),
                result=i / 100,
            )
            for i in range(1, 5)
        ]
    )
    first = moving_block_bootstrap(clusters, resamples=50, block_size=2, rng=random.Random(44))
    second = moving_block_bootstrap(clusters, resamples=50, block_size=2, rng=random.Random(44))
    assert first == second


def _bar(
    *,
    at: datetime,
    open_bid: float = 10.0,
    open_ask: float = 10.1,
    high_bid: float = 10.5,
    low_bid: float = 9.5,
    high_ask: float = 10.6,
    low_ask: float = 9.6,
    price_basis: str = "raw",
) -> QuoteBar:
    return QuoteBar(
        observed_at=at,
        open_bid=open_bid,
        open_ask=open_ask,
        high_bid=high_bid,
        low_bid=low_bid,
        high_ask=high_ask,
        low_ask=low_ask,
        close_bid=open_bid,
        close_ask=open_ask,
        price_basis=price_basis,
    )


def test_quote_aware_fill_waits_for_latency_and_uses_stop_gap_not_chart_stop() -> None:
    scenario = PaperScenario(
        direction=TradeDirection.LONG,
        quantity=100,
        decision_at=NOW,
        measured_latency=timedelta(seconds=2),
        target_price=10.5,
        stop_price=9.8,
        entry_quotes=(
            Quote(NOW + timedelta(seconds=1), bid=9.9, ask=10.0),
            Quote(NOW + timedelta(seconds=2), bid=10.0, ask=10.1),
        ),
        path=(_bar(at=NOW + timedelta(minutes=1), open_bid=9.4, low_bid=9.2),),
    )
    result = simulate_costed_scenario(scenario, CostScenario("stressed", fixed_fees=1))
    assert result.filled
    assert result.entry_at == NOW + timedelta(seconds=2)
    assert result.entry_price == pytest.approx(10.1)  # buy at ask
    assert result.exit_reason is ExitReason.STOP
    assert result.exit_price == pytest.approx(9.2)  # gap-through stop, not an optimistic 9.8
    assert result.net_return_pct is not None and result.net_return_pct < -0.08


def test_unknown_intrabar_ordering_selects_the_adverse_stop_and_split_adjusted_data_fails() -> None:
    scenario = PaperScenario(
        direction=TradeDirection.LONG,
        quantity=1,
        decision_at=NOW,
        measured_latency=timedelta(0),
        target_price=10.4,
        stop_price=9.8,
        entry_quotes=(Quote(NOW, bid=9.9, ask=10.0),),
        path=(_bar(at=NOW + timedelta(minutes=1)),),
    )
    assert simulate_costed_scenario(scenario, CostScenario("base")).exit_reason is ExitReason.STOP
    adjusted = PaperScenario(
        direction=scenario.direction,
        quantity=scenario.quantity,
        decision_at=scenario.decision_at,
        measured_latency=scenario.measured_latency,
        target_price=scenario.target_price,
        stop_price=scenario.stop_price,
        entry_quotes=scenario.entry_quotes,
        path=(_bar(at=NOW + timedelta(minutes=1), price_basis="adjusted"),),
    )
    with pytest.raises(ValueError, match="raw, unadjusted"):
        simulate_costed_scenario(adjusted, CostScenario("base"))


def test_missing_sealed_metrics_explicitly_fail_closed() -> None:
    proof = evaluate_sealed_two_percent_proof(None)
    assert not proof.claimable
    assert proof.research_only
    assert proof.failures == (ProofFailure.MISSING_SEALED_METRICS,)


def test_sealed_proof_requires_costed_preregistered_stable_evidence() -> None:
    metrics = SealedValidationMetrics(
        independent_root_clusters=30,
        net_return_lcb=0.021,
        p_net_2_lcb=0.51,
        brier_score=0.10,
        score_bands_monotone=True,
        regime_metrics=(RegimeMetric(5, 0.021, 0.51),),
        fully_costed=True,
        sealed_window_locked=True,
        experiment_preregistered=True,
    )
    assert evaluate_sealed_two_percent_proof(metrics).claimable
    failed = evaluate_sealed_two_percent_proof(
        SealedValidationMetrics(**{**metrics.__dict__, "net_return_lcb": 0.01}),
        TwoPercentProofRequirements(),
    )
    assert ProofFailure.NET_RETURN_LCB_BELOW_TARGET in failed.failures


def test_experiment_registry_is_append_only_and_blocks_calibration_after_sealed_results() -> None:
    registry = ExperimentRegistry()
    registered = registry.append(
        ExperimentRecord(
            record_id="r1",
            experiment_id="exp",
            strategy_family="relationship-v1",
            stage=ExperimentStage.REGISTERED,
            recorded_at=NOW,
            hypothesis="costed edge",
            code_hash="code",
            configuration_hash="config",
            data_snapshot_hash="discovery",
            cost_model_version="cost-v1",
            feature_version="features-v1",
            sealed_window_start=NOW + timedelta(days=10),
        )
    )
    sealed = registry.append(
        ExperimentRecord(
            record_id="r2",
            experiment_id="exp",
            strategy_family="relationship-v1",
            stage=ExperimentStage.SEALED_EVALUATED,
            recorded_at=NOW + timedelta(days=11),
            hypothesis="costed edge",
            code_hash="code",
            configuration_hash="config",
            data_snapshot_hash="sealed",
            cost_model_version="cost-v1",
            feature_version="features-v1",
            parent_record_hash=registered.record_hash,
            metrics={"sealed": {"net_lcb": 0.02}},
        )
    )
    with pytest.raises(ValueError, match="calibration after sealed"):
        registry.append(
            ExperimentRecord(
                record_id="r3",
                experiment_id="exp",
                strategy_family="relationship-v1",
                stage=ExperimentStage.CALIBRATED,
                recorded_at=NOW + timedelta(days=12),
                hypothesis="costed edge",
                code_hash="code",
                configuration_hash="config",
                data_snapshot_hash="calibration",
                cost_model_version="cost-v1",
                feature_version="features-v1",
                parent_record_hash=sealed.record_hash,
            )
        )
