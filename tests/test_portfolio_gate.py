from __future__ import annotations

from market_intelligence.signals.portfolio_gate import (
    CandidateRisk,
    PortfolioPolicy,
    RiskReservation,
    evaluate,
)


def _policy() -> PortfolioPolicy:
    return PortfolioPolicy(100_000, 75_000, 40_000, 5_000, 30_000)


def _candidate(**overrides: object) -> CandidateRisk:
    data: dict[str, object] = {
        "opportunity_id": "opp-1",
        "ticker": "ABC",
        "sector": "technology",
        "catalyst_key": "filing-1",
        "signed_notional": 20_000.0,
        "max_loss": 1_000.0,
    }
    data.update(overrides)
    return CandidateRisk(**data)  # type: ignore[arg-type]


def test_portfolio_gate_allows_measurable_capacity() -> None:
    assert evaluate(_candidate(), [], _policy()).allowed


def test_portfolio_gate_fails_closed_for_unknown_concentration() -> None:
    result = evaluate(_candidate(sector=None, catalyst_key=None), [], _policy())
    assert not result.allowed
    assert set(result.suppression_reasons) == {"unknown_sector", "unknown_catalyst"}


def test_portfolio_gate_blocks_duplicate_and_kill_switch() -> None:
    reservation = RiskReservation("opp-1", "ABC", "technology", "filing-1", 20_000, 1_000)
    result = evaluate(_candidate(), [reservation], _policy(), kill_switch_active=True)
    assert not result.allowed
    assert "duplicate_portfolio_reservation" in result.suppression_reasons
    assert "kill_switch_active" in result.suppression_reasons


def test_portfolio_gate_checks_sector_catalyst_and_loss_limits() -> None:
    reservation = RiskReservation("old", "DEF", "technology", "filing-1", 25_000, 4_500)
    result = evaluate(_candidate(), [reservation], _policy())
    assert not result.allowed
    assert {"sector_exposure_limit", "catalyst_overlap_limit", "daily_loss_limit"} <= set(
        result.suppression_reasons
    )
