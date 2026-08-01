"""Cost-aware Gate B economics tests."""

from __future__ import annotations

import pytest

from market_intelligence.schemas.opportunities import TradeEconomics
from market_intelligence.signals.net_return import calculate_net_economics


def test_long_net_reward_and_conservative_gap_risk_are_costed_at_quantity() -> None:
    result = calculate_net_economics(
        TradeEconomics(
            direction="long",
            quantity=100,
            entry_vwap=10,
            target_exit_vwap=10.5,
            stop_exit_vwap=9.8,
            fees=2,
            borrow=1,
            p95_spread=3,
            p95_slippage=4,
            p95_impact=5,
            p95_exit_slippage=6,
            gap_r_p95=30,
        )
    )
    assert result.valid
    assert result.entry_notional == 1_000
    assert result.net_reward == 35  # 50 gross less every entry/exit cost in the policy formula
    assert result.risk_r == pytest.approx(29)  # stop distance + fees + borrow + exit slippage
    assert result.reward_to_r == 35 / 30
    assert result.remaining_net_pct == 0.035


def test_short_direction_is_symmetric_and_bad_stop_geometry_fails_closed() -> None:
    short = calculate_net_economics(
        TradeEconomics(
            direction="short",
            quantity=100,
            entry_vwap=10,
            target_exit_vwap=9.5,
            stop_exit_vwap=10.2,
        )
    )
    assert short.valid
    assert short.net_reward == 50
    assert short.risk_r == pytest.approx(20)

    invalid = calculate_net_economics(
        TradeEconomics(
            direction="long",
            quantity=100,
            entry_vwap=10,
            target_exit_vwap=9,
            stop_exit_vwap=11,
        )
    )
    assert not invalid.valid
