"""Cost-aware, deterministic trade economics for relationship opportunities."""

from __future__ import annotations

import math

from market_intelligence.schemas.opportunities import Direction, NetEconomics, TradeEconomics


def calculate_net_economics(economics: TradeEconomics) -> NetEconomics:
    """Compute fully costed reward and conservative risk at the planned quantity.

    Invalid or non-finite values return ``valid=False`` rather than emitting a
    partially meaningful number.  The caller can then fail closed with a stable
    suppression reason.
    """
    numbers = (
        economics.entry_vwap,
        economics.target_exit_vwap,
        economics.stop_exit_vwap,
        economics.fees,
        economics.borrow,
        economics.p95_spread,
        economics.p95_slippage,
        economics.p95_impact,
        economics.p95_exit_slippage,
        economics.gap_r_p95,
    )
    if any(not math.isfinite(number) for number in numbers) or economics.entry_vwap <= 0:
        return NetEconomics(valid=False)

    sign = 1.0 if economics.direction == Direction.LONG.value else -1.0
    quantity = float(economics.quantity)
    entry_notional = economics.entry_vwap * quantity
    net_reward = (
        sign * (economics.target_exit_vwap - economics.entry_vwap) * quantity
        - economics.fees
        - economics.borrow
        - economics.p95_spread
        - economics.p95_slippage
        - economics.p95_impact
    )
    base_risk = (
        sign * (economics.entry_vwap - economics.stop_exit_vwap) * quantity
        + economics.fees
        + economics.borrow
        + economics.p95_exit_slippage
    )
    conservative_risk = max(base_risk, economics.gap_r_p95)
    if base_risk <= 0 or conservative_risk <= 0 or entry_notional <= 0:
        return NetEconomics(valid=False)

    return NetEconomics(
        valid=True,
        entry_notional=entry_notional,
        net_reward=net_reward,
        risk_r=base_risk,
        gap_r_p95=economics.gap_r_p95,
        reward_to_r=net_reward / conservative_risk,
        remaining_net_pct=net_reward / entry_notional,
    )


__all__ = ["calculate_net_economics"]
