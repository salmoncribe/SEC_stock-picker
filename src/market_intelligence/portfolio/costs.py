"""Transaction costs. Sign conventions mirror ``analytics/backtest.py::_fill``.

Parity with the single-book replay is the whole point of this module: the
portfolio simulator is compared against ``run_backtest`` on the same signals,
and that comparison is meaningless if the two disagree about what a trade
costs. 5bps default slippage is the backtest's value, not an independent guess.

Borrow/financing is absent because the account is long-and-cash only
(Robinhood has no stock shorting). The SPY-hedge variant runs as a comparison
experiment, and if it is ever promoted this module grows a borrow term.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Per-trade friction. ``commission_per_share`` is 0.0 for Robinhood."""

    slippage_bps: float = 5.0
    commission_per_share: float = 0.0


def fill_price(raw: float, *, is_buy: bool, model: CostModel) -> float:
    """Slippage always works against the trader: buys fill up, sells fill down.

    Must agree with ``analytics/backtest.py::_fill`` to the last bit -- that
    function is the reference, this is the restatement.  The reference is keyed
    on ``(direction, entry)`` rather than on side, and the two are the same
    statement::

        is_buy == ((direction > 0) == entry)

    Opening a long buys and closing it sells; a short is the mirror image, so
    the four reference branches collapse to two.  For the long-and-cash account
    this module actually serves, that reduces to entry=buy, exit=sell.

    The arithmetic deliberately matches the reference operation for operation --
    divide by 10_000 first, then one multiply -- because ``raw * (1.0 + bps /
    10_000.0)`` and ``raw + raw * bps / 10_000.0`` round differently, and the
    head-to-head comparison against ``run_backtest`` is only meaningful if the
    two books agree exactly.
    """
    rate = model.slippage_bps / 10_000.0
    return raw * (1.0 + rate) if is_buy else raw * (1.0 - rate)


def commission(shares: float, model: CostModel) -> float:
    """Absolute dollar commission for ``shares`` (sign-independent).

    Costs are a drag on either side, so a signed share count -- negative for a
    sell -- must not turn into a rebate.
    """
    return abs(shares) * model.commission_per_share


def round_trip_cost_bps(model: CostModel) -> float:
    """Buy + sell slippage in bps -- the optimizer's turnover penalty kappa.

    A weight change of ``dw`` costs roughly ``dw * round_trip_cost_bps``, so
    this is what makes ``kappa * ||w - w_cur||_1`` denominated in real money
    rather than in an arbitrary regularization unit.

    Commission is absent because it is per-share, not per-dollar, so it has no
    bps expression without a price; at the Robinhood default of 0.0 it is also
    exactly nothing.
    """
    return 2.0 * model.slippage_bps


__all__ = ["CostModel", "commission", "fill_price", "round_trip_cost_bps"]
