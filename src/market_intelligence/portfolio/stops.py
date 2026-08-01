"""Per-position stop losses: fixed-percentage and ATR-multiple.

Before this module there was no stop of any kind in ``portfolio/``: a single
name could fall 60% and nothing would intervene until the portfolio-level
``risk.drawdown_governor`` fired at -10% *account-wide*. This closes the
individual-name gap. Wired into ``simulator.execute_day`` (see
``_stopped_positions`` / ``_apply_stop_exits``), evaluated once per simulated
day independent of ``rebalance_days`` -- a stop must not have to wait for the
next scheduled rebalance to fire.

Two independently selectable kinds (``PortfolioConfig.stop_loss_kind``):

* ``"fixed_pct"`` -- exit once the mark has fallen more than ``stop_loss_pct``
  below the position's entry reference.
* ``"atr_multiple"`` -- exit once the mark has fallen more than
  ``stop_loss_atr_multiple`` Wilder ATRs below the entry reference. Reuses
  :func:`~market_intelligence.signals.trade_plan.wilder_atr` -- the ATR
  implementation ``signals.trade_alerts`` already ships -- via
  :func:`trailing_atr`, rather than inventing a second one, and mirrors
  ``trade_plan.build_trade_plan``'s ``entry - atr_stop_multiple * atr``
  geometry exactly: a stop fixed once at entry, not a trailing/chandelier
  stop that ratchets up behind a winner.

Both kinds are evaluated against the position's *average cost basis*
(``cost_basis / shares``), not a per-lot entry price. This account has no
lot-level history (see ``account.Position``), so a stop that measures from
average cost adjusts exactly the way adding to a loser lowers its own average
cost -- the same math a broker's own average-cost stop would apply.

Long-only (repo design decision): every stop here is a downside stop. There is
no short book to stop on the other side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np

from market_intelligence.signals.trade_plan import Bar, wilder_atr

StopKind = Literal["none", "fixed_pct", "atr_multiple"]


@dataclass(frozen=True)
class StopCheck:
    """One symbol's stop evaluation, returned even when it did not trigger.

    Kept as a real value (not just a bool) so a caller can audit *why* a
    position that looks stop-worthy was not flagged -- no ATR, no reference
    price, or the kind is ``"none"`` -- rather than only ever seeing the
    positive case.
    """

    symbol: str
    kind: StopKind
    entry_price: float
    reference_price: float
    stop_price: float | None
    triggered: bool


def average_cost(cost_basis: float, shares: float) -> float:
    """Average cost per share -- the account's only entry-price-like number.

    ``account.Position`` keeps no lot-level history, so this is a stop on the
    *position*, not on any one fill. Returns 0.0 for a non-positive share
    count rather than raising: the caller has already excluded closed/absent
    positions by construction, and 0.0 reads downstream as "cannot evaluate."
    """
    return cost_basis / shares if shares > 0.0 else 0.0


def fixed_pct_stop_price(entry_price: float, stop_pct: float) -> float:
    """Price ``stop_pct`` below ``entry_price``. Long-only, so always below."""
    if entry_price <= 0.0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if not 0.0 < stop_pct < 1.0:
        raise ValueError(f"stop_pct must be in (0, 1), got {stop_pct}")
    return entry_price * (1.0 - stop_pct)


def atr_stop_price(entry_price: float, atr: float, atr_multiple: float) -> float:
    """Price ``atr_multiple`` Wilder ATRs below ``entry_price``.

    Mirrors ``signals.trade_plan.build_trade_plan``'s ``entry -
    atr_stop_multiple * atr`` exactly, so "ATR stop" means the same geometry
    in the live alert path and here -- one convention, not two.
    """
    if entry_price <= 0.0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if atr < 0.0:
        raise ValueError(f"atr must be non-negative, got {atr}")
    if atr_multiple <= 0.0:
        raise ValueError(f"atr_multiple must be positive, got {atr_multiple}")
    return entry_price - atr_multiple * atr


def trailing_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, *, period: int
) -> float | None:
    """Wilder ATR over an oldest-first ``[T]`` OHLC window, via ``trade_plan.wilder_atr``.

    One ATR implementation for the whole codebase: this builds synthetic
    :class:`~market_intelligence.signals.trade_plan.Bar` rows with
    ``adj_close == close``, so ``wilder_atr``'s own split/dividend factor is
    exactly 1 and every row passes through unchanged -- the caller is expected
    to hand already-adjusted-space prices (``signals.trade_plan``'s own "all
    range math runs in adjusted price space" rule; see
    ``simulator._trailing_atr_raw`` for how the result gets back to raw
    space). Non-finite rows are dropped by ``wilder_atr`` itself, which
    returns ``None`` -- never a fabricated number -- once fewer than
    ``period + 1`` valid rows remain.

    The caller owns the no-lookahead bound: pass a window ending strictly
    before the decision date, exactly as
    ``feed.MarketPanel.trailing_returns`` requires of its own callers.
    """
    length = len(close)
    if len(high) != length or len(low) != length:
        raise ValueError("high, low, close must be the same length")
    bars = [
        Bar(
            # A synthetic, strictly increasing date: wilder_atr sorts by date,
            # and this guarantees the oldest-first order the caller passed in
            # survives regardless of Python's sort-stability being relied on.
            date=date.min + timedelta(days=index),
            open=float(close[index]),
            high=float(high[index]),
            low=float(low[index]),
            close=float(close[index]),
            adj_close=float(close[index]),
        )
        for index in range(length)
    ]
    return wilder_atr(bars, period)


def evaluate_stop(
    *,
    symbol: str,
    kind: StopKind,
    entry_price: float,
    reference_price: float,
    atr: float | None,
    stop_pct: float,
    atr_multiple: float,
) -> StopCheck:
    """One symbol's stop check. ``kind == "none"`` never triggers.

    ``reference_price`` and ``entry_price`` must already share a price space
    -- this codebase's ledger convention is raw (see
    ``account.Position.cost_basis`` and ``simulator._mark_prices``), so a
    caller combining an ATR computed in adjusted space must rescale it first,
    exactly as ``simulator._raw_open`` rescales the panel's adjusted open.

    Never raises: a non-positive or missing price/ATR reads as "cannot
    evaluate today" (not triggered) rather than an error, matching this
    package's "a degraded day is visible, not a crashed one" convention.
    """
    if kind == "none" or entry_price <= 0.0 or reference_price <= 0.0:
        return StopCheck(symbol, kind, entry_price, reference_price, None, False)

    if kind == "fixed_pct":
        stop_price = fixed_pct_stop_price(entry_price, stop_pct)
    elif kind == "atr_multiple":
        if atr is None or not math.isfinite(atr) or atr <= 0.0:
            return StopCheck(symbol, kind, entry_price, reference_price, None, False)
        stop_price = atr_stop_price(entry_price, atr, atr_multiple)
    else:  # pragma: no cover -- Literal is exhaustive above
        raise ValueError(f"unknown stop kind {kind!r}")

    triggered = reference_price <= stop_price
    return StopCheck(symbol, kind, entry_price, reference_price, stop_price, triggered)


__all__ = [
    "StopCheck",
    "StopKind",
    "atr_stop_price",
    "average_cost",
    "evaluate_stop",
    "fixed_pct_stop_price",
    "trailing_atr",
]
