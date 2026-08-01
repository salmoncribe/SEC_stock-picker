"""Event-sourced immutable ledger for the simulated account.

The account is a *fold over fills*, never a mutable balance. Every state is
reachable by replaying the fill log from genesis, which is what makes
``portfolio status`` able to rebuild the account bit-for-bit from
``paper_fills`` and what makes a resumed ``portfolio step`` provably continuous
with the replay that preceded it.

Invariants enforced here, not merely tested:
  * ``equity == cash + sum(shares * price)`` for every marked state (1e-9)
  * ``cash >= 0`` -- an overdraft raises rather than silently borrowing
  * a sell of more shares than held raises rather than opening a short
  * a fill dated before the account's ``as_of`` raises -- no backdating
  * ``replay(fills)`` reproduces any state exactly

Fractional shares are the default (Michael's decision; Robinhood supports
them). Integer mode exists as a config comparison, not as the real mode: a
$10,000 account trading integer shares measures rounding error, not edge.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from types import MappingProxyType

# Fractional share quantities round here. Finer than any real broker's
# precision, coarse enough that float64 accumulation over 2,500 days cannot
# drift a position into a phantom dust holding.
SHARE_PRECISION = 1e-4

# The equity identity is asserted at this tolerance every simulated day.
EQUITY_TOLERANCE = 1e-9

# Decimal places implied by SHARE_PRECISION. Rounding by digit count rather
# than by dividing through the constant avoids re-introducing the float
# artefact the quantisation exists to remove.
_SHARE_DIGITS = max(0, round(-math.log10(SHARE_PRECISION)))

# The deposit "symbol". A deposit is one dollar at price 1.0 repeated ``cash``
# times, so ``shares * price`` is the amount for every side without a special
# case in the fold or in the ledger schema.
_CASH_SYMBOL = "CASH"


class Side(StrEnum):
    """``DEPOSIT`` is the genesis convention: the opening cash arrives as a fill.

    Modelling the initial $10,000 as a deposit fill rather than as a magic
    starting balance means ``replay()`` has exactly one code path and the
    ledger's first row is auditable like every other row.
    """

    BUY = "BUY"
    SELL = "SELL"
    DEPOSIT = "DEPOSIT"


@dataclass(frozen=True)
class Position:
    """A long holding. ``cost_basis`` is total dollars paid, not per-share."""

    symbol: str
    shares: float
    cost_basis: float


@dataclass(frozen=True)
class Order:
    """An intent. Orders are proposed by the optimizer and may go unfilled."""

    order_id: str
    symbol: str
    side: Side
    shares: float
    as_of: date
    reason: str


@dataclass(frozen=True)
class Fill:
    """An executed order. The only thing that changes account state."""

    fill_id: str
    order_id: str
    symbol: str
    side: Side
    shares: float
    price: float
    commission: float
    as_of: date


@dataclass(frozen=True)
class AccountState:
    """Cash, holdings, and the high-water mark, as of the close of ``as_of``."""

    as_of: date
    cash: float
    positions: Mapping[str, Position]
    high_water_mark: float


@dataclass(frozen=True)
class MarkedAccount:
    """An ``AccountState`` valued at a specific set of prices."""

    state: AccountState
    prices: Mapping[str, float]
    equity: float
    drawdown: float


class AccountError(RuntimeError):
    """Raised when a fill would violate an account invariant."""


_NO_POSITIONS: Mapping[str, Position] = MappingProxyType({})


def _quantise(shares: float, *, fractional: bool) -> float:
    """Snap a fill quantity to the tradeable grid.

    Integer mode *rejects* a fractional quantity rather than flooring it. A
    floor silently fills something other than what the optimizer sized, and the
    resulting cash is then wrong by a fraction of a share every trade -- exactly
    the rounding error integer mode exists to measure, contaminated by a second
    unrecorded one.
    """
    if not fractional:
        whole = round(shares)
        if abs(shares - whole) > EQUITY_TOLERANCE:
            raise AccountError(f"integer mode cannot fill {shares} shares")
        return float(whole)
    rounded = round(shares, _SHARE_DIGITS)
    return 0.0 if rounded == 0.0 else rounded


def genesis(cash: float, as_of: date) -> AccountState:
    """Open an account with ``cash``, no positions, high-water mark == cash."""
    if cash < 0:
        raise AccountError(f"opening cash cannot be negative: {cash}")
    return AccountState(
        as_of=as_of, cash=float(cash), positions=_NO_POSITIONS, high_water_mark=float(cash)
    )


def genesis_fill(cash: float, as_of: date, simulation_version: str) -> Fill:
    """The DEPOSIT fill that ``store`` persists so the ledger is self-contained.

    Ids are deterministic and namespaced by ``simulation_version`` so that
    re-running a step rewrites the opening row instead of opening a second
    account, and so a scratch experiment cannot collide with the ledger of
    record.
    """
    if cash < 0:
        raise AccountError(f"opening cash cannot be negative: {cash}")
    key = f"{_CASH_SYMBOL}:{as_of.isoformat()}:{Side.DEPOSIT.value}:{simulation_version}"
    return Fill(
        fill_id=f"pf:{key}",
        order_id=f"po:{key}",
        symbol=_CASH_SYMBOL,
        side=Side.DEPOSIT,
        shares=float(cash),
        price=1.0,
        commission=0.0,
        as_of=as_of,
    )


def apply_fill(state: AccountState, fill: Fill, *, fractional: bool = True) -> AccountState:
    """Fold one fill into the account, returning a new state.

    Raises ``AccountError`` on overdraft, oversell, or a backdated fill. The
    high-water mark does not move here -- it is a function of *marked* equity,
    so it updates in ``mark``. The single exception is ``DEPOSIT``, which is
    capital arriving rather than a gain: it lifts the mark by its own amount so
    that new money is not reported forever after as a recovered drawdown, and
    so ``replay`` from the genesis deposit reproduces ``genesis`` exactly.

    Cost basis is *total dollars paid*, and a partial sell reduces it
    proportionally: a sale of ``s`` of ``S`` shares leaves ``basis * (S-s)/S``.
    Proportional reduction is the only convention that keeps the remaining
    basis independent of sale ordering, which matters because the optimizer
    trims positions in whatever order the solver returns them. A buy
    capitalises its commission into the basis (it is dollars paid); a sell
    commission comes out of proceeds and leaves the remaining basis alone.
    """
    if fill.as_of < state.as_of:
        raise AccountError(f"backdated fill {fill.fill_id}: {fill.as_of} < {state.as_of}")
    if fill.shares <= 0:
        raise AccountError(f"fill {fill.fill_id} has non-positive quantity {fill.shares}")
    if fill.price < 0:
        raise AccountError(f"fill {fill.fill_id} has negative price {fill.price}")
    if fill.commission < 0:
        raise AccountError(f"fill {fill.fill_id} has negative commission {fill.commission}")

    if fill.side is Side.DEPOSIT:
        amount = fill.shares * fill.price - fill.commission
        if amount < 0:
            raise AccountError(f"deposit {fill.fill_id} costs more than it delivers")
        return AccountState(
            as_of=fill.as_of,
            cash=state.cash + amount,
            positions=state.positions,
            high_water_mark=state.high_water_mark + amount,
        )

    shares = _quantise(fill.shares, fractional=fractional)
    positions = dict(state.positions)

    if fill.side is Side.BUY:
        cost = shares * fill.price + fill.commission
        cash = state.cash - cost
        if cash < -EQUITY_TOLERANCE:
            raise AccountError(
                f"overdraft on {fill.fill_id}: costs {cost} against cash {state.cash}"
            )
        held = positions.get(fill.symbol)
        positions[fill.symbol] = Position(
            symbol=fill.symbol,
            shares=_quantise((held.shares if held else 0.0) + shares, fractional=fractional),
            cost_basis=(held.cost_basis if held else 0.0) + cost,
        )
    elif fill.side is Side.SELL:
        held = positions.get(fill.symbol)
        if held is None or shares > held.shares:
            have = held.shares if held else 0.0
            raise AccountError(
                f"oversell on {fill.fill_id}: {shares} of {fill.symbol} against {have} held"
            )
        cash = state.cash + shares * fill.price - fill.commission
        if cash < -EQUITY_TOLERANCE:
            raise AccountError(f"sell {fill.fill_id} overdraws cash on commission")
        remaining = _quantise(held.shares - shares, fractional=fractional)
        if remaining <= 0.0:
            # Below half a share-precision unit the holding is dust, not a
            # position; carrying it would mark equity against a phantom.
            positions.pop(fill.symbol, None)
        else:
            positions[fill.symbol] = Position(
                symbol=fill.symbol,
                shares=remaining,
                cost_basis=held.cost_basis * remaining / held.shares,
            )
    else:  # pragma: no cover -- StrEnum is exhaustive above
        raise AccountError(f"unhandled side {fill.side}")

    return AccountState(
        as_of=fill.as_of,
        cash=max(cash, 0.0),
        positions=MappingProxyType(positions),
        high_water_mark=state.high_water_mark,
    )


def decay_high_water_mark(
    high_water_mark: float, equity: float, *, halflife_days: int, periods: int = 1
) -> float:
    """Let the mark forget a drawdown at a fixed rate, so a halt can end.

    A ratcheting mark plus a flatten-at-15% rule is an **absorbing state**, and
    the replay measured it: the account peaked at $15,942 on 2020-02-14, fell to
    $12,917 by 2020-03-12 -- a 18.97% drawdown -- and the governor flattened it
    to cash. Equity then moved on 0 of the following 1,603 sessions and the
    governor returned gross on 0 of 1,604. It could not: a flat account cannot
    make a new high, so the mark could not move, so the drawdown stayed pinned
    at 18.97% forever. Recovery required being invested and the rule forbade
    being invested. 6.4 years dead, ~$25,000 of forgone recovery.

    The rule here closes that loop. While equity sits *below* the mark, half of
    the remaining gap is forgotten every ``halflife_days``, so a drawdown heals
    even in cash. A new high still ratchets the mark up immediately and without
    decay, exactly as it always did.

    Decay rather than the two obvious alternatives, and the reason is the same
    in both cases -- neither of them makes re-entry a function of the damage:

      * *Reset the mark on flatten* lets the mark chase losses down, so a slow
        bleed re-baselines at every halt and never trips a second one.
      * *A fixed time-box* re-enters on a calendar date whatever happened.

    Under decay, how long the halt lasts depends on how deep the hole is, and
    slow-bleed protection survives untouched: the mark can only forget at one
    fixed rate, so an account losing faster than that keeps sinking deeper and
    stays halted. For the shipped 63-session half-life that break-even is a
    loss of ``1 - 0.5 ** (1/63)`` = 1.09% a session.

    Pure arithmetic on floats, no state. ``periods`` exists so a caller marking
    weekly, or catching up after a gap, gets the same answer as one marking
    daily: the decay composes, ``f(f(h, 1), 1) == f(h, 2)``.
    """
    if halflife_days <= 0:
        raise ValueError(f"halflife_days must be positive, got {halflife_days}")
    if periods < 0:
        raise ValueError(f"periods cannot be negative, got {periods}")
    if equity >= high_water_mark:
        # A new high. There is no drawdown to forget, and the ratchet is
        # unconditional on the half-life -- the mark IS equity.
        return float(equity)
    gap = high_water_mark - equity
    forgotten = 1.0 - 0.5 ** (periods / halflife_days)
    # ``max`` is not decoration: for a large ``periods`` the power underflows to
    # 0.0 and float cancellation in ``hwm - gap`` can land a hair below equity,
    # which would report a *negative* drawdown from a peak the account never hit.
    return float(max(high_water_mark - gap * forgotten, equity))


def mark(
    state: AccountState,
    prices: Mapping[str, float],
    *,
    stale_ok: bool = False,
    hwm_decay_halflife_days: int | None = None,
) -> MarkedAccount:
    """Value the account. A missing price raises unless ``stale_ok``.

    Silently carrying a stale price is how a delisted holding keeps reporting
    equity it does not have, so the caller must say so explicitly. Under
    ``stale_ok`` the holding is valued at its own average cost -- the only
    price-like number the state carries -- which reports no gain rather than an
    invented one. The returned ``prices`` are the ones actually used, so the
    identity is auditable from the ``MarkedAccount`` alone.

    ``hwm_decay_halflife_days`` maintains the high-water mark: ``None`` is the
    pure ratchet this account has always used, and an int applies one period of
    :func:`decay_high_water_mark` when equity is under the mark. It defaults to
    ``None`` so that every caller predating the decay keeps the mark it had --
    a marking function that silently changed the drawdown it reported would
    change every governor decision downstream of it.
    """
    effective = dict(prices)
    for symbol, position in state.positions.items():
        price = prices.get(symbol)
        if price is None:
            if not stale_ok:
                raise AccountError(f"no price for held symbol {symbol} on {state.as_of}")
            effective[symbol] = position.cost_basis / position.shares if position.shares else 0.0
        elif price < 0:
            raise AccountError(f"negative price {price} for {symbol} on {state.as_of}")

    equity = state.cash + sum(
        position.shares * effective[symbol] for symbol, position in state.positions.items()
    )
    if hwm_decay_halflife_days is None:
        high_water = max(state.high_water_mark, equity)
    else:
        high_water = decay_high_water_mark(
            state.high_water_mark, equity, halflife_days=hwm_decay_halflife_days
        )
    drawdown = (high_water - equity) / high_water if high_water > 0 else 0.0
    return MarkedAccount(
        state=AccountState(
            as_of=state.as_of,
            cash=state.cash,
            positions=state.positions,
            high_water_mark=high_water,
        ),
        prices=MappingProxyType(effective),
        equity=equity,
        drawdown=max(drawdown, 0.0),
    )


def replay(fills: Iterable[Fill], *, fractional: bool = True) -> AccountState:
    """Fold a fill log from genesis. Must reproduce the live state to 1e-9.

    The log must open with its own ``DEPOSIT``. An empty log is an error rather
    than an empty account, because "no rows" and "an account that was never
    funded" are the same picture and only one of them is safe to trade from.
    """
    ordered = list(fills)
    if not ordered:
        raise AccountError("cannot replay an empty ledger: genesis is a DEPOSIT fill")
    if ordered[0].side is not Side.DEPOSIT:
        raise AccountError(f"ledger opens with {ordered[0].side}, not the genesis DEPOSIT")
    state = AccountState(
        as_of=ordered[0].as_of, cash=0.0, positions=_NO_POSITIONS, high_water_mark=0.0
    )
    for fill in ordered:
        state = apply_fill(state, fill, fractional=fractional)
    return state


def equity_identity_holds(marked: MarkedAccount, *, tolerance: float = EQUITY_TOLERANCE) -> bool:
    """``equity == cash + sum(shares * price)``. Asserted every simulated day.

    A holding with no price fails the check rather than being skipped: an
    unpriceable position is precisely the case where reported equity is a
    fiction, so it must not be able to pass by omission.
    """
    total = marked.state.cash
    for symbol, position in marked.state.positions.items():
        price = marked.prices.get(symbol)
        if price is None:
            return False
        total += position.shares * price
    return abs(marked.equity - total) <= tolerance


__all__ = [
    "EQUITY_TOLERANCE",
    "SHARE_PRECISION",
    "AccountError",
    "AccountState",
    "Fill",
    "MarkedAccount",
    "Order",
    "Position",
    "Side",
    "apply_fill",
    "decay_high_water_mark",
    "equity_identity_holds",
    "genesis",
    "genesis_fill",
    "mark",
    "replay",
]
