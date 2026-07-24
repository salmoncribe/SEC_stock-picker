"""Trade-plan math: from a signal and price history to stop, size, and exits.

Pure functions over in-memory bars so the whole module tests without a
database. Two deliberate choices from the design doc:

* **All range math runs in adjusted price space.** Each bar's open/high/low is
  scaled by its own ``adj_close / close`` factor before the true range is
  computed, so a split or large dividend inside the ATR window cannot inflate
  the stop distance. The most recent bar's factor is ~1, so the result is in
  current-price terms and composes directly with a raw entry reference.
* **A plan that cannot be sized is still a plan.** ``shares == 0`` (account too
  small for the risk budget at this stop distance) sets ``unsizeable`` rather
  than suppressing the alert — the geometry is still information.

A degenerate plan is different from an unsizeable one, though: if the stop or
target would land at or below zero (ATR too wide for how cheap the instrument
is), ``build_trade_plan`` returns ``None`` outright rather than a clamped
plan — a bad geometry is not information, it is a sign the volatility model
does not fit this instrument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Bar:
    """One daily OHLC bar; ``adj_close`` carries the split/dividend factor."""

    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float


@dataclass(frozen=True)
class TradePlan:
    entry_ref: float
    stop: float
    target: float
    shares: int
    notional: float
    risk_amount: float
    time_exit_date: date
    atr: float
    unsizeable: bool


def _factor(bar: Bar) -> float:
    if bar.close <= 0:
        return 1.0
    return bar.adj_close / bar.close


def _is_valid_bar(bar: Bar) -> bool:
    """A provider bar is usable only if every field is a finite, positive-priced value."""
    return (
        math.isfinite(bar.open)
        and math.isfinite(bar.high)
        and math.isfinite(bar.low)
        and math.isfinite(bar.close)
        and math.isfinite(bar.adj_close)
        and bar.close > 0
        and bar.adj_close > 0
    )


def wilder_atr(bars: list[Bar], period: int) -> float | None:
    """ATR over adjusted OHLC with Wilder's smoothing; None if history is thin.

    A corrupt provider bar (a non-finite field, or a non-positive close/
    adj_close) is dropped before the length check and the smoothing loop,
    rather than allowed to poison the window — one bad tick from a feed
    should not collapse the whole ATR toward zero or blow it up toward NaN.
    """
    bars = [b for b in bars if _is_valid_bar(b)]
    if len(bars) < period + 1:
        return None
    ordered = sorted(bars, key=lambda b: b.date)
    highs, lows, closes = [], [], []
    for bar in ordered:
        f = _factor(bar)
        highs.append(bar.high * f)
        lows.append(bar.low * f)
        closes.append(bar.close * f)
    ranges = [
        max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        for i in range(1, len(ordered))
    ]
    atr = sum(ranges[:period]) / period
    for tr in ranges[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def build_trade_plan(
    *,
    bars: list[Bar],
    direction: int,
    predicted_move: float,
    horizon_days: int,
    as_of: date,
    account_equity: float,
    risk_pct_per_trade: float,
    atr_period: int,
    atr_stop_multiple: float,
    max_position_pct: float,
    entry_override: float | None = None,
) -> TradePlan | None:
    """Turn one directional signal into a full plan, or None without history.

    ``entry_override`` anchors the plan at a live quote instead of the last
    bar's close — the gap path passes the morning quote here, because prior
    close and the quote differ by exactly the gap being traded.

    Returns ``None`` (rather than a clamped plan) whenever the stop or target
    would land at or below zero — that combination of entry, ATR, and stop
    multiple means the volatility model does not fit this instrument, and no
    plan is more honest than a fake one.
    """
    atr = wilder_atr(bars, atr_period)
    if atr is None or not math.isfinite(atr) or atr <= 0 or direction == 0:
        return None
    # Entry reads the same validity-filtered view of history as the ATR does:
    # a stored NaN close on the single most-recent bar must not become an
    # entry that sails through every <= 0 guard (NaN compares false).
    valid_bars = [b for b in bars if _is_valid_bar(b)]
    if entry_override is not None:
        entry = entry_override
    elif valid_bars:
        entry = sorted(valid_bars, key=lambda b: b.date)[-1].close
    else:
        return None
    if not math.isfinite(entry) or entry <= 0:
        return None

    stop_distance = atr_stop_multiple * atr
    if direction > 0:
        stop = entry - stop_distance
        target = entry * (1.0 + abs(predicted_move))
    else:
        stop = entry + stop_distance
        target = entry * (1.0 - abs(predicted_move))

    if stop <= 0 or target <= 0:
        return None

    risk_budget = account_equity * (risk_pct_per_trade / 100.0)
    shares = math.floor(risk_budget / stop_distance)
    max_notional = account_equity * (max_position_pct / 100.0)
    if shares * entry > max_notional:
        shares = math.floor(max_notional / entry)
    shares = max(shares, 0)

    return TradePlan(
        entry_ref=entry,
        stop=stop,
        target=target,
        shares=shares,
        notional=shares * entry,
        risk_amount=shares * stop_distance,
        time_exit_date=as_of + timedelta(days=horizon_days),
        atr=atr,
        unsizeable=shares == 0,
    )


__all__ = ["Bar", "TradePlan", "build_trade_plan", "wilder_atr"]
