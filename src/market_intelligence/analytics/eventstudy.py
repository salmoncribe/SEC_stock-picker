"""Event-study sample construction. Pure and deterministic.

Turns "an event became public at company A" plus "here is B's return series"
into "here is B's abnormal return over the N trading days after anyone could
have acted on it". That row is the unit every later statistic is computed from,
so the two rules below are the load-bearing part of this module.

**t=0 is the first trading day strictly after the event became public.**
Filing dates carry no time of day: a Form 4 stamped 2024-03-15 may have been
filed at 09:00 or at 20:00, and there is no way to tell from the data. Treating
the filing date itself as tradeable would, for every after-hours filing, credit
the model with a move it could not have captured. Being one day conservative
costs a little measured edge; being one day optimistic manufactures edge that
does not exist, and the second error is unrecoverable because it looks like
success.

**Splits are chronological and purged, never random.** A random train/test split
leaks: adjacent days share information, so a neighbouring day in the training
set tells you most of what the test day will do. Worse, a sample near the
boundary has a forward window that physically overlaps the other side, so its
label is partly determined by prices from the opposite split. Those straddling
samples are dropped rather than assigned, which is what makes a holdout number
mean what it claims.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from market_intelligence.analytics.returns import ReturnPoint


class SampleSplit(StrEnum):
    """Which half of the chronological split a sample belongs to.

    ``DISCOVERY`` decides what is admitted; ``HOLDOUT`` decides what the
    reported track record is. Keeping those on disjoint data is the difference
    between a hit rate and a description of the data it was chosen on.
    """

    DISCOVERY = "discovery"
    HOLDOUT = "holdout"


@dataclass(frozen=True)
class ForwardWindow:
    """The realized forward window for one event against one symbol."""

    t0: date
    window_end: date
    trading_days: int
    cumulative_abnormal_return: float


def first_trading_day_after(trading_days: list[date], available_on: date) -> date | None:
    """The first day in ``trading_days`` strictly after ``available_on``.

    ``trading_days`` must be sorted ascending. Strictly after, never on: see the
    module docstring on why an event's own publication date is not tradeable.
    """
    for day in trading_days:
        if day > available_on:
            return day
    return None


class ReturnSeries:
    """One symbol's return series, prepared for repeated window lookups.

    Sorted once at construction and indexed by binary search, because the
    dataset builder asks for millions of windows against the same few hundred
    series. Sorting per lookup instead turns a minutes-long build into a
    days-long one.

    This is the only implementation of the window rule; :func:`forward_window`
    delegates to it. Keeping a second "fast path" beside the tested one is how
    an optimised copy quietly drifts from the reference -- and the rule being
    implemented here is the leakage boundary, so a drift would not be a
    slowdown, it would be a wrong number that still looked plausible.
    """

    __slots__ = ("_dates", "_points")

    def __init__(self, points: list[ReturnPoint]) -> None:
        self._points = sorted(points, key=lambda p: p.price_date)
        self._dates = [p.price_date for p in self._points]

    def __len__(self) -> int:
        return len(self._points)

    def forward(self, available_on: date, horizon_days: int) -> ForwardWindow | None:
        """Abnormal return summed over ``horizon_days`` trading days after t=0.

        The window starts at the first trading day strictly after
        ``available_on`` and runs for ``horizon_days`` *trading* days, taken
        from this symbol's own observed series rather than a calendar, so
        holidays and halts need no special handling.

        Returns ``None`` rather than a partial result when the window cannot be
        fully formed: too few trading days remain, or any day in it has no
        abnormal return. A 5-day CAR summed from 3 usable days is not a 5-day
        CAR, and quietly treating it as one would understate the variance of
        every statistic built on top of it.
        """
        if horizon_days < 1:
            return None

        # bisect_right lands past any entry equal to available_on, which is what
        # makes t=0 strictly after publication rather than on it.
        start = bisect_right(self._dates, available_on)
        window = self._points[start : start + horizon_days]

        if len(window) < horizon_days:
            return None

        total = 0.0
        for point in window:
            if point.abnormal_return is None:
                return None
            total += point.abnormal_return

        return ForwardWindow(
            t0=window[0].price_date,
            window_end=window[-1].price_date,
            trading_days=len(window),
            cumulative_abnormal_return=total,
        )


def forward_window(
    points: list[ReturnPoint],
    available_on: date,
    horizon_days: int,
) -> ForwardWindow | None:
    """One-shot convenience wrapper over :class:`ReturnSeries`.

    Builds an index and discards it, so it is O(n log n) per call. Fine for a
    handful of lookups; callers issuing many against one symbol should hold a
    ``ReturnSeries`` instead.
    """
    return ReturnSeries(points).forward(available_on, horizon_days)


def assign_split(window: ForwardWindow, split_date: date) -> SampleSplit | None:
    """Place a sample in the discovery or holdout half, or drop it.

    A sample is ``DISCOVERY`` only if its *entire* forward window closes before
    ``split_date``, and ``HOLDOUT`` only if it *opens* on or after it. Anything
    straddling the boundary returns ``None`` and must be discarded: its label is
    computed from prices on both sides, so counting it in either half would let
    discovery-period information into the holdout number that is supposed to be
    independent of it.

    This is the "purge" in purged cross-validation, and it is why the boundary
    costs a handful of samples rather than being a free line on a calendar.
    """
    if window.window_end < split_date:
        return SampleSplit.DISCOVERY
    if window.t0 >= split_date:
        return SampleSplit.HOLDOUT
    return None


__all__ = [
    "ForwardWindow",
    "ReturnSeries",
    "SampleSplit",
    "assign_split",
    "first_trading_day_after",
    "forward_window",
]
