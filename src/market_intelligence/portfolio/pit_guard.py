"""A sealed clock: point-in-time reads that raise instead of leaking.

The three canaries in ``test_portfolio_simulator.py`` (t+1 execution, the
covariance window, the shuffle) are spot checks. Each proves one specific path
through ``execute_day`` does not look ahead, on the fixture that path was
written against. None of them says anything about a path added tomorrow --
a new feature that reads ``panel.close`` a row too far, or folds a signal in
by the wrong dict key, would not fail a single one of the three existing
canaries, because none of them runs against the new code at all.

This module is the structural answer: instead of testing that a lookahead
bug's *symptom* (a wrong number) did not appear on one fixture, it makes the
*mechanism* -- reading data dated after "now" -- raise, on every fixture,
including ones nobody has written yet. A caller declares ``PointInTime(as_of=D)``
and reads through the wrappers below instead of the raw ``MarketPanel`` /
``signals_by_day``; anything they touch that is dated after ``D`` raises
``LookaheadError`` instead of silently returning.

**Opt-in and zero-cost when unused.** Nothing in ``simulator.py`` constructs a
``PointInTime`` or a sealed reader, so ``replay`` and ``execute_day`` are
byte-for-byte unchanged -- this module is additive, imported by nothing in the
production path. It exists for tests (see ``test_pit_guard.py``) and for any
future feature that wants the stronger guarantee while it is being built,
without asking the whole simulator to pay for a check it does not need.

What it catches, precisely:
  * **Bar-level.** ``SealedMarketPanel.bar`` refuses a single day's OHLC dated
    after the clock.
  * **Window-level.** ``SealedMarketPanel.trailing_returns`` refuses a window
    whose ``end_exclusive`` reaches past the clock -- the shape a corrupted
    mean or covariance would take, since every such estimate in this codebase
    is built from that one method's output (``feed.MarketPanel.trailing_returns``'s
    own docstring: "the no-lookahead choke point for the entire system").
  * **Signal-level.** ``SealedSignalFeed`` refuses a signal whose own
    ``available_on`` exceeds the clock, checked against the signal's own field
    rather than the dict key it happens to be filed under -- so a signal
    misfiled a day early is still caught, not just a signal asked for on the
    wrong day.

What it CANNOT catch, precisely -- read this before trusting it for anything
it was not built for:
  * **Anything read around it.** The seal is a property of the object a
    caller chose to read through, not a property of the data. A second,
    unwrapped reference to the same ``MarketPanel`` (or a caller who never
    builds a sealed reader at all) is invisible to this module -- there is
    nothing here to intercept a call that never reaches it. This is the same
    limitation every opt-in wrapper has, stated because an opt-in guard that
    is believed to be mandatory is worse than no guard.
  * **Corpus-level / aggregate statistics fit across many dated records in one
    shot** -- a TF-IDF vectorizer, a cross-sectional z-score, a discovery-split
    cutoff picked by eye. These are not a single out-of-order READ; they are a
    STATISTIC whose formula mixes many records' information into one number
    before any "as of D" question is ever posed of it. A sealed reader can
    enforce "every record you touch is dated on or before D"; it cannot
    retroactively repair a fit that already ran over records from every date
    at once, because by the time the number exists, there was no per-record
    read left to guard. ``SealedCorpus`` below offers the discipline for this
    case (fit only on ``.through(as_of)``, refit per day rather than once);
    ``test_pit_guard.py`` proves, with a control that fits the same corpus
    directly and is not caught, that the discipline works only when followed
    and that nothing forces it to be.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from market_intelligence.analytics.backtest import BacktestSignal
from market_intelligence.portfolio.feed import MarketPanel


@dataclass(frozen=True)
class PointInTime:
    """"It is now this date." The one fact every guard below checks against.

    Deliberately just a date, not a context manager or a thread-local. Every
    other object this package passes around (``CostModel``, ``PortfolioConfig``)
    is handed down explicitly rather than reached for out of ambient state, and
    a "current simulated date" living in a global would make two concurrent
    replays -- the live step and a stress-suite scenario, say, which
    ``run_stress_suite`` already runs one after another over the same process
    -- able to stomp on each other's clock. Pass it down like everything else.
    """

    as_of: date


class LookaheadError(RuntimeError):
    """Raised when a sealed reader is asked for data dated after its clock.

    Carries both dates so a test assertion or a CI log does not have to
    re-derive which side of the boundary broke.
    """

    def __init__(self, *, requested: date, as_of: date, what: str) -> None:
        self.requested = requested
        self.as_of = as_of
        super().__init__(
            f"{what} dated {requested.isoformat()} is not yet knowable "
            f"as of {as_of.isoformat()}"
        )


@dataclass(frozen=True)
class SealedBar:
    """One symbol's OHLC + raw close for one day, read through the seal."""

    open: float
    high: float
    low: float
    close: float
    raw_close: float


@dataclass(frozen=True)
class SealedMarketPanel:
    """A ``MarketPanel`` that refuses to hand back a bar or a window dated
    after its clock.

    Wraps rather than replaces: every method here has an unguarded twin
    already on ``MarketPanel`` (``trailing_returns``, ``last_known_price``).
    Construct one with a real panel and a clock; a caller that still holds the
    unwrapped panel, or never builds this at all, is not protected -- see the
    module docstring's "what it cannot catch."
    """

    panel: MarketPanel
    clock: PointInTime

    def bar(self, symbol: str, day: date) -> SealedBar:
        """One day's OHLC + raw close. Raises if ``day`` is after the clock."""
        if day > self.clock.as_of:
            raise LookaheadError(requested=day, as_of=self.clock.as_of, what=f"{symbol} bar")
        row = self.panel.index_of(day)
        column = self.panel.column_of(symbol)
        return SealedBar(
            open=float(self.panel.open[row, column]),
            high=float(self.panel.high[row, column]),
            low=float(self.panel.low[row, column]),
            close=float(self.panel.close[row, column]),
            raw_close=float(self.panel.raw_close[row, column]),
        )

    def trailing_returns(
        self, *, end_exclusive: date, lookback: int, symbols: tuple[str, ...] | None = None
    ) -> np.ndarray:
        """Guarded ``MarketPanel.trailing_returns``.

        The naive check -- comparing ``end_exclusive`` to ``as_of`` by
        calendar-day arithmetic -- is wrong, and provably so on this panel's
        own calendar: the simulator's real usage (``simulator._live_views``)
        calls this with ``end_exclusive`` set to *today*, a real trading day,
        to reach yesterday's close. On a Friday-to-Monday boundary "today" is
        three calendar days after "yesterday," so a fixed-offset check either
        rejects that legitimate call or accepts a two-session lookahead across
        a shorter gap -- wrong in both directions. So this reruns the exact
        ``searchsorted`` the wrapped method uses to find the last row the call
        could touch, and checks THAT row's own calendar date -- the same
        boundary the real method enforces, not an approximation of it.
        """
        calendar = self.panel.calendar
        end_row = int(np.searchsorted(calendar, np.datetime64(end_exclusive, "D"), "left"))
        if end_row > 0:
            last_touched: date = calendar[end_row - 1].astype(object)
            if last_touched > self.clock.as_of:
                raise LookaheadError(
                    requested=last_touched,
                    as_of=self.clock.as_of,
                    what="trailing_returns window ending",
                )
        return self.panel.trailing_returns(
            end_exclusive=end_exclusive, lookback=lookback, symbols=symbols
        )

    def last_known_price(self, symbol: str, *, as_of: date) -> float | None:
        """Guarded ``MarketPanel.last_known_price``.

        The wrapped method's own ``as_of`` is inclusive of that day (it reads
        ``daily_prices`` with a right-side ``searchsorted``), so the bound
        checked here is ``as_of`` itself, not ``as_of - 1`` -- matching the
        method it guards rather than the stricter, exclusive convention
        ``trailing_returns`` uses.
        """
        if as_of > self.clock.as_of:
            raise LookaheadError(requested=as_of, as_of=self.clock.as_of, what="last_known_price")
        return self.panel.last_known_price(symbol, as_of=as_of)

    def index_of(self, day: date) -> int:
        """Delegates unguarded: naming a row number reveals nothing about its
        price, only whether ``day`` is a trading day at all."""
        return self.panel.index_of(day)

    def column_of(self, symbol: str) -> int:
        """Delegates unguarded, for the same reason as ``index_of``."""
        return self.panel.column_of(symbol)


@dataclass(frozen=True)
class SealedSignalFeed:
    """A ``signals_by_day`` mapping that refuses to hand back a signal whose
    own ``available_on`` is after the clock.

    Checked against the signal's own field, not just the dict key it is filed
    under -- ``load_feed`` files every signal under ``signal.available_on``
    itself (see ``feed.load_feed``), so in practice the two agree, but a
    caller building a fixture by hand (as the tests here do) can misfile one,
    and this catches that too rather than only catching a bad *lookup*.
    """

    signals_by_day: Mapping[date, Sequence[BacktestSignal]]
    clock: PointInTime

    def on_day(self, day: date) -> tuple[BacktestSignal, ...]:
        """Every signal filed under exactly ``day``. Raises if ``day`` itself
        is after the clock, before even looking at what is filed there."""
        if day > self.clock.as_of:
            raise LookaheadError(requested=day, as_of=self.clock.as_of, what="signal day")
        return self._checked(self.signals_by_day.get(day, ()))

    def through(self, floor_day: date, ceil_day: date) -> tuple[BacktestSignal, ...]:
        """Every signal with ``floor_day <= available_on <= ceil_day``.

        Mirrors ``simulator._signal_window``'s own range query, which is the
        real caller this is modelled on.
        """
        if ceil_day > self.clock.as_of:
            raise LookaheadError(requested=ceil_day, as_of=self.clock.as_of, what="signal window")
        window = (
            signal
            for day, signals in self.signals_by_day.items()
            if floor_day <= day <= ceil_day
            for signal in signals
        )
        return self._checked(window)

    def _checked(self, signals: Iterable[BacktestSignal]) -> tuple[BacktestSignal, ...]:
        """The defense-in-depth pass: every signal actually returned must
        still satisfy the clock on its own field, independent of which dict
        key or day argument was used to reach it."""
        out = tuple(signals)
        for signal in out:
            if signal.available_on > self.clock.as_of:
                raise LookaheadError(
                    requested=signal.available_on,
                    as_of=self.clock.as_of,
                    what=f"signal {signal.symbol} available_on",
                )
        return out


@dataclass(frozen=True)
class SealedCorpus:
    """A day-bucketed document store, for the corpus-level leak canary only.

    Nothing in this codebase fits a TF-IDF vectorizer or any other
    document-frequency statistic today -- there is no NLP/corpus feature to
    wrap. This class exists solely so ``test_pit_guard.py`` can run a concrete,
    executable case of the leak class the module docstring warns about:
    fitting a corpus-wide statistic leaks future documents into a past
    decision even though no single document is ever read out of order. Read
    that docstring's "what it cannot catch" section before assuming this class
    solves the problem generally -- it only helps a caller who chooses to fit
    through ``.through()`` instead of over the raw mapping.
    """

    documents_by_day: Mapping[date, Sequence[tuple[str, ...]]]
    clock: PointInTime

    def through(self, ceil_day: date) -> tuple[tuple[str, ...], ...]:
        """Every document dated on or before ``ceil_day``. Raises if
        ``ceil_day`` itself is after the clock -- checked before any document
        is read, exactly like ``SealedSignalFeed.through``."""
        if ceil_day > self.clock.as_of:
            raise LookaheadError(requested=ceil_day, as_of=self.clock.as_of, what="corpus window")
        return tuple(
            document
            for day, documents in self.documents_by_day.items()
            if day <= ceil_day
            for document in documents
        )


def document_frequency(corpus: Sequence[Sequence[str]]) -> dict[str, int]:
    """How many documents each token appears in at least once.

    The one building block the corpus leak-canary test needs to demonstrate
    the failure mode; nothing upstream calls this. Deliberately the simplest
    possible corpus-wide statistic -- IDF is ``log(N / df)`` of exactly this
    number -- so the test's arithmetic is checkable by hand rather than
    routed through a library's own smoothing conventions.
    """
    counts: dict[str, int] = {}
    for document in corpus:
        for token in set(document):
            counts[token] = counts.get(token, 0) + 1
    return counts


__all__ = [
    "LookaheadError",
    "PointInTime",
    "SealedBar",
    "SealedCorpus",
    "SealedMarketPanel",
    "SealedSignalFeed",
    "document_frequency",
]
