"""Proof that the sealed clock (``portfolio.pit_guard``) actually seals.

Every test here follows the same shape the module's own docstring promises:
plant a leak that a correct point-in-time reader must refuse, show the guard
refuses it, and then show a *control* -- the identical mechanism used on
genuinely past data -- goes through untouched. A guard that blocks everything
is indistinguishable from a broken import; the controls are what prove this
one is actually checking a date rather than just always raising.

The final section is different on purpose. It demonstrates a class of leak
(a corpus-wide statistic, such as document frequency / IDF) that this module's
own docstring says it CANNOT catch, and proves that claim rather than merely
asserting it: the disciplined path (``SealedCorpus.through``) is shown to both
work and refuse over-reach, and then the naive path -- fitting directly over
the raw mapping, exactly as an unsuspecting new feature would -- is shown to
succeed with no exception at all, producing measurably different (leaked)
statistics. That silence is the honest limitation, made concrete instead of
asserted in prose.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from market_intelligence.analytics.backtest import BacktestSignal
from market_intelligence.portfolio.feed import MarketPanel, build_panel
from market_intelligence.portfolio.pit_guard import (
    LookaheadError,
    PointInTime,
    SealedCorpus,
    SealedMarketPanel,
    SealedSignalFeed,
    document_frequency,
)
from market_intelligence.signals.trade_plan import Bar

# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _sessions(count: int, start: date = date(2024, 1, 2)) -> list[date]:
    """``count`` weekday sessions from ``start``."""
    days: list[date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


#: Tue, Wed, Thu, Fri, Mon, Tue -- a real weekend gap sits between index 3 and
#: 4, which is exactly what makes the naive "add one calendar day" boundary
#: check wrong and forces ``SealedMarketPanel.trailing_returns`` to check the
#: panel's own calendar instead (see that method's docstring).
_DAYS = _sessions(6)


def _bar(day: date, close: float) -> Bar:
    return Bar(date=day, open=close, high=close, low=close, close=close, adj_close=close)


def _spike_panel() -> MarketPanel:
    """AAA drifts gently through day 3, then jumps 5x on day 4 -- a return only
    a lookahead could see. BBB is flat and present every session purely to
    keep day 2 (AAA's deliberate gap) on the shared calendar, so
    ``last_known_price`` has a real gap to walk backward across.
    """
    aaa = [
        _bar(_DAYS[0], 10.0),
        _bar(_DAYS[1], 11.0),
        # _DAYS[2]: no AAA bar at all -- the last_known_price gap.
        _bar(_DAYS[3], 13.0),
        _bar(_DAYS[4], 65.0),  # the spike: a decision on day 3 must never see this
        _bar(_DAYS[5], 66.0),
    ]
    bbb = [_bar(day, 5.0) for day in _DAYS]
    return build_panel({"AAA": aaa, "BBB": bbb}, ("AAA", "BBB"))


#: "Today" is day 3 (a Friday): day 4's Monday spike is not yet knowable.
_CLOCK = PointInTime(as_of=_DAYS[3])


def _signal(symbol: str, available_on: date) -> BacktestSignal:
    return BacktestSignal(
        symbol=symbol,
        available_on=available_on,
        direction=1,
        predicted_move=0.05,
        horizon_days=5,
        confidence=80,
    )


# --------------------------------------------------------------------------- #
# 1. bar-level: a single day's OHLC                                            #
# --------------------------------------------------------------------------- #
def test_sealed_panel_refuses_a_bar_dated_after_the_clock() -> None:
    """Reading day 4's bar from a clock stopped at day 3 must raise."""
    sealed = SealedMarketPanel(panel=_spike_panel(), clock=_CLOCK)

    with pytest.raises(LookaheadError) as excinfo:
        sealed.bar("AAA", _DAYS[4])

    assert excinfo.value.requested == _DAYS[4]
    assert excinfo.value.as_of == _CLOCK.as_of


def test_sealed_panel_permits_the_boundary_and_everything_before_it() -> None:
    """Control: the clock's own day, and every day before it, must NOT raise.

    Without this, ``test_sealed_panel_refuses_a_bar_dated_after_the_clock``
    would be equally consistent with a guard that raises unconditionally --
    which would "catch" every leak by also catching everything real.
    """
    sealed = SealedMarketPanel(panel=_spike_panel(), clock=_CLOCK)

    boundary = sealed.bar("AAA", _DAYS[3])
    assert boundary.close == pytest.approx(13.0)

    earliest = sealed.bar("AAA", _DAYS[0])
    assert earliest.close == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# 2. window-level: trailing_returns, and a mean computed over it               #
# --------------------------------------------------------------------------- #
def test_a_covariance_style_mean_over_a_window_reaching_the_future_must_raise() -> None:
    """The exact shape a corrupted covariance/volatility estimate would take:
    a window whose last row is the spike. The mean must never be computable
    at all -- the guard raises before ``trailing_returns`` returns anything,
    not after, so there is no NaN-laundered or silently-wrong number to catch
    downstream.
    """
    sealed = SealedMarketPanel(panel=_spike_panel(), clock=_CLOCK)

    with pytest.raises(LookaheadError) as excinfo:
        np.nanmean(sealed.trailing_returns(end_exclusive=_DAYS[5], lookback=3))

    # the row actually identified as the leak is day 4 -- the spike itself,
    # not day 5, which is what a caller most needs to see in a failure message
    assert excinfo.value.requested == _DAYS[4]


def test_sealed_trailing_returns_permits_the_legitimate_decision_window() -> None:
    """Control: this is ``simulator._live_views``'s own calling convention --
    ``end_exclusive`` set to *today* (day 4, a real trading day) to reach
    yesterday's close (day 3, exactly the clock).

    Two things are checked, not one: that this does not raise at all, and
    that the values it returns are bit-identical to the unwrapped panel's own
    ``trailing_returns`` -- a guard that silently returned zeros instead of
    forwarding the real read would pass the first check and still be useless.
    """
    panel = _spike_panel()
    sealed = SealedMarketPanel(panel=panel, clock=_CLOCK)

    guarded = sealed.trailing_returns(end_exclusive=_DAYS[4], lookback=3)
    direct = panel.trailing_returns(end_exclusive=_DAYS[4], lookback=3)

    assert np.array_equal(guarded, direct, equal_nan=True)
    mean_return = float(np.nanmean(guarded))
    assert math.isfinite(mean_return)
    assert mean_return < 0.5, "the legitimate window must not contain the 5x spike"


# --------------------------------------------------------------------------- #
# 3. last_known_price -- inclusive of its own day, and still backward-only     #
# --------------------------------------------------------------------------- #
def test_sealed_last_known_price_refuses_a_future_as_of() -> None:
    sealed = SealedMarketPanel(panel=_spike_panel(), clock=_CLOCK)

    with pytest.raises(LookaheadError):
        sealed.last_known_price("AAA", as_of=_DAYS[4])


def test_sealed_last_known_price_permits_the_boundary_and_walks_back_across_a_gap() -> None:
    """Control, in two parts. ``as_of`` equal to the clock is inclusive (it
    matches the wrapped method's own convention, not ``trailing_returns``'s
    stricter one) and a legitimate query still walks backward across AAA's
    deliberate day-2 gap to day 1's real print, never forward to day 3's.
    """
    sealed = SealedMarketPanel(panel=_spike_panel(), clock=_CLOCK)

    assert sealed.last_known_price("AAA", as_of=_DAYS[3]) == pytest.approx(13.0)
    assert sealed.last_known_price("AAA", as_of=_DAYS[2]) == pytest.approx(11.0)


# --------------------------------------------------------------------------- #
# 4. signal-level: available_on, checked on the signal itself                  #
# --------------------------------------------------------------------------- #
def test_sealed_signal_feed_refuses_a_signal_dated_after_the_clock() -> None:
    signals_by_day = {_DAYS[1]: [_signal("AAA", _DAYS[1])], _DAYS[4]: [_signal("AAA", _DAYS[4])]}
    sealed = SealedSignalFeed(signals_by_day=signals_by_day, clock=_CLOCK)

    with pytest.raises(LookaheadError):
        sealed.on_day(_DAYS[4])

    with pytest.raises(LookaheadError):
        sealed.through(_DAYS[0], _DAYS[4])


def test_sealed_signal_feed_catches_a_signal_misfiled_under_an_earlier_day() -> None:
    """Defense in depth: the signal's OWN ``available_on`` is checked, not
    just the dict key or the day argument used to reach it. A signal for day 4
    filed under day 2's bucket -- a data bug, not a lookup bug -- must still
    be caught by a query that only asked about day 2.
    """
    misfiled = {_DAYS[2]: [_signal("AAA", _DAYS[4])]}
    sealed = SealedSignalFeed(signals_by_day=misfiled, clock=_CLOCK)

    with pytest.raises(LookaheadError) as excinfo:
        sealed.on_day(_DAYS[2])  # day 2 itself is legitimate; the signal inside is not
    assert excinfo.value.requested == _DAYS[4]


def test_sealed_signal_feed_permits_signals_on_or_before_the_clock() -> None:
    """Control: correctly-filed past signals must come through untouched."""
    signal = _signal("AAA", _DAYS[1])
    signals_by_day = {_DAYS[1]: [signal], _DAYS[4]: [_signal("AAA", _DAYS[4])]}
    sealed = SealedSignalFeed(signals_by_day=signals_by_day, clock=_CLOCK)

    assert sealed.on_day(_DAYS[1]) == (signal,)
    assert sealed.through(_DAYS[0], _DAYS[3]) == (signal,)  # day 4's signal excluded, not raised


# --------------------------------------------------------------------------- #
# 5. corpus-level statistics -- the class this module CANNOT catch by         #
#    construction, proved rather than asserted                                #
# --------------------------------------------------------------------------- #
#: A term present in literally every past document (uninformative by
#: definition -- IDF should be exactly zero) and a term that does not exist
#: until the future. Neither fact about the past changes when the future
#: documents are added; both statistics change anyway if fit over everything.
_CORPUS = {
    _DAYS[0]: (("alpha",),),
    _DAYS[1]: (("alpha",),),
    _DAYS[2]: (("alpha", "common"),),
    _DAYS[4]: (("beta",),),
    _DAYS[5]: (("beta", "common"),),
}
_CORPUS_CLOCK = PointInTime(as_of=_DAYS[2])


def _idf(term: str, corpus: tuple[tuple[str, ...], ...]) -> float:
    """``log(N / df)`` -- the textbook inverse document frequency, spelled out
    so the test's expectation is a formula, not a number copied from a run."""
    df = document_frequency(corpus).get(term, 0)
    if df == 0:
        return math.inf
    return math.log(len(corpus) / df)


def test_sealed_corpus_confines_document_frequency_to_the_clock() -> None:
    """The disciplined path: fit only on ``.through(as_of)``.

    "alpha" appears in every one of the 3 past documents, so its correct,
    point-in-time IDF is exactly zero -- it carries no information among
    documents actually seen. "beta" does not exist yet at all: it is out of
    vocabulary, not merely rare.
    """
    corpus = SealedCorpus(documents_by_day=_CORPUS, clock=_CORPUS_CLOCK)

    past = corpus.through(_DAYS[2])
    assert len(past) == 3
    assert ("beta",) not in past and ("beta", "common") not in past

    past_df = document_frequency(past)
    assert "beta" not in past_df
    assert _idf("alpha", past) == pytest.approx(0.0)
    assert _idf("common", past) == pytest.approx(math.log(3 / 1))

    with pytest.raises(LookaheadError) as excinfo:
        corpus.through(_DAYS[4])
    assert excinfo.value.requested == _DAYS[4]


def test_a_whole_corpus_fit_bypasses_the_seal_and_leaks_future_vocabulary() -> None:
    """The honest limitation, proved rather than asserted.

    This is exactly what an NLP feature written without point-in-time
    discipline in mind would naturally do: flatten the corpus and fit one
    vectorizer over all of it. No ``PointInTime`` is ever declared, no
    ``SealedCorpus`` is ever constructed, and nothing here raises -- there is
    no per-record read for a guard to intercept, because the caller never
    asked ``pit_guard`` anything at all. That silence is the point.
    """
    all_documents = tuple(document for documents in _CORPUS.values() for document in documents)
    past_documents = SealedCorpus(documents_by_day=_CORPUS, clock=_CORPUS_CLOCK).through(_DAYS[2])

    whole_df = document_frequency(all_documents)  # no exception -- nothing was ever asked

    # a term that will not exist for two more sessions is already a first-class
    # feature dimension, indistinguishable at fit time from a real one
    assert whole_df["beta"] == 2

    # and "alpha" -- present in literally every document anyone could have
    # observed by day 2 -- is handed a nonzero importance purely because two
    # unrelated future documents joined the corpus. Nothing about alpha's own,
    # already-observed occurrences changed.
    past_idf_alpha = _idf("alpha", past_documents)
    whole_idf_alpha = _idf("alpha", all_documents)
    assert past_idf_alpha == pytest.approx(0.0)
    assert whole_idf_alpha == pytest.approx(math.log(5 / 3))
    assert whole_idf_alpha != pytest.approx(past_idf_alpha)

    # "common" genuinely present in one past and one future document: even a
    # term whose PAST count is fixed at 1 gets a different weight depending on
    # a document count (N) that only the future changed.
    past_idf_common = _idf("common", past_documents)
    whole_idf_common = _idf("common", all_documents)
    assert past_idf_common == pytest.approx(math.log(3 / 1))
    assert whole_idf_common == pytest.approx(math.log(5 / 2))
    assert whole_idf_common != pytest.approx(past_idf_common)
