"""Grading: close the loop on open trade alerts, in split-adjusted price space.

A fired alert's ``entry_ref``/``stop``/``target`` are stored in *fired-day raw
price terms* -- whatever the ticker's raw close was the day the plan was
built. Left ungraded that way, a split or a large special dividend that lands
during the hold would corrupt every comparison against later raw bars: a 2:1
split alone would make a perfectly healthy position look like it gapped
through its stop. So every comparison here happens in adjusted space instead:

* ``f0`` -- the fired-day factor (``adj_close / close`` of the last bar at or
  before the fired date) -- converts the stored, still-raw ``entry_ref`` /
  ``stop`` / ``target`` into adjusted terms once, up front.
* Each later bar is converted with *its own* factor ``f_t``, so a split that
  happens mid-window rescales both sides of every comparison the same way and
  the cliff in raw prices never appears in adjusted space.

This mirrors ``signals.trade_plan``'s own adjusted-space convention (see that
module's docstring) applied to grading instead of planning. ``Bar`` and
``_is_valid_bar`` are imported directly from there rather than re-implemented
-- the underscore is crossed deliberately because both modules live in
``market_intelligence.signals`` and the bad-bar definition must not drift
between the two.

**Conservative tie-break.** A single bar whose range touches both the stop and
the target is graded ``hit_stop`` -- there is no way to know which happened
first intraday from daily OHLC alone, so grading assumes the worse path
through the bar rather than guessing in the alert's favor.

**Time exit.** Once the walk reaches ``time_exit_date`` without a touch, the
position is graded ``expired`` using the close of the *last bar at or before*
``time_exit_date`` (not the nearest bar to ``as_of``) -- price action after
the planned exit is irrelevant, because the position was already closed by
then. A row with no bars at all through its exit date is left ``still_open``
rather than guessed at; grading requires data, not an assumption.

**A partial window must not fabricate an expiry.** ``expired`` is only ever
finalized when the ticker's price history demonstrably continues on or past
``time_exit_date`` -- i.e. a bar exists with ``date >= time_exit_date``, not
merely a bar somewhere before it. Without that, ``as_of >= time_exit_date``
alone is not proof the exit-date price ever happened: it is equally
consistent with a mid-hold delisting or a provider gap in the tail, where the
feed simply stopped a few days early. Grading a stale last-known close as a
real exit in that case would write a permanent, fabricated outcome onto the
ledger. So the completeness check looks one step past the exit-bounded
``relevant`` window (at bars between ``time_exit_date`` and ``as_of``,
inclusive of the exit date itself) purely to confirm data exists there; the
close actually used still comes from ``relevant``, i.e. never later than
``time_exit_date``. When that confirming bar is absent, the row stays
``still_open`` -- forever, in v1, for a genuinely delisted ticker. That is a
deliberate trade-off: an honest "insufficient data" beats a fabricated exit
price. A delisting-aware terminal state (e.g. grading from the last trade
once a ticker is confirmed delisted) is future work, not this module's job.

Already-graded rows (``outcome != 'open'``) are never re-read here -- the
ledger's first-firing-wins philosophy extends to grading too: once a row has
an outcome, it is a historical fact, not something a later run can revise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from market_intelligence.collectors import pipeline_run
from market_intelligence.logging_config import get_logger
from market_intelligence.schemas.common import utcnow
from market_intelligence.signals.trade_plan import Bar, _is_valid_bar

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.collectors import RunSummary
    from market_intelligence.config import Config

logger = get_logger(__name__)

#: Outcomes ``grade_open_alerts`` can produce, and the zero-filled shape of
#: its return value -- callers can rely on every key being present even when
#: a run grades nothing of that kind.
_OUTCOMES: tuple[str, ...] = ("hit_stop", "hit_target", "expired", "still_open")

#: Outcomes that write an UPDATE to the ledger -- everything except
#: ``still_open``. Shared between the write guard in ``grade_open_alerts``
#: and the ``summary.updated`` count in ``grade`` so the two never drift.
_TERMINAL_OUTCOMES: frozenset[str] = frozenset({"hit_stop", "hit_target", "expired"})


@dataclass(frozen=True)
class _GradeResult:
    outcome: str
    outcome_return: float | None


def _bar_factor(bar: Bar) -> float:
    """``adj_close / close`` for one bar; 1.0 (no adjustment) if close <= 0."""
    if bar.close <= 0:
        return 1.0
    return bar.adj_close / bar.close


def _fired_day_factor(con: duckdb.DuckDBPyConnection, ticker: str, fired_date: date) -> float:
    """f0: the adjustment factor in effect on the day the plan was priced.

    The last bar at or before ``fired_date`` -- falling back to 1.0 (no
    adjustment) when that bar is missing, so a ticker with no price history
    yet before the alert fired still grades: the stop/target simply stay in
    the raw terms they were already stored in.
    """
    row = con.execute(
        "SELECT close, adj_close FROM daily_prices "
        "WHERE symbol = ? AND price_date <= ? ORDER BY price_date DESC LIMIT 1",
        [ticker, fired_date],
    ).fetchone()
    if row is None:
        return 1.0
    close, adj_close = row
    if close is None or adj_close is None:
        return 1.0
    if not math.isfinite(close) or not math.isfinite(adj_close) or close <= 0:
        return 1.0
    return adj_close / close


def _bars_after(
    con: duckdb.DuckDBPyConnection, ticker: str, fired_date: date, as_of: date
) -> list[Bar]:
    """Bars strictly after ``fired_date`` through ``as_of``, ascending.

    Same NULL-row-skipping idiom as ``signals.trade_alerts._load_bars``: a row
    missing any OHLC field cannot become a ``Bar`` (every field is a required
    float) and is dropped before it reaches ``_is_valid_bar``'s finite/positive
    check, rather than crashing the row build.
    """
    rows = con.execute(
        "SELECT price_date, open, high, low, close, adj_close FROM daily_prices "
        "WHERE symbol = ? AND price_date > ? AND price_date <= ? ORDER BY price_date ASC",
        [ticker, fired_date, as_of],
    ).fetchall()
    bars: list[Bar] = []
    for price_date, o, h, low, c, adj in rows:
        if None in (o, h, low, c, adj):
            continue
        bars.append(Bar(date=price_date, open=o, high=h, low=low, close=c, adj_close=adj))
    return [b for b in bars if _is_valid_bar(b)]


def _grade_one(
    con: duckdb.DuckDBPyConnection,
    *,
    ticker: str,
    fired_date: date,
    direction: int,
    entry_ref: float,
    stop: float,
    target: float,
    time_exit_date: date | None,
    as_of: date,
) -> _GradeResult:
    """Walk one alert's bars in adjusted space and grade its outcome."""
    f0 = _fired_day_factor(con, ticker, fired_date)
    entry_adj = entry_ref * f0
    stop_adj = stop * f0
    target_adj = target * f0

    bars = _bars_after(con, ticker, fired_date, as_of)
    # Price action after the planned exit cannot touch a position that is
    # already closed by then -- restrict the walk to bars at/before the exit.
    relevant = [b for b in bars if time_exit_date is None or b.date <= time_exit_date]

    for bar in relevant:
        f_t = _bar_factor(bar)
        high_adj = bar.high * f_t
        low_adj = bar.low * f_t
        if direction > 0:
            stop_hit = low_adj <= stop_adj
            target_hit = high_adj >= target_adj
        else:
            stop_hit = high_adj >= stop_adj
            target_hit = low_adj <= target_adj

        # Stop checked first so a bar touching both grades as hit_stop --
        # the conservative, worst-path-through-the-bar assumption.
        if stop_hit:
            ret = direction * (stop_adj - entry_adj) / entry_adj
            return _GradeResult("hit_stop", ret)
        if target_hit:
            ret = direction * (target_adj - entry_adj) / entry_adj
            return _GradeResult("hit_target", ret)

    if time_exit_date is not None and as_of >= time_exit_date and relevant:
        # Completeness check: the exit-date close is only trustworthy if the
        # ticker's history demonstrably continues on/past time_exit_date --
        # otherwise a delisting or a provider gap in the tail is
        # indistinguishable from a genuine, quiet expiry (see module
        # docstring). This peeks past `relevant` on purpose; the price used
        # below still comes only from `relevant` (never later than the exit
        # date).
        window_complete = any(b.date >= time_exit_date for b in bars)
        if window_complete:
            last_bar = relevant[-1]
            close_adj = last_bar.close * _bar_factor(last_bar)
            ret = direction * (close_adj - entry_adj) / entry_adj
            return _GradeResult("expired", ret)

    return _GradeResult("still_open", None)


def grade_open_alerts(con: duckdb.DuckDBPyConnection, as_of: date) -> dict[str, int]:
    """Grade every ``outcome = 'open'`` row against bars through ``as_of``.

    Returns counts per outcome (always all four keys, zero-filled). Rows that
    stay open are left untouched in the database -- only rows that resolve to
    ``hit_stop`` / ``hit_target`` / ``expired`` are updated, and each exactly
    once (the ``WHERE outcome = 'open'`` read means an already-graded row is
    never re-read, let alone re-graded).
    """
    counts: dict[str, int] = dict.fromkeys(_OUTCOMES, 0)

    rows = con.execute(
        "SELECT alert_id, ticker, CAST(fired_at AS DATE) AS fired_date, direction, "
        "entry_ref, stop, target, time_exit_date "
        "FROM trade_alerts WHERE outcome = 'open'"
    ).fetchall()

    for alert_id, ticker, fired_date, direction, entry_ref, stop, target, time_exit_date in rows:
        if (
            fired_date is None
            or direction is None
            or entry_ref is None
            or stop is None
            or target is None
            or entry_ref == 0
        ):
            # Not enough on the row to grade against; leave it open rather
            # than guess.
            counts["still_open"] += 1
            continue

        result = _grade_one(
            con,
            ticker=ticker,
            fired_date=fired_date,
            direction=direction,
            entry_ref=entry_ref,
            stop=stop,
            target=target,
            time_exit_date=time_exit_date,
            as_of=as_of,
        )
        counts[result.outcome] += 1
        if result.outcome in _TERMINAL_OUTCOMES:
            con.execute(
                "UPDATE trade_alerts SET outcome = ?, outcome_return = ?, graded_at = ? "
                "WHERE alert_id = ?",
                [result.outcome, result.outcome_return, utcnow(), alert_id],
            )

    return counts


def grade(config: Config, as_of: date | None = None) -> RunSummary:
    """Grade today's open alerts as a pipeline step.

    ``as_of`` defaults to today (UTC date) and is injectable for tests. The
    outcome counts land in ``summary.stage``; ``summary.collected`` is the
    number of open alerts examined (the sum of those counts), matching the
    house convention that ``collected`` means "offered to this stage", not
    "changed by it". ``summary.updated`` is the number of rows that actually
    received a database write -- ``hit_stop`` + ``hit_target`` + ``expired``,
    per the house ``RunSummary`` contract that ``updated`` means rows
    modified; ``still_open`` rows are read but never written, so they do not
    count.
    """
    grading_date = as_of if as_of is not None else datetime.now(tz=UTC).date()
    with pipeline_run(config, "signals.grade-trade-alerts") as (con, summary):
        counts = grade_open_alerts(con, grading_date)
        summary.stage.update(counts)
        summary.collected = sum(counts.values())
        summary.updated = sum(n for outcome, n in counts.items() if outcome in _TERMINAL_OUTCOMES)
    return summary


__all__ = ["grade", "grade_open_alerts"]
