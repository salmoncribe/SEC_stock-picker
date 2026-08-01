"""Morning gap scan: overnight price gaps -> the trade-alert ledger.

This is the second trigger path onto the trade-alert chain (the first is the
daily briefing's reaction-lag alerts, ``signals/trade_alerts.build_records``
via the orchestrator). Where that path has a model prediction and a measured
holdout track record behind it, this one has neither: it looks at this
morning's live quote against yesterday's close and asks a much blunter
question -- did this gap enough, on enough volume, to be worth a look.

**Continuation, not mean-reversion.** ``direction`` is the sign of the gap
itself (up-gap -> long, down-gap -> short). This is a deliberate strategy
choice, not the only reasonable one -- a mean-reversion scanner would fade
the gap instead -- but it is the one this scanner encodes, and the message
should read as "this moved and may keep moving," never as a prediction of
where it settles.

**Entry anchors at the live quote, never the prior close.** The two differ by
exactly the gap being traded, so ``build_trade_plan`` is called with
``entry_override=<morning quote>``: passing the prior close instead would
size and target the position off a price nobody can transact at anymore. The
profit target has no model behind it either (there is no ``predicted_car``
on this path), so it is expressed the only way that fits the same code path
as the daily alerts: an R-multiple of the stop distance
(``target = entry +/- gap_target_r_multiple * stop_distance``). Rather than
duplicating that arithmetic, ``wilder_atr`` is called once here (on the same
bars and period ``build_trade_plan`` will use internally) purely to derive
``predicted_move = gap_target_r_multiple * stop_distance / entry`` up front,
so both the ATR that sizes ``predicted_move`` and the ATR ``build_trade_plan``
uses to place the stop come from one deterministic function fed identical
inputs -- there is no second, hand-rolled copy of the target math to drift
out of sync with the first. Note that ``as_of`` only ever controls which
prior-day bars and events are read: the quote itself is always a live call,
so ``scan(as_of=<a past date>)`` is not a true historical replay.

**Confidence starts capped, then earns its way out per bucket.**
``ConfidenceInputs.has_track_record`` was always ``False`` here until
:mod:`market_intelligence.signals.gap_calibration` existed to accumulate real
hit/miss history for ``kind="price_gap"`` rows. Now each candidate is looked
up against its own ``(direction, gap size, catalyst present)`` bucket in
``gap_calibration_stats`` (via ``gap_calibration.lookup_track_record``,
using the same connection this write phase already has open): a bucket with
at least one decisive graded outcome (``hit_target``/``hit_stop``) flips
``has_track_record=True`` and passes that bucket's shrunk hit rate and
sample size through; an empty or undecided bucket falls back to exactly
today's original behaviour -- ``has_track_record=False``, capped at 50. Spec
Sec.5 sketches gap-specific substitute inputs (gap size, liquidity) for the
score directly; those are deliberately *not* implemented as separate blend
terms -- the bucket lookup is the substitute. The only input this path adds
beyond that is ``times_asserted=1 if catalyst else 0`` -- a same-week
filing/event on the ticker nudges the score up a little as corroboration,
exactly the same mechanism the daily path uses for independently-asserted
edges. The evidence's ``note`` field says which case applied (no track
record yet, vs. calibrated on n decisive outcomes) so a human reading it
never mistakes a capped 50 for an earned one, or vice versa. Because the
no-track-record cap sits *below* ``trading.min_confidence`` (60 by default),
gap alerts are gated against their own floor -- ``gap_scanner.min_confidence``
(40 by default) -- rather than the daily path's, or they would never clear
the bar to text at all; a bucket that has earned a track record can, once
its shrunk hit rate is high enough, clear even the daily path's floor.

**Three phases, so the DuckDB lock is never held across network I/O.**
DuckDB is single-writer, and a live quote fetch is comparatively slow and
unreliable (an unofficial endpoint, one HTTP round trip per symbol -- see
below); holding the platform's one write lock for the whole ~500-ticker fetch
would starve every other collector and the relationship extractor for the
run's entire duration. So the work is split into three strictly-sequential
steps with the connection closed in between:

1. ``_read_survivors`` -- a short-lived plain ``database.connection``: pull
   the universe, each ticker's prior bars and 20-day liquidity, and filter to
   the tickers worth quoting (history present, fresh enough, liquid enough).
   Connection closes before this function returns.
2. ``_fetch_quotes`` -- no database connection open at all. One paced
   ``get_latest_price`` call per survivor.
3. ``_persist_candidates`` -- a fresh connection via ``pipeline_run``: now
   that quotes exist, do the actual candidate math (gap threshold, catalyst
   join, ATR, plan, confidence) and persist/gate. This is also where the
   ``pipeline_runs`` bookkeeping row is written, so it reflects the write
   phase's real counts, not the read phase's.

Each phase's entry is wrapped in a bounded retry (``_with_db_retry``: 3
attempts, 30s apart) catching ``duckdb.IOException`` specifically, because a
collector (or the relationship extractor) may already hold the lock when the
morning LaunchAgent fires. A missed scan is a missed message, not a crash:
final failure on either phase logs ``gap_scan_skipped_db_busy`` and returns a
``skipped`` ``RunSummary`` rather than raising.

**Quotes are fetched one symbol at a time, paced.** ``YFinanceMarketDataProvider.
get_latest_price`` (see ``clients/market_yfinance.py``) makes one HTTP fetch
per symbol; across a ~500-ticker survivor set that is up to ~500 sequential
calls, each separated by a short pause (``quote_pause_seconds``, default
0.15s) so this process does not hammer the provider back-to-back. That is
acceptable for v1 -- the scan runs once, well before the trading day gets
interesting, and quoting is isolated behind the ``MarketDataProvider`` ABC so
a batched endpoint can replace it later without this module changing. If more
than half of the attempted quotes fail (a provider outage, not a few dead
symbols), the run still completes but is marked ``status="partial"`` with a
note -- a mostly-throttled scan must not report a clean, silent success.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import duckdb

from market_intelligence import database
from market_intelligence.autopilot import notify as notify_module
from market_intelligence.clients.market_yfinance import YFinanceMarketDataProvider
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.logging_config import get_logger
from market_intelligence.schemas.common import utcnow
from market_intelligence.signals import gap_calibration
from market_intelligence.signals.confidence import ConfidenceInputs, score
from market_intelligence.signals.trade_alerts import (
    TradeAlertRecord,
    gated,
    mark_delivered,
    mark_gated,
    persist_new,
    sendable,
)
from market_intelligence.signals.trade_plan import Bar, build_trade_plan, wilder_atr

if TYPE_CHECKING:
    from market_intelligence.clients.market import LatestPrice, MarketDataProvider
    from market_intelligence.config import Config

_log = get_logger("collectors.gaps")

# Bounded retry around each DB connection entry when another process holds
# the DuckDB single-writer lock -- see the module docstring.
_DB_BUSY_MAX_ATTEMPTS = 3
_DB_BUSY_RETRY_SECONDS = 30.0

# "20-day average dollar volume" per the design spec / gap_scanner config.
_LIQUIDITY_WINDOW = 20

# A prior bar older than this many calendar days before as_of is too stale to
# call an "overnight gap" -- a lagging price sync, not a real gap.
_MAX_STALE_DAYS = 4

# Pause between sequential quote calls (see the module docstring).
_QUOTE_PAUSE_SECONDS = 0.15


@dataclass(frozen=True)
class _Survivor:
    """A ticker that passed the read-only filters and is worth quoting."""

    ticker: str
    bars: list[Bar]
    prior_close: float
    avg_dollar_volume: float


@dataclass(frozen=True)
class _QuoteFetchResult:
    """Live quotes fetched for survivors, plus enough to judge run health."""

    quotes: dict[str, LatestPrice]
    attempted: int
    errors: int


def _with_db_retry[T](fn: Callable[[], T], *, sleep: Callable[[float], None]) -> T:
    """Call ``fn()``, retrying on ``duckdb.IOException`` (the lock is busy).

    Re-raises the last ``IOException`` once ``_DB_BUSY_MAX_ATTEMPTS`` is
    exhausted -- the caller decides what "give up" means (``scan`` turns it
    into a ``skipped`` summary rather than a crash).
    """
    last_error: duckdb.IOException | None = None
    for attempt in range(1, _DB_BUSY_MAX_ATTEMPTS + 1):
        try:
            return fn()
        except duckdb.IOException as exc:
            last_error = exc
            if attempt < _DB_BUSY_MAX_ATTEMPTS:
                sleep(_DB_BUSY_RETRY_SECONDS)
    assert last_error is not None  # loop always sets it before falling through
    raise last_error


def _load_prior_history(
    con: duckdb.DuckDBPyConnection, ticker: str, as_of: date, limit: int
) -> tuple[list[Bar], float]:
    """Bars at/before ``as_of`` for ATR, plus the trailing avg dollar volume.

    Same NULL-row-skipping idiom as ``signals.trade_alerts._load_bars``: a row
    missing any OHLC field cannot become a ``Bar`` and is dropped rather than
    crashing the query or the ATR window. Volume travels on the same rows, so
    the liquidity floor (``close x volume`` over the latest <= 20 bars) comes
    from this one query rather than a second round trip.

    An empty result -- no row at/before ``as_of`` at all -- is the ticker's
    no-op path: a fresh listing, a stale/removed symbol, or (the one case the
    spec calls out explicitly) a weekend/holiday scan where "yesterday's"
    close simply is not there yet relative to a thin seed. The caller treats
    an empty bar list as "skip silently", never as zero bars worth of ATR.
    """
    rows = con.execute(
        "SELECT price_date, open, high, low, close, adj_close, volume "
        "FROM daily_prices WHERE symbol = ? AND price_date <= ? "
        "ORDER BY price_date DESC LIMIT ?",
        [ticker, as_of, limit],
    ).fetchall()

    bars: list[Bar] = []
    dollar_volumes: list[float] = []
    for price_date, o, h, low, c, adj, volume in reversed(rows):
        if None not in (o, h, low, c, adj):
            bars.append(Bar(date=price_date, open=o, high=h, low=low, close=c, adj_close=adj))
        if c is not None and volume is not None:
            dollar_volumes.append(c * volume)

    recent = dollar_volumes[-_LIQUIDITY_WINDOW:]
    avg_dollar_volume = sum(recent) / len(recent) if recent else 0.0
    return bars, avg_dollar_volume


def _read_survivors(config: Config, as_of: date) -> tuple[list[_Survivor], dict[str, int]]:
    """Read-only phase: universe -> prior history -> freshness/liquidity filters.

    No network I/O happens here or between this and the write phase -- the
    exclusive DuckDB lock must never be held across a quote fetch (see the
    module docstring). Returns the tickers worth quoting plus a stage-counter
    dict for the ones filtered out here, merged into the write phase's
    ``RunSummary`` once ``pipeline_run`` creates one (this phase runs before
    that connection exists, so there is nowhere to bump counters onto yet).
    """
    trading = config.settings.trading
    gap_cfg = config.settings.gap_scanner
    limit = trading.atr_period * 4
    stage: dict[str, int] = {}

    def bump(key: str) -> None:
        stage[key] = stage.get(key, 0) + 1

    survivors: list[_Survivor] = []
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        universe = [
            str(row[0])
            for row in con.execute(
                "SELECT DISTINCT ticker FROM index_constituents WHERE removed_date IS NULL"
            ).fetchall()
            if row[0]
        ]

        for ticker in universe:
            bars, avg_dollar_volume = _load_prior_history(con, ticker, as_of, limit)
            if not bars:
                bump("gap_scan_no_prior_bar")
                continue
            if (as_of - bars[-1].date).days > _MAX_STALE_DAYS:
                # A lagging price sync must not grade a multi-day drift as an
                # overnight gap.
                bump("gap_scan_stale_history")
                continue
            prior_close = bars[-1].close
            if prior_close <= 0:
                bump("gap_scan_invalid_prior_close")
                continue
            if avg_dollar_volume < gap_cfg.min_avg_dollar_volume:
                bump("gap_scan_illiquid")
                continue
            survivors.append(
                _Survivor(
                    ticker=ticker,
                    bars=bars,
                    prior_close=prior_close,
                    avg_dollar_volume=avg_dollar_volume,
                )
            )

    return survivors, stage


def _fetch_quotes(
    provider: MarketDataProvider,
    survivors: list[_Survivor],
    *,
    sleep: Callable[[float], None],
    quote_pause_seconds: float,
) -> _QuoteFetchResult:
    """Fetch one live quote per survivor, paced, with no DB connection open.

    This is the network phase: it runs strictly between the read connection
    (``_read_survivors``) closing and the write connection
    (``_persist_candidates``) opening. A short pause between calls keeps a
    several-hundred-ticker survivor set from hammering the provider
    back-to-back; the caller compares ``errors`` to ``attempted`` to flag a
    mostly-throttled run rather than reporting a clean success that quietly
    skipped most of the universe.
    """
    quotes: dict[str, LatestPrice] = {}
    errors = 0
    for i, survivor in enumerate(survivors):
        if i > 0:
            sleep(quote_pause_seconds)
        try:
            quotes[survivor.ticker] = provider.get_latest_price(survivor.ticker)
        except Exception as exc:  # one dead/erroring symbol must not sink the scan
            errors += 1
            _log.warning("gap_scan_provider_error", ticker=survivor.ticker, error=str(exc))
    return _QuoteFetchResult(quotes=quotes, attempted=len(survivors), errors=errors)


def _find_catalyst(
    con: duckdb.DuckDBPyConnection, ticker: str, as_of: date, lookback_days: int
) -> dict[str, Any] | None:
    """Most recent qualifying event for ``ticker`` within the lookback window.

    Uses the same ``> as_of - lookback`` / ``<= as_of`` window as
    ``autopilot.briefing.event_alerts`` uses for the daily path, so "within N
    days before as_of" means the same thing on both trigger paths.
    """
    row = con.execute(
        """
        SELECT event_id, event_type, event_subtype, available_time
        FROM events
        WHERE ticker = ?
          AND available_time IS NOT NULL
          AND CAST(available_time AS DATE) > CAST(? AS DATE) - ?
          AND CAST(available_time AS DATE) <= CAST(? AS DATE)
        ORDER BY available_time DESC
        LIMIT 1
        """,
        [ticker, as_of, lookback_days, as_of],
    ).fetchone()
    if row is None:
        return None
    event_id, event_type, event_subtype, available_time = row
    return {
        "event_id": str(event_id),
        "event_type": event_type,
        "event_subtype": event_subtype,
        "available": available_time.isoformat(),
    }


def _build_candidate(
    con: duckdb.DuckDBPyConnection,
    survivor: _Survivor,
    quote: LatestPrice,
    config: Config,
    as_of: date,
    summary: RunSummary,
) -> TradeAlertRecord | None:
    """One survivor's gap check now that a quote exists, or ``None`` with the
    reason bumped onto ``summary``. History/liquidity/freshness were already
    judged in ``_read_survivors``; this is only the quote-dependent math."""
    trading = config.settings.trading
    gap_cfg = config.settings.gap_scanner

    gap_pct = (quote.price - survivor.prior_close) / survivor.prior_close
    if abs(gap_pct) < gap_cfg.min_gap_pct / 100.0:
        # Includes the market-closed no-op: an unchanged quote (quote ==
        # prior_close) always lands here with gap_pct == 0.
        summary.bump("gap_scan_below_threshold")
        return None

    direction = 1 if gap_pct > 0 else -1

    catalyst = _find_catalyst(con, survivor.ticker, as_of, gap_cfg.gap_catalyst_lookback_days)
    if gap_cfg.require_catalyst and catalyst is None:
        summary.bump("gap_scan_no_catalyst")
        return None

    atr = wilder_atr(survivor.bars, trading.atr_period)
    if atr is None or atr <= 0:
        summary.bump("gap_scan_thin_history")
        return None

    stop_distance = trading.atr_stop_multiple * atr
    predicted_move = gap_cfg.gap_target_r_multiple * stop_distance / quote.price

    plan = build_trade_plan(
        bars=survivor.bars,
        direction=direction,
        predicted_move=predicted_move,
        horizon_days=gap_cfg.gap_max_hold_days,
        as_of=as_of,
        account_equity=trading.account_equity,
        risk_pct_per_trade=trading.risk_pct_per_trade,
        atr_period=trading.atr_period,
        atr_stop_multiple=trading.atr_stop_multiple,
        max_position_pct=trading.max_position_pct,
        entry_override=quote.price,
    )
    if plan is None:
        summary.bump("gap_scan_no_plan")
        return None

    track_record = gap_calibration.lookup_track_record(
        con, direction=direction, gap_pct=gap_pct, catalyst_present=catalyst is not None
    )
    if track_record is None:
        confidence = score(
            ConfidenceInputs(
                hit_rate=None,
                n_clusters=0,
                times_asserted=1 if catalyst else 0,
                extraction_confidence=None,
                has_track_record=False,
            )
        )
        track_record_note = "no track record yet"
    else:
        bucket_hit_rate, bucket_n_decisive = track_record
        confidence = score(
            ConfidenceInputs(
                hit_rate=bucket_hit_rate,
                n_clusters=bucket_n_decisive,
                times_asserted=1 if catalyst else 0,
                extraction_confidence=None,
                has_track_record=True,
            )
        )
        track_record_note = (
            f"calibrated on n={bucket_n_decisive} decisive outcomes "
            f"(bucket hit rate {bucket_hit_rate:.0%})"
        )

    evidence: dict[str, Any] = {
        "gap_pct": gap_pct,
        "prior_close": survivor.prior_close,
        "quote": quote.price,
        "avg_dollar_volume": survivor.avg_dollar_volume,
        "catalyst": catalyst if catalyst is not None else "no catalyst",
        "note": track_record_note,
    }

    return TradeAlertRecord(
        alert_id=uuid4().hex,
        kind="price_gap",
        ticker=survivor.ticker,
        trigger_key=as_of.isoformat(),
        fired_at=utcnow(),
        direction=direction,
        plan=plan,
        confidence=confidence,
        evidence=evidence,
        event_id=catalyst["event_id"] if catalyst else None,
        edge_id=None,
    )


def _persist_candidates(
    config: Config,
    survivors: list[_Survivor],
    fetch: _QuoteFetchResult,
    as_of: date,
    read_stage: dict[str, int],
) -> tuple[RunSummary, list[TradeAlertRecord]]:
    """Write phase: cheap candidate math (now that quotes exist) + persist/gate.

    Runs inside one ``pipeline_run`` connection -- the second and only other
    place this module touches the database, so the lock is held only for
    this and the read phase, never for the quote fetch in between. Gating
    uses ``gap_scanner.min_confidence``, not ``trading.min_confidence``: gap
    alerts cap at 50 (see the module docstring), so gating against the daily
    path's floor would mean they never clear the bar to text.
    """
    with pipeline_run(config, "gaps.scan") as (con, summary):
        summary.stage.update(read_stage)
        if fetch.errors:
            summary.stage["gap_scan_provider_error"] = fetch.errors

        candidates: list[TradeAlertRecord] = []
        for survivor in survivors:
            quote = fetch.quotes.get(survivor.ticker)
            if quote is None:
                continue  # already counted in fetch.errors
            record = _build_candidate(con, survivor, quote, config, as_of, summary)
            if record is not None:
                candidates.append(record)

        summary.collected = len(candidates)
        new_records = persist_new(
            con, candidates, schema_version=config.settings.app.schema_version
        )
        summary.inserted = len(new_records)

        min_conf = config.settings.gap_scanner.min_confidence
        to_send = sendable(new_records, min_conf)
        mark_gated(con, gated(new_records, min_conf))

        if fetch.attempted > 0 and fetch.errors * 2 > fetch.attempted:
            # A mostly-throttled run must not report a clean, silent success.
            summary.status = "partial"
            summary.note(
                f"gap_scan_degraded: provider errors {fetch.errors}/{fetch.attempted} quotes"
            )

    return summary, to_send


def scan(
    config: Config,
    provider: MarketDataProvider | None = None,
    as_of: date | None = None,
    notify: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    quote_pause_seconds: float = _QUOTE_PAUSE_SECONDS,
) -> RunSummary:
    """Run the morning gap scan once: read, quote, plan, score, persist, alert.

    Three phases, deliberately separated so the DuckDB single-writer lock is
    never held across a network call (see the module docstring for why):
    ``_read_survivors`` (DB only) -> ``_fetch_quotes`` (network only, no
    connection open) -> ``_persist_candidates`` (DB only, inside
    ``pipeline_run`` so the run gets recorded). Each DB phase's connection
    entry is independently retried via ``_with_db_retry``; final failure on
    either phase returns a ``skipped`` summary rather than raising.

    ``provider`` defaults to a real ``YFinanceMarketDataProvider``; tests
    inject a stub implementing ``MarketDataProvider``. ``sleep`` is
    injectable and used on two paths: the DB-busy retry (30s apart) and the
    pacing pause between quote calls (``quote_pause_seconds``, default
    0.15s) -- tests never actually wait for either.
    """
    active_provider = provider or YFinanceMarketDataProvider()
    scan_date = as_of or date.today()

    try:
        survivors, read_stage = _with_db_retry(
            lambda: _read_survivors(config, scan_date), sleep=sleep
        )
    except duckdb.IOException as exc:
        _log.error(
            "gap_scan_skipped_db_busy",
            phase="read",
            attempts=_DB_BUSY_MAX_ATTEMPTS,
            error=str(exc),
        )
        return RunSummary(pipeline_name="gaps.scan", status="skipped")

    fetch = _fetch_quotes(
        active_provider, survivors, sleep=sleep, quote_pause_seconds=quote_pause_seconds
    )

    try:
        summary, to_send = _with_db_retry(
            lambda: _persist_candidates(config, survivors, fetch, scan_date, read_stage),
            sleep=sleep,
        )
    except duckdb.IOException as exc:
        _log.error(
            "gap_scan_skipped_db_busy",
            phase="write",
            attempts=_DB_BUSY_MAX_ATTEMPTS,
            error=str(exc),
        )
        return RunSummary(pipeline_name="gaps.scan", status="skipped")

    if notify and to_send:
        sent = notify_module.send_trade_alerts(config, to_send)
        try:
            with database.connection(config.paths.database_path) as con:
                mark_delivered(con, to_send, delivered=sent)
        except Exception as exc:  # a failed stamp must not affect the run's summary
            _log.error("gap_scan_mark_delivered_failed", error=str(exc))

    return summary


__all__ = ["scan"]
