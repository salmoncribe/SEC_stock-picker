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
out of sync with the first.

**Confidence is deliberately capped.** ``ConfidenceInputs.has_track_record``
is always ``False`` here, which caps the blended score at 50 regardless of
gap size or liquidity -- this scanner has no graded outcomes yet to earn a
higher number from. Spec Sec.5 sketches gap-specific substitute inputs (gap
size, liquidity) for the score; those are deliberately *not* implemented as
separate blend terms. Until :mod:`market_intelligence.signals.grading`
accumulates real hit/miss history for ``kind="price_gap"`` rows, the only
input this path adds beyond the no-track-record cap is
``times_asserted=1 if catalyst else 0`` -- a same-week filing/event on the
ticker nudges the score up a little as corroboration, exactly the same
mechanism the daily path uses for independently-asserted edges. The message
still carries "no track record yet" in evidence so a human reading it never
mistakes 50 for a real number.

**Runs as a short-lived sequenced process, not a daemon.** Like every other
collector, ``scan`` opens one DuckDB connection via ``pipeline_run``, does its
work, and exits -- DuckDB is single-writer, and a long-running gap-scan
process would be exactly the kind of standing lock-holder the rest of the
platform is built to avoid. Because a collector (or the relationship
extractor) may already hold that lock when the morning LaunchAgent fires,
entering ``pipeline_run`` is wrapped in a bounded retry (3 attempts, 30s
apart by default) that catches ``duckdb.IOException`` specifically. A missed
scan is a missed message, not a crash: final failure logs
``gap_scan_skipped_db_busy`` and returns a ``skipped`` ``RunSummary`` rather
than raising.

**Quotes are fetched one symbol at a time.** ``YFinanceMarketDataProvider.
get_latest_price`` (see ``clients/market_yfinance.py``) makes one HTTP fetch
per symbol; across a ~500-ticker universe that is ~500 sequential calls. That
is acceptable for v1 -- the scan runs once, well before the trading day gets
interesting -- and is isolated behind the ``MarketDataProvider`` ABC, so a
batched quote endpoint can replace it later without this module changing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
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

# Bounded retry around entering pipeline_run when another process holds the
# DuckDB single-writer lock -- see the module docstring.
_DB_BUSY_MAX_ATTEMPTS = 3
_DB_BUSY_RETRY_SECONDS = 30.0

# "20-day average dollar volume" per the design spec / gap_scanner config.
_LIQUIDITY_WINDOW = 20


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
    no-op path: a fresh listing, a stale/removed symbol, or (the one case
    the spec calls out explicitly) a weekend/holiday scan where "yesterday's"
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
    ticker: str,
    quote: LatestPrice,
    config: Config,
    as_of: date,
    summary: RunSummary,
) -> TradeAlertRecord | None:
    """One ticker's gap check, or ``None`` with the reason bumped onto ``summary``."""
    trading = config.settings.trading
    gap_cfg = config.settings.gap_scanner
    limit = trading.atr_period * 4

    bars, avg_dollar_volume = _load_prior_history(con, ticker, as_of, limit)
    if not bars:
        summary.bump("gap_scan_no_prior_bar")
        return None

    prior_close = bars[-1].close
    if prior_close <= 0:
        summary.bump("gap_scan_invalid_prior_close")
        return None

    if avg_dollar_volume < gap_cfg.min_avg_dollar_volume:
        summary.bump("gap_scan_illiquid")
        return None

    gap_pct = (quote.price - prior_close) / prior_close
    if abs(gap_pct) < gap_cfg.min_gap_pct / 100.0:
        # Includes the market-closed no-op: an unchanged quote (quote ==
        # prior_close) always lands here with gap_pct == 0.
        summary.bump("gap_scan_below_threshold")
        return None

    direction = 1 if gap_pct > 0 else -1

    catalyst = _find_catalyst(con, ticker, as_of, gap_cfg.gap_catalyst_lookback_days)
    if gap_cfg.require_catalyst and catalyst is None:
        summary.bump("gap_scan_no_catalyst")
        return None

    atr = wilder_atr(bars, trading.atr_period)
    if atr is None or atr <= 0:
        summary.bump("gap_scan_thin_history")
        return None

    stop_distance = trading.atr_stop_multiple * atr
    predicted_move = gap_cfg.gap_target_r_multiple * stop_distance / quote.price

    plan = build_trade_plan(
        bars=bars,
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

    confidence = score(
        ConfidenceInputs(
            hit_rate=None,
            n_clusters=0,
            times_asserted=1 if catalyst else 0,
            extraction_confidence=None,
            has_track_record=False,
        )
    )

    evidence: dict[str, Any] = {
        "gap_pct": gap_pct,
        "prior_close": prior_close,
        "quote": quote.price,
        "avg_dollar_volume": avg_dollar_volume,
        "catalyst": catalyst if catalyst is not None else "no catalyst",
        "note": "no track record yet",
    }

    return TradeAlertRecord(
        alert_id=uuid4().hex,
        kind="price_gap",
        ticker=ticker,
        trigger_key=as_of.isoformat(),
        fired_at=utcnow(),
        direction=direction,
        plan=plan,
        confidence=confidence,
        evidence=evidence,
        event_id=catalyst["event_id"] if catalyst else None,
        edge_id=None,
    )


def _run_once(
    con: duckdb.DuckDBPyConnection,
    summary: RunSummary,
    config: Config,
    provider: MarketDataProvider,
    as_of: date,
) -> list[TradeAlertRecord]:
    """Scan the universe, persist+gate new candidates, return the sendable ones.

    Everything here runs inside one ``pipeline_run`` connection, so the
    ``pipeline_runs`` row reflects real counts. The network send happens
    after that connection closes (see ``scan``), matching the orchestrator's
    persist-before-send discipline: a Telegram outage can only under-report
    delivery, never lose the ledger row.
    """
    universe = [
        str(row[0])
        for row in con.execute(
            "SELECT DISTINCT ticker FROM index_constituents WHERE removed_date IS NULL"
        ).fetchall()
        if row[0]
    ]

    candidates: list[TradeAlertRecord] = []
    for ticker in universe:
        try:
            quote = provider.get_latest_price(ticker)
        except Exception as exc:  # one dead/erroring symbol must not sink the scan
            summary.bump("gap_scan_provider_error")
            _log.warning("gap_scan_provider_error", ticker=ticker, error=str(exc))
            continue

        record = _build_candidate(con, ticker, quote, config, as_of, summary)
        if record is not None:
            candidates.append(record)

    summary.collected = len(candidates)
    new_records = persist_new(con, candidates, schema_version=config.settings.app.schema_version)
    summary.inserted = len(new_records)

    min_conf = config.settings.trading.min_confidence
    to_send = sendable(new_records, min_conf)
    mark_gated(con, gated(new_records, min_conf))
    return to_send


def scan(
    config: Config,
    provider: MarketDataProvider | None = None,
    as_of: date | None = None,
    notify: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> RunSummary:
    """Run the morning gap scan once: detect, plan, score, persist, alert.

    ``provider`` defaults to a real ``YFinanceMarketDataProvider``; tests
    inject a stub implementing ``MarketDataProvider`` so nothing here ever
    touches the network. ``sleep`` is injectable for the same reason on the
    DB-busy retry path -- tests never actually wait 30 seconds.

    See the module docstring for why entry anchors at the live quote, why
    confidence is capped, and why the DB-busy retry exists.
    """
    active_provider = provider or YFinanceMarketDataProvider()
    scan_date = as_of or date.today()

    summary: RunSummary | None = None
    to_send: list[TradeAlertRecord] = []
    last_error: duckdb.IOException | None = None

    for attempt in range(1, _DB_BUSY_MAX_ATTEMPTS + 1):
        try:
            with pipeline_run(config, "gaps.scan") as (con, run_summary):
                to_send = _run_once(con, run_summary, config, active_provider, scan_date)
            summary = run_summary
            break
        except duckdb.IOException as exc:
            last_error = exc
            if attempt < _DB_BUSY_MAX_ATTEMPTS:
                sleep(_DB_BUSY_RETRY_SECONDS)

    if summary is None:
        _log.error(
            "gap_scan_skipped_db_busy",
            attempts=_DB_BUSY_MAX_ATTEMPTS,
            error=str(last_error),
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
