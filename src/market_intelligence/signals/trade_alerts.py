"""Trade-alert assembly: an ``EventAlert`` + price history -> a ledger row.

This is the glue between the signal layer and the trade-alert ledger. It has
no opinion about *when* it runs or what happens if it fails -- the
never-raise, persist-before-send philosophy lives in the orchestrator (Task
9). This module is plain functions over a DuckDB connection:

* :func:`build_records` turns each fired :class:`EventAlert` into a
  :class:`TradeAlertRecord` -- loading recent bars, running them through
  :func:`~market_intelligence.signals.trade_plan.build_trade_plan` for the
  entry/stop/target/size, and scoring confidence. An alert whose ticker has
  no usable price history produces no record, only a note.
* :func:`persist_new` writes rows through the ledger's first-firing-wins
  insert (:func:`~market_intelligence.storage.duckdb.insert_new_trade_alerts`)
  and hands back only the records that were actually new.
* :func:`sendable` / :func:`gated` split a batch on the confidence floor;
  :func:`mark_gated` and :func:`mark_delivered` stamp the ledger after the
  fact so ``delivered`` / ``delivery_note`` always reflect what really
  happened to the message.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from market_intelligence.logging_config import get_logger
from market_intelligence.schemas.common import utcnow
from market_intelligence.signals.confidence import ConfidenceInputs, score
from market_intelligence.signals.trade_plan import Bar, TradePlan, build_trade_plan
from market_intelligence.storage import duckdb as duckdb_store

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.autopilot.types import EventAlert
    from market_intelligence.config import Config

_log = get_logger("signals.trade_alerts")


@dataclass(frozen=True)
class TradeAlertRecord:
    """One fired trade alert: identity + plan + evidence, pre-persistence.

    ``plan`` is the full :class:`~market_intelligence.signals.trade_plan.TradePlan`
    rather than its flattened fields -- ``persist_new`` flattens it at the DB
    boundary, and the notifier (Task 8) reads it directly for rendering.
    """

    alert_id: str
    kind: str  # "reaction_lag" | "price_gap"
    ticker: str
    trigger_key: str
    fired_at: datetime
    direction: int
    plan: TradePlan
    confidence: int
    evidence: dict[str, Any]  # rendered to JSON at persist time
    event_id: str | None
    edge_id: str | None
    delivered: bool = False
    delivery_note: str | None = None


def _load_bars(
    con: duckdb.DuckDBPyConnection, ticker: str, as_of: date, limit: int
) -> tuple[list[Bar], int]:
    """Last ``limit`` bars at/before ``as_of``, oldest first; skip NULL rows.

    A row with a NULL in any of the six price columns cannot become a
    ``Bar`` (every field is a required float) -- it is dropped rather than
    allowed to crash the query, and counted so the caller can note how many
    were unusable.
    """
    rows = con.execute(
        "SELECT price_date, open, high, low, close, adj_close "
        "FROM daily_prices WHERE symbol = ? AND price_date <= ? "
        "ORDER BY price_date DESC LIMIT ?",
        [ticker, as_of, limit],
    ).fetchall()
    bars: list[Bar] = []
    invalid = 0
    for price_date, o, h, low, c, adj in reversed(rows):
        if None in (o, h, low, c, adj):
            invalid += 1
            continue
        bars.append(Bar(date=price_date, open=o, high=h, low=low, close=c, adj_close=adj))
    return bars, invalid


def build_records(
    con: duckdb.DuckDBPyConnection,
    alerts: Sequence[EventAlert],
    config: Config,
    as_of: date,
) -> tuple[list[TradeAlertRecord], list[str]]:
    """Turn each fired self-edge alert into a record, or a skip note.

    ``confidence`` uses ``times_asserted=0`` and ``has_track_record=True``:
    a reaction-lag alert is graded on its own cell's holdout history, not on
    how many filings independently asserted the underlying edge.

    Each alert is isolated in its own ``try/except``: the orchestrator can
    only catch failure at the granularity of the whole call, so one alert
    whose bars are corrupt, or whose plan math blows up, must not cost the
    rest of the briefing its records. A failure here is reported the same
    way a thin-history skip is -- a note naming the ticker -- not raised.
    """
    trading = config.settings.trading
    limit = trading.atr_period * 4
    records: list[TradeAlertRecord] = []
    notes: list[str] = []

    for alert in alerts:
        try:
            if not alert.event_id:
                _log.warning("trade_alert_event_id_missing", ticker=alert.ticker)

            bars, invalid = _load_bars(con, alert.ticker, as_of, limit)
            if invalid:
                _log.debug("trade_alert_bars_invalid", ticker=alert.ticker, count=invalid)

            plan = build_trade_plan(
                bars=bars,
                direction=alert.direction,
                predicted_move=alert.predicted_car,
                horizon_days=alert.horizon_days,
                as_of=as_of,
                account_equity=trading.account_equity,
                risk_pct_per_trade=trading.risk_pct_per_trade,
                atr_period=trading.atr_period,
                atr_stop_multiple=trading.atr_stop_multiple,
                max_position_pct=trading.max_position_pct,
            )
            if plan is None:
                notes.append(f"trade-alert skipped {alert.ticker}: no viable plan")
                continue

            confidence = score(
                ConfidenceInputs(
                    hit_rate=alert.hit_rate,
                    n_clusters=alert.n_clusters,
                    times_asserted=0,
                    extraction_confidence=alert.extraction_confidence,
                    has_track_record=True,
                )
            )
            evidence = {
                "event_type": alert.event_type,
                "event_subtype": alert.event_subtype,
                "available_on": alert.available_on.isoformat(),
                "basis": alert.basis,
                "hit_rate": alert.hit_rate,
                "n_clusters": alert.n_clusters,
                "predicted_car": alert.predicted_car,
                "extraction_confidence": alert.extraction_confidence,
            }
            records.append(
                TradeAlertRecord(
                    alert_id=uuid4().hex,
                    kind="reaction_lag",
                    ticker=alert.ticker,
                    trigger_key=f"{alert.event_id or 'unknown'}:self:{alert.horizon_days}",
                    fired_at=utcnow(),
                    direction=alert.direction,
                    plan=plan,
                    confidence=confidence,
                    evidence=evidence,
                    event_id=alert.event_id or None,
                    edge_id="self",
                )
            )
        except Exception as exc:  # one bad ticker must not sink the rest
            _log.warning(
                "trade_alert_build_failed",
                ticker=alert.ticker,
                error=str(exc),
            )
            notes.append(f"trade-alert skipped {alert.ticker}: {type(exc).__name__}: {exc}")
            continue

    return records, notes


def _to_row(record: TradeAlertRecord, schema_version: str) -> dict[str, Any]:
    plan = record.plan
    return {
        "alert_id": record.alert_id,
        "kind": record.kind,
        "ticker": record.ticker,
        "trigger_key": record.trigger_key,
        "fired_at": record.fired_at,
        "direction": record.direction,
        "entry_ref": plan.entry_ref,
        "stop": plan.stop,
        "target": plan.target,
        "shares": plan.shares,
        "notional": plan.notional,
        "risk_amount": plan.risk_amount,
        "time_exit_date": plan.time_exit_date,
        "confidence": record.confidence,
        "evidence": json.dumps(record.evidence),
        "event_id": record.event_id,
        "edge_id": record.edge_id,
        "delivered": record.delivered,
        "delivery_note": record.delivery_note,
        "outcome": "open",
        "unsizeable": plan.unsizeable,
        "schema_version": schema_version,
        "collected_time": record.fired_at,
    }


def persist_new(
    con: duckdb.DuckDBPyConnection,
    records: Sequence[TradeAlertRecord],
    schema_version: str = "1.0.0",
) -> list[TradeAlertRecord]:
    """Insert unseen records into the ledger; return only the ones that landed.

    Delegates to :func:`~market_intelligence.storage.duckdb.insert_new_trade_alerts`
    for first-firing-wins semantics, then maps the inserted rows back to the
    ``TradeAlertRecord`` objects that produced them (by ``alert_id``) so the
    caller keeps working with the richer in-memory shape rather than raw rows.
    """
    if not records:
        return []
    by_id = {record.alert_id: record for record in records}
    rows = [_to_row(record, schema_version) for record in records]
    inserted = duckdb_store.insert_new_trade_alerts(con, rows)
    return [by_id[row["alert_id"]] for row in inserted]


def sendable(
    records: Sequence[TradeAlertRecord], min_confidence: int
) -> list[TradeAlertRecord]:
    """Records confident enough to notify."""
    return [record for record in records if record.confidence >= min_confidence]


def gated(
    records: Sequence[TradeAlertRecord], min_confidence: int
) -> list[TradeAlertRecord]:
    """Records held back for being below the confidence floor."""
    return [record for record in records if record.confidence < min_confidence]


def mark_gated(con: duckdb.DuckDBPyConnection, records: Sequence[TradeAlertRecord]) -> None:
    """Stamp gated rows so the ledger explains why they were never sent."""
    for record in records:
        con.execute(
            "UPDATE trade_alerts SET delivery_note = 'gated_below_min_confidence' "
            "WHERE alert_id = ?",
            [record.alert_id],
        )


def mark_delivered(
    con: duckdb.DuckDBPyConnection,
    records: Sequence[TradeAlertRecord],
    *,
    delivered: bool,
) -> None:
    """Stamp the outcome of an actual send attempt.

    ``delivered=True`` only flips the flag -- a prior ``delivery_note`` (e.g.
    from ``mark_gated``) is not expected here since gated records are never
    sent. ``delivered=False`` also records why via ``'telegram_failed'``.
    """
    if delivered:
        for record in records:
            con.execute(
                "UPDATE trade_alerts SET delivered = TRUE WHERE alert_id = ?",
                [record.alert_id],
            )
    else:
        for record in records:
            con.execute(
                "UPDATE trade_alerts SET delivered = FALSE, "
                "delivery_note = 'telegram_failed' WHERE alert_id = ?",
                [record.alert_id],
            )


__all__ = [
    "TradeAlertRecord",
    "build_records",
    "gated",
    "mark_delivered",
    "mark_gated",
    "persist_new",
    "sendable",
]
