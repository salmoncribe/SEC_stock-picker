"""A lock-friendly, human-readable projection of the live research ledger.

DuckDB permits one process to own a database file at a time.  The daily
autopilot legitimately holds that lock while it computes, so a browser process
must not try to compete with it merely to render a screen.  This module runs
inside the already-authorized autopilot connection and writes a compact JSON
projection for :mod:`market_intelligence.dashboard` to serve.

The projection is intentionally a *display cache*, never an input to the
model.  It makes the dashboard continuously available without giving it a
second, conflicting database connection.
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from market_intelligence import database
from market_intelligence.autopilot.types import Briefing

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.autopilot.types import Briefing
    from market_intelligence.config import Config


SNAPSHOT_FILENAME = "dashboard_snapshot.json"


def snapshot_path(config: Config) -> Path:
    """Return the dashboard's read-only projection path."""
    return config.paths.logs_dir / SNAPSHOT_FILENAME


def build_snapshot(
    con: duckdb.DuckDBPyConnection,
    config: Config,
    briefing: Briefing,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the complete dashboard payload from the current ledger state."""
    captured_at = now or datetime.now(tz=UTC)
    open_alerts = _open_alerts(con, config)
    resolved_alerts = _resolved_alerts(con, config)
    active_signals = _active_signals(con)
    recent_events = _recent_events(con, config)
    health = _health(con)

    outcomes = con.execute(
        """
        SELECT
            count(*) FILTER (WHERE outcome = 'open'),
            count(*) FILTER (WHERE outcome = 'hit_target'),
            count(*) FILTER (WHERE outcome = 'hit_stop'),
            count(*) FILTER (WHERE outcome = 'expired')
        FROM trade_alerts
        """
    ).fetchone()
    price_row = con.execute("SELECT max(price_date) FROM daily_prices").fetchone()

    return {
        "version": 1,
        "captured_at": captured_at.isoformat(),
        "briefing": {
            "as_of": briefing.as_of.isoformat(),
            "status": briefing.run_status,
            "notes": briefing.notes[-6:],
            "ingest": briefing.ingest,
        },
        "vault": _vault_links(config, briefing.as_of),
        "metrics": {
            "open_alerts": int(outcomes[0] or 0),
            "active_signals": len(active_signals),
            "targets_hit": int(outcomes[1] or 0),
            "stops_hit": int(outcomes[2] or 0),
            "expired": int(outcomes[3] or 0),
            "latest_price_date": _iso(price_row[0] if price_row else None),
        },
        "open_alerts": open_alerts,
        "resolved_alerts": resolved_alerts,
        "active_signals": active_signals,
        "recent_events": recent_events,
        "system_health": health,
    }


def write_snapshot(
    config: Config,
    payload: dict[str, Any],
) -> Path:
    """Atomically replace the display cache so browsers never read half JSON."""
    path = snapshot_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".dashboard-", delete=False
    ) as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)
    return path


def refresh_snapshot(config: Config, *, now: datetime | None = None) -> Path:
    """Refresh the display cache without running or changing the model.

    This is used when ``PAGE`` is opened.  It only reads DuckDB and writes the
    separate JSON cache; when an autopilot pass owns the lock, callers simply
    retain the last good cache rather than interrupting the pass.
    """
    captured_at = now or datetime.now(tz=UTC)
    with database.connection(config.paths.database_path) as con:
        latest_run = con.execute(
            """
            SELECT started_time, completed_time, status, error_message
            FROM pipeline_runs
            WHERE pipeline_name = 'autopilot'
            ORDER BY started_time DESC NULLS LAST
            LIMIT 1
            """
        ).fetchone()
        if latest_run is None:
            briefing = Briefing(
                as_of=captured_at.date(),
                run_status="unknown",
                notes=["No completed autopilot run has been recorded yet."],
            )
        else:
            started_at, completed_at, status, error_message = latest_run
            as_of = (completed_at or started_at or captured_at).date()
            notes = [str(error_message)] if error_message else []
            briefing = Briefing(as_of=as_of, run_status=str(status or "unknown"), notes=notes)
        return write_snapshot(config, build_snapshot(con, config, briefing, now=captured_at))


def _open_alerts(con: duckdb.DuckDBPyConnection, config: Config) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        WITH latest AS (
            SELECT symbol, price_date, close, adj_close,
                   row_number() OVER (PARTITION BY symbol ORDER BY price_date DESC) AS rn
            FROM daily_prices
        ), fired_factors AS (
            SELECT a.alert_id,
                   p.close AS fired_close,
                   p.adj_close AS fired_adj_close
            FROM trade_alerts a
            LEFT JOIN LATERAL (
                SELECT close, adj_close
                FROM daily_prices
                WHERE symbol = a.ticker AND price_date <= CAST(a.fired_at AS DATE)
                ORDER BY price_date DESC
                LIMIT 1
            ) p ON true
        )
        SELECT a.alert_id, a.ticker, a.kind, a.fired_at, a.direction, a.entry_ref,
               a.stop, a.target, a.time_exit_date, a.confidence, a.delivery_note,
               l.price_date, l.close, l.adj_close, f.fired_close, f.fired_adj_close
        FROM trade_alerts a
        LEFT JOIN latest l ON l.symbol = a.ticker AND l.rn = 1
        LEFT JOIN fired_factors f ON f.alert_id = a.alert_id
        WHERE a.outcome = 'open'
        ORDER BY a.fired_at DESC NULLS LAST
        LIMIT 100
        """
    ).fetchall()
    alerts: list[dict[str, Any]] = []
    for row in rows:
        (
            alert_id, ticker, kind, fired_at, direction, entry, stop, target, exit_date, confidence,
            delivery_note, price_date, close, adj_close, fired_close, fired_adj_close,
        ) = row
        actual_return = _actual_return(
            direction, entry, close, adj_close, fired_close, fired_adj_close
        )
        expected_return = _expected_return(direction, entry, target)
        progress = (
            actual_return / expected_return
            if actual_return is not None and expected_return not in (None, 0.0)
            else None
        )
        alerts.append(
            {
                "id": str(alert_id),
                "ticker": str(ticker),
                "kind": str(kind),
                "fired_at": _iso(fired_at),
                "direction": "long" if int(direction or 0) > 0 else "short",
                "entry": _number(entry),
                "stop": _number(stop),
                "target": _number(target),
                "exit_date": _iso(exit_date),
                "confidence": int(confidence) if confidence is not None else None,
                "delivery_note": str(delivery_note) if delivery_note else None,
                "price_date": _iso(price_date),
                "last_price": _number(close),
                "actual_return": actual_return,
                "expected_return": expected_return,
                "progress_to_target": progress,
                "vault_url": _company_vault_url(config, str(ticker)),
                "quote_url": _quote_url(str(ticker)),
            }
        )
    return alerts


def _resolved_alerts(con: duckdb.DuckDBPyConnection, config: Config) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT ticker, kind, fired_at, direction, entry_ref, target, confidence,
               outcome, outcome_return, graded_at
        FROM trade_alerts
        WHERE outcome IS NOT NULL AND outcome != 'open'
        ORDER BY graded_at DESC NULLS LAST, fired_at DESC NULLS LAST
        LIMIT 20
        """
    ).fetchall()
    return [
        {
            "ticker": str(ticker),
            "kind": str(kind),
            "fired_at": _iso(fired_at),
            "direction": "long" if int(direction or 0) > 0 else "short",
            "entry": _number(entry),
            "target": _number(target),
            "confidence": int(confidence) if confidence is not None else None,
            "outcome": str(outcome),
            "actual_return": _number(outcome_return),
            "graded_at": _iso(graded_at),
            "vault_url": _company_vault_url(config, str(ticker)),
            "quote_url": _quote_url(str(ticker)),
        }
        for (
            ticker,
            kind,
            fired_at,
            direction,
            entry,
            target,
            confidence,
            outcome,
            outcome_return,
            graded_at,
        ) in rows
    ]


def _active_signals(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT event_type, event_subtype, edge_type, horizon_days, mean_car,
               hit_rate, n_clusters, confirm_streak, last_evaluated_time, last_reason
        FROM signal_status
        WHERE status = 'active'
        ORDER BY abs(mean_car) DESC NULLS LAST, hit_rate DESC NULLS LAST
        """
    ).fetchall()
    return [
        {
            "label": f"{event_type}/{subtype or '-'}/{edge_type}/{horizon}d",
            "direction": "long" if float(mean_car or 0) >= 0 else "short",
            "mean_car": _number(mean_car),
            "hit_rate": _number(hit_rate),
            "clusters": int(clusters) if clusters is not None else 0,
            "confirm_streak": int(streak) if streak is not None else 0,
            "evaluated_at": _iso(evaluated_at),
            "reason": str(reason) if reason else None,
        }
        for (
            event_type,
            subtype,
            edge_type,
            horizon,
            mean_car,
            hit_rate,
            clusters,
            streak,
            evaluated_at,
            reason,
        ) in rows
    ]


def _recent_events(con: duckdb.DuckDBPyConnection, config: Config) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT ticker, event_type, event_subtype, available_time, direction, magnitude
        FROM events
        WHERE ticker IS NOT NULL AND available_time IS NOT NULL
        ORDER BY available_time DESC
        LIMIT 12
        """
    ).fetchall()
    return [
        {
            "ticker": str(ticker),
            "event_type": str(event_type),
            "event_subtype": str(subtype) if subtype else None,
            "available_at": _iso(available_at),
            "direction": int(direction) if direction is not None else None,
            "magnitude": _number(magnitude),
            "vault_url": _company_vault_url(config, str(ticker)),
            "quote_url": _quote_url(str(ticker)),
        }
        for ticker, event_type, subtype, available_at, direction, magnitude in rows
    ]


def _health(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT component, status, observed_at, details
        FROM (
            SELECT component, status, observed_at, details,
                   row_number() OVER (PARTITION BY component ORDER BY observed_at DESC) AS rn
            FROM system_health
        )
        WHERE rn = 1
        ORDER BY component
        LIMIT 20
        """
    ).fetchall()
    return [
        {
            "component": str(component),
            "status": str(status),
            "observed_at": _iso(observed_at),
            "details": str(details) if details else None,
        }
        for component, status, observed_at, details in rows
    ]


def _vault_links(config: Config, briefing_date: date) -> dict[str, str]:
    vault_name = config.paths.obsidian_vault_dir.name
    return {
        "vault_url": f"obsidian://open?vault={quote(vault_name)}",
        "briefing_url": _vault_file_url(config, f"briefings/{briefing_date.isoformat()}.md"),
        "vault_path": str(config.paths.obsidian_vault_dir),
    }


def _company_vault_url(config: Config, ticker: str) -> str:
    return _vault_file_url(config, f"companies/{ticker}.md")


def _vault_file_url(config: Config, file_path: str) -> str:
    return (
        f"obsidian://open?vault={quote(config.paths.obsidian_vault_dir.name)}"
        f"&file={quote(file_path)}"
    )


def _quote_url(ticker: str) -> str:
    return f"https://finance.yahoo.com/quote/{quote(ticker, safe='')}/"


def _actual_return(
    direction: Any,
    entry: Any,
    close: Any,
    adj_close: Any,
    fired_close: Any,
    fired_adj_close: Any,
) -> float | None:
    """Return direction-adjusted mark-to-market in split-adjusted space."""
    values = (direction, entry, close, adj_close, fired_close, fired_adj_close)
    if any(value is None for value in values):
        return None
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric) or fired_close <= 0 or entry <= 0:
        return None
    entry_adjusted = float(entry) * float(fired_adj_close) / float(fired_close)
    if entry_adjusted <= 0:
        return None
    return _number(float(direction) * (float(adj_close) - entry_adjusted) / entry_adjusted)


def _expected_return(direction: Any, entry: Any, target: Any) -> float | None:
    if any(value is None for value in (direction, entry, target)):
        return None
    if float(entry) <= 0:
        return None
    return _number(float(direction) * (float(target) - float(entry)) / float(entry))


def _number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return round(number, 8) if math.isfinite(number) else None


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


__all__ = [
    "SNAPSHOT_FILENAME",
    "build_snapshot",
    "refresh_snapshot",
    "snapshot_path",
    "write_snapshot",
]
