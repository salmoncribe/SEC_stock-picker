"""Turn a run's before/after state into the day-over-day briefing.

The briefing reports *changes*, not the standing model: what newly earned the
right to fire, what broke, and which of today's events an already-trusted cell
predicts. A dump of every cell every morning would bury the one line that
matters; the diff is the product.

Change detection compares a snapshot of ``signal_status`` taken before the gate
ran against the table after it ran. The orchestrator holds both, so "what
changed" is a set difference on the ladder status of each cell -- no history
table required, and no dependence on wall-clock beyond the run itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from market_intelligence.autopilot.types import (
    Briefing,
    ChangeKind,
    EventAlert,
    SignalChange,
)
from market_intelligence.signals.promotion import LadderStatus

if TYPE_CHECKING:
    from datetime import date

    import duckdb

#: The self edge -- an event scored against its own issuer. Alerts in phase 1
#: fire only through self-edge cells; propagation edges join here once they exist.
SELF_EDGE = "self"

#: Cell identity used to line up before/after snapshots.
CellKey = tuple[str, "str | None", str, int]


def snapshot_statuses(con: duckdb.DuckDBPyConnection) -> dict[CellKey, str]:
    """Current ladder status of every tracked cell, keyed by the cell.

    Taken before the gate runs and again (implicitly, via the table) after, so a
    transition is just a change in this mapping.
    """
    rows = con.execute(
        """
        SELECT event_type, event_subtype, edge_type, horizon_days, status
        FROM signal_status
        """
    ).fetchall()
    return {(str(r[0]), r[1], str(r[2]), int(r[3])): str(r[4]) for r in rows}


def _classify(before: str | None, after: str) -> str | None:
    """Name the transition, or ``None`` when nothing worth reporting changed.

    Only the four transitions a reader should act on are surfaced. A cell that
    stays put, or churns between two non-firing states, produces no line.
    """
    if before == after:
        return None
    active = LadderStatus.ACTIVE.value
    retired = LadderStatus.RETIRED.value
    dormant = LadderStatus.DORMANT.value

    if after == active and before != active:
        return ChangeKind.ACTIVATED
    if before == active and after == dormant:
        return ChangeKind.DEMOTED
    if after == retired and before in (active, dormant):
        return ChangeKind.RETIRED
    if before is None:
        return ChangeKind.NEW_CANDIDATE
    return None


def _change_from_row(kind: str, row: dict[str, Any]) -> SignalChange:
    return SignalChange(
        event_type=str(row["event_type"]),
        event_subtype=row["event_subtype"],
        edge_type=str(row["edge_type"]),
        horizon_days=int(row["horizon_days"]),
        kind=kind,
        mean_car=float(row["mean_car"]) if row["mean_car"] is not None else 0.0,
        hit_rate=float(row["hit_rate"]) if row["hit_rate"] is not None else 0.0,
        n_clusters=int(row["n_clusters"]) if row["n_clusters"] is not None else 0,
        reason=str(row["last_reason"] or ""),
    )


def _status_rows(con: duckdb.DuckDBPyConnection) -> dict[CellKey, dict[str, Any]]:
    columns = (
        "event_type", "event_subtype", "edge_type", "horizon_days",
        "status", "mean_car", "hit_rate", "n_clusters", "direction", "last_reason",
    )  # fmt: skip
    rows = con.execute(f"SELECT {', '.join(columns)} FROM signal_status").fetchall()
    out: dict[CellKey, dict[str, Any]] = {}
    for r in rows:
        record = dict(zip(columns, r, strict=True))
        key = (str(r[0]), r[1], str(r[2]), int(r[3]))
        out[key] = record
    return out


def detect_changes(
    con: duckdb.DuckDBPyConnection,
    before: dict[CellKey, str],
) -> list[SignalChange]:
    """Diff the ladder against the pre-run snapshot into reportable changes."""
    after = _status_rows(con)
    changes: list[SignalChange] = []
    for key, row in after.items():
        kind = _classify(before.get(key), str(row["status"]))
        if kind is not None:
            changes.append(_change_from_row(kind, row))
    return changes


def active_roster(con: duckdb.DuckDBPyConnection) -> list[SignalChange]:
    """Every currently-active cell, as context for the standing model.

    Reported with the ``ACTIVATED`` kind so a reader sees the trusted set, not
    only this morning's deltas.
    """
    rows = _status_rows(con)
    return [
        _change_from_row(ChangeKind.ACTIVATED, row)
        for row in rows.values()
        if row["status"] == LadderStatus.ACTIVE.value
    ]


def event_alerts(
    con: duckdb.DuckDBPyConnection,
    *,
    as_of: date,
    lookback_days: int,
) -> list[EventAlert]:
    """Predictions fired by recent events through currently-active cells.

    An event that became public within the lookback window, at a ticker, whose
    (type, subtype) matches an active self-edge cell, yields one alert per
    matching horizon. The prediction is that cell's measured mean move and sign;
    the basis quotes its out-of-sample track record.

    The lookback covers more than one day on purpose: a daily loop that misses a
    weekend or a run must not silently drop the events in the gap.
    """
    rows = con.execute(
        """
        SELECT e.ticker, e.event_type, e.event_subtype, e.available_time,
               s.horizon_days, s.direction, s.mean_car, s.hit_rate
        FROM events e
        JOIN signal_status s
          ON s.event_type = e.event_type
         AND s.event_subtype IS NOT DISTINCT FROM e.event_subtype
         AND s.edge_type = ?
         AND s.status = ?
        WHERE e.ticker IS NOT NULL
          AND e.available_time IS NOT NULL
          AND CAST(e.available_time AS DATE) > CAST(? AS DATE) - ?
          AND CAST(e.available_time AS DATE) <= CAST(? AS DATE)
        ORDER BY e.available_time DESC, e.ticker
        """,
        [SELF_EDGE, LadderStatus.ACTIVE.value, as_of, lookback_days, as_of],
    ).fetchall()

    alerts: list[EventAlert] = []
    for ticker, etype, subtype, available_time, horizon, direction, mean_car, hit_rate in rows:
        hit_pct = f"{float(hit_rate):.1%}" if hit_rate is not None else "n/a"
        alerts.append(
            EventAlert(
                ticker=str(ticker),
                event_type=str(etype),
                event_subtype=subtype,
                available_on=available_time.date(),
                horizon_days=int(horizon),
                direction=int(direction) if direction is not None else 0,
                predicted_car=float(mean_car) if mean_car is not None else 0.0,
                basis=f"active cell, holdout hit {hit_pct}",
            )
        )
    return alerts


def build(
    con: duckdb.DuckDBPyConnection,
    *,
    as_of: date,
    run_status: str,
    ingest: dict[str, int],
    before: dict[CellKey, str],
    notes: list[str],
    alert_lookback_days: int = 4,
) -> Briefing:
    """Assemble the full briefing from the post-run database state."""
    return Briefing(
        as_of=as_of,
        run_status=run_status,
        ingest=ingest,
        changes=detect_changes(con, before),
        alerts=event_alerts(con, as_of=as_of, lookback_days=alert_lookback_days),
        active_signals=active_roster(con),
        notes=notes,
    )


__all__ = [
    "SELF_EDGE",
    "active_roster",
    "build",
    "detect_changes",
    "event_alerts",
    "snapshot_statuses",
]
