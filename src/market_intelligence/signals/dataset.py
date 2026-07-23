"""Dataset builder: join stored events to the forward returns that grade them.

Reads ``events`` and ``daily_returns`` and writes ``event_samples`` -- one row
per (event, target company, horizon). Nothing here fetches; it is pure
computation over data already collected.

**Everything is keyed on ``available_time``, never ``event_time``.** An insider
trades on day D and the Form 4 lands up to two business days later; scoring the
trade from D would measure a move against information nobody had. The window
then starts on the first trading day *strictly after* the filing date, because
filing dates carry no time of day and an after-hours filing is not tradeable
that session. See ``analytics.eventstudy`` for the full argument.

Work is flushed per symbol so an interrupted build keeps what it finished,
matching the price collector. The alternative -- buffering several million rows
for one terminal write -- would make a long build all-or-nothing.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.analytics.eventstudy import ReturnSeries, assign_split
from market_intelligence.analytics.returns import ReturnPoint
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.schemas.common import utcnow
from market_intelligence.schemas.samples import SELF_EDGE, EventSampleRecord
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config

#: Horizons measured by default. Short enough that a days-to-weeks diffusion
#: effect is still visible, spread enough to show whether an effect decays
#: (real, slowly priced in) or reverses (noise or liquidity, not information).
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20)

#: Chronological boundary between the half that decides what is admitted and
#: the half that reports the track record. Roughly a 70/30 split of the price
#: history, late enough that discovery covers a full market cycle.
DEFAULT_SPLIT_DATE = date(2023, 1, 1)


def _load_returns(con: duckdb.DuckDBPyConnection, symbol: str) -> list[ReturnPoint]:
    """One symbol's abnormal-return series, oldest first."""
    rows = con.execute(
        """
        SELECT price_date, total_return, abnormal_return
        FROM daily_returns
        WHERE symbol = ?
        ORDER BY price_date
        """,
        [symbol],
    ).fetchall()
    return [
        ReturnPoint(
            symbol=symbol,
            price_date=row[0],
            total_return=row[1] if row[1] is not None else 0.0,
            abnormal_return=row[2],
        )
        for row in rows
    ]


def _symbols_with_events(
    con: duckdb.DuckDBPyConnection, event_types: list[str] | None, subtypes: list[str] | None
) -> list[str]:
    """Tickers that have both events and a return series, so a join is possible."""
    clauses = ["e.ticker IS NOT NULL"]
    params: list[Any] = []
    if event_types:
        clauses.append(f"e.event_type IN ({', '.join('?' * len(event_types))})")
        params.extend(event_types)
    if subtypes:
        clauses.append(f"e.event_subtype IN ({', '.join('?' * len(subtypes))})")
        params.extend(subtypes)

    rows = con.execute(
        f"""
        SELECT DISTINCT e.ticker
        FROM events e
        WHERE {" AND ".join(clauses)}
          AND EXISTS (SELECT 1 FROM daily_returns r WHERE r.symbol = e.ticker)
        ORDER BY e.ticker
        """,
        params,
    ).fetchall()
    return [str(row[0]) for row in rows]


def _events_for_symbol(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    event_types: list[str] | None,
    subtypes: list[str] | None,
) -> list[tuple]:
    clauses = ["ticker = ?", "available_time IS NOT NULL"]
    params: list[Any] = [symbol]
    if event_types:
        clauses.append(f"event_type IN ({', '.join('?' * len(event_types))})")
        params.extend(event_types)
    if subtypes:
        clauses.append(f"event_subtype IN ({', '.join('?' * len(subtypes))})")
        params.extend(subtypes)

    return con.execute(
        f"""
        SELECT event_id, event_type, event_subtype, available_time, magnitude, direction
        FROM events
        WHERE {" AND ".join(clauses)}
        """,
        params,
    ).fetchall()


def build(
    config: Config,
    *,
    event_types: list[str] | None = None,
    subtypes: list[str] | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    split_date: date = DEFAULT_SPLIT_DATE,
) -> RunSummary:
    """Build (event, target, horizon) samples for every event with a return series.

    Each event is scored against its own issuer via the ``self`` edge -- the
    positive control. Propagation edges reuse this same builder and table once
    the relationship graph exists.
    """
    with pipeline_run(config, "signals.dataset") as (con, summary):
        schema_version = config.settings.app.schema_version
        symbols = _symbols_with_events(con, event_types, subtypes)
        summary.note(f"{len(symbols)} symbols have both events and returns")

        collected = 0
        inserted = 0
        updated = 0
        unmeasurable = 0
        purged = 0

        for symbol in symbols:
            points = _load_returns(con, symbol)
            if not points:
                continue
            # Indexed once per symbol; the inner loop issues thousands of
            # lookups against it.
            returns = ReturnSeries(points)

            rows: list[dict[str, Any]] = []
            for (
                event_id,
                etype,
                subtype,
                available_time,
                magnitude,
                direction,
            ) in _events_for_symbol(con, symbol, event_types, subtypes):
                available_on = available_time.date()

                for horizon in horizons:
                    collected += 1
                    window = returns.forward(available_on, horizon)
                    if window is None:
                        # Not enough measured trading days after the event --
                        # near the end of the series, or a gap in returns.
                        unmeasurable += 1
                        continue

                    split = assign_split(window, split_date)
                    if split is None:
                        # Straddles the boundary: its label draws on prices from
                        # both halves, so it belongs to neither.
                        purged += 1
                        continue

                    record = EventSampleRecord(
                        sample_id=hashing.content_hash("sample", event_id, SELF_EDGE, horizon),
                        event_id=event_id,
                        edge_id=SELF_EDGE,
                        event_type=etype,
                        event_subtype=subtype,
                        source_ticker=symbol,
                        target_ticker=symbol,
                        horizon_days=horizon,
                        available_on=available_on,
                        t0=window.t0,
                        window_end=window.window_end,
                        forward_abnormal_return=window.cumulative_abnormal_return,
                        magnitude=magnitude,
                        direction=direction,
                        split=split.value,
                        features={"magnitude_usd": magnitude, "direction": direction},
                        schema_version=schema_version,
                        collected_time=utcnow(),
                    )
                    rows.append(record.to_row())

            if rows:
                result = duckdb_store.upsert_event_samples(con, rows)
                inserted += result.inserted
                updated += result.updated
                parquet.write_records(
                    config.paths.parquet_dir,
                    "event_samples",
                    rows,
                    ["sample_id"],
                    partition_col="event_type",
                )

        summary.collected = collected
        summary.inserted = inserted
        summary.updated = updated
        summary.bump("unmeasurable", unmeasurable)
        summary.bump("purged_at_split", purged)
        summary.note(
            f"{unmeasurable} unmeasurable (no complete forward window), "
            f"{purged} purged at the {split_date} split boundary"
        )

    return summary


def _validatable_edges(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    """Resolved edges whose both ends have a return series, grouped-ready.

    Returns ``(source_ticker, target_ticker, edge_type)``. Only resolved edges
    with a real target and a source distinct from the target can be measured --
    the same is-validatable test the collector records, re-checked here against
    the return series so a target with no prices is never queued.
    """
    rows = con.execute(
        """
        SELECT DISTINCT e.source_ticker, e.target_ticker, e.edge_type
        FROM company_edges e
        WHERE e.resolution_status = 'resolved'
          AND e.target_ticker IS NOT NULL AND e.target_ticker <> ''
          AND e.source_ticker IS NOT NULL AND e.source_ticker <> ''
          AND e.source_ticker <> e.target_ticker
          AND EXISTS (SELECT 1 FROM daily_returns r WHERE r.symbol = e.source_ticker)
          AND EXISTS (SELECT 1 FROM daily_returns r WHERE r.symbol = e.target_ticker)
        ORDER BY e.target_ticker, e.source_ticker, e.edge_type
        """
    ).fetchall()
    return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]


def build_propagation(
    config: Config,
    *,
    event_types: list[str] | None = None,
    subtypes: list[str] | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    split_date: date = DEFAULT_SPLIT_DATE,
) -> RunSummary:
    """Build propagation samples: an event at the source, the target's return.

    For every validatable edge, each of the *source's* events is scored against
    the *target's* forward return, with ``edge_id`` set to the relationship type.
    Those samples land in the same ``event_samples`` table the self-control uses,
    so the gate and the promotion ladder judge propagation cells (e.g.
    ``insider_transaction/P/customer/20d``) with no additional machinery.

    Indexed by target: each target's return series is built once and reused for
    every edge pointing into it, and rows flush per target so an interrupted
    build keeps what it finished.
    """
    with pipeline_run(config, "signals.dataset.propagation") as (con, summary):
        schema_version = config.settings.app.schema_version
        edges = _validatable_edges(con)
        by_target: dict[str, list[tuple[str, str]]] = {}
        for source, target, edge_type in edges:
            by_target.setdefault(target, []).append((source, edge_type))
        summary.note(f"{len(edges)} validatable edges into {len(by_target)} targets")

        collected = inserted = updated = unmeasurable = purged = 0

        for target, incoming in by_target.items():
            points = _load_returns(con, target)
            if not points:
                continue
            returns = ReturnSeries(points)

            rows: list[dict[str, Any]] = []
            for source, edge_type in incoming:
                for (
                    event_id,
                    etype,
                    subtype,
                    available_time,
                    magnitude,
                    direction,
                ) in _events_for_symbol(con, source, event_types, subtypes):
                    available_on = available_time.date()
                    for horizon in horizons:
                        collected += 1
                        window = returns.forward(available_on, horizon)
                        if window is None:
                            unmeasurable += 1
                            continue
                        split = assign_split(window, split_date)
                        if split is None:
                            purged += 1
                            continue

                        record = EventSampleRecord(
                            # Target is in the id: one source event points at
                            # several targets, each a distinct sample.
                            sample_id=hashing.content_hash(
                                "prop", event_id, edge_type, target, horizon
                            ),
                            event_id=event_id,
                            edge_id=edge_type,
                            event_type=etype,
                            event_subtype=subtype,
                            source_ticker=source,
                            target_ticker=target,
                            horizon_days=horizon,
                            available_on=available_on,
                            t0=window.t0,
                            window_end=window.window_end,
                            forward_abnormal_return=window.cumulative_abnormal_return,
                            magnitude=magnitude,
                            direction=direction,
                            split=split.value,
                            features={"edge_type": edge_type, "source": source},
                            schema_version=schema_version,
                            collected_time=utcnow(),
                        )
                        rows.append(record.to_row())

            if rows:
                result = duckdb_store.upsert_event_samples(con, rows)
                inserted += result.inserted
                updated += result.updated
                parquet.write_records(
                    config.paths.parquet_dir,
                    "event_samples",
                    rows,
                    ["sample_id"],
                    partition_col="event_type",
                )

        summary.collected = collected
        summary.inserted = inserted
        summary.updated = updated
        summary.bump("unmeasurable", unmeasurable)
        summary.bump("purged_at_split", purged)
        summary.bump("edges", len(edges))
    return summary


__all__ = ["DEFAULT_HORIZONS", "DEFAULT_SPLIT_DATE", "build", "build_propagation"]
