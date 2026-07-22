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


__all__ = ["DEFAULT_HORIZONS", "DEFAULT_SPLIT_DATE", "build"]
