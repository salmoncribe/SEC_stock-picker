"""The validation gate: measure every cell, and decide which may fire alerts.

Reads ``event_samples``, groups them into (event type, subtype, edge type,
horizon) cells, computes clustered statistics per split, and writes a verdict
to ``impact_stats``. Pure computation over stored data; nothing fetches.

Admission is decided on the **discovery** split and the track record reported
from **holdout**. Those must stay disjoint: choosing the winners and grading
them on the same sample turns a hit rate into a description of the data it was
chosen on, which is the single easiest way to produce a number that looks like
an edge and is not one.

The gate also runs a **placebo control** on request. It draws fake events at
random company-days from the real return series and measures them exactly like
real ones. Those cells must not be admitted. This is the check that the gate
can still say no; the positive control -- insider purchases, which are known to
predict -- is the check that it can still say yes. A gate with only the first
passes by rejecting everything, which is indistinguishable from being broken.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.analytics.eventstudy import ReturnSeries
from market_intelligence.analytics.impact import (
    AdmissionThresholds,
    ImpactCell,
    Verdict,
    judge,
    summarize,
)
from market_intelligence.analytics.returns import ReturnPoint
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.schemas.common import Source, utcnow
from market_intelligence.storage import duckdb as duckdb_store

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config

#: Edge type used for placebo cells, so they can never collide with a real one.
PLACEBO_EDGE = "placebo"

DISCOVERY = "discovery"
HOLDOUT = "holdout"


def _load_cell_keys(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str | None, str, int]]:
    rows = con.execute(
        """
        SELECT DISTINCT event_type, event_subtype, edge_id, horizon_days
        FROM event_samples
        ORDER BY event_type, event_subtype, edge_id, horizon_days
        """
    ).fetchall()
    return [(str(r[0]), r[1], str(r[2]), int(r[3])) for r in rows]


def _load_samples(
    con: duckdb.DuckDBPyConnection,
    event_type: str,
    subtype: str | None,
    edge_type: str,
    horizon: int,
    split: str,
) -> list[tuple[str, Any, float]]:
    rows = con.execute(
        """
        SELECT target_ticker, t0, forward_abnormal_return
        FROM event_samples
        WHERE event_type = ?
          AND event_subtype IS NOT DISTINCT FROM ?
          AND edge_id = ?
          AND horizon_days = ?
          AND split = ?
          AND forward_abnormal_return IS NOT NULL
        """,
        [event_type, subtype, edge_type, horizon, split],
    ).fetchall()
    return [(str(r[0]), r[1], float(r[2])) for r in rows]


def _placebo_samples(
    con: duckdb.DuckDBPyConnection,
    horizon: int,
    split: str,
    *,
    n: int,
    seed: int,
) -> list[tuple[str, Any, float]]:
    """Fake events at random company-days drawn from the *unconditional* returns.

    Sampled from ``daily_returns`` -- every company-day in the universe --
    rather than from ``event_samples``.

    That distinction is the whole control, and getting it wrong silently breaks
    it. ``event_samples`` is 92% insider *sales*, which genuinely average about
    -1% over 20 days. Drawing random rows from it reproduces that mean exactly,
    because resampling any population preserves its population mean: the
    "random" control inherits the very effect it is supposed to be blind to, and
    is duly admitted. A placebo built that way certifies the gate as working
    while proving nothing.

    What a null requires is breaking the *pairing* between event and outcome,
    not merely choosing rows at random. Company-days here are picked without
    regard to whether anything happened on them, so the expected effect is the
    unconditional mean abnormal return -- about -0.003% per day, i.e. zero by
    construction, since abnormal returns are already demeaned against the market
    model.

    The forward window is summed the same way the real builder sums it, so the
    placebo inherits the real return distribution, fat tails and all. A placebo
    drawn from a normal distribution would be easier to reject than reality and
    would prove less.
    """
    boundary = _split_boundary(con)
    if boundary is None:
        return []

    # Confine drawn days to the same era the real cell covers, so the control
    # is measured over the same market regime rather than a different one.
    window_sql = "AND price_date < ?" if split == DISCOVERY else "AND price_date >= ?"

    candidates = con.execute(
        f"""
        SELECT symbol, price_date
        FROM daily_returns
        WHERE abnormal_return IS NOT NULL {window_sql}
        """,
        [boundary],
    ).fetchall()
    if not candidates:
        return []

    rng = random.Random(seed)
    picks = rng.sample(candidates, min(n, len(candidates)))

    # Group by symbol so each return series is indexed once, then reuse the
    # exact window code the real builder uses. A second implementation here
    # could drift from it, and a control measured differently from the thing it
    # controls proves nothing about it.
    by_symbol: dict[str, list[Any]] = {}
    for symbol, day in picks:
        by_symbol.setdefault(str(symbol), []).append(day)

    samples: list[tuple[str, Any, float]] = []
    for symbol, day_list in by_symbol.items():
        series = ReturnSeries(_load_return_points(con, symbol))
        for day in day_list:
            window = series.forward(day, horizon)
            if window is not None:
                samples.append((symbol, window.t0, window.cumulative_abnormal_return))
    return samples


def _split_boundary(con: duckdb.DuckDBPyConnection) -> Any:
    """The discovery/holdout boundary actually used when samples were built.

    Read back from the data rather than re-imported from a default, so the
    control cannot silently be measured against a different boundary than the
    real cells were.
    """
    row = con.execute("SELECT min(t0) FROM event_samples WHERE split = ?", [HOLDOUT]).fetchone()
    return row[0] if row else None


def _load_return_points(con: duckdb.DuckDBPyConnection, symbol: str) -> list[ReturnPoint]:
    rows = con.execute(
        """
        SELECT price_date, total_return, abnormal_return
        FROM daily_returns WHERE symbol = ? ORDER BY price_date
        """,
        [symbol],
    ).fetchall()
    return [
        ReturnPoint(
            symbol=symbol,
            price_date=r[0],
            total_return=r[1] if r[1] is not None else 0.0,
            abnormal_return=r[2],
        )
        for r in rows
    ]


def _row(cell: ImpactCell, verdict: Verdict, reason: str, schema_version: str) -> dict[str, Any]:
    return {
        "stat_id": hashing.content_hash(
            "impact",
            cell.event_type,
            cell.event_subtype,
            cell.edge_type,
            cell.horizon_days,
            cell.split,
        ),
        "event_type": cell.event_type,
        "event_subtype": cell.event_subtype,
        "edge_type": cell.edge_type,
        "horizon_days": cell.horizon_days,
        "split": cell.split,
        "n_samples": cell.n_samples,
        "n_clusters": cell.n_clusters,
        "mean_car": cell.mean_car,
        "median_car": cell.median_car,
        "std_car": cell.std_car,
        "hit_rate": cell.hit_rate,
        "t_stat": cell.t_stat,
        "verdict": verdict.value,
        "verdict_reason": reason,
        "source": Source.DERIVED.value,
        "schema_version": schema_version,
        "validation_status": "valid",
        "validation_errors": "[]",
        "collected_time": utcnow(),
    }


def evaluate(
    config: Config,
    *,
    thresholds: AdmissionThresholds | None = None,
    with_placebo: bool = True,
    placebo_size: int = 5000,
    placebo_seed: int = 20260722,
) -> RunSummary:
    """Measure every cell and record which may fire alerts."""
    limits = thresholds or AdmissionThresholds()

    with pipeline_run(config, "signals.impact") as (con, summary):
        schema_version = config.settings.app.schema_version
        rows: list[dict[str, Any]] = []
        verdict_counts: dict[str, int] = {}

        for event_type, subtype, edge_type, horizon in _load_cell_keys(con):
            splits = {
                split: summarize(
                    _load_samples(con, event_type, subtype, edge_type, horizon, split),
                    event_type=event_type,
                    event_subtype=subtype,
                    edge_type=edge_type,
                    horizon_days=horizon,
                    split=split,
                )
                for split in (DISCOVERY, HOLDOUT)
            }

            verdict, reason = judge(splits[DISCOVERY], splits[HOLDOUT], limits)
            verdict_counts[verdict.value] = verdict_counts.get(verdict.value, 0) + 1

            for cell in splits.values():
                if cell is not None:
                    rows.append(_row(cell, verdict, reason, schema_version))

            summary.note(
                f"{event_type}/{subtype or '-'}/{edge_type}/{horizon}d: {verdict.value} — {reason}"
            )

        if with_placebo:
            for horizon in sorted({key[3] for key in _load_cell_keys(con)}):
                placebo = {
                    split: summarize(
                        _placebo_samples(
                            con, horizon, split, n=placebo_size, seed=placebo_seed + horizon
                        ),
                        event_type="placebo",
                        event_subtype=None,
                        edge_type=PLACEBO_EDGE,
                        horizon_days=horizon,
                        split=split,
                    )
                    for split in (DISCOVERY, HOLDOUT)
                }
                verdict, reason = judge(placebo[DISCOVERY], placebo[HOLDOUT], limits)

                # A placebo that clears the gate means the gate is broken, and
                # every admitted cell beside it is suspect. Surface it loudly
                # rather than filing it as one row among many.
                if verdict is Verdict.ADMITTED:
                    summary.bump("placebo_admitted")
                    summary.note(
                        f"CONTROL FAILURE: placebo at {horizon}d was ADMITTED — {reason}. "
                        "The gate is not rejecting noise; treat every admitted cell as unproven."
                    )
                else:
                    summary.note(f"placebo/{horizon}d: {verdict.value} (expected non-admission)")

                for cell in placebo.values():
                    if cell is not None:
                        rows.append(_row(cell, verdict, reason, schema_version))

        result = duckdb_store.upsert_impact_stats(con, rows)
        summary.collected = len(rows)
        summary.inserted = result.inserted
        summary.updated = result.updated
        for name, count in sorted(verdict_counts.items()):
            summary.bump(name, count)

    return summary


__all__ = ["PLACEBO_EDGE", "evaluate"]
