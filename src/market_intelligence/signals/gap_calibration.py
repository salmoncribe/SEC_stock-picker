"""Gap calibration: turn graded ``price_gap`` outcomes into a track record.

``collectors/gaps.py`` fires alerts with ``ConfidenceInputs.has_track_record``
hardcoded to ``False`` (see that module's docstring, "Confidence is
deliberately capped") because it has no graded history to earn a higher score
from. ``signals/grading.py`` already grades every ``trade_alerts`` row,
``price_gap`` included -- it just has nowhere gap-specific to land. This
module is that landing place: it buckets graded gap alerts by the
characteristics known *at fire time* (direction, gap size, whether a catalyst
was attached) and aggregates each bucket's hit rate and mean return into
``gap_calibration_stats``. ``collectors/gaps.py`` then looks a candidate's
bucket up before scoring it, so ``has_track_record`` flips to ``True`` only
once real graded evidence exists for that specific kind of gap.

Deliberately **not** the ``impact_stats``/``signal_status`` gate
(``signals/impact.py``): that machinery is keyed on ``event_samples`` rows
(event_type, event_subtype, edge_id, horizon_days), and a ``price_gap`` alert
has no ``event_id`` to join on -- it cannot populate that table, so this is a
separate, simpler path rather than a forced fit.

**No network I/O, unlike ``gaps.py``.** Everything here reads and writes
``trade_alerts`` / ``gap_calibration_stats`` only, so -- unlike the gap
scanner's three-phase split -- there is no reason to hold the DuckDB
connection open across anything slow. ``recompute`` takes an already-open
connection and does its read and its write on it, the same one-connection
shape ``signals/grading.py``'s ``grade_open_alerts`` uses. ``recompute_step``
wraps that in ``pipeline_run`` for the orchestrator, exactly like
``grading.grade`` wraps ``grade_open_alerts``.

**Delete + reinsert, not upsert.** The bucket count is small (direction x
size bucket x catalyst-present is at most 2 x 3 x 2 = 12 rows) and every run
recomputes every bucket from the full graded history, so there is no
"existing row vs. new row" distinction worth tracking -- an incremental
upsert would only add bug surface for no real savings. The whole table is
rewritten inside one transaction so a reader never observes a half-deleted
table.

**Shrinkage does the small-sample protection, not a hard n-gate.**
``signals/confidence.shrunk_hit_rate`` already pulls a thin bucket's hit rate
toward 0.5 as its sample shrinks (``SHRINKAGE_N = 20`` pseudo-observations),
so a bucket with ``n_decisive = 1`` contributes almost nothing to the blended
score even though it technically "has a track record" now. This module does
not duplicate that protection with its own minimum-n floor -- it hands
``gaps.py`` the raw ``(hit_rate, n_decisive)`` pair and trusts
``signals/confidence.py`` to do the pulling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from market_intelligence.collectors import pipeline_run
from market_intelligence.logging_config import get_logger
from market_intelligence.schemas.common import utcnow

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.collectors import RunSummary
    from market_intelligence.config import Config

_log = get_logger("signals.gap_calibration")

# Outcomes that count as a settled win/loss, vs. a time exit with no clean
# touch. Mirrors signals/grading.py's _TERMINAL_OUTCOMES split between "hit
# something" and "expired" -- "still_open" rows are never read here at all
# because they never leave outcome='open' (see the WHERE clause below).
_WIN_OUTCOME = "hit_target"
_LOSS_OUTCOME = "hit_stop"
_EXPIRED_OUTCOME = "expired"

# TODO(michael): these gap-size cutoffs are a judgment call, not a discovery
# -- "small"/"medium"/"large" as fractions of the prior close, picked to be
# human-legible buckets rather than derived from anything statistical. Tune
# them once real gap history accumulates and you can see where the hit rate
# actually breaks.
_SMALL_GAP_MAX = 0.03
_MEDIUM_GAP_MAX = 0.06


def gap_size_bucket(gap_pct: float) -> str:
    """``"small"`` / ``"medium"`` / ``"large"`` from ``|gap_pct|``. See the TODO above."""
    magnitude = abs(gap_pct)
    if magnitude < _SMALL_GAP_MAX:
        return "small"
    if magnitude < _MEDIUM_GAP_MAX:
        return "medium"
    return "large"


def bucket_id(direction: int, size_bucket: str, catalyst_present: bool) -> str:
    """The deterministic natural key shared by ``recompute`` and the ``gaps.py`` lookup.

    Both sides must compute this identically or a candidate can silently miss
    its own bucket's history -- so neither module re-implements the format
    string; both call this function.
    """
    return f"{direction}:{size_bucket}:{catalyst_present}"


@dataclass
class GapCalibrationRow:
    """One bucket's aggregate track record -- the read-side shape of a row."""

    bucket_id: str
    direction: int
    gap_size_bucket: str
    catalyst_present: bool
    n_decisive: int
    n_expired: int
    hit_rate: float | None
    mean_return: float | None
    last_updated: Any = None


@dataclass
class _Accumulator:
    """Mutable per-bucket running totals while walking graded rows."""

    direction: int
    gap_size_bucket: str
    catalyst_present: bool
    n_hit_target: int = 0
    n_hit_stop: int = 0
    n_expired: int = 0
    _return_sum: float = field(default=0.0, repr=False)
    _return_count: int = field(default=0, repr=False)

    def add(self, outcome: str, outcome_return: float | None) -> None:
        if outcome == _WIN_OUTCOME:
            self.n_hit_target += 1
        elif outcome == _LOSS_OUTCOME:
            self.n_hit_stop += 1
        elif outcome == _EXPIRED_OUTCOME:
            self.n_expired += 1
        # mean_return spans every graded row in the bucket, expired included
        # -- see the module docstring's column description.
        if outcome_return is not None:
            self._return_sum += outcome_return
            self._return_count += 1

    @property
    def n_decisive(self) -> int:
        return self.n_hit_target + self.n_hit_stop

    @property
    def hit_rate(self) -> float | None:
        decisive = self.n_decisive
        return (self.n_hit_target / decisive) if decisive else None

    @property
    def mean_return(self) -> float | None:
        return (self._return_sum / self._return_count) if self._return_count else None


def _catalyst_present(evidence: dict[str, Any]) -> bool:
    """``True`` only when ``evidence["catalyst"]`` is the dict gaps.py writes for a real hit.

    ``gaps.py`` always writes the key -- either a catalyst dict or the
    literal string ``"no catalyst"`` (see ``_build_candidate``'s evidence
    block) -- but this stays defensive against a missing key too, so an
    older or hand-edited row degrades to "no catalyst" rather than raising.
    """
    return isinstance(evidence.get("catalyst"), dict)


def _load_graded_gap_rows(
    con: duckdb.DuckDBPyConnection,
) -> list[tuple[int | None, str | None, str | None, float | None]]:
    """``(direction, evidence_json, outcome, outcome_return)`` for every graded gap alert."""
    return con.execute(
        "SELECT direction, evidence, outcome, outcome_return "
        "FROM trade_alerts WHERE kind = 'price_gap' AND outcome != 'open'"
    ).fetchall()


def _aggregate(
    rows: list[tuple[int | None, str | None, str | None, float | None]],
) -> dict[str, _Accumulator]:
    """Bucket the graded rows in memory; skip anything too malformed to bucket.

    A row missing ``direction``/``evidence``, evidence that fails to parse, or
    evidence missing ``gap_pct`` cannot be assigned a bucket -- it is skipped
    (logged, not raised) rather than crashing the whole recompute over one bad
    row. This should never happen for rows this module itself wrote via
    ``gaps.py``, but grading never re-validates the evidence blob it is
    handed, so staying defensive here costs nothing.
    """
    buckets: dict[str, _Accumulator] = {}
    for direction, evidence_text, outcome, outcome_return in rows:
        if direction is None or evidence_text is None or outcome is None:
            continue
        try:
            evidence = json.loads(evidence_text)
        except (TypeError, ValueError):
            _log.warning("gap_calibration_evidence_unparseable", evidence=evidence_text)
            continue
        gap_pct = evidence.get("gap_pct")
        if not isinstance(gap_pct, int | float):
            continue

        size_bucket = gap_size_bucket(float(gap_pct))
        catalyst_present = _catalyst_present(evidence)
        key = bucket_id(direction, size_bucket, catalyst_present)

        acc = buckets.setdefault(
            key,
            _Accumulator(
                direction=direction,
                gap_size_bucket=size_bucket,
                catalyst_present=catalyst_present,
            ),
        )
        acc.add(outcome, outcome_return)
    return buckets


def recompute(con: duckdb.DuckDBPyConnection) -> int:
    """Rebuild ``gap_calibration_stats`` from every graded ``price_gap`` alert.

    Read phase is a single plain ``SELECT`` (no writes); the write phase
    deletes and reinserts the whole table inside one explicit transaction, so
    a crash mid-write leaves the previous run's table intact rather than
    half-empty. Returns the number of bucket rows written -- ``0`` on a
    genuinely quiet ledger (no graded gap alerts yet), which is the correct,
    inert state today: this function is called every day by the orchestrator
    but has nothing to do until grading resolves the first ``price_gap`` row.
    """
    buckets = _aggregate(_load_graded_gap_rows(con))
    now = utcnow()

    write_rows = [
        (
            key,
            acc.direction,
            acc.gap_size_bucket,
            acc.catalyst_present,
            acc.n_decisive,
            acc.n_expired,
            acc.hit_rate,
            acc.mean_return,
            now,
        )
        for key, acc in buckets.items()
    ]

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DELETE FROM gap_calibration_stats")
        if write_rows:
            con.executemany(
                "INSERT INTO gap_calibration_stats "
                "(bucket_id, direction, gap_size_bucket, catalyst_present, "
                "n_decisive, n_expired, hit_rate, mean_return, last_updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                write_rows,
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    return len(write_rows)


def recompute_step(config: Config) -> RunSummary:
    """Pipeline-step wrapper: recompute inside one ``pipeline_run`` connection.

    Mirrors ``signals.grading.grade``'s shape exactly -- one ``pipeline_run``
    call, no phase split, because (unlike ``gaps.py``) nothing here is slow
    or network-bound. ``collected``/``inserted`` both report the bucket count:
    every run is a full rewrite, so every row written this run is, honestly,
    a fresh insert (see ``recompute``'s delete-then-reinsert docstring) --
    there is no separate "updated" count to report.
    """
    with pipeline_run(config, "signals.calibrate-gap-confidence") as (con, summary):
        n = recompute(con)
        summary.collected = n
        summary.inserted = n
    return summary


def lookup_track_record(
    con: duckdb.DuckDBPyConnection,
    *,
    direction: int,
    gap_pct: float,
    catalyst_present: bool,
) -> tuple[float, int] | None:
    """``(hit_rate, n_decisive)`` for a candidate's bucket, or ``None``.

    ``None`` covers both cases ``gaps.py`` must fall back on: the bucket has
    no row yet (table empty, or this exact combination has never graded), or
    a row exists but ``n_decisive == 0`` (every graded row in the bucket so
    far expired without a clean hit, so ``hit_rate`` is NULL). Either way the
    caller's fallback is identical -- the pre-existing hardcoded
    ``has_track_record=False`` -- which is what keeps today's scoring
    unchanged until real decisive evidence exists for this specific bucket.
    """
    key = bucket_id(direction, gap_size_bucket(gap_pct), catalyst_present)
    row = con.execute(
        "SELECT hit_rate, n_decisive FROM gap_calibration_stats WHERE bucket_id = ?",
        [key],
    ).fetchone()
    if row is None:
        return None
    hit_rate, n_decisive = row
    if hit_rate is None or not n_decisive:
        return None
    return float(hit_rate), int(n_decisive)


def load_all(con: duckdb.DuckDBPyConnection) -> list[GapCalibrationRow]:
    """Every current bucket row, for the Obsidian calibration table."""
    rows = con.execute(
        "SELECT bucket_id, direction, gap_size_bucket, catalyst_present, "
        "n_decisive, n_expired, hit_rate, mean_return, last_updated "
        "FROM gap_calibration_stats ORDER BY bucket_id"
    ).fetchall()
    return [
        GapCalibrationRow(
            bucket_id=r[0],
            direction=r[1],
            gap_size_bucket=r[2],
            catalyst_present=r[3],
            n_decisive=r[4],
            n_expired=r[5],
            hit_rate=r[6],
            mean_return=r[7],
            last_updated=r[8],
        )
        for r in rows
    ]


__all__ = [
    "GapCalibrationRow",
    "bucket_id",
    "gap_size_bucket",
    "load_all",
    "lookup_track_record",
    "recompute",
    "recompute_step",
]
