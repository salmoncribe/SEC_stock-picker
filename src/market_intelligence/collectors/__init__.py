"""Collectors orchestrate: client -> validate -> raw + Parquet + DuckDB.

Shared machinery lives here so every collector records a ``pipeline_runs`` row
consistently. Source-specific collectors are in ``sec.py`` / ``fred.py``.

Usage pattern inside a collector::

    def sync_companies(config):
        with pipeline_run(config, "sec.sync-companies") as (con, summary):
            ...                       # fetch + validate + build rows
            summary.collected = n
            result = duckdb_store.upsert_companies(con, rows)
            summary.inserted, summary.updated = result.inserted, result.updated
            parquet.write_records(...)
        return summary
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from market_intelligence import database, hashing
from market_intelligence.schemas.common import utcnow
from market_intelligence.storage import duckdb as duckdb_store

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config


@dataclass
class RunSummary:
    """Result of a single collection run, surfaced to the CLI.

    The counters are deliberately distinct rather than derived, because each
    answers a different question during reconciliation:

    ``collected``  records built from the source and offered to storage;
    ``downloaded`` payloads successfully fetched over the network;
    ``stored``     *new* raw files written to disk (first sighting of bytes);
    ``skipped``    fetched payloads already byte-identical in the raw store —
                   the idempotency signal, not an error;
    ``inserted`` / ``updated`` rows created vs. modified in DuckDB;
    ``deduped``   rows collapsed inside one batch sharing a natural key;
    ``rejected``  records that failed a hard invariant and were not persisted.

    ``stage`` holds free-form per-stage counters (e.g. ``sections_extracted``)
    that a pipeline needs for its own reconciliation but that are not universal
    enough to be first-class fields.
    """

    pipeline_name: str
    collected: int = 0
    inserted: int = 0
    updated: int = 0
    rejected: int = 0
    downloaded: int = 0
    stored: int = 0
    skipped: int = 0
    deduped: int = 0
    status: str = "success"
    error_message: str | None = None
    notes: list[str] = field(default_factory=list)
    stage: dict[str, int] = field(default_factory=dict)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def bump(self, key: str, amount: int = 1) -> None:
        """Increment a free-form stage counter."""
        self.stage[key] = self.stage.get(key, 0) + amount


def make_run_id() -> str:
    return uuid.uuid4().hex


def config_hash(config: Config) -> str:
    return hashing.sha256_json(config.fingerprint())


@contextmanager
def pipeline_run(
    config: Config, pipeline_name: str
) -> Iterator[tuple[duckdb.DuckDBPyConnection, RunSummary]]:
    """Open a DuckDB connection and bookkeep a ``pipeline_runs`` row.

    Yields ``(connection, summary)``. The connection is shared by all upserts
    in the run so counts and the run record are written together. On exception
    the run is marked ``failed`` and the error re-raised.
    """
    summary = RunSummary(pipeline_name=pipeline_name)
    run_id = make_run_id()
    started = utcnow()
    chash = config_hash(config)

    config.paths.ensure()
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.start_pipeline_run(con, run_id, pipeline_name, started, chash)
        try:
            yield con, summary
        except Exception as exc:  # recorded to pipeline_runs, then re-raised
            summary.status = "failed"
            summary.error_message = str(exc)
            duckdb_store.finish_pipeline_run(
                con,
                run_id,
                status="failed",
                completed_time=utcnow(),
                collected=summary.collected,
                inserted=summary.inserted,
                updated=summary.updated,
                rejected=summary.rejected,
                downloaded=summary.downloaded,
                stored=summary.stored,
                skipped=summary.skipped,
                deduped=summary.deduped,
                stage_counts=json.dumps(summary.stage, sort_keys=True),
                error_message=str(exc),
            )
            raise
        else:
            duckdb_store.finish_pipeline_run(
                con,
                run_id,
                status=summary.status,
                completed_time=utcnow(),
                collected=summary.collected,
                inserted=summary.inserted,
                updated=summary.updated,
                rejected=summary.rejected,
                downloaded=summary.downloaded,
                stored=summary.stored,
                skipped=summary.skipped,
                deduped=summary.deduped,
                stage_counts=json.dumps(summary.stage, sort_keys=True),
                error_message=summary.error_message,
            )


__all__ = [
    "RunSummary",
    "config_hash",
    "make_run_id",
    "pipeline_run",
]
