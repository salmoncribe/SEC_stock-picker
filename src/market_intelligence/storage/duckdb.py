"""Idempotent DuckDB writes.

The generic ``upsert`` reads the existing natural keys for a table, splits the
incoming batch into inserts vs. updates, and applies each. This gives:

* idempotency  — re-running a collection never duplicates rows;
* accurate accounting — exact inserted / updated counts for pipeline_runs;
* immutable columns — e.g. ``companies.first_seen_time`` is set once.

Rows are plain dicts (typically ``ProvenanceModel.to_row()`` output). Date/
datetime values must be real ``date``/``datetime`` objects so they compare
equal to what DuckDB returns for the key columns.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import duckdb

Row = dict[str, Any]


@dataclass(frozen=True)
class UpsertResult:
    table: str
    inserted: int
    updated: int
    deduped: int = 0

    @property
    def total(self) -> int:
        return self.inserted + self.updated

    @property
    def offered(self) -> int:
        """Rows handed to the upsert, i.e. ``inserted + updated + deduped``.

        Exposed so a caller can close the accounting identity: without
        ``deduped``, rows collapsed by :func:`_dedupe_last_wins` vanish with no
        counter explaining why ``inserted + updated`` fell short of the batch.
        """
        return self.inserted + self.updated + self.deduped


def _dedupe_last_wins(rows: list[Row], key_cols: Sequence[str]) -> list[Row]:
    """Collapse rows sharing a natural key, keeping the last occurrence."""
    collapsed: dict[tuple[Any, ...], Row] = {}
    for row in rows:
        key = tuple(row.get(col) for col in key_cols)
        collapsed[key] = row
    return list(collapsed.values())


def _existing_keys(
    con: duckdb.DuckDBPyConnection,
    table: str,
    key_cols: Sequence[str],
) -> set[tuple[Any, ...]]:
    columns = ", ".join(f'"{col}"' for col in key_cols)
    result = con.execute(f'SELECT {columns} FROM "{table}"').fetchall()
    return {tuple(record) for record in result}


def _insert(
    con: duckdb.DuckDBPyConnection,
    table: str,
    rows: list[Row],
    columns: Sequence[str],
) -> None:
    col_list = ", ".join(f'"{col}"' for col in columns)
    placeholders = ", ".join(["?"] * len(columns))
    sql = f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})'
    con.executemany(sql, [[row.get(col) for col in columns] for row in rows])


def _update(
    con: duckdb.DuckDBPyConnection,
    table: str,
    rows: list[Row],
    key_cols: Sequence[str],
    columns: Sequence[str],
    immutable_on_update: Sequence[str],
) -> None:
    frozen = set(key_cols) | set(immutable_on_update)
    set_cols = [col for col in columns if col not in frozen]
    if not set_cols:
        return
    set_clause = ", ".join(f'"{col}" = ?' for col in set_cols)
    # IS NOT DISTINCT FROM so NULL key parts (e.g. realtime bounds) match.
    where_clause = " AND ".join(f'"{col}" IS NOT DISTINCT FROM ?' for col in key_cols)
    sql = f'UPDATE "{table}" SET {set_clause} WHERE {where_clause}'
    params = [
        [row.get(col) for col in set_cols] + [row.get(col) for col in key_cols] for row in rows
    ]
    con.executemany(sql, params)


def upsert(
    con: duckdb.DuckDBPyConnection,
    table: str,
    rows: Iterable[Row],
    key_cols: Sequence[str],
    *,
    columns: Sequence[str] | None = None,
    immutable_on_update: Sequence[str] = (),
) -> UpsertResult:
    """Insert new rows and update existing ones, keyed on ``key_cols``."""
    materialized = [dict(row) for row in rows]
    if not materialized:
        return UpsertResult(table, 0, 0)

    offered = len(materialized)
    materialized = _dedupe_last_wins(materialized, key_cols)
    deduped = offered - len(materialized)

    if columns is None:
        ordered: list[str] = []
        for row in materialized:
            for col in row:
                if col not in ordered:
                    ordered.append(col)
        columns = ordered

    existing = _existing_keys(con, table, key_cols)
    to_insert: list[Row] = []
    to_update: list[Row] = []
    for row in materialized:
        key = tuple(row.get(col) for col in key_cols)
        (to_update if key in existing else to_insert).append(row)

    if to_insert:
        _insert(con, table, to_insert, columns)
    if to_update:
        _update(con, table, to_update, key_cols, columns, immutable_on_update)

    return UpsertResult(table, len(to_insert), len(to_update), deduped)


# --------------------------------------------------------------------------- #
# Table-specific wrappers                                                      #
# --------------------------------------------------------------------------- #
# Explicit column lists (matching database.py DDL) so records may carry extra
# base-provenance fields (e.g. event_time) without breaking inserts: only these
# columns are written, and any absent column binds to NULL.
COMPANY_COLUMNS: tuple[str, ...] = (
    "company_id", "ticker", "company_name", "cik", "exchange", "is_active",
    "first_seen_time", "last_seen_time", "source", "source_url", "content_hash",
    "schema_version", "validation_status", "validation_errors", "collected_time",
)  # fmt: skip

FILING_COLUMNS: tuple[str, ...] = (
    "filing_id", "company_id", "cik", "accession_number", "form", "filing_date",
    "report_date", "acceptance_time", "primary_document", "filing_url",
    "raw_file_path", "content_hash", "collected_time", "validation_status",
    "validation_errors", "source", "source_url", "schema_version",
)  # fmt: skip

FILING_DOCUMENT_COLUMNS: tuple[str, ...] = (
    "document_id", "filing_id", "company_id", "cik", "accession_number", "form",
    "document_name", "document_url", "document_type", "byte_size", "declared_size",
    "sha256", "raw_file_path", "content_type", "downloaded_time", "integrity_status",
    "section_count", "source", "source_url", "content_hash", "schema_version",
    "validation_status", "validation_errors", "collected_time",
)  # fmt: skip

FILING_SECTION_COLUMNS: tuple[str, ...] = (
    "section_id", "document_id", "filing_id", "company_id", "cik", "accession_number",
    "form", "report_date", "item_code", "item_title", "section_order", "char_count",
    "word_count", "text_path", "text_sha256", "preview", "extraction_method",
    "source", "source_url", "content_hash", "schema_version", "validation_status",
    "validation_errors", "collected_time",
)  # fmt: skip

SERIES_COLUMNS: tuple[str, ...] = (
    "series_id", "title", "units", "units_short", "frequency", "frequency_short",
    "seasonal_adjustment", "seasonal_adjustment_short", "observation_start",
    "observation_end", "last_updated", "popularity", "notes", "source",
    "source_url", "content_hash", "schema_version", "validation_status",
    "validation_errors", "collected_time",
)  # fmt: skip

OBSERVATION_COLUMNS: tuple[str, ...] = (
    "observation_id", "series_id", "observation_date", "value", "realtime_start",
    "realtime_end", "collected_time", "raw_file_path", "content_hash", "source",
    "source_url", "schema_version", "validation_status", "validation_errors",
)  # fmt: skip


def upsert_companies(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    return upsert(
        con,
        "companies",
        rows,
        key_cols=["cik"],
        columns=COMPANY_COLUMNS,
        immutable_on_update=["first_seen_time"],
    )


def upsert_filings(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    return upsert(con, "filings", rows, key_cols=["accession_number"], columns=FILING_COLUMNS)


def upsert_filing_documents(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    return upsert(
        con,
        "filing_documents",
        rows,
        key_cols=["accession_number", "document_name"],
        columns=FILING_DOCUMENT_COLUMNS,
    )


def upsert_filing_sections(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    return upsert(
        con,
        "filing_sections",
        rows,
        key_cols=["accession_number", "item_code"],
        columns=FILING_SECTION_COLUMNS,
    )


def upsert_series(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    return upsert(con, "economic_series", rows, key_cols=["series_id"], columns=SERIES_COLUMNS)


def upsert_observations(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    return upsert(
        con,
        "economic_observations",
        rows,
        key_cols=["series_id", "observation_date", "realtime_start", "realtime_end"],
        columns=OBSERVATION_COLUMNS,
    )


# --------------------------------------------------------------------------- #
# pipeline_runs bookkeeping                                                    #
# --------------------------------------------------------------------------- #
def start_pipeline_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    pipeline_name: str,
    started_time: Any,
    config_hash: str,
) -> None:
    con.execute(
        """
        INSERT INTO pipeline_runs (
            run_id, pipeline_name, started_time, status, config_hash,
            records_collected, records_inserted, records_updated, records_rejected
        ) VALUES (?, ?, ?, 'running', ?, 0, 0, 0, 0)
        """,
        [run_id, pipeline_name, started_time, config_hash],
    )


def finish_pipeline_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    *,
    status: str,
    completed_time: Any,
    collected: int,
    inserted: int,
    updated: int,
    rejected: int,
    downloaded: int = 0,
    stored: int = 0,
    skipped: int = 0,
    deduped: int = 0,
    stage_counts: str | None = None,
    error_message: str | None = None,
) -> None:
    con.execute(
        """
        UPDATE pipeline_runs
        SET completed_time = ?, status = ?, records_collected = ?,
            records_inserted = ?, records_updated = ?, records_rejected = ?,
            records_downloaded = ?, records_stored = ?, records_skipped = ?,
            records_deduped = ?, stage_counts = ?, error_message = ?
        WHERE run_id = ?
        """,
        [
            completed_time,
            status,
            collected,
            inserted,
            updated,
            rejected,
            downloaded,
            stored,
            skipped,
            deduped,
            stage_counts,
            error_message,
            run_id,
        ],
    )
