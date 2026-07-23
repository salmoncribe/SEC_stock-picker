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


CONSTITUENT_COLUMNS: tuple[str, ...] = (
    "constituent_id", "index_id", "company_id", "cik", "ticker", "company_name",
    "added_date", "removed_date", "source", "source_url", "content_hash",
    "schema_version", "validation_status", "validation_errors", "collected_time",
)  # fmt: skip

DAILY_PRICE_COLUMNS: tuple[str, ...] = (
    "price_id", "symbol", "price_date", "open", "high", "low", "close",
    "adj_close", "volume", "provider", "is_delisted_gap", "source", "source_url",
    "content_hash", "schema_version", "validation_status", "validation_errors",
    "collected_time",
)  # fmt: skip

DAILY_RETURN_COLUMNS: tuple[str, ...] = (
    "return_id", "symbol", "price_date", "total_return", "market_return",
    "sector_return", "abnormal_return", "beta", "alpha", "method",
    "estimation_window_start", "source", "source_url", "content_hash",
    "schema_version", "validation_status", "validation_errors", "collected_time",
)  # fmt: skip


EVENT_COLUMNS: tuple[str, ...] = (
    "event_id", "company_id", "cik", "ticker", "event_type", "event_subtype",
    "event_key", "accession_number", "filing_id", "event_time", "available_time",
    "magnitude", "direction", "payload", "extraction_method",
    "extraction_confidence", "source", "source_url", "content_hash",
    "schema_version", "validation_status", "validation_errors", "collected_time",
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


def upsert_constituents(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    """Upsert index-membership windows.

    Keyed on ``(index_id, ticker, added_date)`` rather than ``(index_id,
    ticker)``: a company that leaves an index and later rejoins has two
    distinct membership windows, and collapsing them onto one key would
    silently overwrite the first with the second, erasing the gap between them.
    """
    return upsert(
        con,
        "index_constituents",
        rows,
        key_cols=["index_id", "ticker", "added_date"],
        columns=CONSTITUENT_COLUMNS,
    )


def upsert_daily_prices(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    return upsert(
        con,
        "daily_prices",
        rows,
        key_cols=["symbol", "price_date"],
        columns=DAILY_PRICE_COLUMNS,
    )


def upsert_daily_returns(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    return upsert(
        con,
        "daily_returns",
        rows,
        key_cols=["symbol", "price_date"],
        columns=DAILY_RETURN_COLUMNS,
    )


SAMPLE_COLUMNS: tuple[str, ...] = (
    "sample_id", "event_id", "edge_id", "event_type", "event_subtype",
    "source_ticker", "target_ticker", "horizon_days", "available_on", "t0",
    "window_end", "forward_abnormal_return", "magnitude", "direction", "split",
    "features", "source", "source_url", "content_hash", "schema_version",
    "validation_status", "validation_errors", "collected_time",
)  # fmt: skip


IMPACT_STAT_COLUMNS: tuple[str, ...] = (
    "stat_id", "event_type", "event_subtype", "edge_type", "horizon_days",
    "split", "n_samples", "n_clusters", "mean_car", "median_car", "std_car",
    "hit_rate", "t_stat", "verdict", "verdict_reason", "source", "source_url",
    "content_hash", "schema_version", "validation_status", "validation_errors",
    "collected_time",
)  # fmt: skip


def upsert_impact_stats(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    return upsert(
        con,
        "impact_stats",
        rows,
        key_cols=["event_type", "event_subtype", "edge_type", "horizon_days", "split"],
        columns=IMPACT_STAT_COLUMNS,
    )


SIGNAL_STATUS_COLUMNS: tuple[str, ...] = (
    "signal_id", "event_type", "event_subtype", "edge_type", "horizon_days",
    "status", "confirm_streak", "fail_streak", "holdout_clusters", "last_verdict",
    "last_reason", "mean_car", "hit_rate", "n_clusters", "direction",
    "first_seen_time", "became_active_time", "last_evaluated_time", "schema_version",
)  # fmt: skip


COMPANY_EDGE_COLUMNS: tuple[str, ...] = (
    "edge_id", "edge_key", "source_cik", "source_ticker", "source_company_id",
    "target", "target_name", "target_cik", "target_ticker", "edge_type",
    "resolution_status", "resolution_confidence", "evidence",
    "extraction_confidence", "extraction_method", "extraction_model",
    "accession_number", "filing_id", "report_date", "times_asserted",
    "first_seen_time", "last_seen_time", "source", "source_url",
    "source_record_id", "event_time", "published_time", "content_hash",
    "schema_version", "validation_status", "validation_errors", "collected_time",
)  # fmt: skip


def upsert_company_edges(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    """Upsert relationship edges, keyed on ``edge_key`` = (source, target, type).

    One row per relationship, however many filings assert it -- the dedup the
    "don't get bloated with the same info added four times" requirement asks
    for. ``first_seen_time`` is immutable on update so the row keeps the moment
    the claim was first observed; the collector carries ``times_asserted``
    forward so the count reflects total assertions, not just the latest run.
    """
    return upsert(
        con,
        "company_edges",
        rows,
        key_cols=["edge_key"],
        columns=COMPANY_EDGE_COLUMNS,
        immutable_on_update=["first_seen_time"],
    )


def upsert_signal_status(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    """Upsert a cell's ladder position, keyed on the cell (not the split).

    ``first_seen_time`` is immutable on update -- it records when a cell was
    first tracked and must survive every later run's clock. ``became_active_time``
    is deliberately *not* immutable here: a cell is first inserted as a candidate
    with no activation time, so the value has to be settable later, on the run
    that promotes it. The caller computes it (set once on the transition into
    active, carried forward thereafter) so the upsert can write it plainly.
    """
    return upsert(
        con,
        "signal_status",
        rows,
        key_cols=["event_type", "event_subtype", "edge_type", "horizon_days"],
        columns=SIGNAL_STATUS_COLUMNS,
        immutable_on_update=["first_seen_time"],
    )


def upsert_event_samples(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    return upsert(
        con,
        "event_samples",
        rows,
        key_cols=["event_id", "edge_id", "horizon_days"],
        columns=SAMPLE_COLUMNS,
    )


def upsert_events(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    """Upsert typed events, keyed on ``(event_type, event_key)``.

    The key excludes ``accession_number`` on purpose: one filing routinely
    reports several transactions, so the filing alone does not identify an
    event. ``event_key`` is the producer's per-event identifier within its type.
    """
    return upsert(
        con,
        "events",
        rows,
        key_cols=["event_type", "event_key"],
        columns=EVENT_COLUMNS,
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
