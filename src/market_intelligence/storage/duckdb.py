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

# _existing_keys and _update each build one SQL statement embedding a
# VALUES row per item in the batch, plus a flat ? parameter per value. Doing
# that for the whole batch in one shot is fine for a few thousand rows, but
# for a million-plus-row batch (e.g. upsert_daily_returns recomputing the
# full estimation window) DuckDB has to materialize the entire VALUES
# relation and parameter set before it can even start the join -- a live
# incident hit ~16GB RSS updating 1.56M rows in one statement and had to be
# killed before it took down the machine. Chunking bounds memory to
# O(chunk_size) regardless of how large the caller's batch is; see the
# empirical sizing note above `_BATCH_CHUNK_ROWS`.
_BATCH_CHUNK_ROWS = 10_000


def _chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


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
    candidates: Sequence[tuple[Any, ...]],
) -> set[tuple[Any, ...]]:
    """Existing key tuples for ``table``, scoped to rows that could match ``candidates``.

    A plain ``SELECT key_cols FROM table`` re-fetches and rebuilds a Python set
    from every row in the table on every call -- cheap while a table is a few
    thousand rows, a multi-minute stall once it is millions, because callers
    that flush incrementally (per symbol, per target) pay that full cost once
    per flush. Only rows matching something in this batch can ever affect the
    insert/update split, so a join scoped to the batch's own keys is the same
    answer for O(batch) instead of O(table). ``IS NOT DISTINCT FROM`` (not
    ``=``) so a NULL key part still matches, same as the old Python-side
    ``key in existing`` check did.

    Chunked into ``_BATCH_CHUNK_ROWS``-sized JOIN queries -- see the module
    comment above that constant -- so a million-plus-row batch doesn't force
    DuckDB to build one giant VALUES relation in memory.
    """
    if not candidates:
        return set()
    columns = ", ".join(f'"{col}"' for col in key_cols)
    value_cols = [f"c{i}" for i in range(len(key_cols))]
    join_cond = " AND ".join(
        f't."{col}" IS NOT DISTINCT FROM b.{value_col}'
        for col, value_col in zip(key_cols, value_cols, strict=True)
    )
    existing: set[tuple[Any, ...]] = set()
    for chunk in _chunked(candidates, _BATCH_CHUNK_ROWS):
        row_placeholder = "(" + ", ".join("?" for _ in key_cols) + ")"
        values_clause = ", ".join([row_placeholder] * len(chunk))
        sql = f"""
            SELECT {columns} FROM "{table}" t
            JOIN (VALUES {values_clause}) AS b({", ".join(value_cols)})
              ON {join_cond}
        """
        params = [value for candidate in chunk for value in candidate]
        result = con.execute(sql, params).fetchall()
        existing.update(tuple(record) for record in result)
    return existing


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
    """Apply the whole batch as one ``UPDATE ... FROM (VALUES ...)`` statement.

    The previous implementation fired one ``UPDATE ... WHERE key IS NOT
    DISTINCT FROM ?`` per row via ``executemany``. DuckDB has no point-lookup
    index for arbitrary key columns, so each row was a full table scan --
    O(updates x table_size), which stalled for hours against a multi-GB
    table. Joining the whole batch against the table in one pass, the same
    "VALUES clause of candidates" pattern :func:`_existing_keys` uses above,
    is O(batch) instead.

    That single-statement version is itself chunked into
    ``_BATCH_CHUNK_ROWS``-sized statements -- see the module comment above
    that constant. Without chunking, a million-plus-row batch (e.g.
    ``upsert_daily_returns`` recomputing a rolling window over the whole
    symbol universe) makes DuckDB materialize one giant VALUES relation plus
    a matching flat parameter list in a single call, which is what drove a
    live process to ~16GB RSS and had to be killed.
    """
    frozen = set(key_cols) | set(immutable_on_update)
    set_cols = [col for col in columns if col not in frozen]
    if not set_cols or not rows:
        return
    key_value_cols = [f"k{i}" for i in range(len(key_cols))]
    set_value_cols = [f"s{i}" for i in range(len(set_cols))]
    all_value_cols = key_value_cols + set_value_cols
    set_clause = ", ".join(
        f'"{col}" = b.{value_col}' for col, value_col in zip(set_cols, set_value_cols, strict=True)
    )
    # IS NOT DISTINCT FROM so NULL key parts (e.g. realtime bounds) match.
    join_cond = " AND ".join(
        f't."{col}" IS NOT DISTINCT FROM b.{value_col}'
        for col, value_col in zip(key_cols, key_value_cols, strict=True)
    )
    for chunk in _chunked(rows, _BATCH_CHUNK_ROWS):
        row_placeholder = "(" + ", ".join("?" for _ in all_value_cols) + ")"
        values_clause = ", ".join([row_placeholder] * len(chunk))
        sql = f"""
            UPDATE "{table}" AS t
            SET {set_clause}
            FROM (VALUES {values_clause}) AS b({", ".join(all_value_cols)})
            WHERE {join_cond}
        """
        params = [
            value
            for row in chunk
            for value in ([row.get(col) for col in key_cols] + [row.get(col) for col in set_cols])
        ]
        con.execute(sql, params)


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

    candidates = [tuple(row.get(col) for col in key_cols) for row in materialized]
    existing = _existing_keys(con, table, key_cols, candidates)
    to_insert: list[Row] = []
    to_update: list[Row] = []
    for row, key in zip(materialized, candidates, strict=True):
        (to_update if key in existing else to_insert).append(row)

    if to_insert:
        _insert(con, table, to_insert, columns)
    if to_update:
        _update(con, table, to_update, key_cols, columns, immutable_on_update)

    return UpsertResult(table, len(to_insert), len(to_update), deduped)


TRADE_ALERT_KEY = ("kind", "ticker", "trigger_key")


def insert_append_only(
    con: duckdb.DuckDBPyConnection,
    table: str,
    rows: Iterable[Row],
    *,
    key_cols: Sequence[str],
) -> list[Row]:
    """Insert only previously unseen rows using an explicit natural key.

    This is intentionally narrower than :func:`upsert`.  Decision attempts,
    source receipts, health reports and kill-switch history are audit facts;
    changing a previous one would make a later reconstruction impossible.
    The caller supplies the key so every append-only table documents its
    idempotency rule at its call site.
    """
    materialized = [dict(row) for row in rows]
    if not materialized:
        return []
    materialized = _dedupe_last_wins(materialized, key_cols)
    candidates = [tuple(row.get(c) for c in key_cols) for row in materialized]
    existing = _existing_keys(con, table, key_cols, candidates)
    fresh = [row for row in materialized if tuple(row.get(c) for c in key_cols) not in existing]
    if fresh:
        columns = [row[1] for row in con.execute(f'PRAGMA table_info("{table}")').fetchall()]
        _insert(con, table, fresh, columns)
    return fresh


OPPORTUNITY_KEY = ("event_id", "edge_id", "target_ticker", "horizon_days", "strategy_version")
DECISION_KEY = ("opportunity_id", "input_hash", "prompt_version", "model_version", "attempt_number")


def upsert_relationship_opportunities(
    con: duckdb.DuckDBPyConnection, rows: Iterable[Row]
) -> UpsertResult:
    """Persist the current evaluation state without rewriting its evidence hash.

    A natural-key repeat represents a re-evaluation of the same strategy
    family. The original creation time remains immutable, while a changed
    evidence packet creates a new hash and an explicit updated state.
    """
    columns = [
        row[1]
        for row in con.execute('PRAGMA table_info("relationship_opportunities")').fetchall()
    ]
    return upsert(
        con,
        "relationship_opportunities",
        rows,
        key_cols=OPPORTUNITY_KEY,
        columns=columns,
        immutable_on_update=("opportunity_id", "created_at"),
    )


def insert_opportunity_decisions(
    con: duckdb.DuckDBPyConnection, rows: Iterable[Row]
) -> list[Row]:
    """Append model attempts; failed/duplicate attempts are still auditable."""
    return insert_append_only(con, "opportunity_decisions", rows, key_cols=DECISION_KEY)


def insert_new_trade_alerts(
    con: duckdb.DuckDBPyConnection, rows: Iterable[Row]
) -> list[Row]:
    """Insert rows whose (kind, ticker, trigger_key) is unseen; return them.

    Insert-only, first-firing-wins semantics -- deliberately not the generic
    ``upsert``. An alert is a discrete prediction: once fired, its plan must
    never change underneath it. The generic ``upsert`` would overwrite the
    original plan on re-fire (e.g. the daily briefing's lookback window
    re-surfacing the same event), which is exactly what the ledger must never
    do. Rows whose key already exists in the table are silently dropped.

    "First firing wins" describes behaviour *across calls*: once a key is
    committed, no later call can change it. Within a single call, the house
    ``_dedupe_last_wins`` helper collapses same-key duplicates by keeping the
    *last* occurrence in the batch -- that collapse happens before the
    against-the-table existence check, so it only decides which of several
    simultaneous duplicates in one batch is offered for insertion, not
    whether an already-persisted alert can be overwritten.
    """
    materialized = [dict(r) for r in rows]
    if not materialized:
        return []
    materialized = _dedupe_last_wins(materialized, TRADE_ALERT_KEY)
    candidates = [tuple(r.get(c) for c in TRADE_ALERT_KEY) for r in materialized]
    existing = _existing_keys(con, "trade_alerts", TRADE_ALERT_KEY, candidates)
    fresh = [
        r for r in materialized
        if tuple(r.get(c) for c in TRADE_ALERT_KEY) not in existing
    ]
    if fresh:
        columns = [r[1] for r in con.execute('PRAGMA table_info("trade_alerts")').fetchall()]
        _insert(con, "trade_alerts", fresh, columns)
    return fresh


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


PEOPLE_COLUMNS: tuple[str, ...] = (
    "person_id", "reporting_owner_cik", "canonical_name", "name_variants",
    "first_seen_filing_date", "last_seen_filing_date", "source", "source_url",
    "content_hash", "schema_version", "validation_status", "validation_errors",
    "collected_time",
)  # fmt: skip


ROLE_MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "role_id", "person_id", "company_id", "is_officer", "is_director",
    "is_ten_pct_owner", "latest_officer_title", "first_seen", "last_seen",
    "source_filing_count", "source", "source_url", "content_hash",
    "schema_version", "validation_status", "validation_errors", "collected_time",
)  # fmt: skip


def upsert_people(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    """Upsert person identities, keyed on ``reporting_owner_cik``.

    ``first_seen_filing_date`` is immutable on update so the row keeps the
    moment this person was first observed, mirroring ``upsert_companies``.
    """
    return upsert(
        con,
        "people",
        rows,
        key_cols=["reporting_owner_cik"],
        columns=PEOPLE_COLUMNS,
        immutable_on_update=["first_seen_filing_date"],
    )


def upsert_role_memberships(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    """Upsert person-to-company roles, keyed on ``(person_id, company_id)``."""
    return upsert(
        con,
        "role_memberships",
        rows,
        key_cols=["person_id", "company_id"],
        columns=ROLE_MEMBERSHIP_COLUMNS,
        immutable_on_update=["first_seen"],
    )


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


#: ``event_samples``' natural key, matching the table's own UNIQUE constraint.
#: ``target_ticker`` is part of it because propagation fans one event out to
#: several targets under the same edge type: an insider buy at NKE scores both
#: ONON and LULU as "competitor" at the same horizon, and those are distinct
#: observations, not the same row overwritten. For a self edge the target equals
#: the source, so this changes nothing.
#:
#: Exported rather than inlined below so a producer that has to collapse its own
#: batch before offering it (``signals.dataset.build_propagation``) keys that
#: collapse on the same tuple this upsert does, instead of a hand-copied one
#: that can drift out of step with the constraint.
EVENT_SAMPLE_KEY: tuple[str, ...] = ("event_id", "edge_id", "horizon_days", "target_ticker")


def upsert_event_samples(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    return upsert(
        con,
        "event_samples",
        rows,
        key_cols=EVENT_SAMPLE_KEY,
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


PAPER_ORDER_COLUMNS: tuple[str, ...] = (
    "paper_order_id", "opportunity_id", "symbol", "side", "quantity",
    "quantity_exact", "reason", "limit_price", "submitted_at", "status",
    "simulation_version",
)  # fmt: skip

PAPER_FILL_COLUMNS: tuple[str, ...] = (
    "paper_fill_id", "paper_order_id", "symbol", "fill_price", "quantity",
    "quantity_exact", "filled_at", "fees", "borrow_cost", "slippage",
    "fill_assumptions",
)  # fmt: skip

#: ``paper_orders``' natural key, matching the table's own UNIQUE constraint.
PAPER_ORDER_KEY: tuple[str, ...] = ("opportunity_id", "simulation_version")

#: ``paper_fills``' natural key, matching the table's own UNIQUE constraint.
PAPER_FILL_KEY: tuple[str, ...] = ("paper_order_id", "filled_at")


def upsert_paper_orders(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    """Upsert simulated orders, keyed on ``(opportunity_id, simulation_version)``.

    Upsert rather than ``insert_append_only``: a paper order is a *derived*
    artefact of replaying a deterministic strategy over a fixed date, not an
    audit fact, so a re-run must be able to rewrite it in place. The version is
    part of the key so a scratch experiment writes beside the ledger of record
    instead of over it.
    """
    return upsert(
        con,
        "paper_orders",
        rows,
        key_cols=PAPER_ORDER_KEY,
        columns=PAPER_ORDER_COLUMNS,
    )


def upsert_paper_fills(con: duckdb.DuckDBPyConnection, rows: Iterable[Row]) -> UpsertResult:
    """Upsert simulated fills, keyed on ``(paper_order_id, filled_at)``.

    ``paper_fills`` carries no ``simulation_version`` column of its own; the
    version travels on the parent order, and every fill id is already version
    suffixed, so this key cannot collide across versions.
    """
    return upsert(
        con,
        "paper_fills",
        rows,
        key_cols=PAPER_FILL_KEY,
        columns=PAPER_FILL_COLUMNS,
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
