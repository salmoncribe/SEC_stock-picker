"""DuckDB connection management and schema definition (DDL).

``init_db`` creates the analytical tables (idempotent — ``CREATE TABLE IF NOT
EXISTS``). Per-table write/upsert logic lives in ``storage/duckdb.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

TABLES: tuple[str, ...] = (
    "companies",
    "filings",
    "filing_documents",
    "filing_sections",
    "economic_series",
    "economic_observations",
    "index_constituents",
    "daily_prices",
    "daily_returns",
    "pipeline_runs",
)

# One statement per table so we can execute them individually.
SCHEMA_STATEMENTS: dict[str, str] = {
    "companies": """
        CREATE TABLE IF NOT EXISTS companies (
            company_id        TEXT PRIMARY KEY,
            ticker            TEXT,
            company_name      TEXT,
            cik               TEXT NOT NULL UNIQUE,
            exchange          TEXT,
            is_active         BOOLEAN,
            first_seen_time   TIMESTAMPTZ,
            last_seen_time    TIMESTAMPTZ,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ
        )
    """,
    "filings": """
        CREATE TABLE IF NOT EXISTS filings (
            filing_id         TEXT PRIMARY KEY,
            company_id        TEXT,
            cik               TEXT,
            accession_number  TEXT NOT NULL UNIQUE,
            form              TEXT,
            filing_date       DATE,
            report_date       DATE,
            acceptance_time   TIMESTAMPTZ,
            primary_document  TEXT,
            filing_url        TEXT,
            raw_file_path     TEXT,
            content_hash      TEXT,
            collected_time    TIMESTAMPTZ,
            validation_status TEXT,
            validation_errors TEXT,
            source            TEXT,
            source_url        TEXT,
            schema_version    TEXT
        )
    """,
    "filing_documents": """
        CREATE TABLE IF NOT EXISTS filing_documents (
            document_id       TEXT PRIMARY KEY,
            filing_id         TEXT,
            company_id        TEXT,
            cik               TEXT,
            accession_number  TEXT NOT NULL,
            form              TEXT,
            document_name     TEXT NOT NULL,
            document_url      TEXT,
            document_type     TEXT,
            byte_size         BIGINT,
            declared_size     BIGINT,
            sha256            TEXT,
            raw_file_path     TEXT,
            content_type      TEXT,
            downloaded_time   TIMESTAMPTZ,
            integrity_status  TEXT,
            section_count     INTEGER,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ,
            UNIQUE (accession_number, document_name)
        )
    """,
    "filing_sections": """
        CREATE TABLE IF NOT EXISTS filing_sections (
            section_id        TEXT PRIMARY KEY,
            document_id       TEXT,
            filing_id         TEXT,
            company_id        TEXT,
            cik               TEXT,
            accession_number  TEXT NOT NULL,
            form              TEXT,
            report_date       DATE,
            item_code         TEXT NOT NULL,
            item_title        TEXT,
            section_order     INTEGER,
            char_count        INTEGER,
            word_count        INTEGER,
            text_path         TEXT,
            text_sha256       TEXT,
            preview           TEXT,
            extraction_method TEXT,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ,
            UNIQUE (accession_number, item_code)
        )
    """,
    "economic_series": """
        CREATE TABLE IF NOT EXISTS economic_series (
            series_id                 TEXT PRIMARY KEY,
            title                     TEXT,
            units                     TEXT,
            units_short               TEXT,
            frequency                 TEXT,
            frequency_short           TEXT,
            seasonal_adjustment       TEXT,
            seasonal_adjustment_short TEXT,
            observation_start         DATE,
            observation_end           DATE,
            last_updated              TEXT,
            popularity                INTEGER,
            notes                     TEXT,
            source                    TEXT,
            source_url                TEXT,
            content_hash              TEXT,
            schema_version            TEXT,
            validation_status         TEXT,
            validation_errors         TEXT,
            collected_time            TIMESTAMPTZ
        )
    """,
    "economic_observations": """
        CREATE TABLE IF NOT EXISTS economic_observations (
            observation_id    TEXT PRIMARY KEY,
            series_id         TEXT NOT NULL,
            observation_date  DATE NOT NULL,
            value             DOUBLE,
            realtime_start    DATE,
            realtime_end      DATE,
            collected_time    TIMESTAMPTZ,
            raw_file_path     TEXT,
            content_hash      TEXT,
            source            TEXT,
            source_url        TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            UNIQUE (series_id, observation_date, realtime_start, realtime_end)
        )
    """,
    # Point-in-time index membership. `removed_date IS NULL` means "still a
    # member". Storing the window (rather than a flat current list) is what lets
    # a backtest ask "who was in the index on 2019-03-14?" instead of "who is in
    # it now?" -- the difference between a real result and survivorship bias.
    "index_constituents": """
        CREATE TABLE IF NOT EXISTS index_constituents (
            constituent_id    TEXT PRIMARY KEY,
            index_id          TEXT NOT NULL,
            company_id        TEXT,
            cik               TEXT,
            ticker            TEXT NOT NULL,
            company_name      TEXT,
            added_date        DATE NOT NULL,
            removed_date      DATE,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ,
            UNIQUE (index_id, ticker, added_date)
        )
    """,
    "daily_prices": """
        CREATE TABLE IF NOT EXISTS daily_prices (
            price_id          TEXT PRIMARY KEY,
            symbol            TEXT NOT NULL,
            price_date        DATE NOT NULL,
            open              DOUBLE,
            high              DOUBLE,
            low               DOUBLE,
            close             DOUBLE,
            adj_close         DOUBLE,
            volume            BIGINT,
            provider          TEXT,
            is_delisted_gap   BOOLEAN,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ,
            UNIQUE (symbol, price_date)
        )
    """,
    # `abnormal_return` is the label the whole signal layer is graded against.
    # `estimation_window_start` records which trailing window produced `beta`,
    # so a stored row can be re-derived and audited for lookahead.
    "daily_returns": """
        CREATE TABLE IF NOT EXISTS daily_returns (
            return_id               TEXT PRIMARY KEY,
            symbol                  TEXT NOT NULL,
            price_date              DATE NOT NULL,
            total_return            DOUBLE,
            market_return           DOUBLE,
            sector_return           DOUBLE,
            abnormal_return         DOUBLE,
            beta                    DOUBLE,
            alpha                   DOUBLE,
            method                  TEXT,
            estimation_window_start DATE,
            source                  TEXT,
            source_url              TEXT,
            content_hash            TEXT,
            schema_version          TEXT,
            validation_status       TEXT,
            validation_errors       TEXT,
            collected_time          TIMESTAMPTZ,
            UNIQUE (symbol, price_date)
        )
    """,
    "pipeline_runs": """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id            TEXT PRIMARY KEY,
            pipeline_name     TEXT,
            started_time      TIMESTAMPTZ,
            completed_time    TIMESTAMPTZ,
            status            TEXT,
            records_collected INTEGER,
            records_inserted  INTEGER,
            records_updated   INTEGER,
            records_rejected  INTEGER,
            error_message     TEXT,
            config_hash       TEXT
        )
    """,
}


def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open (creating parent dirs) a DuckDB connection to ``db_path``.

    The session timezone is pinned to UTC so ``TIMESTAMPTZ`` columns round-trip
    as UTC (the platform's internal clock) rather than the host's local zone.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("SET TimeZone='UTC'")
    return con


@contextmanager
def connection(db_path: str | Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context-managed DuckDB connection."""
    con = connect(db_path)
    try:
        yield con
    finally:
        con.close()


# Columns added to existing tables after their first release. ``CREATE TABLE IF
# NOT EXISTS`` will not add them to a database that already exists, so they are
# applied additively by ``migrate_db``.
COLUMN_MIGRATIONS: dict[str, dict[str, str]] = {
    "pipeline_runs": {
        "records_downloaded": "INTEGER",
        "records_stored": "INTEGER",
        "records_skipped": "INTEGER",
        "records_deduped": "INTEGER",
        # Free-form per-stage counters as JSON, so a run stays reconcilable
        # after the process that produced it is gone.
        "stage_counts": "TEXT",
    },
}


def migrate_db(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Additively add any missing columns. Idempotent; returns what it added."""
    applied: list[str] = []
    existing_tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    for table, columns in COLUMN_MIGRATIONS.items():
        if table not in existing_tables:
            continue
        present = {row[1] for row in con.execute(f'PRAGMA table_info("{table}")').fetchall()}
        for column, ddl in columns.items():
            if column not in present:
                con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {ddl}')
                applied.append(f"{table}.{column}")
    return applied


def init_db(con: duckdb.DuckDBPyConnection) -> tuple[str, ...]:
    """Create all tables if they do not exist, then apply column migrations."""
    for statement in SCHEMA_STATEMENTS.values():
        con.execute(statement)
    migrate_db(con)
    return TABLES


def table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Row count per table. Missing tables report -1."""
    counts: dict[str, int] = {}
    existing = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    for table in TABLES:
        if table in existing:
            result = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()
            counts[table] = int(result[0]) if result else 0
        else:
            counts[table] = -1
    return counts
