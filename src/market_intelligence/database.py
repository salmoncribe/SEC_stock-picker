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
    "events",
    "event_samples",
    "impact_stats",
    "signal_status",
    "company_edges",
    "processed_relationship_sections",
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
    # One typed, timestamped thing that happened at one company.
    #
    # Two clocks, deliberately separate, because conflating them is the single
    # easiest way to fabricate a backtest result:
    #
    #   event_time     -- when it happened in the world (e.g. the trade date on
    #                     a Form 4). Unknowable to the market at the time.
    #   available_time -- when the public could first have known. This is the
    #                     only clock a feature or a label may key on.
    #
    # For an insider trade those differ by up to two business days, and using
    # event_time as t=0 would be trading on information nobody had yet.
    #
    # `event_type` is a plain string, not an enum, so a new kind of event is a
    # config entry and a re-run rather than a schema migration.
    "events": """
        CREATE TABLE IF NOT EXISTS events (
            event_id              TEXT PRIMARY KEY,
            company_id            TEXT,
            cik                   TEXT,
            ticker                TEXT,
            event_type            TEXT NOT NULL,
            event_subtype         TEXT,
            event_key             TEXT NOT NULL,
            accession_number      TEXT,
            filing_id             TEXT,
            event_time            TIMESTAMPTZ,
            available_time        TIMESTAMPTZ,
            magnitude             DOUBLE,
            direction             INTEGER,
            payload               TEXT,
            extraction_method     TEXT,
            extraction_confidence DOUBLE,
            source                TEXT,
            source_url            TEXT,
            content_hash          TEXT,
            schema_version        TEXT,
            validation_status     TEXT,
            validation_errors     TEXT,
            collected_time        TIMESTAMPTZ,
            UNIQUE (event_type, event_key)
        )
    """,
    # One (event, target company, horizon) observation: what happened to the
    # target over the N trading days after the event became actionable.
    #
    # This is the single contract every statistic downstream is computed from,
    # so it stores the *derivation* alongside the answer: `available_on` (when
    # the public could know), `t0` (the first tradeable day after that), and
    # `window_end`. Keeping all three means a stored row can be re-derived and
    # audited for leakage rather than trusted.
    #
    # `edge_id` is 'self' for a single-company event such as an insider trade,
    # and a graph edge id once propagation events arrive -- so the same table
    # serves the positive control and the real hypothesis without a second
    # schema.
    "event_samples": """
        CREATE TABLE IF NOT EXISTS event_samples (
            sample_id               TEXT PRIMARY KEY,
            event_id                TEXT NOT NULL,
            edge_id                 TEXT NOT NULL,
            event_type              TEXT,
            event_subtype           TEXT,
            source_ticker           TEXT,
            target_ticker           TEXT,
            horizon_days            INTEGER NOT NULL,
            available_on            DATE,
            t0                      DATE,
            window_end              DATE,
            forward_abnormal_return DOUBLE,
            magnitude               DOUBLE,
            direction               INTEGER,
            split                   TEXT,
            features                TEXT,
            source                  TEXT,
            source_url              TEXT,
            content_hash            TEXT,
            schema_version          TEXT,
            validation_status       TEXT,
            validation_errors       TEXT,
            collected_time          TIMESTAMPTZ,
            UNIQUE (event_id, edge_id, horizon_days, target_ticker)
        )
    """,
    # Measured behaviour of one (event kind, edge kind, horizon) combination on
    # one split, plus the verdict on whether it may fire alerts.
    #
    # n_samples and n_clusters are both stored and they are not redundant:
    # n_clusters is the number of independent observations (one per company-day)
    # and is what every statistic here is computed from, while n_samples is the
    # raw row count. Keeping both makes the clustering visible rather than
    # implicit -- a large gap between them is exactly the condition under which
    # an unclustered statistic would have been badly overconfident.
    #
    # The verdict is a property of the cell, not of a split: it is decided on
    # discovery and stamped on both rows so either can be read alone.
    "impact_stats": """
        CREATE TABLE IF NOT EXISTS impact_stats (
            stat_id           TEXT PRIMARY KEY,
            event_type        TEXT NOT NULL,
            event_subtype     TEXT,
            edge_type         TEXT NOT NULL,
            horizon_days      INTEGER NOT NULL,
            split             TEXT NOT NULL,
            n_samples         INTEGER,
            n_clusters        INTEGER,
            mean_car          DOUBLE,
            median_car        DOUBLE,
            std_car           DOUBLE,
            hit_rate          DOUBLE,
            t_stat            DOUBLE,
            verdict           TEXT,
            verdict_reason    TEXT,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ,
            UNIQUE (event_type, event_subtype, edge_type, horizon_days, split)
        )
    """,
    "signal_status": """
        CREATE TABLE IF NOT EXISTS signal_status (
            signal_id           TEXT PRIMARY KEY,
            event_type          TEXT NOT NULL,
            event_subtype       TEXT,
            edge_type           TEXT NOT NULL,
            horizon_days        INTEGER NOT NULL,
            status              TEXT NOT NULL,
            confirm_streak      INTEGER NOT NULL,
            fail_streak         INTEGER NOT NULL,
            holdout_clusters    INTEGER,
            last_verdict        TEXT,
            last_reason         TEXT,
            mean_car            DOUBLE,
            hit_rate            DOUBLE,
            n_clusters          INTEGER,
            direction           INTEGER,
            first_seen_time     TIMESTAMPTZ,
            became_active_time  TIMESTAMPTZ,
            last_evaluated_time TIMESTAMPTZ,
            schema_version      TEXT,
            UNIQUE (event_type, event_subtype, edge_type, horizon_days)
        )
    """,
    "company_edges": """
        CREATE TABLE IF NOT EXISTS company_edges (
            edge_id               TEXT PRIMARY KEY,
            edge_key              TEXT NOT NULL,
            source_cik            TEXT NOT NULL,
            source_ticker         TEXT,
            source_company_id     TEXT,
            target                TEXT NOT NULL,
            target_name           TEXT NOT NULL,
            target_cik            TEXT,
            target_ticker         TEXT,
            edge_type             TEXT NOT NULL,
            resolution_status     TEXT,
            resolution_confidence DOUBLE,
            evidence              TEXT,
            extraction_confidence DOUBLE,
            extraction_method     TEXT,
            extraction_model      TEXT,
            accession_number      TEXT,
            filing_id             TEXT,
            report_date           DATE,
            times_asserted        INTEGER,
            first_seen_time       TIMESTAMPTZ,
            last_seen_time        TIMESTAMPTZ,
            source                TEXT,
            source_url            TEXT,
            source_record_id      TEXT,
            event_time            TIMESTAMPTZ,
            published_time        TIMESTAMPTZ,
            content_hash          TEXT,
            schema_version        TEXT,
            validation_status     TEXT,
            validation_errors     TEXT,
            collected_time        TIMESTAMPTZ,
            UNIQUE (edge_key)
        )
    """,
    "processed_relationship_sections": """
        CREATE TABLE IF NOT EXISTS processed_relationship_sections (
            accession_number   TEXT NOT NULL,
            item_code          TEXT NOT NULL,
            source_ticker      TEXT,
            processed_time     TIMESTAMPTZ,
            edge_count         INTEGER,
            schema_version     TEXT,
            PRIMARY KEY (accession_number, item_code)
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
    "signal_status": {
        # Holdout cluster count at the last evaluation. The promotion ladder
        # advances a confirmation streak only when this grows -- i.e. when new
        # out-of-sample evidence actually arrived -- so re-running the gate on
        # unchanged data cannot manufacture a streak.
        "holdout_clusters": "INTEGER",
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
