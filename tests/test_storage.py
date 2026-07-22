"""Storage layer: raw hashing, DuckDB idempotency, Parquet merge, DB init."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from market_intelligence import database
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet
from market_intelligence.storage.raw import save_raw


def _company_row(cik: str, *, ticker: str, first_seen: datetime, last_seen: datetime) -> dict:
    return {
        "company_id": f"id-{cik}",
        "ticker": ticker,
        "company_name": f"Company {cik}",
        "cik": cik,
        "exchange": "NASDAQ",
        "is_active": True,
        "first_seen_time": first_seen,
        "last_seen_time": last_seen,
        "source": "sec",
        "source_url": "https://example.test",
        "content_hash": "h",
        "schema_version": "1.0.0",
        "validation_status": "valid",
        "validation_errors": json.dumps([]),
        "collected_time": last_seen,
    }


def _observation_row(series_id: str, obs_date: date, value: float | None) -> dict:
    return {
        "observation_id": f"{series_id}-{obs_date.isoformat()}",
        "series_id": series_id,
        "observation_date": obs_date,
        "value": value,
        "realtime_start": date(2026, 7, 21),
        "realtime_end": date(2026, 7, 21),
        "collected_time": datetime(2026, 7, 21, tzinfo=UTC),
        "raw_file_path": "/tmp/x.json",
        "content_hash": "h",
        "source": "fred",
        "source_url": "https://example.test",
        "schema_version": "1.0.0",
        "validation_status": "valid",
        "validation_errors": json.dumps([]),
    }


# --------------------------------------------------------------------------- #
# raw store                                                                    #
# --------------------------------------------------------------------------- #
def test_save_raw_is_idempotent_on_identical_content(tmp_path: Path) -> None:
    first = save_raw(
        tmp_path, "sec", "submissions", "NVDA", b'{"a":1}', collected_date=date(2026, 7, 21)
    )
    second = save_raw(
        tmp_path, "sec", "submissions", "NVDA", b'{"a":1}', collected_date=date(2026, 7, 21)
    )
    assert first.was_new is True
    assert second.was_new is False
    assert first.content_hash == second.content_hash
    assert first.path == second.path
    assert Path(first.path).exists()


def test_save_raw_does_not_clobber_different_content(tmp_path: Path) -> None:
    first = save_raw(
        tmp_path, "sec", "submissions", "NVDA", b"one", collected_date=date(2026, 7, 21)
    )
    second = save_raw(
        tmp_path, "sec", "submissions", "NVDA", b"two", collected_date=date(2026, 7, 21)
    )
    assert first.path != second.path
    assert Path(first.path).exists()
    assert Path(second.path).exists()
    assert Path(first.path).read_bytes() == b"one"
    assert Path(second.path).read_bytes() == b"two"


# --------------------------------------------------------------------------- #
# database init                                                                #
# --------------------------------------------------------------------------- #
def test_init_db_creates_all_tables() -> None:
    con = duckdb.connect(":memory:")
    tables = database.init_db(con)
    present = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert set(tables) == present
    assert "companies" in present and "pipeline_runs" in present
    con.close()


def test_table_counts_reports_zero_for_empty(memory_db: duckdb.DuckDBPyConnection) -> None:
    counts = database.table_counts(memory_db)
    assert counts["companies"] == 0
    assert set(counts) == set(database.TABLES)


# --------------------------------------------------------------------------- #
# idempotent upsert                                                            #
# --------------------------------------------------------------------------- #
def test_upsert_companies_idempotent(memory_db: duckdb.DuckDBPyConnection) -> None:
    now = datetime(2026, 7, 21, tzinfo=UTC)
    rows = [_company_row("0000320193", ticker="AAPL", first_seen=now, last_seen=now)]

    first = duckdb_store.upsert_companies(memory_db, rows)
    second = duckdb_store.upsert_companies(memory_db, rows)

    assert (first.inserted, first.updated) == (1, 0)
    assert (second.inserted, second.updated) == (0, 1)
    assert memory_db.execute("SELECT count(*) FROM companies").fetchone()[0] == 1


def test_upsert_updates_mutable_but_keeps_first_seen(memory_db: duckdb.DuckDBPyConnection) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 7, 21, tzinfo=UTC)
    duckdb_store.upsert_companies(
        memory_db, [_company_row("0000320193", ticker="AAPL", first_seen=t0, last_seen=t0)]
    )
    duckdb_store.upsert_companies(
        memory_db, [_company_row("0000320193", ticker="APPL-NEW", first_seen=t1, last_seen=t1)]
    )

    ticker, first_seen, last_seen = memory_db.execute(
        "SELECT ticker, first_seen_time, last_seen_time FROM companies WHERE cik = '0000320193'"
    ).fetchone()
    assert ticker == "APPL-NEW"
    assert first_seen == t0  # immutable_on_update kept the original
    assert last_seen == t1


def test_upsert_observations_composite_key_idempotent(memory_db: duckdb.DuckDBPyConnection) -> None:
    rows = [
        _observation_row("DGS10", date(2026, 7, 14), 4.45),
        _observation_row("DGS10", date(2026, 7, 15), None),
    ]
    first = duckdb_store.upsert_observations(memory_db, rows)
    second = duckdb_store.upsert_observations(memory_db, rows)

    assert (first.inserted, first.updated) == (2, 0)
    assert (second.inserted, second.updated) == (0, 2)
    assert memory_db.execute("SELECT count(*) FROM economic_observations").fetchone()[0] == 2
    null_count = memory_db.execute(
        "SELECT count(*) FROM economic_observations WHERE value IS NULL"
    ).fetchone()[0]
    assert null_count == 1


def test_upsert_dedupes_within_batch(memory_db: duckdb.DuckDBPyConnection) -> None:
    now = datetime(2026, 7, 21, tzinfo=UTC)
    rows = [
        _company_row("0000320193", ticker="AAPL", first_seen=now, last_seen=now),
        _company_row("0000320193", ticker="AAPL-DUP", first_seen=now, last_seen=now),
    ]
    result = duckdb_store.upsert_companies(memory_db, rows)
    assert result.inserted == 1
    assert memory_db.execute("SELECT ticker FROM companies").fetchone()[0] == "AAPL-DUP"


# --------------------------------------------------------------------------- #
# parquet                                                                      #
# --------------------------------------------------------------------------- #
def test_parquet_merge_is_idempotent(tmp_path: Path) -> None:
    now = datetime(2026, 7, 21, tzinfo=UTC)
    rows = [_company_row("0000320193", ticker="AAPL", first_seen=now, last_seen=now)]
    paths = parquet.write_records(tmp_path, "companies", rows, ["cik"])
    parquet.write_records(tmp_path, "companies", rows, ["cik"])
    frame = pd.read_parquet(paths[0])
    assert len(frame) == 1


def test_parquet_partitioned_write(tmp_path: Path) -> None:
    rows = [
        _observation_row("DGS10", date(2026, 7, 14), 4.45),
        _observation_row("UNRATE", date(2026, 7, 1), 3.8),
    ]
    written = parquet.write_records(
        tmp_path, "economic_observations", rows, ["observation_id"], partition_col="series_id"
    )
    assert len(written) == 2
    assert any("series_id=DGS10" in path for path in written)
    assert any("series_id=UNRATE" in path for path in written)


def test_parquet_empty_records_is_noop(tmp_path: Path) -> None:
    assert parquet.write_records(tmp_path, "companies", [], ["cik"]) == []
