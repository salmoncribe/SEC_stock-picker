"""DuckDB database connection and table management."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import duckdb


def hash_id(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:16]


def connect(db_path: Path | str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path_str = str(db_path)
    if path_str != ":memory:":
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(database=path_str, read_only=read_only)


@contextmanager
def connection(db_path: Path | str, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    con = connect(db_path, read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    """Create analytical tables if missing."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            company_id TEXT PRIMARY KEY,
            ticker TEXT,
            company_name TEXT,
            cik TEXT NOT NULL UNIQUE,
            first_seen_time TIMESTAMPTZ,
            last_seen_time TIMESTAMPTZ,
            collected_time TIMESTAMPTZ
        );

        CREATE TABLE IF NOT EXISTS filings (
            filing_id TEXT PRIMARY KEY,
            company_id TEXT,
            cik TEXT,
            accession_number TEXT NOT NULL UNIQUE,
            form TEXT,
            filing_date DATE,
            report_date DATE,
            primary_document TEXT,
            filing_url TEXT,
            collected_time TIMESTAMPTZ
        );

        CREATE TABLE IF NOT EXISTS filing_documents (
            document_id TEXT PRIMARY KEY,
            filing_id TEXT,
            company_id TEXT,
            cik TEXT,
            accession_number TEXT NOT NULL,
            form TEXT,
            document_name TEXT NOT NULL,
            document_url TEXT,
            byte_size BIGINT,
            raw_file_path TEXT,
            collected_time TIMESTAMPTZ,
            UNIQUE (accession_number, document_name)
        );

        CREATE TABLE IF NOT EXISTS sync_log (
            id TEXT PRIMARY KEY,
            action TEXT,
            items_synced INTEGER,
            status TEXT,
            message TEXT,
            created_at TIMESTAMPTZ
        );

        CREATE TABLE IF NOT EXISTS filing_sections (
            section_id TEXT PRIMARY KEY,
            filing_id TEXT,
            accession_number TEXT NOT NULL,
            section_name TEXT NOT NULL,
            section_title TEXT,
            clean_text TEXT,
            word_count INTEGER,
            extracted_at TIMESTAMPTZ
        );

        ALTER TABLE filing_sections ADD COLUMN IF NOT EXISTS clean_text TEXT;
        ALTER TABLE filing_sections ADD COLUMN IF NOT EXISTS section_name TEXT;
        ALTER TABLE filing_sections ADD COLUMN IF NOT EXISTS section_title TEXT;
        ALTER TABLE filing_sections ADD COLUMN IF NOT EXISTS extracted_at TIMESTAMPTZ;

        UPDATE filing_documents
        SET form = f.form
        FROM filings f
        WHERE filing_documents.accession_number = f.accession_number
          AND (filing_documents.form IS NULL OR filing_documents.form = '');

        CREATE TABLE IF NOT EXISTS filing_metrics (
            metric_id TEXT PRIMARY KEY,
            filing_id TEXT,
            accession_number TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value DOUBLE,
            text_value TEXT,
            extracted_at TIMESTAMPTZ,
            UNIQUE (accession_number, metric_name)
        );

        CREATE TABLE IF NOT EXISTS filing_grades (
            grade_id TEXT PRIMARY KEY,
            filing_id TEXT,
            cik TEXT,
            ticker TEXT,
            accession_number TEXT NOT NULL UNIQUE,
            form TEXT,
            overall_grade TEXT,
            overall_score DOUBLE,
            sentiment_score DOUBLE,
            transparency_score DOUBLE,
            financial_score DOUBLE,
            summary_notes TEXT,
            graded_at TIMESTAMPTZ
        );
    """)


def get_table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    counts: dict[str, int] = {}
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    for t in ["companies", "filings", "filing_documents", "sync_log", "filing_sections", "filing_metrics", "filing_grades"]:
        if t in tables:
            counts[t] = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        else:
            counts[t] = 0
    return counts


def upsert_companies(con: duckdb.DuckDBPyConnection, rows: list[tuple[str, str, str]]) -> int:
    """Upsert company ticker map tuples: (cik, ticker, title)."""
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    data = [(hash_id("company", r[0]), r[1], r[2], r[0], now, now, now) for r in rows]
    con.executemany("""
        INSERT INTO companies (company_id, ticker, company_name, cik, first_seen_time, last_seen_time, collected_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (cik) DO UPDATE SET
            ticker = EXCLUDED.ticker,
            company_name = EXCLUDED.company_name,
            last_seen_time = EXCLUDED.last_seen_time,
            collected_time = EXCLUDED.collected_time
    """, data)
    return len(rows)


def upsert_filings(con: duckdb.DuckDBPyConnection, rows: list[tuple[str, str, str, str, str, str, str]]) -> int:
    """Upsert filings: (accession_number, cik, form, filing_date, report_date, primary_document, filing_url)."""
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    data = [
        (hash_id("filing", r[0]), hash_id("company", r[1]), r[1], r[0], r[2], r[3], r[4], r[5], r[6], now)
        for r in rows
    ]
    con.executemany("""
        INSERT INTO filings (filing_id, company_id, cik, accession_number, form, filing_date, report_date, primary_document, filing_url, collected_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (accession_number) DO NOTHING
    """, data)
    return len(rows)


def upsert_document(con: duckdb.DuckDBPyConnection, accession_number: str, cik: str, document_name: str, url: str, byte_size: int, raw_path: str, form: str = "") -> None:
    now = datetime.now(timezone.utc)
    doc_id = hash_id("doc", accession_number, document_name)
    company_id = hash_id("company", cik)
    filing_id = hash_id("filing", accession_number)
    con.execute("""
        INSERT INTO filing_documents (document_id, filing_id, company_id, cik, accession_number, form, document_name, document_url, byte_size, raw_file_path, collected_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (accession_number, document_name) DO UPDATE SET
            form = EXCLUDED.form,
            byte_size = EXCLUDED.byte_size,
            raw_file_path = EXCLUDED.raw_file_path,
            collected_time = EXCLUDED.collected_time
    """, [doc_id, filing_id, company_id, cik, accession_number, form, document_name, url, byte_size, raw_path, now])



def upsert_sections(con: duckdb.DuckDBPyConnection, rows: list[tuple[str, str, str, str, int]]) -> int:
    """Upsert filing sections: (accession_number, section_name, section_title, clean_text, word_count)."""
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    data = []
    for r in rows:
        acc, sec_name, title, text, w_count = r
        sec_id = hash_id("section", acc, sec_name)
        filing_id = hash_id("filing", acc)
        data.append((sec_id, filing_id, acc, sec_name, sec_name, title, title, text, w_count, now))

    con.executemany("""
        INSERT INTO filing_sections (section_id, filing_id, accession_number, section_name, item_code, section_title, item_title, clean_text, word_count, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
    """, data)
    return len(rows)




def upsert_metrics(con: duckdb.DuckDBPyConnection, rows: list[tuple[str, str, float | None, str | None]]) -> int:
    """Upsert filing metrics: (accession_number, metric_name, metric_value, text_value)."""
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    data = []
    for r in rows:
        acc, metric_name, val, text_val = r
        m_id = hash_id("metric", acc, metric_name)
        filing_id = hash_id("filing", acc)
        data.append((m_id, filing_id, acc, metric_name, val, text_val, now))

    con.executemany("""
        INSERT INTO filing_metrics (metric_id, filing_id, accession_number, metric_name, metric_value, text_value, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (accession_number, metric_name) DO UPDATE SET
            metric_value = EXCLUDED.metric_value,
            text_value = EXCLUDED.text_value,
            extracted_at = EXCLUDED.extracted_at
    """, data)
    return len(rows)


def upsert_grade(
    con: duckdb.DuckDBPyConnection,
    accession_number: str,
    cik: str,
    ticker: str,
    form: str,
    overall_grade: str,
    overall_score: float,
    sentiment_score: float,
    transparency_score: float,
    financial_score: float,
    summary_notes: str,
) -> None:
    now = datetime.now(timezone.utc)
    grade_id = hash_id("grade", accession_number)
    filing_id = hash_id("filing", accession_number)
    con.execute("""
        INSERT INTO filing_grades (grade_id, filing_id, cik, ticker, accession_number, form, overall_grade, overall_score, sentiment_score, transparency_score, financial_score, summary_notes, graded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (accession_number) DO UPDATE SET
            overall_grade = EXCLUDED.overall_grade,
            overall_score = EXCLUDED.overall_score,
            sentiment_score = EXCLUDED.sentiment_score,
            transparency_score = EXCLUDED.transparency_score,
            financial_score = EXCLUDED.financial_score,
            summary_notes = EXCLUDED.summary_notes,
            graded_at = EXCLUDED.graded_at
    """, [grade_id, filing_id, cik, ticker, accession_number, form, overall_grade, overall_score, sentiment_score, transparency_score, financial_score, summary_notes, now])


def get_ungraded_filings(con: duckdb.DuckDBPyConnection, limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve downloaded filing documents that have not yet been graded."""
    query = """
        SELECT
            d.accession_number,
            d.cik,
            c.ticker,
            f.form,
            f.filing_date,
            d.raw_file_path,
            d.document_name
        FROM filing_documents d
        JOIN filings f ON f.accession_number = d.accession_number
        LEFT JOIN companies c ON c.cik = d.cik
        LEFT JOIN filing_grades g ON g.accession_number = d.accession_number
        WHERE d.raw_file_path IS NOT NULL AND g.accession_number IS NULL
        LIMIT ?
    """
    rows = con.execute(query, [limit]).fetchall()
    return [
        {
            "accession_number": r[0],
            "cik": r[1],
            "ticker": r[2] or "UNKNOWN",
            "form": r[3] or "UNKNOWN",
            "filing_date": str(r[4]) if r[4] else "",
            "raw_file_path": r[5],
            "document_name": r[6],
        }
        for r in rows
    ]


def log_sync(con: duckdb.DuckDBPyConnection, sync_id: str, action: str, items_synced: int, status: str, message: str = "") -> None:
    now = datetime.now(timezone.utc)
    con.execute("""
        INSERT INTO sync_log (id, action, items_synced, status, message, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, [sync_id, action, items_synced, status, message, now])

