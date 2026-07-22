"""Pipeline count reconciliation.

Answers one question precisely: *do the numbers add up, and if not, why?*

Two independent families of checks:

* **run identities** — arithmetic that must hold inside a single recorded run
  (e.g. every downloaded payload was either newly stored or skipped as already
  present). These read ``pipeline_runs`` only.
* **storage agreement** — the database's claims checked against what is
  actually on disk and in the other tables (raw files present, section text
  present and matching its recorded hash, per-document section counts
  consistent).

A failing check is reported with its residual rather than raised, because the
point is to *explain* a discrepancy, not to hide it behind an exception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from market_intelligence.hashing import sha256_bytes

if TYPE_CHECKING:
    import duckdb

INGEST_PIPELINE = "sec.ingest-documents"


@dataclass(frozen=True)
class Check:
    """One reconciliation identity: ``left`` should equal ``right``."""

    name: str
    identity: str
    left: int
    right: int
    explanation: str

    @property
    def ok(self) -> bool:
        return self.left == self.right

    @property
    def residual(self) -> int:
        return self.left - self.right


def _scalar(con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None) -> int:
    """Run a single-value aggregate query, returning 0 when it yields no row."""
    result = con.execute(sql, params or []).fetchone()
    return int(result[0]) if result else 0


def latest_run(con: duckdb.DuckDBPyConnection, pipeline_name: str) -> dict[str, Any] | None:
    """Most recent run row for ``pipeline_name``, as a dict, or ``None``."""
    result = con.execute(
        """
        SELECT run_id, pipeline_name, started_time, completed_time, status,
               records_collected, records_inserted, records_updated,
               records_rejected, records_downloaded, records_stored,
               records_skipped, records_deduped, stage_counts
        FROM pipeline_runs
        WHERE pipeline_name = ?
        ORDER BY started_time DESC
        LIMIT 1
        """,
        [pipeline_name],
    ).fetchone()
    if result is None:
        return None

    columns = (
        "run_id", "pipeline_name", "started_time", "completed_time", "status",
        "collected", "inserted", "updated", "rejected", "downloaded", "stored",
        "skipped", "deduped", "stage_counts",
    )  # fmt: skip
    run: dict[str, Any] = dict(zip(columns, result, strict=True))
    for key in ("collected", "inserted", "updated", "rejected", "downloaded",
                "stored", "skipped", "deduped"):  # fmt: skip
        run[key] = int(run[key] or 0)
    try:
        run["stage"] = json.loads(run["stage_counts"] or "{}")
    except (TypeError, ValueError):
        run["stage"] = {}
    return run


def run_checks(run: dict[str, Any]) -> list[Check]:
    """Arithmetic identities that must hold within one ingest-documents run."""
    stage: dict[str, int] = run.get("stage") or {}

    def s(key: str) -> int:
        return int(stage.get(key, 0))

    return [
        Check(
            name="candidate disposition",
            identity="candidates = downloaded + fetch_failed + skipped_non_textual",
            left=s("candidates"),
            right=run["downloaded"] + s("fetch_failed") + s("skipped_non_textual"),
            explanation=(
                "Every filing selected for ingestion either downloaded, failed to "
                "fetch, or was skipped because its primary document is not a "
                "textual format we can section."
            ),
        ),
        Check(
            name="raw-store disposition",
            identity="downloaded = stored + skipped",
            left=run["downloaded"],
            right=run["stored"] + run["skipped"],
            explanation=(
                "Each downloaded payload was either written to the raw store for "
                "the first time (stored) or found already present byte-for-byte "
                "(skipped). 'skipped' is the idempotency signal, not an error."
            ),
        ),
        Check(
            name="document validation",
            identity="downloaded = collected + rejected",
            left=run["downloaded"],
            right=run["collected"] + run["rejected"],
            explanation=(
                "A downloaded document either became a collected record or was "
                "rejected by integrity/identity validation and not persisted."
            ),
        ),
        Check(
            name="document upsert",
            identity="collected = inserted + updated + deduped",
            left=run["collected"],
            right=run["inserted"] + run["updated"] + run["deduped"],
            explanation=(
                "Each collected document row was inserted as new, updated in "
                "place, or collapsed into a same-key row within the batch."
            ),
        ),
        Check(
            name="section validation",
            identity="sections_extracted = sections_offered + sections_rejected",
            left=s("sections_extracted"),
            right=s("sections_offered") + s("sections_rejected"),
            explanation=(
                "Every extracted Item section was either offered to storage or "
                "rejected (empty body or broken identity)."
            ),
        ),
        Check(
            name="section upsert",
            identity="sections_offered = sections_inserted + sections_updated + sections_deduped",
            left=s("sections_offered"),
            right=s("sections_inserted") + s("sections_updated") + s("sections_deduped"),
            explanation=(
                "Each offered section row was inserted, updated, or collapsed "
                "into a same-key row within the batch."
            ),
        ),
        Check(
            name="section file writes",
            identity="sections_extracted = section_files_written + section_files_unchanged",
            left=s("sections_extracted"),
            right=s("section_files_written") + s("section_files_unchanged"),
            explanation=(
                "Each extracted section's text was written to the normalized "
                "store, or left alone because the file already held identical "
                "text (a re-run extracting the same content)."
            ),
        ),
    ]


def storage_checks(
    con: duckdb.DuckDBPyConnection, *, verify_hashes: bool = True
) -> tuple[list[Check], dict[str, int]]:
    """Check the database's claims against the filesystem and sibling tables.

    Returns ``(checks, anomaly_counts)``. Hash verification re-reads every
    section text file, so it is optional for large stores.
    """
    doc_total = _scalar(con, "SELECT count(*) FROM filing_documents")
    section_total = _scalar(con, "SELECT count(*) FROM filing_sections")

    raw_paths = [
        row[0]
        for row in con.execute(
            "SELECT raw_file_path FROM filing_documents WHERE raw_file_path IS NOT NULL"
        ).fetchall()
    ]
    raw_present = sum(1 for path in raw_paths if Path(path).exists())

    section_rows = con.execute(
        "SELECT text_path, text_sha256 FROM filing_sections WHERE text_path IS NOT NULL"
    ).fetchall()
    text_present = 0
    hash_matches = 0
    hash_checked = 0
    for path, expected in section_rows:
        file_path = Path(path)
        if not file_path.exists():
            continue
        text_present += 1
        if verify_hashes and expected:
            hash_checked += 1
            if sha256_bytes(file_path.read_bytes()) == expected:
                hash_matches += 1

    # Does each document's recorded section_count match the rows actually stored?
    mismatched_counts = _scalar(
        con,
        """
        SELECT count(*) FROM (
            SELECT d.document_id,
                   COALESCE(d.section_count, 0) AS claimed,
                   COALESCE(s.actual, 0)        AS actual
            FROM filing_documents d
            LEFT JOIN (
                SELECT document_id, count(*) AS actual
                FROM filing_sections GROUP BY document_id
            ) s ON s.document_id = d.document_id
            WHERE COALESCE(d.section_count, 0) <> COALESCE(s.actual, 0)
        )
        """,
    )

    orphan_sections = _scalar(
        con,
        """
        SELECT count(*) FROM filing_sections s
        LEFT JOIN filing_documents d ON d.document_id = s.document_id
        WHERE d.document_id IS NULL
        """,
    )

    checks = [
        Check(
            name="raw files present",
            identity="filing_documents rows = raw files on disk",
            left=len(raw_paths),
            right=raw_present,
            explanation="Every stored document row points at a raw file that still exists.",
        ),
        Check(
            name="section text present",
            identity="filing_sections rows = section text files on disk",
            left=len(section_rows),
            right=text_present,
            explanation="Every section row points at a normalized text file that still exists.",
        ),
        Check(
            name="document section counts",
            identity="documents with mismatched section_count = 0",
            left=mismatched_counts,
            right=0,
            explanation=(
                "filing_documents.section_count agrees with the number of "
                "filing_sections rows actually linked to that document."
            ),
        ),
        Check(
            name="section parentage",
            identity="orphan sections = 0",
            left=orphan_sections,
            right=0,
            explanation="Every section links to a document row that exists.",
        ),
    ]

    if verify_hashes:
        checks.append(
            Check(
                name="section text integrity",
                identity="section files hashed = hashes matching recorded sha256",
                left=hash_checked,
                right=hash_matches,
                explanation=(
                    "Re-hashing each section text file reproduces the sha256 "
                    "recorded when it was extracted — the text has not drifted."
                ),
            )
        )

    anomalies = {
        "filing_documents": doc_total,
        "filing_sections": section_total,
        "raw_files_missing": len(raw_paths) - raw_present,
        "section_files_missing": len(section_rows) - text_present,
    }
    return checks, anomalies


def integrity_breakdown(con: duckdb.DuckDBPyConnection) -> list[tuple[str, int]]:
    """Document count per ``integrity_status``, most common first."""
    return [
        (str(status), int(count))
        for status, count in con.execute(
            """
            SELECT COALESCE(integrity_status, 'unknown'), count(*)
            FROM filing_documents GROUP BY 1 ORDER BY 2 DESC
            """
        ).fetchall()
    ]


def section_coverage(con: duckdb.DuckDBPyConnection, limit: int = 15) -> list[tuple[str, int, int]]:
    """Per-item-code coverage: ``(item_code, section_count, distinct_filings)``."""
    return [
        (str(code), int(sections), int(filings))
        for code, sections, filings in con.execute(
            """
            SELECT item_code, count(*), count(DISTINCT accession_number)
            FROM filing_sections
            GROUP BY 1 ORDER BY 2 DESC LIMIT ?
            """,
            [limit],
        ).fetchall()
    ]


__all__ = [
    "INGEST_PIPELINE",
    "Check",
    "integrity_breakdown",
    "latest_run",
    "run_checks",
    "section_coverage",
    "storage_checks",
]
