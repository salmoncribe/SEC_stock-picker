"""Offline tests for filing-document ingestion, integrity, and reconciliation."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from market_intelligence import database, reconciliation
from market_intelligence.collectors import documents as document_collector
from market_intelligence.config import Config
from market_intelligence.hashing import sha256_bytes
from market_intelligence.schemas.sec import (
    FilingDocumentRecord,
    FilingSectionRecord,
    IntegrityStatus,
    build_filing_index_url,
    parse_filing_index,
)
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import normalized
from market_intelligence.validators.sec import validate_document, validate_section

CIK = "0001045810"
ACCESSION = "0001045810-24-000029"
ACC_NODASH = ACCESSION.replace("-", "")
DOCUMENT_NAME = "nvda-20240128.htm"
DOC_URL = f"https://www.sec.gov/Archives/edgar/data/1045810/{ACC_NODASH}/{DOCUMENT_NAME}"
INDEX_URL = build_filing_index_url(CIK, ACCESSION)


def _body(topic: str) -> str:
    """A block long enough to clear the extractor's minimum-body threshold."""
    return f"This section discusses {topic} in detail for the fiscal year. " * 12


FILING_HTML = f"""
<html><body>
  <p>Table of Contents</p>
  <p>Item 1. Business .................. 3</p>
  <p>Item 1A. Risk Factors ............. 10</p>
  <p>Item 7. Management's Discussion ... 30</p>

  <p>Item 1. Business</p>
  <p>{_body("our business operations and product lines")}</p>

  <p>Item 1A. Risk Factors</p>
  <p>{_body("the material risks facing the company")}</p>

  <p>Item 7. Management's Discussion and Analysis of Financial Condition</p>
  <p>{_body("management analysis of results of operations")}</p>
</body></html>
""".strip().encode("utf-8")


def _index_payload(size: int, name: str = DOCUMENT_NAME) -> dict[str, Any]:
    return {
        "directory": {
            "item": [
                {"name": name, "type": "10-K", "size": str(size)},
                {"name": "exhibit-21.htm", "type": "EX-21.1", "size": "2048"},
            ]
        }
    }


def _handler(
    *,
    index_payload: dict[str, Any] | None = None,
    index_status: int = 200,
    document: bytes = FILING_HTML,
) -> Callable[[httpx.Request], httpx.Response]:
    payload = _index_payload(len(document)) if index_payload is None else index_payload

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("index.json"):
            if index_status != 200:
                return httpx.Response(index_status, text="not found")
            return httpx.Response(200, json=payload)
        if url.endswith(DOCUMENT_NAME):
            return httpx.Response(200, content=document, headers={"Content-Type": "text/html"})
        return httpx.Response(404, text=f"unexpected url: {url}")

    return handle


def _seed_filing(config: Config, *, form: str = "10-K") -> None:
    """Insert one filing row so the collector has a candidate to work from."""
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_filings(
            con,
            [
                {
                    "filing_id": "filing-1",
                    "company_id": "company-1",
                    "cik": CIK,
                    "accession_number": ACCESSION,
                    "form": form,
                    "filing_date": None,
                    "report_date": None,
                    "primary_document": DOCUMENT_NAME,
                    "filing_url": DOC_URL,
                    "validation_status": "valid",
                    "source": "sec",
                }
            ],
        )


def _ingest(
    config: Config, handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> Any:
    return document_collector.ingest_documents(
        config, transport=httpx.MockTransport(handler), **kwargs
    )


def _seed_batch(config: Config, count: int) -> list[str]:
    """Insert ``count`` distinct filings, oldest index first. Returns accessions."""
    rows = []
    accessions = []
    for i in range(count):
        accession = f"0001045810-24-{i:06d}"
        acc_nodash = accession.replace("-", "")
        name = f"doc-{i}.htm"
        accessions.append(accession)
        rows.append(
            {
                "filing_id": f"filing-{i}",
                "company_id": "company-1",
                "cik": CIK,
                "accession_number": accession,
                "form": "10-K",
                # Ascending dates, so the newest-first candidate order is the
                # reverse of this list and the interruption point is knowable.
                "filing_date": date(2024, 1, 1 + i),
                "report_date": None,
                "primary_document": name,
                "filing_url": (
                    f"https://www.sec.gov/Archives/edgar/data/1045810/{acc_nodash}/{name}"
                ),
                "validation_status": "valid",
                "source": "sec",
            }
        )
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_filings(con, rows)
    return accessions


def _batch_handler(*, interrupt_on: str | None = None) -> Callable[[httpx.Request], httpx.Response]:
    """Serves any ``doc-N.htm``; optionally hard-stops on one of them.

    ``interrupt_on`` raises ``KeyboardInterrupt`` -- a ``BaseException``, so it
    passes straight through the collector's per-filing ``except Exception``
    guard the way a real Ctrl-C or SIGINT would, rather than being caught and
    counted as a skipped filing.
    """

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("index.json"):
            index = int(url.rsplit("/", 2)[-2][-6:])
            return httpx.Response(
                200, json=_index_payload(len(FILING_HTML), name=f"doc-{index}.htm")
            )
        name = url.rsplit("/", 1)[-1]
        if interrupt_on is not None and name == interrupt_on:
            raise KeyboardInterrupt("simulated interruption")
        if name.startswith("doc-"):
            return httpx.Response(200, content=FILING_HTML, headers={"Content-Type": "text/html"})
        return httpx.Response(404, text=f"unexpected url: {url}")

    return handle


# --------------------------------------------------------------------------- #
# index parsing                                                                #
# --------------------------------------------------------------------------- #
def test_parse_filing_index_flattens_items() -> None:
    rows = parse_filing_index(_index_payload(1234))
    assert rows[0] == {"name": DOCUMENT_NAME, "type": "10-K", "size": 1234}
    assert len(rows) == 2


def test_parse_filing_index_tolerates_garbage() -> None:
    assert parse_filing_index(None) == []
    assert parse_filing_index({"directory": {"item": "nope"}}) == []
    assert parse_filing_index({"directory": {"item": [{"name": "a.htm", "size": "x"}]}}) == [
        {"name": "a.htm", "type": None, "size": None}
    ]


def test_build_filing_index_url_strips_dashes_and_pads() -> None:
    url = build_filing_index_url(CIK, ACCESSION)
    assert url == f"https://www.sec.gov/Archives/edgar/data/1045810/{ACC_NODASH}/index.json"


# --------------------------------------------------------------------------- #
# normalized text store                                                        #
# --------------------------------------------------------------------------- #
def test_write_section_text_is_idempotent(tmp_path) -> None:
    first = normalized.write_section_text(
        tmp_path, cik=CIK, accession_number=ACCESSION, item_code="1A", text="risk text"
    )
    second = normalized.write_section_text(
        tmp_path, cik=CIK, accession_number=ACCESSION, item_code="1A", text="risk text"
    )
    assert first.changed is True
    assert second.changed is False
    assert first.path == second.path
    assert first.text_sha256 == second.text_sha256
    assert first.char_count == len("risk text")


def test_write_section_text_replaces_changed_content(tmp_path) -> None:
    first = normalized.write_section_text(
        tmp_path, cik=CIK, accession_number=ACCESSION, item_code="1A", text="old"
    )
    second = normalized.write_section_text(
        tmp_path, cik=CIK, accession_number=ACCESSION, item_code="1A", text="new"
    )
    assert second.changed is True
    assert first.text_sha256 != second.text_sha256


# --------------------------------------------------------------------------- #
# validation                                                                   #
# --------------------------------------------------------------------------- #
def _document(**overrides: Any) -> FilingDocumentRecord:
    base: dict[str, Any] = {
        "document_id": "doc-1",
        "cik": CIK,
        "accession_number": ACCESSION,
        "form": "10-K",
        "document_name": DOCUMENT_NAME,
        "byte_size": 100,
        "sha256": "abc",
        "integrity_status": IntegrityStatus.VERIFIED.value,
    }
    base.update(overrides)
    return FilingDocumentRecord(**base)


def test_verified_document_passes() -> None:
    assert validate_document(_document()).is_rejected is False


def test_size_mismatch_rejects() -> None:
    record = validate_document(
        _document(integrity_status=IntegrityStatus.SIZE_MISMATCH.value, declared_size=99)
    )
    assert record.is_rejected
    assert any("size_mismatch" in err for err in record.validation_errors)


def test_missing_from_index_rejects() -> None:
    record = validate_document(_document(integrity_status=IntegrityStatus.NOT_IN_INDEX.value))
    assert record.is_rejected


def test_unverified_document_warns_but_is_kept() -> None:
    record = validate_document(_document(integrity_status=IntegrityStatus.UNVERIFIED.value))
    assert record.is_rejected is False
    assert "integrity_unverified" in record.validation_errors


def test_empty_document_rejects() -> None:
    assert validate_document(_document(byte_size=0)).is_rejected


def test_short_section_warns_and_empty_section_rejects() -> None:
    def section(**over: Any) -> FilingSectionRecord:
        base: dict[str, Any] = {
            "section_id": "s1",
            "cik": CIK,
            "accession_number": ACCESSION,
            "form": "10-K",
            "item_code": "1A",
            "char_count": 5000,
            "text_path": "/tmp/x.txt",
            "text_sha256": "deadbeef",
        }
        base.update(over)
        return FilingSectionRecord(**base)

    assert validate_section(section()).is_rejected is False
    assert validate_section(section(char_count=0)).is_rejected
    short = validate_section(section(char_count=10))
    assert short.is_rejected is False
    assert any("short_section" in err for err in short.validation_errors)


# --------------------------------------------------------------------------- #
# upsert accounting                                                            #
# --------------------------------------------------------------------------- #
def test_upsert_reports_deduped_rows(memory_db) -> None:
    def row(item_code: str, marker: str) -> dict[str, Any]:
        return {
            "section_id": f"section-{item_code}",
            "accession_number": ACCESSION,
            "item_code": item_code,
            "cik": CIK,
            "form": "10-K",
            "preview": marker,
        }

    # Two rows share (accession_number, item_code): last must win, and the
    # collapse must be reported rather than silently vanishing.
    rows = [row("1A", "first"), row("1A", "second"), row("7", "only")]
    result = duckdb_store.upsert_filing_sections(memory_db, rows)
    assert (result.inserted, result.updated, result.deduped) == (2, 0, 1)
    assert result.offered == len(rows)

    kept = memory_db.execute(
        "SELECT preview FROM filing_sections WHERE item_code = '1A'"
    ).fetchone()[0]
    assert kept == "second"


# --------------------------------------------------------------------------- #
# end-to-end ingestion                                                         #
# --------------------------------------------------------------------------- #
def test_ingestion_downloads_verifies_and_sections(tmp_config: Config) -> None:
    _seed_filing(tmp_config)
    summary = _ingest(tmp_config, _handler())

    assert summary.status == "success"
    assert summary.downloaded == 1
    assert summary.stored == 1
    assert summary.skipped == 0
    assert summary.collected == 1
    assert summary.inserted == 1
    assert summary.rejected == 0
    assert summary.stage["sections_extracted"] >= 3

    with database.connection(tmp_config.paths.database_path) as con:
        status, sections = con.execute(
            "SELECT integrity_status, section_count FROM filing_documents"
        ).fetchone()
        assert status == IntegrityStatus.VERIFIED.value
        assert sections >= 3

        codes = {row[0] for row in con.execute("SELECT item_code FROM filing_sections").fetchall()}
        assert {"1", "1A", "7"} <= codes


def test_table_of_contents_is_not_stored_as_a_section(tmp_config: Config) -> None:
    _seed_filing(tmp_config)
    _ingest(tmp_config, _handler())

    with database.connection(tmp_config.paths.database_path) as con:
        rows = con.execute(
            "SELECT item_code, char_count FROM filing_sections ORDER BY item_code"
        ).fetchall()

    # Each accepted section holds real prose, not a one-line contents entry.
    assert rows
    assert all(char_count > 200 for _, char_count in rows)


def test_refetching_the_same_bytes_does_not_duplicate(tmp_config: Config) -> None:
    """Forced re-ingestion is idempotent: same bytes, same rows, no duplicates."""
    _seed_filing(tmp_config)
    first = _ingest(tmp_config, _handler())
    second = _ingest(tmp_config, _handler(), skip_ingested=False)

    assert first.stored == 1 and first.skipped == 0
    assert second.stored == 0 and second.skipped == 1
    assert second.inserted == 0 and second.updated == 1

    with database.connection(tmp_config.paths.database_path) as con:
        assert con.execute("SELECT count(*) FROM filing_documents").fetchone()[0] == 1
        before = con.execute("SELECT count(*) FROM filing_sections").fetchone()[0]
    assert before == first.stage["sections_extracted"]


def test_rerunning_does_not_refetch_already_ingested_filings(tmp_config: Config) -> None:
    """The resumability property: a caught-up run does no network work at all.

    Without this, restarting a 22k-filing backfill would spend hours of SEC
    rate limit re-downloading bytes it already has.
    """
    _seed_filing(tmp_config)
    _ingest(tmp_config, _handler())

    # Any request at all would 404 against this handler and fail the run.
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"already-ingested filing was refetched: {request.url}")

    second = _ingest(tmp_config, refuse)

    assert second.status == "success"
    assert second.stage["candidates"] == 0
    assert second.downloaded == 0


def test_interruption_keeps_the_batches_already_completed(tmp_config: Config) -> None:
    """A killed backfill must retain finished work, not lose the whole run.

    This is the regression guard for buffering every row in memory and writing
    once at the end -- behaviour indistinguishable from correct at small n, and
    total data loss at the scale this collector actually runs at.
    """
    _seed_batch(tmp_config, 5)

    # Newest-first ordering means doc-0 is processed last; with flush_every=2
    # the runs before it have already committed four documents.
    with pytest.raises(KeyboardInterrupt):
        _ingest(tmp_config, _batch_handler(interrupt_on="doc-0.htm"), flush_every=2)

    with database.connection(tmp_config.paths.database_path) as con:
        stored = con.execute("SELECT count(*) FROM filing_documents").fetchone()[0]
        sections = con.execute("SELECT count(*) FROM filing_sections").fetchone()[0]

    assert stored == 4, "completed batches were lost when the run was interrupted"
    assert sections > 0


def test_resuming_after_an_interruption_finishes_the_remainder(tmp_config: Config) -> None:
    """Interruption plus resumption equals one complete run."""
    _seed_batch(tmp_config, 5)

    with pytest.raises(KeyboardInterrupt):
        _ingest(tmp_config, _batch_handler(interrupt_on="doc-0.htm"), flush_every=2)

    resumed = _ingest(tmp_config, _batch_handler(), flush_every=2)

    # Only the one filing that never completed is retried.
    assert resumed.stage["candidates"] == 1
    assert resumed.downloaded == 1

    with database.connection(tmp_config.paths.database_path) as con:
        assert con.execute("SELECT count(*) FROM filing_documents").fetchone()[0] == 5


def test_counts_accumulate_across_flushes(tmp_config: Config) -> None:
    """Batched writes must report the whole run, not just the final batch."""
    _seed_batch(tmp_config, 5)

    summary = _ingest(tmp_config, _batch_handler(), flush_every=2)

    assert summary.collected == 5
    assert summary.inserted == 5
    assert summary.stage["sections_offered"] == summary.stage["sections_inserted"]


def test_size_mismatch_is_rejected_end_to_end(tmp_config: Config) -> None:
    _seed_filing(tmp_config)
    summary = _ingest(tmp_config, _handler(index_payload=_index_payload(999_999)))

    assert summary.downloaded == 1
    assert summary.rejected == 1
    assert summary.collected == 0
    with database.connection(tmp_config.paths.database_path) as con:
        assert con.execute("SELECT count(*) FROM filing_documents").fetchone()[0] == 0


def test_unavailable_index_still_stores_document_as_unverified(tmp_config: Config) -> None:
    _seed_filing(tmp_config)
    summary = _ingest(tmp_config, _handler(index_status=404))

    assert summary.collected == 1
    assert summary.stage["index_unavailable"] == 1
    with database.connection(tmp_config.paths.database_path) as con:
        status = con.execute("SELECT integrity_status FROM filing_documents").fetchone()[0]
    assert status == IntegrityStatus.UNVERIFIED.value


def test_document_missing_from_index_is_rejected(tmp_config: Config) -> None:
    _seed_filing(tmp_config)
    payload = _index_payload(len(FILING_HTML), name="something-else.htm")
    summary = _ingest(tmp_config, _handler(index_payload=payload))

    assert summary.rejected == 1
    assert summary.collected == 0


def test_raw_bytes_are_preserved_verbatim(tmp_config: Config) -> None:
    _seed_filing(tmp_config)
    _ingest(tmp_config, _handler())

    with database.connection(tmp_config.paths.database_path) as con:
        path, sha = con.execute("SELECT raw_file_path, sha256 FROM filing_documents").fetchone()

    stored_bytes = Path(path).read_bytes()
    assert stored_bytes == FILING_HTML
    assert sha256_bytes(stored_bytes) == sha


# --------------------------------------------------------------------------- #
# reconciliation                                                               #
# --------------------------------------------------------------------------- #
def test_reconciliation_identities_hold_after_a_run(tmp_config: Config) -> None:
    _seed_filing(tmp_config)
    _ingest(tmp_config, _handler())

    with database.connection(tmp_config.paths.database_path) as con:
        run = reconciliation.latest_run(con, reconciliation.INGEST_PIPELINE)
        assert run is not None
        failures = [c for c in reconciliation.run_checks(run) if not c.ok]
        assert failures == [], [(c.name, c.left, c.right) for c in failures]

        checks, anomalies = reconciliation.storage_checks(con, verify_hashes=True)
        storage_failures = [c for c in checks if not c.ok]
        assert storage_failures == [], [(c.name, c.left, c.right) for c in storage_failures]
        assert anomalies["raw_files_missing"] == 0
        assert anomalies["section_files_missing"] == 0


def test_reconciliation_detects_a_deleted_section_file(tmp_config: Config) -> None:
    _seed_filing(tmp_config)
    _ingest(tmp_config, _handler())

    with database.connection(tmp_config.paths.database_path) as con:
        path = con.execute("SELECT text_path FROM filing_sections LIMIT 1").fetchone()[0]

    import os

    os.remove(path)

    with database.connection(tmp_config.paths.database_path) as con:
        checks, anomalies = reconciliation.storage_checks(con, verify_hashes=True)

    assert anomalies["section_files_missing"] == 1
    presence = next(c for c in checks if c.name == "section text present")
    assert presence.ok is False
    assert presence.residual == 1


def test_stage_counts_survive_to_the_database(tmp_config: Config) -> None:
    _seed_filing(tmp_config)
    _ingest(tmp_config, _handler())

    with database.connection(tmp_config.paths.database_path) as con:
        raw_json = con.execute(
            "SELECT stage_counts FROM pipeline_runs WHERE pipeline_name = ?",
            [reconciliation.INGEST_PIPELINE],
        ).fetchone()[0]

    assert json.loads(raw_json)["candidates"] == 1


def test_non_sectionable_forms_are_not_candidates(tmp_config: Config) -> None:
    _seed_filing(tmp_config, form="8-K")
    summary = _ingest(tmp_config, _handler())
    assert summary.stage.get("candidates", 0) == 0
    assert summary.downloaded == 0


@pytest.mark.parametrize("form", ["10-K", "10-Q"])
def test_both_periodic_forms_are_candidates(tmp_config: Config, form: str) -> None:
    _seed_filing(tmp_config, form=form)
    summary = _ingest(tmp_config, _handler())
    assert summary.stage["candidates"] == 1
    assert summary.downloaded == 1
