"""Filing-document ingestion: download -> preserve -> verify -> extract sections.

This collector turns filing *metadata* (already in the ``filings`` table) into
filing *documents* and the clean Item sections a later AI-extraction phase will
consume. Nothing here interprets meaning — extraction is a deterministic split
on the official 10-K/10-Q Item taxonomy.

Per candidate filing:

1. fetch the filing folder's ``index.json`` (the document manifest);
2. fetch the primary document and preserve its bytes byte-for-byte;
3. cross-check received bytes against the size the manifest declares;
4. split the document into Item sections and write each section's text;
5. upsert document + section records into DuckDB and Parquet.

Count accounting
----------------
The first-class :class:`RunSummary` counters describe the **document** stage;
every **section**-stage count is namespaced under ``summary.stage`` with a
``sections_`` prefix. Keeping them apart is what makes the run reconcilable:
one filing yields exactly one document but many sections, so a single blended
"records" number could never balance.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.clients.sec import SECClient
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.extraction import sections as section_extractor
from market_intelligence.schemas.common import utcnow
from market_intelligence.schemas.sec import (
    FilingDocumentRecord,
    FilingSectionRecord,
    IntegrityStatus,
    parse_filing_index,
)
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import normalized, parquet, raw
from market_intelligence.validators.sec import validate_document, validate_section

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config

# Forms with a stable, officially-defined Item taxonomy. Only these can be
# split deterministically, so only these are candidates for ingestion.
SECTIONABLE_FORMS: tuple[str, ...] = ("10-K", "10-K/A", "10-Q", "10-Q/A")

# Document extensions we can turn into text. Anything else (PDF, image exhibit)
# is skipped rather than stored as an unparseable blob.
TEXTUAL_SUFFIXES: tuple[str, ...] = (".htm", ".html", ".txt")

PREVIEW_CHARS = 300

#: Filings ingested between writes. Small enough that an interruption costs
#: little, large enough that the Parquet merge-write is not the bottleneck.
DEFAULT_FLUSH_EVERY = 100


def _preview(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:PREVIEW_CHARS]


def _extension(document_name: str) -> str:
    _, _, suffix = document_name.rpartition(".")
    return suffix.lower() if suffix and suffix != document_name else "htm"


def select_candidates(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str] | None = None,
    forms: list[str] | None = None,
    limit: int | None = None,
    skip_ingested: bool = True,
) -> list[dict[str, Any]]:
    """Read the filings eligible for document ingestion, newest first.

    ``skip_ingested`` subtracts filings already present in ``filing_documents``,
    which is what makes a long backfill resumable: each run picks up where the
    last one stopped instead of re-downloading from the top. Persisting results
    incrementally is only half of resumability -- without this subtraction a
    restart would faithfully store everything it already had, spending hours of
    SEC rate limit to learn nothing.

    Matched on ``accession_number`` rather than ``filing_id`` because the
    accession is SEC's own identifier and is always populated, whereas
    ``filing_id`` is ours and may be absent on older rows.
    """
    wanted = list(forms) if forms else list(SECTIONABLE_FORMS)
    params: list[Any] = list(wanted)
    sql = f"""
        SELECT filing_id, company_id, cik, accession_number, form,
               primary_document, filing_url, report_date
        FROM filings
        WHERE form IN ({", ".join(["?"] * len(wanted))})
          AND primary_document IS NOT NULL
          AND primary_document <> ''
          AND validation_status <> 'rejected'
    """
    if skip_ingested:
        sql += """
          AND NOT EXISTS (
              SELECT 1 FROM filing_documents d
              WHERE d.accession_number = filings.accession_number
          )
        """
    if tickers:
        uppered = [t.upper() for t in tickers]
        placeholders = ", ".join(["?"] * len(uppered))
        sql += f" AND cik IN (SELECT cik FROM companies WHERE upper(ticker) IN ({placeholders}))"
        params.extend(uppered)
    sql += " ORDER BY filing_date DESC NULLS LAST"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    columns = (
        "filing_id", "company_id", "cik", "accession_number", "form",
        "primary_document", "filing_url", "report_date",
    )  # fmt: skip
    return [dict(zip(columns, row, strict=True)) for row in con.execute(sql, params).fetchall()]


def _resolve_integrity(
    index_rows: list[dict[str, Any]] | None,
    document_name: str,
    byte_size: int,
) -> tuple[str, int | None, str | None]:
    """Compare the received bytes with the filing manifest.

    Returns ``(integrity_status, declared_size, document_type)``. A manifest we
    could not fetch yields ``unverified`` — an absence of evidence, not
    evidence of corruption — whereas a manifest that positively disagrees
    yields a rejecting status.
    """
    if index_rows is None:
        return IntegrityStatus.UNVERIFIED.value, None, None

    match = next((r for r in index_rows if r.get("name") == document_name), None)
    if match is None:
        return IntegrityStatus.NOT_IN_INDEX.value, None, None

    declared = match.get("size")
    doc_type = match.get("type")
    if declared is None:
        return IntegrityStatus.UNVERIFIED.value, None, doc_type
    if int(declared) != byte_size:
        return IntegrityStatus.SIZE_MISMATCH.value, int(declared), doc_type
    return IntegrityStatus.VERIFIED.value, int(declared), doc_type


def _ingest_one(
    config: Config,
    client: SECClient,
    summary: RunSummary,
    candidate: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Download, verify, and section a single filing.

    Returns ``(document_row, section_rows)``; ``document_row`` is ``None`` when
    the document could not be fetched or was rejected.
    """
    accession = str(candidate["accession_number"])
    cik = str(candidate["cik"])
    form = str(candidate["form"])
    document_name = str(candidate["primary_document"])
    schema_version = config.settings.app.schema_version

    if not document_name.lower().endswith(TEXTUAL_SUFFIXES):
        summary.bump("skipped_non_textual")
        summary.note(f"non_textual_document:{accession}:{document_name}")
        return None, []

    # 1. Manifest first, so integrity is decided by evidence gathered
    #    independently of the document response itself.
    index_rows: list[dict[str, Any]] | None
    try:
        index_rows = parse_filing_index(client.fetch_filing_index(cik, accession).data)
    except Exception as exc:
        index_rows = None
        summary.bump("index_unavailable")
        summary.note(f"index_unavailable:{accession}:{exc}")

    # 2. The document itself.
    url = candidate.get("filing_url") or ""
    fetched = client.fetch_filing_document(str(url))
    summary.downloaded += 1

    saved = raw.save_raw(
        config.paths.raw_dir,
        "sec",
        "filing_document",
        accession.replace("-", ""),
        fetched.raw,
        ext=_extension(document_name),
    )
    if saved.was_new:
        summary.stored += 1
    else:
        summary.skipped += 1

    integrity_status, declared_size, doc_type = _resolve_integrity(
        index_rows, document_name, saved.size
    )
    collected_time = utcnow()

    document = FilingDocumentRecord(
        document_id=hashing.content_hash("document", accession, document_name),
        filing_id=candidate.get("filing_id"),
        company_id=candidate.get("company_id"),
        cik=cik,
        accession_number=accession,
        form=form,
        document_name=document_name,
        document_url=fetched.url,
        document_type=doc_type,
        byte_size=saved.size,
        declared_size=declared_size,
        sha256=saved.content_hash,
        raw_file_path=saved.path,
        content_type=fetched.content_type,
        downloaded_time=collected_time,
        integrity_status=integrity_status,
        source_url=fetched.url,
        content_hash=saved.content_hash,
        collected_time=collected_time,
        schema_version=schema_version,
    )
    validate_document(document)
    if document.is_rejected:
        summary.rejected += 1
        summary.note(f"document_rejected:{accession}:{','.join(document.validation_errors)}")
        return None, []

    # 3. Deterministic section split. Only a document that passed integrity
    #    validation is allowed to produce sections.
    extracted = section_extractor.extract_sections(fetched.raw, form)
    summary.bump("sections_extracted", len(extracted))
    if not extracted:
        summary.bump("documents_without_sections")
        summary.note(f"no_sections:{accession}:{form}")

    report_date = candidate.get("report_date")
    if isinstance(report_date, str):
        try:
            report_date = date.fromisoformat(report_date[:10])
        except ValueError:
            report_date = None

    section_rows: list[dict[str, Any]] = []
    for item in extracted:
        written = normalized.write_section_text(
            config.paths.normalized_dir,
            cik=cik,
            accession_number=accession,
            item_code=item.item_code,
            text=item.text,
        )
        summary.bump("section_files_written" if written.changed else "section_files_unchanged")

        record = FilingSectionRecord(
            section_id=hashing.content_hash("section", accession, item.item_code),
            document_id=document.document_id,
            filing_id=candidate.get("filing_id"),
            company_id=candidate.get("company_id"),
            cik=cik,
            accession_number=accession,
            form=form,
            report_date=report_date,
            item_code=item.item_code,
            item_title=item.item_title,
            section_order=item.order,
            char_count=written.char_count,
            word_count=written.word_count,
            text_path=written.path,
            text_sha256=written.text_sha256,
            preview=_preview(item.text),
            extraction_method=section_extractor.EXTRACTION_METHOD,
            source_url=fetched.url,
            content_hash=written.text_sha256,
            collected_time=collected_time,
            schema_version=schema_version,
        )
        validate_section(record)
        if record.is_rejected:
            summary.bump("sections_rejected")
            continue
        section_rows.append(record.to_row())

    document.section_count = len(section_rows)
    return document.to_row(), section_rows


def _flush(
    config: Config,
    con: duckdb.DuckDBPyConnection,
    summary: RunSummary,
    document_rows: list[dict[str, Any]],
    section_rows: list[dict[str, Any]],
) -> None:
    """Persist one batch and fold its counts into the running summary.

    Counters accumulate with ``+=`` rather than being assigned, because this is
    called many times per run; assigning would silently report only the last
    batch and make the run look far smaller than it was.
    """
    if not document_rows and not section_rows:
        return

    summary.collected += len(document_rows)

    doc_result = duckdb_store.upsert_filing_documents(con, document_rows)
    summary.inserted += doc_result.inserted
    summary.updated += doc_result.updated
    summary.deduped += doc_result.deduped

    section_result = duckdb_store.upsert_filing_sections(con, section_rows)
    summary.bump("sections_offered", len(section_rows))
    summary.bump("sections_inserted", section_result.inserted)
    summary.bump("sections_updated", section_result.updated)
    summary.bump("sections_deduped", section_result.deduped)

    parquet.write_records(
        config.paths.parquet_dir,
        "filing_documents",
        document_rows,
        ["accession_number", "document_name"],
    )
    parquet.write_records(
        config.paths.parquet_dir,
        "filing_sections",
        section_rows,
        ["accession_number", "item_code"],
    )


def ingest_documents(
    config: Config,
    *,
    tickers: list[str] | None = None,
    forms: list[str] | None = None,
    limit: int | None = None,
    transport: Any = None,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    skip_ingested: bool = True,
) -> RunSummary:
    """Download and section the filings already recorded in ``filings``.

    Results are written every ``flush_every`` filings rather than once at the
    end. A full backfill runs for hours against a rate-limited SEC, so an
    end-of-run write would mean an interruption at filing 22,000 of 22,892
    persisted nothing at all. Flushing in batches bounds the loss from any
    interruption to at most one batch, and combined with ``skip_ingested`` in
    :func:`select_candidates` it makes the backfill restartable to completion.
    """
    with pipeline_run(config, "sec.ingest-documents") as (con, summary):
        candidates = select_candidates(
            con, tickers=tickers, forms=forms, limit=limit, skip_ingested=skip_ingested
        )
        summary.bump("candidates", len(candidates))

        document_rows: list[dict[str, Any]] = []
        section_rows: list[dict[str, Any]] = []

        with SECClient.from_config(config, transport=transport) as client:
            for candidate in candidates:
                accession = candidate.get("accession_number")
                try:
                    document, sections = _ingest_one(config, client, summary, candidate)
                except Exception as exc:  # one bad filing must not end the run
                    summary.bump("fetch_failed")
                    summary.note(f"fetch_failed:{accession}:{exc}")
                    continue
                if document is not None:
                    document_rows.append(document)
                    section_rows.extend(sections)

                if len(document_rows) >= flush_every:
                    _flush(config, con, summary, document_rows, section_rows)
                    document_rows = []
                    section_rows = []

        _flush(config, con, summary, document_rows, section_rows)
    return summary


__all__ = [
    "DEFAULT_FLUSH_EVERY",
    "SECTIONABLE_FORMS",
    "ingest_documents",
    "select_candidates",
]
