# Filing-Document Ingestion — Design

**Date:** 2026-07-22
**Status:** Approved
**Scope:** Expand the platform from filing *metadata* into filing *document* ingestion.
**Non-goal:** No trading signals, no AI extraction. This phase produces clean section
records that a later AI-extraction phase will consume.

---

## 1. Problem

Phase 1 collects filing **metadata** (1,003 filings across 12 CIKs) but never downloads
the filings themselves. The raw store holds only submission JSON and the ticker map.
This phase adds: document download → byte preservation → integrity validation →
deterministic section extraction → clean section records.

## 2. Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| A | Section extractor | **bs4/lxml + deterministic Item splitter** | Reuses the platform's paced client, raw store, and provenance. Only the fiddly HTML→text step is delegated to boring OSS. Fully deterministic and auditable. |
| B | Ingestion depth | **All 10-K/10-Q already in DB** (~235 docs) | Most complete first run; the universe is already collected. |
| C | Section text location | **Files in `data/normalized/` + metadata in DuckDB/Parquet** | Uses the existing (empty) `normalized_dir`; keeps DB lean; mirrors the raw/normalized split. |

Section extraction targets **10-K and 10-Q only** — they have a stable, official Item
taxonomy that supports deterministic splitting. 8-K/S-1 documents are out of scope here.

## 3. Architecture

```
filings (existing, 10-K/10-Q)
  └─> SECClient.fetch_filing_index()      # index.json — declared name + size
  └─> SECClient.fetch_filing_document()   # the primary document bytes
        └─> raw.save_raw(...)             # byte-for-byte, hash-named, idempotent
              └─> integrity validation     # sha256 + index cross-check
                    └─> filing_documents  (DuckDB + Parquet)
                          └─> extraction.sections.extract_sections()   # pure, deterministic
                                └─> normalized.write_section_text()    # data/normalized/sections/
                                      └─> filing_sections (DuckDB + Parquet)
```

New/changed modules:

- `clients/sec.py` — `fetch_filing_index()`, `fetch_filing_document()`
- `extraction/sections.py` — **new**, pure: `html_to_text()`, `extract_sections()`
- `storage/normalized.py` — **new**: atomic section-text writes
- `schemas/sec.py` — `FilingDocumentRecord`, `FilingSectionRecord`
- `validators/sec.py` — `validate_document()`, `validate_section()`
- `database.py` — `filing_documents`, `filing_sections` tables + column migration
- `storage/duckdb.py` — upsert wrappers; `UpsertResult.deduped`
- `collectors/documents.py` — **new**: orchestration + full count accounting
- `cli.py` — `sec ingest-documents`, `reconcile`

## 4. Schema

**`filing_documents`** — natural key `(accession_number, document_name)`

`document_id` (PK), `filing_id`, `company_id`, `cik`, `accession_number`, `form`,
`document_name`, `document_url`, `document_type`, `byte_size`, `declared_size`,
`sha256`, `raw_file_path`, `content_type`, `downloaded_time`, `integrity_status`,
`section_count`, + provenance columns.

**`filing_sections`** — natural key `(accession_number, item_code)`

`section_id` (PK), `document_id`, `filing_id`, `company_id`, `cik`, `accession_number`,
`form`, `report_date`, `item_code`, `item_title`, `section_order`, `char_count`,
`word_count`, `text_path`, `text_sha256`, `preview`, `extraction_method`,
+ provenance columns.

## 5. Integrity validation

`integrity_status` is one of:

| Status | Meaning |
|--------|---------|
| `verified` | Document appears in the filing's `index.json` and the downloaded byte count equals the declared size. |
| `size_mismatch` | Present in the index, but byte count differs from declared size → **rejected**. |
| `not_in_index` | Index fetched, document absent from it → **rejected**. |
| `unverified` | Index unavailable (404/error); bytes preserved and hashed, but uncorroborated → warning, still stored. |

Every document also records its own `sha256`, so any later re-download is provably
identical or provably different.

## 6. Count model & reconciliation

`RunSummary` gains `downloaded`, `stored`, `skipped`; `UpsertResult` gains `deduped`
(previously a silent gap — `_dedupe_last_wins()` could collapse rows with no counter
explaining why `inserted + updated < collected`).

Enforced identities:

```
# Download stage
candidates   = downloaded + fetch_failed
downloaded   = stored + skipped              # skipped = bytes already in raw store
downloaded   = doc_inserted + doc_updated + doc_rejected + doc_deduped

# Section stage
sections_extracted = sec_inserted + sec_updated + sec_rejected + sec_deduped
```

Term definitions (these are what the final report explains):

- **collected** — records built from the source and offered to storage.
- **stored** — *new* raw files written to disk this run (`was_new=True`).
- **skipped** — fetched bytes already byte-identical in the raw store; no write. This
  is the idempotency signal, not an error.
- **inserted / updated** — DuckDB rows created vs. modified by the upsert.
- **deduped** — rows collapsed within one batch sharing a natural key.
- **rejected** — failed a hard invariant; kept for audit, not persisted.

`market-intelligence reconcile` recomputes these from DuckDB + the filesystem and
asserts each identity, printing any residual.

## 7. Testing

- Pure unit tests for `extract_sections()` against trimmed real 10-K/10-Q fixtures.
- Offline `respx` tests for the new client methods and the collector.
- Integrity tests: size mismatch → rejected; missing index → `unverified`.
- Idempotency test: running the collector twice yields `stored=0, skipped=N` on the
  second pass and no duplicate rows.
- Reconciliation test asserting every identity above.

## 8. Out of scope

Trading signals, AI/LLM extraction, XBRL financial facts, exhibits beyond the primary
document, 8-K/S-1 sectioning.
