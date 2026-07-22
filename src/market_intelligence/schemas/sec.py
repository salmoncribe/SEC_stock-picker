"""SEC EDGAR normalized record schemas and parsing helpers.

Four record types map onto the SEC DuckDB tables:

* :class:`CompanyRecord` — a company/ticker/CIK identity;
* :class:`FilingRecord` — a single EDGAR filing (metadata);
* :class:`FilingDocumentRecord` — the downloaded primary document for a filing;
* :class:`FilingSectionRecord` — one deterministically extracted Item section.

The parsing helpers turn SEC's raw JSON shapes (``company_tickers.json``, the
per-company ``submissions`` document, and a filing's ``index.json``) into flat
dicts the collectors can build records from. They are pure and side-effect free
so they can be unit tested offline against fixtures.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from market_intelligence.schemas.common import ProvenanceModel, Source

# Accession numbers look like ``0001045810-24-000029`` (10-2-6 digits).
ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")


class CompanyRecord(ProvenanceModel):
    """A company identity keyed on its 10-digit zero-padded CIK."""

    source: Source = Source.SEC
    company_id: str
    ticker: str | None
    company_name: str | None
    cik: str
    exchange: str | None = None
    is_active: bool | None = True
    first_seen_time: datetime | None = None
    last_seen_time: datetime | None = None


class FilingRecord(ProvenanceModel):
    """A single EDGAR filing keyed on its accession number."""

    source: Source = Source.SEC
    filing_id: str
    company_id: str | None
    cik: str
    accession_number: str
    form: str
    filing_date: date | None = None
    report_date: date | None = None
    acceptance_time: datetime | None = None
    primary_document: str | None = None
    filing_url: str | None = None


class IntegrityStatus(StrEnum):
    """Outcome of cross-checking a downloaded document against its filing index.

    ``verified``      -> the document is listed in ``index.json`` and the byte
                         count we received equals the size SEC declared.
    ``size_mismatch`` -> listed, but the byte counts disagree (hard reject).
    ``not_in_index``  -> the index was readable and does not list this document
                         (hard reject).
    ``unverified``    -> the index could not be fetched; the bytes are still
                         preserved and hashed, but nothing corroborates them.
    """

    VERIFIED = "verified"
    SIZE_MISMATCH = "size_mismatch"
    NOT_IN_INDEX = "not_in_index"
    UNVERIFIED = "unverified"


class FilingDocumentRecord(ProvenanceModel):
    """A single downloaded filing document, keyed on (accession, document_name).

    Carries both what we received (``byte_size``, ``sha256``) and what SEC
    declared (``declared_size``) so integrity is auditable after the fact.
    """

    source: Source = Source.SEC
    document_id: str
    filing_id: str | None = None
    company_id: str | None = None
    cik: str
    accession_number: str
    form: str
    document_name: str
    document_url: str | None = None
    document_type: str | None = None
    byte_size: int | None = None
    declared_size: int | None = None
    sha256: str | None = None
    raw_file_path: str | None = None
    content_type: str | None = None
    downloaded_time: datetime | None = None
    integrity_status: str = IntegrityStatus.UNVERIFIED.value
    section_count: int = 0


class FilingSectionRecord(ProvenanceModel):
    """One extracted Item section, keyed on (accession_number, item_code).

    The section body lives on disk (``text_path``); this record carries the
    addressing, size, and hash so a consumer can locate and verify the text
    without the database holding megabytes of prose.
    """

    source: Source = Source.SEC
    section_id: str
    document_id: str | None = None
    filing_id: str | None = None
    company_id: str | None = None
    cik: str
    accession_number: str
    form: str
    report_date: date | None = None
    item_code: str
    item_title: str | None = None
    section_order: int = 0
    char_count: int = 0
    word_count: int = 0
    text_path: str | None = None
    text_sha256: str | None = None
    preview: str | None = None
    extraction_method: str | None = None


def normalize_cik(value: str | int) -> str:
    """Coerce a CIK to its canonical 10-digit zero-padded string form.

    Accepts ints, plain digit strings, and a leading ``CIK`` prefix. Raises
    ``ValueError`` when the value is empty, non-numeric, or longer than 10
    digits.
    """
    text = str(value).strip()
    if text[:3].upper() == "CIK":
        text = text[3:].strip()
    if not text:
        raise ValueError(f"CIK is empty: {value!r}")
    if not text.isdigit():
        raise ValueError(f"CIK is not numeric: {value!r}")
    if len(text) > 10:
        raise ValueError(f"CIK has more than 10 digits: {value!r}")
    return text.zfill(10)


def is_valid_accession(value: str) -> bool:
    """True when ``value`` matches the ``##########-##-######`` accession form."""
    return isinstance(value, str) and ACCESSION_RE.match(value) is not None


def parse_ticker_map(data: Any) -> list[dict[str, str]]:
    """Normalize ``company_tickers.json`` into a flat list of rows.

    SEC ships either a dict-of-rows (``{"0": {...}, "1": {...}}``) or, in some
    mirrors, a list-of-rows. Each row carries ``cik_str``/``cik``, ``ticker``,
    and ``title``. Rows missing a usable CIK or ticker are skipped.
    """
    if isinstance(data, dict):
        rows: list[Any] = list(data.values())
    elif isinstance(data, list):
        rows = list(data)
    else:
        return []

    result: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cik_raw = row.get("cik_str", row.get("cik"))
        ticker = row.get("ticker")
        if cik_raw is None or ticker is None:
            continue
        try:
            cik = normalize_cik(cik_raw)
        except ValueError:
            continue
        title = row.get("title")
        result.append(
            {
                "cik": cik,
                "ticker": str(ticker),
                "title": "" if title is None else str(title),
            }
        )
    return result


def iter_submission_filings(submissions: dict[str, Any]) -> list[dict[str, Any]]:
    """Zip the parallel ``filings.recent`` arrays into per-filing dicts.

    Robust to missing or short arrays: the row count is driven by
    ``accessionNumber`` and every other column is read defensively.
    """
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    accession = recent.get("accessionNumber") or []
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    report_dates = recent.get("reportDate") or []
    acceptance = recent.get("acceptanceDateTime") or []
    primary_docs = recent.get("primaryDocument") or []
    primary_descs = recent.get("primaryDocDescription") or []

    def _at(seq: list[Any], index: int) -> Any:
        return seq[index] if index < len(seq) else None

    result: list[dict[str, Any]] = []
    for i in range(len(accession)):
        result.append(
            {
                "accession_number": accession[i],
                "form": _at(forms, i),
                "filing_date": _at(filing_dates, i),
                "report_date": _at(report_dates, i),
                "acceptance_time": _at(acceptance, i),
                "primary_document": _at(primary_docs, i),
                "primary_doc_description": _at(primary_descs, i),
            }
        )
    return result


def filter_by_forms(
    filings: list[dict[str, Any]], forms: Iterable[str] | None
) -> list[dict[str, Any]]:
    """Keep only filings whose ``form`` is in ``forms``; ``None`` keeps all."""
    if forms is None:
        return list(filings)
    wanted = set(forms)
    return [f for f in filings if f.get("form") in wanted]


def build_filing_url(cik: str, accession_number: str, primary_document: str | None) -> str:
    """Build the canonical EDGAR archive URL for a filing's primary document."""
    acc_nodash = accession_number.replace("-", "")
    doc = primary_document or ""
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"


def build_filing_index_url(cik: str, accession_number: str) -> str:
    """Build the EDGAR ``index.json`` URL listing every file in a filing folder."""
    acc_nodash = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/index.json"


def parse_filing_index(data: Any) -> list[dict[str, Any]]:
    """Flatten a filing's ``index.json`` into ``{name, type, size}`` rows.

    SEC nests the listing under ``directory.item``. ``size`` is returned as an
    ``int`` when it parses cleanly and ``None`` otherwise, so a malformed size
    degrades to "unknown" rather than poisoning the integrity comparison.
    """
    items = ((data or {}).get("directory") or {}).get("item") or []
    if not isinstance(items, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        size: int | None
        try:
            size = int(str(item.get("size")).strip())
        except (TypeError, ValueError):
            size = None
        rows.append({"name": str(name), "type": item.get("type"), "size": size})
    return rows


__all__ = [
    "ACCESSION_RE",
    "CompanyRecord",
    "FilingDocumentRecord",
    "FilingRecord",
    "FilingSectionRecord",
    "IntegrityStatus",
    "build_filing_index_url",
    "build_filing_url",
    "filter_by_forms",
    "is_valid_accession",
    "iter_submission_filings",
    "normalize_cik",
    "parse_filing_index",
    "parse_ticker_map",
]
