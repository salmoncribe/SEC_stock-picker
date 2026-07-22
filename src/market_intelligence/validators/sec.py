"""SEC record validation.

Validators never raise on data problems — they record anomalies on the record
via :meth:`ProvenanceModel.add_error`, escalating status to ``warning`` or
``rejected``. Rejected records are still returned (kept for audit) but the
collectors skip persisting them.
"""

from __future__ import annotations

from market_intelligence.schemas.sec import (
    CompanyRecord,
    FilingDocumentRecord,
    FilingRecord,
    FilingSectionRecord,
    IntegrityStatus,
    is_valid_accession,
)

_CIK_LENGTH = 10

# A 10-K/10-Q Item body this short is almost certainly a stray header match or
# a cross-reference stub rather than real disclosure; flagged, not dropped.
_MIN_SECTION_CHARS = 200


def _is_ten_digit_cik(cik: str | None) -> bool:
    return cik is not None and len(cik) == _CIK_LENGTH and cik.isdigit()


def validate_company(record: CompanyRecord) -> CompanyRecord:
    """Reject on a missing/malformed CIK; warn on missing ticker or name."""
    if not _is_ten_digit_cik(record.cik):
        record.add_error("invalid_cik", reject=True)
    if not record.ticker:
        record.add_error("missing_ticker")
    if not record.company_name:
        record.add_error("missing_company_name")
    return record


def validate_filing(record: FilingRecord, supported_forms: set[str]) -> FilingRecord:
    """Reject on a bad accession or CIK; warn on unsupported form / missing metadata."""
    if not is_valid_accession(record.accession_number):
        record.add_error("invalid_accession", reject=True)
    if not _is_ten_digit_cik(record.cik):
        record.add_error("invalid_cik", reject=True)
    if record.form not in supported_forms:
        record.add_error(f"unsupported_form:{record.form}")
    if record.filing_date is None:
        record.add_error("missing_filing_date")
    if not record.source_url and not record.filing_url:
        record.add_error("missing_source_url")
    return record


def validate_document(record: FilingDocumentRecord) -> FilingDocumentRecord:
    """Reject on a broken identity, empty payload, or failed integrity check.

    ``unverified`` (the filing index was unreachable) is a *warning*: the bytes
    are still preserved and hashed, so the download stays auditable even though
    nothing corroborates it. A positive contradiction — the index lists a
    different size, or omits the document entirely — is a hard reject.
    """
    if not is_valid_accession(record.accession_number):
        record.add_error("invalid_accession", reject=True)
    if not _is_ten_digit_cik(record.cik):
        record.add_error("invalid_cik", reject=True)
    if not record.document_name:
        record.add_error("missing_document_name", reject=True)
    if not record.byte_size:
        record.add_error("empty_document", reject=True)
    if not record.sha256:
        record.add_error("missing_sha256", reject=True)

    if record.integrity_status == IntegrityStatus.SIZE_MISMATCH.value:
        record.add_error(
            f"size_mismatch:declared={record.declared_size},received={record.byte_size}",
            reject=True,
        )
    elif record.integrity_status == IntegrityStatus.NOT_IN_INDEX.value:
        record.add_error("not_in_filing_index", reject=True)
    elif record.integrity_status == IntegrityStatus.UNVERIFIED.value:
        record.add_error("integrity_unverified")
    return record


def validate_section(record: FilingSectionRecord) -> FilingSectionRecord:
    """Reject on a broken identity or empty body; warn on a suspiciously short one."""
    if not is_valid_accession(record.accession_number):
        record.add_error("invalid_accession", reject=True)
    if not record.item_code:
        record.add_error("missing_item_code", reject=True)
    if record.char_count <= 0:
        record.add_error("empty_section", reject=True)
    elif record.char_count < _MIN_SECTION_CHARS:
        record.add_error(f"short_section:{record.char_count}")
    if not record.text_path:
        record.add_error("missing_text_path")
    if not record.text_sha256:
        record.add_error("missing_text_hash")
    return record


__all__ = [
    "validate_company",
    "validate_document",
    "validate_filing",
    "validate_section",
]
