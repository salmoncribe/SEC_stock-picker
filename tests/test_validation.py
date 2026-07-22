"""Cross-cutting validation behaviour (SEC + FRED validators + common base)."""

from __future__ import annotations

import json
from datetime import date

from market_intelligence.schemas.common import ProvenanceModel, Source, ValidationStatus
from market_intelligence.schemas.fred import ObservationRecord, SeriesRecord
from market_intelligence.schemas.sec import CompanyRecord, FilingRecord
from market_intelligence.validators.fred import validate_observation, validate_series
from market_intelligence.validators.sec import validate_company, validate_filing


# --------------------------------------------------------------------------- #
# common base                                                                  #
# --------------------------------------------------------------------------- #
def test_add_error_escalates_valid_to_warning_then_reject() -> None:
    record = ProvenanceModel(source=Source.SEC)
    assert record.validation_status == ValidationStatus.VALID
    record.add_error("soft problem")
    assert record.validation_status == ValidationStatus.WARNING
    record.add_error("hard problem", reject=True)
    assert record.validation_status == ValidationStatus.REJECTED
    assert record.is_rejected is True


def test_to_row_serializes_errors_as_json_string() -> None:
    record = ProvenanceModel(source=Source.FRED)
    record.add_error("one")
    record.add_error("two")
    row = record.to_row()
    assert isinstance(row["validation_errors"], str)
    assert json.loads(row["validation_errors"]) == ["one", "two"]
    assert row["source"] == "fred"


# --------------------------------------------------------------------------- #
# SEC validators                                                               #
# --------------------------------------------------------------------------- #
def test_validate_company_rejects_malformed_cik() -> None:
    record = CompanyRecord(company_id="x", cik="12", ticker="AAPL", company_name="Apple")
    validate_company(record)
    assert record.is_rejected is True


def test_validate_company_warns_on_missing_ticker() -> None:
    record = CompanyRecord(company_id="x", cik="0000320193", ticker=None, company_name="Apple")
    validate_company(record)
    assert record.validation_status == ValidationStatus.WARNING
    assert record.is_rejected is False


def test_validate_filing_rejects_malformed_accession() -> None:
    record = FilingRecord(
        filing_id="x", company_id="c", cik="0000320193", accession_number="BAD", form="10-K"
    )
    validate_filing(record, {"10-K"})
    assert record.is_rejected is True


def test_validate_filing_warns_on_unsupported_form() -> None:
    record = FilingRecord(
        filing_id="x",
        company_id="c",
        cik="0000320193",
        accession_number="0000320193-24-000001",
        form="XYZ",
        filing_date=date(2024, 1, 1),
        filing_url="https://sec.gov/x",
    )
    validate_filing(record, {"10-K", "10-Q"})
    assert record.validation_status == ValidationStatus.WARNING
    assert record.is_rejected is False


def test_validate_filing_valid_case() -> None:
    record = FilingRecord(
        filing_id="x",
        company_id="c",
        cik="0000320193",
        accession_number="0000320193-24-000001",
        form="10-K",
        filing_date=date(2024, 1, 1),
        filing_url="https://sec.gov/x",
        source_url="https://sec.gov/x",
    )
    validate_filing(record, {"10-K"})
    assert record.validation_status == ValidationStatus.VALID


# --------------------------------------------------------------------------- #
# FRED validators                                                              #
# --------------------------------------------------------------------------- #
def test_validate_observation_flags_missing_value_as_warning() -> None:
    record = ObservationRecord(
        observation_id="x", series_id="DGS10", observation_date=date(2026, 7, 15), value=None
    )
    validate_observation(record)
    assert record.validation_status == ValidationStatus.WARNING
    assert record.is_rejected is False
    assert any("missing" in err.lower() for err in record.validation_errors)


def test_validate_observation_valid_value() -> None:
    record = ObservationRecord(
        observation_id="x", series_id="DGS10", observation_date=date(2026, 7, 14), value=4.45
    )
    validate_observation(record)
    assert record.validation_status == ValidationStatus.VALID


def test_validate_series_rejects_missing_id() -> None:
    record = SeriesRecord(series_id="")
    validate_series(record)
    assert record.is_rejected is True
