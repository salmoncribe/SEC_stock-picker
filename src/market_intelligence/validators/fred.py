"""FRED record validators.

These mutate a record's ``validation_status`` / ``validation_errors`` and never
raise on data problems, so anomalies are recorded (and auditable) rather than
silently dropped. A missing required key rejects the record; softer gaps (a
missing title, or the FRED ``"."`` value marker) are recorded as warnings.
"""

from __future__ import annotations

from market_intelligence.schemas.fred import ObservationRecord, SeriesRecord


def validate_series(record: SeriesRecord) -> SeriesRecord:
    """Validate a series metadata record in place and return it."""
    if not record.series_id:
        record.add_error("missing_series_id", reject=True)
    if not record.title:
        record.add_error("missing_title")
    if not record.units:
        record.add_error("missing_units")
    return record


def validate_observation(record: ObservationRecord) -> ObservationRecord:
    """Validate an observation record in place and return it.

    A missing value (the FRED ``"."`` gap) is a warning, not a drop — the row
    is still stored so the gap is visible downstream.
    """
    if not record.series_id:
        record.add_error("missing_series_id", reject=True)
    if record.observation_date is None:
        record.add_error("missing_observation_date", reject=True)
    if record.value is None:
        record.add_error("missing_value", reject=False)
    return record


__all__ = ["validate_observation", "validate_series"]
