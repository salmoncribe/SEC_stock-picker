"""FRED (Federal Reserve Economic Data) normalized records + payload parsing.

Two record types mirror the ``economic_series`` / ``economic_observations``
DuckDB tables. Parsing helpers translate the raw FRED JSON envelopes into the
plain shapes the collector assembles into records — kept pure and side-effect
free so they are trivially testable.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from market_intelligence.schemas.common import ProvenanceModel, Source

# FRED encodes a missing observation as a single period.
MISSING_MARKER = "."


class SeriesRecord(ProvenanceModel):
    """Normalized FRED series metadata (one row of ``economic_series``)."""

    series_id: str
    title: str | None = None
    units: str | None = None
    units_short: str | None = None
    frequency: str | None = None
    frequency_short: str | None = None
    seasonal_adjustment: str | None = None
    seasonal_adjustment_short: str | None = None
    observation_start: date | None = None
    observation_end: date | None = None
    last_updated: str | None = None
    popularity: int | None = None
    notes: str | None = None
    source: Source = Source.FRED


class ObservationRecord(ProvenanceModel):
    """A single FRED observation (one row of ``economic_observations``)."""

    observation_id: str
    series_id: str
    observation_date: date
    value: float | None = None
    realtime_start: date | None = None
    realtime_end: date | None = None
    raw_file_path: str | None = None
    source: Source = Source.FRED


def parse_value(raw: str | None) -> float | None:
    """Parse a FRED observation value string.

    FRED encodes a missing value as ``"."``; empty/whitespace is also treated
    as missing. Everything else is parsed as a float.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text or text == MISSING_MARKER:
        return None
    return float(text)


def _parse_date(raw: str | None) -> date | None:
    """Parse an ISO ``YYYY-MM-DD`` date, tolerating missing/empty values."""
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    return date.fromisoformat(text)


def parse_observations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a FRED ``series/observations`` payload into plain dicts.

    Each element carries ``observation_date`` / ``realtime_start`` /
    ``realtime_end`` as ``datetime.date`` (or ``None``) and ``value`` as a
    float (or ``None`` for the FRED gap marker). Robust to missing keys.
    """
    observations = payload.get("observations") or []
    parsed: list[dict[str, Any]] = []
    for obs in observations:
        parsed.append(
            {
                "observation_date": _parse_date(obs.get("date")),
                "value": parse_value(obs.get("value")),
                "realtime_start": _parse_date(obs.get("realtime_start")),
                "realtime_end": _parse_date(obs.get("realtime_end")),
            }
        )
    return parsed


def parse_series_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the first series object from a FRED ``series`` payload.

    The FRED series endpoint wraps metadata in a ``seriess`` list; this returns
    the first element (or ``{}`` when absent/empty).
    """
    seriess = payload.get("seriess") or []
    if not seriess:
        return {}
    first = seriess[0]
    return dict(first) if isinstance(first, dict) else {}


__all__ = [
    "ObservationRecord",
    "SeriesRecord",
    "parse_observations",
    "parse_series_metadata",
    "parse_value",
]
