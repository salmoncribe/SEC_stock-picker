"""Common schema primitives shared by every normalized record.

``ProvenanceModel`` is the base for all normalized records. It carries the
provenance / validation fields the platform tracks uniformly, and a
``to_row()`` helper that flattens a record into a DuckDB-/Parquet-friendly
dict (enum -> value, validation_errors -> JSON string).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


def utcnow() -> datetime:
    """Timezone-aware current time in UTC (the platform's internal clock)."""
    return datetime.now(UTC)


class Source(StrEnum):
    """Origin of a record."""

    SEC = "sec"
    FRED = "fred"
    MARKET = "market"
    # Reference data that is neither a filing, a macro series, nor a price:
    # index membership, sector maps, and similar lookup tables.
    REFERENCE = "reference"
    # Values computed by this platform from other stored records (e.g. returns
    # derived from prices). Distinguishes derived rows from ingested ones.
    DERIVED = "derived"


class ValidationStatus(StrEnum):
    """Outcome of validating a record.

    ``valid``    -> stored, no issues.
    ``warning``  -> stored, but a non-fatal anomaly was recorded.
    ``rejected`` -> a required invariant failed; kept for audit, flagged.
    """

    VALID = "valid"
    WARNING = "warning"
    REJECTED = "rejected"


class ProvenanceModel(BaseModel):
    """Base for all normalized records.

    ``use_enum_values=True`` means enum fields are stored as their string
    values, so ``model_dump()`` / ``to_row()`` produce plain strings ready
    for the database. Comparisons against the enum members still work because
    ``ValidationStatus`` / ``Source`` subclass ``str``.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid", validate_assignment=False)

    source: Source
    source_url: str | None = None
    source_record_id: str | None = None
    event_time: datetime | None = None
    published_time: datetime | None = None
    collected_time: datetime = Field(default_factory=utcnow)
    content_hash: str | None = None
    schema_version: str = SCHEMA_VERSION
    validation_status: str = ValidationStatus.VALID.value
    validation_errors: list[str] = Field(default_factory=list)

    def add_error(self, message: str, *, reject: bool = False) -> None:
        """Record a validation problem and escalate the status.

        A ``reject=True`` problem always wins. Otherwise a first problem
        downgrades ``valid`` -> ``warning`` but never overrides a prior
        ``rejected``.
        """
        self.validation_errors.append(message)
        if reject:
            self.validation_status = ValidationStatus.REJECTED.value
        elif self.validation_status == ValidationStatus.VALID.value:
            self.validation_status = ValidationStatus.WARNING.value

    @property
    def is_rejected(self) -> bool:
        return self.validation_status == ValidationStatus.REJECTED.value

    def to_row(self) -> dict[str, Any]:
        """Flatten to a storage-friendly dict.

        - Enum members become their ``.value`` (defensive; ``use_enum_values``
          already handles the declared fields).
        - ``validation_errors`` is serialized to a JSON string so it maps to a
          single TEXT column and a single Parquet column.
        """
        data: dict[str, Any] = self.model_dump()
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value
        data["validation_errors"] = json.dumps(list(self.validation_errors))
        return data


__all__ = [
    "SCHEMA_VERSION",
    "ProvenanceModel",
    "Source",
    "ValidationStatus",
    "utcnow",
]
