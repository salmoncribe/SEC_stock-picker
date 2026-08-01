"""Parse the small, authoritative header in an EDGAR complete submission.

EDGAR's ``ACCEPTANCE-DATETIME`` is a fourteen-digit *New York civil time*, not
an ISO timestamp.  Keeping its raw value alongside the UTC conversion makes a
later audit independent of parser implementation details.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
_ACCEPTANCE_DATETIME_RE = re.compile(r"(?im)^\s*<ACCEPTANCE-DATETIME>\s*(?P<value>\d{14})\s*$")


@dataclass(frozen=True, slots=True)
class SubmissionHeader:
    """The acceptance timestamp recovered from a complete submission header."""

    acceptance_datetime_raw: str
    acceptance_datetime_utc: datetime


def parse_acceptance_datetime(value: str) -> datetime:
    """Convert a SEC ``YYYYMMDDHHMMSS`` acceptance value to aware UTC.

    SEC submission headers label this value in ``America/New_York``.  During
    the autumn repeated hour, the source supplies no offset/fold indicator, so
    Python's zoneinfo convention (the first occurrence, ``fold=0``) is used
    deterministically.  That limitation remains visible through the preserved
    raw value rather than being hidden by a guessed offset.
    """
    if not re.fullmatch(r"\d{14}", value):
        raise ValueError("ACCEPTANCE-DATETIME must contain exactly 14 digits")
    try:
        local = datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=NEW_YORK)
    except ValueError as exc:
        raise ValueError(f"invalid ACCEPTANCE-DATETIME: {value!r}") from exc
    return local.astimezone(UTC)


def parse_submission_header(raw_submission: bytes | str) -> SubmissionHeader:
    """Extract and convert ``ACCEPTANCE-DATETIME`` from complete-submission bytes.

    The complete submission must be fetched before its primary document.  This
    function intentionally accepts bytes and uses ASCII-compatible decoding so
    a non-text exhibit elsewhere in the submission cannot change the header
    timestamp or prevent the raw package from being retained.
    """
    text = raw_submission.decode("latin-1") if isinstance(raw_submission, bytes) else raw_submission
    match = _ACCEPTANCE_DATETIME_RE.search(text)
    if match is None:
        raise ValueError("complete submission has no ACCEPTANCE-DATETIME header")
    raw_value = match.group("value")
    return SubmissionHeader(raw_value, parse_acceptance_datetime(raw_value))


__all__ = [
    "NEW_YORK",
    "SubmissionHeader",
    "parse_acceptance_datetime",
    "parse_submission_header",
]
