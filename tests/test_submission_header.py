"""Offline tests for complete-submission acceptance-time parsing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from market_intelligence.extraction.submission_header import (
    parse_acceptance_datetime,
    parse_submission_header,
)


def test_parses_complete_submission_header_and_preserves_raw_value() -> None:
    raw = (Path(__file__).parent / "fixtures" / "sec_complete_submission_header.txt").read_bytes()

    header = parse_submission_header(raw)

    assert header.acceptance_datetime_raw == "20240310013000"
    # Before the 2024 spring-forward transition: EST is UTC-5.
    assert header.acceptance_datetime_utc == datetime(2024, 3, 10, 6, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("20240115123000", datetime(2024, 1, 15, 17, 30, tzinfo=UTC)),
        # The same date after spring-forward: EDT is UTC-4.
        ("20240310033000", datetime(2024, 3, 10, 7, 30, tzinfo=UTC)),
        # Ambiguous autumn hour deterministically follows ZoneInfo's fold=0.
        ("20241103013000", datetime(2024, 11, 3, 5, 30, tzinfo=UTC)),
    ],
)
def test_converts_new_york_civil_time_to_utc_across_dst(raw_value: str, expected: datetime) -> None:
    assert parse_acceptance_datetime(raw_value) == expected


def test_missing_or_invalid_header_is_not_silently_guessed() -> None:
    with pytest.raises(ValueError, match="no ACCEPTANCE-DATETIME"):
        parse_submission_header(b"<SEC-HEADER>\nACCESSION NUMBER: 1")
    with pytest.raises(ValueError, match="exactly 14 digits"):
        parse_acceptance_datetime("20240101")
