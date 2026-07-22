"""Offline tests for ``validators.events.validate_event``.

Fully offline: constructs ``EventRecord`` instances directly, no network, no
database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from market_intelligence.schemas.events import EventRecord, EventType
from market_intelligence.validators.events import LATE_AVAILABILITY_THRESHOLD, validate_event

NOW = datetime(2024, 6, 1, tzinfo=UTC)


def _record(**overrides: object) -> EventRecord:
    defaults: dict[str, object] = {
        "event_id": "evt-1",
        "event_type": EventType.INSIDER_TRANSACTION,
        "event_key": "SK-1",
        "event_time": datetime(2024, 1, 15, tzinfo=UTC),
        "available_time": datetime(2024, 1, 17, tzinfo=UTC),
        "direction": 1,
        "magnitude": 1000.0,
    }
    defaults.update(overrides)
    return EventRecord(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# happy path                                                                   #
# --------------------------------------------------------------------------- #
def test_valid_record_passes() -> None:
    record = validate_event(_record(), today=NOW)
    assert record.validation_status == "valid"
    assert not record.is_rejected


# --------------------------------------------------------------------------- #
# reject: missing natural key                                                  #
# --------------------------------------------------------------------------- #
def test_missing_event_type_is_rejected() -> None:
    record = validate_event(_record(event_type=""), today=NOW)
    assert record.is_rejected
    assert any("event_type" in e for e in record.validation_errors)


def test_missing_event_key_is_rejected() -> None:
    record = validate_event(_record(event_key=""), today=NOW)
    assert record.is_rejected
    assert any("event_key" in e for e in record.validation_errors)


def test_blank_event_key_is_rejected() -> None:
    record = validate_event(_record(event_key="   "), today=NOW)
    assert record.is_rejected


# --------------------------------------------------------------------------- #
# reject: missing / unparseable available_time                                #
# --------------------------------------------------------------------------- #
def test_missing_available_time_is_rejected() -> None:
    """The point-in-time clock is mandatory -- an unparseable FILING_DATE
    upstream (clients.insider.parse_sec_date returning None) must not pass
    through as an unusable NULL.
    """
    record = validate_event(_record(available_time=None), today=NOW)
    assert record.is_rejected
    assert any("available_time is required" in e for e in record.validation_errors)


# --------------------------------------------------------------------------- #
# warn: missing event_time                                                     #
# --------------------------------------------------------------------------- #
def test_missing_event_time_is_a_warning_not_a_rejection() -> None:
    record = validate_event(_record(event_time=None), today=NOW)
    assert not record.is_rejected
    assert record.validation_status == "warning"
    assert any("event_time is missing" in e for e in record.validation_errors)


# --------------------------------------------------------------------------- #
# reject: either clock in the future                                           #
# --------------------------------------------------------------------------- #
def test_available_time_in_the_future_is_rejected() -> None:
    record = validate_event(
        _record(
            event_time=datetime(2024, 6, 2, tzinfo=UTC),
            available_time=datetime(2024, 6, 3, tzinfo=UTC),
        ),
        today=NOW,
    )
    assert record.is_rejected


def test_event_time_in_the_future_is_rejected() -> None:
    record = validate_event(
        _record(
            event_time=datetime(2024, 6, 3, tzinfo=UTC),
            available_time=datetime(2024, 6, 4, tzinfo=UTC),
        ),
        today=NOW,
    )
    assert record.is_rejected


# --------------------------------------------------------------------------- #
# reject: inverted clocks                                                      #
# --------------------------------------------------------------------------- #
def test_available_time_before_event_time_is_rejected() -> None:
    """The public cannot know something before it happens -- the two clocks
    crossed somewhere upstream, which is exactly the leak this platform's
    point-in-time discipline exists to catch.
    """
    record = validate_event(
        _record(
            event_time=datetime(2024, 1, 17, tzinfo=UTC),
            available_time=datetime(2024, 1, 15, tzinfo=UTC),
        ),
        today=NOW,
    )
    assert record.is_rejected
    assert any("crossed" in e for e in record.validation_errors)


def test_equal_clocks_are_not_inverted() -> None:
    same = datetime(2024, 1, 15, tzinfo=UTC)
    record = validate_event(_record(event_time=same, available_time=same), today=NOW)
    assert not record.is_rejected


# --------------------------------------------------------------------------- #
# warn: late/amended filing                                                    #
# --------------------------------------------------------------------------- #
def test_late_filing_beyond_threshold_warns_but_is_not_rejected() -> None:
    event_time = datetime(2024, 1, 1, tzinfo=UTC)
    available_time = event_time + LATE_AVAILABILITY_THRESHOLD + timedelta(days=1)
    record = validate_event(
        _record(event_time=event_time, available_time=available_time), today=NOW
    )
    assert not record.is_rejected
    assert record.validation_status == "warning"
    assert any("late/amended filing" in e for e in record.validation_errors)


def test_filing_within_threshold_does_not_warn() -> None:
    event_time = datetime(2024, 1, 1, tzinfo=UTC)
    available_time = event_time + LATE_AVAILABILITY_THRESHOLD - timedelta(days=1)
    record = validate_event(
        _record(event_time=event_time, available_time=available_time), today=NOW
    )
    assert record.validation_status == "valid"


# --------------------------------------------------------------------------- #
# warn: direction / magnitude                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_direction", [2, -5, 100])
def test_direction_outside_range_warns(bad_direction: int) -> None:
    record = validate_event(_record(direction=bad_direction), today=NOW)
    assert not record.is_rejected
    assert record.validation_status == "warning"


@pytest.mark.parametrize("good_direction", [-1, 0, 1])
def test_direction_in_range_does_not_warn(good_direction: int) -> None:
    record = validate_event(_record(direction=good_direction), today=NOW)
    assert record.validation_status == "valid"


def test_negative_magnitude_warns() -> None:
    record = validate_event(_record(magnitude=-500.0), today=NOW)
    assert not record.is_rejected
    assert record.validation_status == "warning"
    assert any("negative" in e for e in record.validation_errors)


def test_none_magnitude_is_fine() -> None:
    """A missing magnitude (e.g. a grant with no price) is the truth, not an
    error -- see schemas/events.py and collectors/insider.py._build_event.
    """
    record = validate_event(_record(magnitude=None), today=NOW)
    assert record.validation_status == "valid"


# --------------------------------------------------------------------------- #
# default `today`                                                              #
# --------------------------------------------------------------------------- #
def test_default_today_is_now() -> None:
    """Without an explicit `today`, a record with far-future times is rejected."""
    far_future = datetime(2999, 1, 1, tzinfo=UTC)
    record = validate_event(_record(event_time=far_future, available_time=far_future))
    assert record.is_rejected
