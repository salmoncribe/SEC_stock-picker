"""Validation rules for typed events (``schemas.events.EventRecord``).

Follows the platform contract: a problem is recorded on the record itself via
``add_error`` and classified ``warning`` (stored, flagged) or ``rejected``
(kept for audit, not persisted). Nothing is ever silently discarded.

The rules here exist to protect the two-clock discipline the whole signal
graph depends on (see ``schemas/events.py`` and
``docs/specs/2026-07-22-signal-graph-design.md`` §7, "Point-in-time
discipline"): ``available_time`` is the only clock a downstream feature or
label may key on, so an event whose ``available_time`` is missing, in the
future, or precedes its own ``event_time`` cannot be trusted at any point
downstream and is rejected rather than passed through with a caveat.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from market_intelligence.schemas.common import utcnow
from market_intelligence.schemas.events import EventRecord

# An amended or late-filed Form 4 can land well after the transaction date;
# SEC's own filing deadline is two business days, so anything past ~10
# calendar days is unusual enough to flag -- still usable (the information did
# eventually become public), but the delay itself is worth surfacing rather
# than treated as ordinary.
LATE_AVAILABILITY_THRESHOLD = timedelta(days=10)

_VALID_DIRECTIONS = (-1, 0, 1)


def validate_event(record: EventRecord, *, today: datetime | None = None) -> EventRecord:
    """Validate one typed event.

    ``today`` is the reference "now" instant (a timezone-aware
    ``datetime.datetime``, mirroring the ``today`` parameter naming used by
    ``validators.market``); defaults to :func:`schemas.common.utcnow`.

    Rejections:

    * ``event_type`` or ``event_key`` missing/blank -- these two fields are
      the natural key; without them the record cannot be identified, let
      alone deduped on re-collection.
    * ``available_time`` missing -- the point-in-time clock every downstream
      join depends on. A record that reached here with no ``available_time``
      usually means its source date was unparseable (see
      ``clients.insider.parse_sec_date``); silently keeping it around as a
      NULL would force every later query to remember to guard against it.
    * ``available_time`` (or ``event_time``) in the future -- a filing dated
      after "now" is a clock or parsing error, not a valid record.
    * ``available_time`` before ``event_time`` -- the public cannot know
      something before it happens. This ordering inversion means the two
      clocks got crossed somewhere upstream, which is exactly the failure
      mode that would otherwise leak future information into a backtest.

    Warnings:

    * ``event_time`` missing -- worth flagging, but ``event_time`` is never
      used as t=0 for anything (only ``available_time`` is), so its absence
      does not make the record unusable the way a missing ``available_time``
      does.
    * ``available_time`` more than ~10 days after ``event_time`` -- a
      late/amended filing; usable, but the information is stale relative to
      when the transaction actually happened.
    * ``direction`` not in ``(-1, 0, 1)``.
    * ``magnitude`` negative.
    """
    horizon = today or utcnow()

    if not record.event_type or not record.event_type.strip():
        record.add_error("event_type is required", reject=True)

    if not record.event_key or not record.event_key.strip():
        record.add_error("event_key is required", reject=True)

    if record.available_time is None:
        record.add_error("available_time is required (point-in-time clock)", reject=True)
    elif record.available_time > horizon:
        record.add_error(f"available_time {record.available_time} is in the future", reject=True)

    if record.event_time is None:
        record.add_error("event_time is missing or unparseable")
    elif record.event_time > horizon:
        record.add_error(f"event_time {record.event_time} is in the future", reject=True)

    if record.event_time is not None and record.available_time is not None:
        if record.available_time < record.event_time:
            record.add_error(
                f"available_time {record.available_time} precedes event_time "
                f"{record.event_time}: the two clocks are crossed",
                reject=True,
            )
        elif record.available_time - record.event_time > LATE_AVAILABILITY_THRESHOLD:
            delay = record.available_time - record.event_time
            record.add_error(
                f"available_time is {delay.days} days after event_time: "
                "a late/amended filing, usable but stale"
            )

    if record.direction is not None and record.direction not in _VALID_DIRECTIONS:
        record.add_error(f"direction {record.direction} is not in (-1, 0, 1)")

    if record.magnitude is not None and record.magnitude < 0:
        record.add_error(f"magnitude {record.magnitude} is negative")

    return record


__all__ = ["LATE_AVAILABILITY_THRESHOLD", "validate_event"]
