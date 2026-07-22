"""Typed events: one thing that happened at one company, at a known time.

An event is the trigger half of the propagation hypothesis -- the "A filed
something" that is expected to move linked company B. Every producer (insider
transactions, 8-K item codes, later LLM-extracted disclosures) writes this same
record type, so the dataset builder downstream has exactly one contract to
consume regardless of where an event came from.

**The two clocks are the important part of this module.** ``event_time`` is
when something happened in the world; ``available_time`` is when the public
could first have known it. They are not the same, and only the second one may
ever be used as t=0 for a feature or a label. An insider's trade date precedes
its Form 4 filing by up to two business days, so keying on ``event_time`` would
be scoring trades against information that did not exist yet -- a backtest that
looks brilliant and is unreproducible with real money.

Event types are open strings backed by a registry, not an enum, so adding a new
kind of event is a config entry and a re-run rather than a schema migration.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import Field

from market_intelligence.schemas.common import ProvenanceModel, Source

# Direction of the economic action an event represents. Kept as a small int
# rather than a string so it multiplies cleanly into a signed expected impact.
DIRECTION_POSITIVE = 1
DIRECTION_NEGATIVE = -1
DIRECTION_NEUTRAL = 0


class EventType:
    """Known event-type identifiers.

    A registry of the types built so far, not a closed set: ``events.event_type``
    is a free string and a producer may introduce a new one without touching
    this class or the schema.
    """

    INSIDER_TRANSACTION = "insider_transaction"
    FILING_ITEM = "filing_item"


class InsiderTransactionCode:
    """SEC Form 4 transaction codes, and which of them carry information.

    This distinction decides whether the signal works at all. Only 5.5% of
    Form 4 transactions are open-market purchases; the bulk are compensation
    mechanics:

    ``P`` open-market purchase -- the informative one. An insider buying with
          their own money at market price has one obvious reason to do it.
    ``S`` open-market sale -- weakly informative at best. Insiders sell for
          diversification, tax, tuition, a house; the reasons are many and
          mostly uninformative, which is why the literature finds purchases
          predict returns far more reliably than sales.
    ``A`` grant/award, ``F`` tax withholding, ``M`` option exercise -- pure
          compensation mechanics, and worse than neutral for signal purposes:
          they fire on *scheduled vesting dates*, so pooling them with ``P``
          both dilutes the informative events and adds a systematic,
          news-uncorrelated component that drags any measured effect toward
          zero.

    Producers must therefore set ``event_subtype`` to the raw code and let the
    validation gate measure each code separately. Never collapse these into a
    single "insider activity" event.
    """

    PURCHASE = "P"
    SALE = "S"
    GRANT = "A"
    TAX_WITHHOLDING = "F"
    OPTION_EXERCISE = "M"
    GIFT = "G"
    CONVERSION = "C"
    DISPOSITION_TO_ISSUER = "D"

    #: Codes reflecting a deliberate, priced decision by the insider.
    DISCRETIONARY = frozenset({PURCHASE, SALE})

    #: Codes that are compensation plumbing rather than a market decision.
    MECHANICAL = frozenset({GRANT, TAX_WITHHOLDING, OPTION_EXERCISE, DISPOSITION_TO_ISSUER})


class EventRecord(ProvenanceModel):
    """One typed event (a row of ``events``).

    ``event_key`` is the producer's stable identifier for this event within its
    type -- for insider transactions, the SEC's own per-transaction surrogate
    key. Together with ``event_type`` it forms the natural key, so re-running a
    producer updates rather than duplicates.

    ``magnitude`` is the event's size in whatever unit the producer documents in
    ``payload`` (dollar value for an insider trade). ``direction`` is +1/-1/0 for
    the sign of the expected effect, kept numeric so it multiplies into a signed
    impact without a lookup.
    """

    event_id: str
    event_type: str
    event_key: str
    company_id: str | None = None
    cik: str | None = None
    ticker: str | None = None
    event_subtype: str | None = None
    accession_number: str | None = None
    filing_id: str | None = None
    event_time: datetime | None = None
    available_time: datetime | None = None
    magnitude: float | None = None
    direction: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    extraction_method: str | None = None
    extraction_confidence: float | None = None
    source: Source = Source.SEC

    def to_row(self) -> dict[str, Any]:
        """Flatten for storage, serializing ``payload`` to a JSON string."""
        data = super().to_row()
        data["payload"] = json.dumps(self.payload, sort_keys=True, default=str)
        return data


__all__ = [
    "DIRECTION_NEGATIVE",
    "DIRECTION_NEUTRAL",
    "DIRECTION_POSITIVE",
    "EventRecord",
    "EventType",
    "InsiderTransactionCode",
]
