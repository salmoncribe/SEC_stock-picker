"""Event-study samples: one (event, target, horizon) observation.

This is the contract between event production and every statistic computed from
it. The impact table reads these rows; a future model reads the same rows. That
single contract is deliberate -- it means adding a model later is a new consumer
rather than a new pipeline, and it means both are measured on identical inputs.

Each record stores its own derivation, not just its answer: ``available_on``
(when the public could first know), ``t0`` (the first tradeable day after that),
and ``window_end``. A stored row can therefore be re-checked for leakage
afterwards instead of being taken on trust -- which matters, because leakage is
invisible in the result and obvious in the derivation.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from pydantic import Field

from market_intelligence.schemas.common import ProvenanceModel, Source

#: Edge id used when an event's target is the company the event happened at --
#: an insider trade predicting its own issuer. Real graph edges carry their own
#: ids, so the same table holds the positive control and the propagation
#: hypothesis without branching the schema.
SELF_EDGE = "self"


class EventSampleRecord(ProvenanceModel):
    """One event scored against one target company over one horizon."""

    sample_id: str
    event_id: str
    edge_id: str = SELF_EDGE
    event_type: str | None = None
    event_subtype: str | None = None
    source_ticker: str | None = None
    target_ticker: str | None = None
    horizon_days: int
    available_on: date | None = None
    t0: date | None = None
    window_end: date | None = None
    forward_abnormal_return: float | None = None
    magnitude: float | None = None
    direction: int | None = None
    split: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    source: Source = Source.DERIVED

    def to_row(self) -> dict[str, Any]:
        data = super().to_row()
        data["features"] = json.dumps(self.features, sort_keys=True, default=str)
        return data


__all__ = ["SELF_EDGE", "EventSampleRecord"]
