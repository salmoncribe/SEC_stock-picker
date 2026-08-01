"""Append-only, storage-neutral contracts for validation experiments.

The registry records every threshold/model/feature attempt as a new immutable
event.  It deliberately has no database dependency: an adapter can persist
``ExperimentRecord`` values in DuckDB, JSONL, or a remote audit store without
changing the rules that prevent sealed-result tuning.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from market_intelligence.hashing import sha256_json


class ExperimentStage(StrEnum):
    REGISTERED = "registered"
    CALIBRATED = "calibrated"
    SEALED_EVALUATED = "sealed_evaluated"


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _freeze_snapshot(value: Any) -> Any:
    """Recursively detach mutable caller input from a record snapshot."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_snapshot(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_snapshot(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_snapshot(item) for item in value)
    return deepcopy(value)


def _plain_snapshot(value: Any) -> Any:
    """Convert a frozen snapshot back to canonical JSON-friendly values."""
    if isinstance(value, Mapping):
        return {key: _plain_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple | frozenset):
        return [_plain_snapshot(item) for item in value]
    return value


@dataclass(frozen=True)
class ExperimentRecord:
    """One immutable state transition for a strategy-family experiment.

    A record never mutates a prior choice.  ``parameters`` is a snapshot, not
    a live configuration pointer, and all data/code/cost fingerprints are
    retained so an apparent result can be reproduced and audited later.
    """

    record_id: str
    experiment_id: str
    strategy_family: str
    stage: ExperimentStage
    recorded_at: datetime
    hypothesis: str
    code_hash: str
    configuration_hash: str
    data_snapshot_hash: str
    cost_model_version: str
    feature_version: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    parent_record_hash: str | None = None
    sealed_window_start: datetime | None = None
    record_hash: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "record_id",
            "experiment_id",
            "strategy_family",
            "hypothesis",
            "code_hash",
            "configuration_hash",
            "data_snapshot_hash",
            "cost_model_version",
            "feature_version",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        _aware(self.recorded_at, "recorded_at")
        if self.sealed_window_start is not None:
            _aware(self.sealed_window_start, "sealed_window_start")
        if self.stage == ExperimentStage.REGISTERED:
            if self.metrics:
                raise ValueError("a registration cannot include tuned or sealed metrics")
            if self.parent_record_hash is not None:
                raise ValueError("the first registration cannot have a parent hash")
            if self.sealed_window_start is None:
                raise ValueError("a registration must declare sealed_window_start")
            if self.recorded_at >= self.sealed_window_start:
                raise ValueError("experiment must be registered before sealed window begins")
        elif self.parent_record_hash is None:
            raise ValueError("non-registration records require a parent hash")
        if self.stage != ExperimentStage.SEALED_EVALUATED and "sealed" in self.metrics:
            raise ValueError("sealed metrics may only be recorded at sealed_evaluated stage")
        # A caller may reuse and later mutate its dictionaries.  Store an
        # independent snapshot before hashing so a historical record cannot be
        # changed through the caller's reference.
        object.__setattr__(self, "parameters", _freeze_snapshot(self.parameters))
        object.__setattr__(self, "metrics", _freeze_snapshot(self.metrics))
        payload = self._hash_payload()
        computed = sha256_json(payload)
        if self.record_hash is not None and self.record_hash != computed:
            raise ValueError("record_hash does not match immutable record contents")
        object.__setattr__(self, "record_hash", computed)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "experiment_id": self.experiment_id,
            "strategy_family": self.strategy_family,
            "stage": self.stage,
            "recorded_at": self.recorded_at,
            "hypothesis": self.hypothesis,
            "code_hash": self.code_hash,
            "configuration_hash": self.configuration_hash,
            "data_snapshot_hash": self.data_snapshot_hash,
            "cost_model_version": self.cost_model_version,
            "feature_version": self.feature_version,
            "parameters": _plain_snapshot(self.parameters),
            "metrics": _plain_snapshot(self.metrics),
            "parent_record_hash": self.parent_record_hash,
            "sealed_window_start": self.sealed_window_start,
        }


class ExperimentRegistry:
    """In-memory append-only verifier suitable for adapters and focused tests."""

    def __init__(self, records: tuple[ExperimentRecord, ...] = ()) -> None:
        self._records: list[ExperimentRecord] = []
        self._by_record_id: dict[str, ExperimentRecord] = {}
        self._by_experiment_id: dict[str, list[ExperimentRecord]] = {}
        for record in records:
            self.append(record)

    @property
    def records(self) -> tuple[ExperimentRecord, ...]:
        return tuple(self._records)

    def records_for(self, experiment_id: str) -> tuple[ExperimentRecord, ...]:
        return tuple(self._by_experiment_id.get(experiment_id, ()))

    def append(self, record: ExperimentRecord) -> ExperimentRecord:
        """Validate and append one immutable transition; never update an old row."""
        if record.record_id in self._by_record_id:
            raise ValueError("record_id already exists; experiment registry is append-only")
        lineage = self._by_experiment_id.get(record.experiment_id, [])
        if not lineage:
            if record.stage != ExperimentStage.REGISTERED:
                raise ValueError("an experiment must begin with a registration")
        else:
            previous = lineage[-1]
            if record.stage == ExperimentStage.REGISTERED:
                raise ValueError("an experiment cannot be registered twice")
            if record.parent_record_hash != previous.record_hash:
                raise ValueError("record must point to the immediately preceding immutable record")
            if record.strategy_family != previous.strategy_family:
                raise ValueError("strategy_family cannot change within an experiment")
            registration = lineage[0]
            assert registration.sealed_window_start is not None
            if record.stage == ExperimentStage.CALIBRATED:
                if previous.stage == ExperimentStage.SEALED_EVALUATED:
                    raise ValueError(
                        "calibration after sealed evaluation would tune on sealed results"
                    )
                if record.recorded_at >= registration.sealed_window_start:
                    raise ValueError("calibration must complete before the sealed window begins")
            elif record.stage == ExperimentStage.SEALED_EVALUATED:
                if any(item.stage == ExperimentStage.SEALED_EVALUATED for item in lineage):
                    raise ValueError("sealed evaluation may be recorded only once per experiment")
                if record.recorded_at < registration.sealed_window_start:
                    raise ValueError("sealed evaluation cannot predate the declared sealed window")
        self._records.append(record)
        self._by_record_id[record.record_id] = record
        self._by_experiment_id.setdefault(record.experiment_id, []).append(record)
        return record

    def is_preregistered_for_sealed_test(self, experiment_id: str) -> bool:
        """True only for a lineage whose registration predates its sealed test."""
        lineage = self._by_experiment_id.get(experiment_id, [])
        if not lineage:
            return False
        first = lineage[0]
        return (
            first.stage == ExperimentStage.REGISTERED
            and first.sealed_window_start is not None
            and first.recorded_at < first.sealed_window_start
        )


__all__ = ["ExperimentRecord", "ExperimentRegistry", "ExperimentStage"]
