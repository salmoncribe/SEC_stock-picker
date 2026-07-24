"""The briefing contract: what the daily loop produces and both projections read.

The orchestrator builds a :class:`Briefing`; the Obsidian renderer and the
Telegram notifier each turn one into their own output. Keeping this contract in
one small module means the two projections never depend on the orchestrator's
internals, only on this shape -- so they can be built and tested in isolation.

Everything here is frozen and free of I/O on purpose: a briefing is a value, so
it can be constructed in a test without a database and rendered without a
network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


class RunStatus:
    """How a daily run ended. Reported verbatim so a partial run is legible."""

    SUCCESS = "success"
    PARTIAL = "partial"  # some steps failed; earlier work still landed
    FAILED = "failed"  # the run could not produce a usable briefing


class ChangeKind:
    """What happened to a signal cell between the last run and this one."""

    ACTIVATED = "activated"  # earned its Kth consecutive confirmation; now fires
    DEMOTED = "demoted"  # was active, weakened below the economic floor
    RETIRED = "retired"  # holdout reversed sign, or dead too long
    NEW_CANDIDATE = "new_candidate"  # cleared discovery for the first time


@dataclass(frozen=True)
class SignalChange:
    """One cell whose trusted status changed on this run.

    The briefing reports *changes*, not the full roster: what newly went active,
    what broke. ``kind`` is a :class:`ChangeKind` value; ``reason`` is the
    human-readable justification carried from the gate.
    """

    event_type: str
    event_subtype: str | None
    edge_type: str
    horizon_days: int
    kind: str
    mean_car: float
    hit_rate: float
    n_clusters: int
    reason: str

    @property
    def label(self) -> str:
        """A compact identifier like ``insider_transaction/P/self/20d``."""
        subtype = self.event_subtype or "-"
        return f"{self.event_type}/{subtype}/{self.edge_type}/{self.horizon_days}d"


@dataclass(frozen=True)
class EventAlert:
    """A prediction fired by today's event through an *active* cell.

    An event landed today at ``ticker`` whose (type, subtype, edge, horizon)
    matches a cell the ladder currently trusts. ``predicted_car`` is that cell's
    measured mean move and ``direction`` its sign; ``basis`` explains which cell
    and how strong its track record is (e.g. "active, holdout hit 58.8%").

    The alert also carries the identifiers and stats the trade-alert layer
    needs: ``event_id`` is the ledger's dedup key, and ``hit_rate``,
    ``n_clusters``, ``extraction_confidence`` feed its confidence score.
    """

    ticker: str
    event_type: str
    event_subtype: str | None
    available_on: date
    horizon_days: int
    direction: int
    predicted_car: float
    basis: str
    event_id: str = ""
    hit_rate: float | None = None
    n_clusters: int = 0
    extraction_confidence: float | None = None


@dataclass(frozen=True)
class Briefing:
    """Everything one morning's note and nudge are rendered from.

    ``ingest`` is a small dict of counts (new prices, filings, events, matured
    samples) for the "what moved" line. ``changes`` and ``alerts`` are the body.
    ``active_signals`` is the current trusted roster, included for context so a
    reader sees the standing model, not only the deltas. ``notes`` carries
    free-form run detail, including which steps failed on a partial run.
    """

    as_of: date
    run_status: str
    ingest: dict[str, int] = field(default_factory=dict)
    changes: list[SignalChange] = field(default_factory=list)
    alerts: list[EventAlert] = field(default_factory=list)
    active_signals: list[SignalChange] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_news(self) -> bool:
        """True when something happened worth a reader's attention today."""
        return bool(self.changes or self.alerts) or self.run_status != RunStatus.SUCCESS


__all__ = [
    "Briefing",
    "ChangeKind",
    "EventAlert",
    "RunStatus",
    "SignalChange",
]
