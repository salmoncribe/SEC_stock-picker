"""Bounded, storage-free SEC filing discovery contracts.

This module only limits and normalizes work handed to a future collector.  It
does not call SEC, persist cursors, or silently backfill missed observations.
Those behaviours belong to the later integration/reconciliation phase.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

DISCOVERABLE_SEC_FORMS = frozenset(
    {
        "4",
        "4/A",
        "6-K",
        "8-K",
        "8-K/A",
        "10-K",
        "10-K/A",
        "10-Q",
        "10-Q/A",
        "13D",
        "13D/A",
        "13G",
        "13G/A",
    }
)
MAX_SEC_DISCOVERY_REQUESTS = 100
MAX_FILINGS_PER_DISCOVERY_REQUEST = 500
MAX_SEC_REQUESTS_PER_SECOND = 9.0


def _normalized_forms(forms: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(form.strip().upper() for form in forms if form.strip()))
    if not normalized:
        raise ValueError("at least one SEC form is required")
    unsupported = set(normalized) - DISCOVERABLE_SEC_FORMS
    if unsupported:
        raise ValueError(f"unsupported SEC discovery forms: {sorted(unsupported)!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class SECDiscoveryRequest:
    """One bounded discovery query; cursor persistence is intentionally absent."""

    cik: str
    forms: tuple[str, ...] = tuple(sorted(DISCOVERABLE_SEC_FORMS))
    max_filings: int = MAX_FILINGS_PER_DISCOVERY_REQUEST
    cursor: str | None = None

    def __post_init__(self) -> None:
        if not self.cik.isdigit() or len(self.cik) > 10:
            raise ValueError("CIK must contain one to ten digits")
        object.__setattr__(self, "cik", self.cik.zfill(10))
        object.__setattr__(self, "forms", _normalized_forms(self.forms))
        if not 1 <= self.max_filings <= MAX_FILINGS_PER_DISCOVERY_REQUEST:
            raise ValueError(f"max_filings must be in 1..{MAX_FILINGS_PER_DISCOVERY_REQUEST}")


@dataclass(frozen=True, slots=True)
class SECDiscoveryCandidate:
    """Minimal metadata that can later be fetched as one complete submission."""

    cik: str
    accession_number: str
    form: str
    discovered_at: datetime

    def __post_init__(self) -> None:
        if not self.cik.isdigit() or len(self.cik) > 10:
            raise ValueError("CIK must contain one to ten digits")
        object.__setattr__(self, "cik", self.cik.zfill(10))
        form = self.form.strip().upper()
        if form not in DISCOVERABLE_SEC_FORMS:
            raise ValueError(f"unsupported SEC discovery form: {form!r}")
        object.__setattr__(self, "form", form)
        offset = self.discovered_at.utcoffset()
        if self.discovered_at.tzinfo is None or offset is None:
            raise ValueError("discovered_at must be timezone-aware UTC")
        if offset.total_seconds() != 0:
            raise ValueError("discovered_at must be normalized to UTC")
        object.__setattr__(self, "discovered_at", self.discovered_at.astimezone(UTC))

    @property
    def key(self) -> tuple[str, str]:
        return self.cik, self.accession_number


class SECDiscoverySource(Protocol):
    """Adapter boundary for SEC full-text/submissions discovery.

    Implementations must honour the request bounds and the platform-wide SEC
    rate limit (at or below ``MAX_SEC_REQUESTS_PER_SECOND``).  This contract is
    deliberately independent of the existing persistent collector.
    """

    def discover(self, request: SECDiscoveryRequest) -> Sequence[SECDiscoveryCandidate]: ...


@dataclass(slots=True)
class BoundedSECDiscoveryQueue:
    """In-memory bounded queue that deduplicates before expensive retrieval."""

    max_pending: int = MAX_SEC_DISCOVERY_REQUESTS
    _pending: deque[SECDiscoveryRequest] = field(default_factory=deque, init=False)
    _keys: set[tuple[str, tuple[str, ...], str | None]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if not 1 <= self.max_pending <= MAX_SEC_DISCOVERY_REQUESTS:
            raise ValueError(f"max_pending must be in 1..{MAX_SEC_DISCOVERY_REQUESTS}")

    def enqueue(self, request: SECDiscoveryRequest) -> bool:
        """Add work once; return false for a duplicate or a full bounded queue."""
        key = request.cik, request.forms, request.cursor
        if key in self._keys or len(self._pending) >= self.max_pending:
            return False
        self._pending.append(request)
        self._keys.add(key)
        return True

    def take(self, *, limit: int) -> tuple[SECDiscoveryRequest, ...]:
        """Remove at most ``limit`` requests without any side effects beyond memory."""
        if not 1 <= limit <= MAX_SEC_DISCOVERY_REQUESTS:
            raise ValueError(f"limit must be in 1..{MAX_SEC_DISCOVERY_REQUESTS}")
        taken: list[SECDiscoveryRequest] = []
        while self._pending and len(taken) < limit:
            request = self._pending.popleft()
            self._keys.discard((request.cik, request.forms, request.cursor))
            taken.append(request)
        return tuple(taken)

    def __len__(self) -> int:
        return len(self._pending)


def discover_bounded(
    source: SECDiscoverySource,
    request: SECDiscoveryRequest,
) -> tuple[SECDiscoveryCandidate, ...]:
    """Defensively cap and filter one adapter response without persisting it."""
    candidates = source.discover(request)
    selected: list[SECDiscoveryCandidate] = []
    seen: set[tuple[str, str]] = set()
    wanted = set(request.forms)
    for candidate in candidates:
        if candidate.cik != request.cik or candidate.form not in wanted or candidate.key in seen:
            continue
        seen.add(candidate.key)
        selected.append(candidate)
        if len(selected) == request.max_filings:
            break
    return tuple(selected)


__all__ = [
    "DISCOVERABLE_SEC_FORMS",
    "MAX_FILINGS_PER_DISCOVERY_REQUEST",
    "MAX_SEC_DISCOVERY_REQUESTS",
    "MAX_SEC_REQUESTS_PER_SECOND",
    "BoundedSECDiscoveryQueue",
    "SECDiscoveryCandidate",
    "SECDiscoveryRequest",
    "SECDiscoverySource",
    "discover_bounded",
]
