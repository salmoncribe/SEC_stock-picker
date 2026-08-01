"""Tests for the bounded, non-persistent SEC discovery adapter boundary."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from market_intelligence.collectors.disclosure_packages import (
    BoundedSECDiscoveryQueue,
    SECDiscoveryCandidate,
    SECDiscoveryRequest,
    discover_bounded,
)

NOW = datetime(2024, 5, 7, tzinfo=UTC)


class _Source:
    def discover(self, request: SECDiscoveryRequest) -> list[SECDiscoveryCandidate]:
        return [
            SECDiscoveryCandidate(request.cik, "0001045810-24-000001", "8-K", NOW),
            SECDiscoveryCandidate(request.cik, "0001045810-24-000001", "8-K", NOW),
            SECDiscoveryCandidate(request.cik, "0001045810-24-000002", "10-Q", NOW),
            SECDiscoveryCandidate("0000000001", "0000000001-24-000003", "8-K", NOW),
        ]


def test_request_is_form_allowlisted_and_hard_bounded() -> None:
    request = SECDiscoveryRequest(cik="1045810", forms=("8-k", "8-K", "10-Q"), max_filings=1)

    assert request.cik == "0001045810"
    assert request.forms == ("8-K", "10-Q")
    with pytest.raises(ValueError, match="unsupported"):
        SECDiscoveryRequest(cik="1", forms=("S-1",))
    with pytest.raises(ValueError, match="max_filings"):
        SECDiscoveryRequest(cik="1", max_filings=501)


def test_discovery_filters_wrong_cik_duplicates_and_applies_result_bound() -> None:
    request = SECDiscoveryRequest(cik="1045810", forms=("8-K", "10-Q"), max_filings=1)

    found = discover_bounded(_Source(), request)

    assert [candidate.accession_number for candidate in found] == ["0001045810-24-000001"]


def test_in_memory_queue_is_bounded_and_deduplicated() -> None:
    queue = BoundedSECDiscoveryQueue(max_pending=1)
    first = SECDiscoveryRequest(cik="1", forms=("8-K",))
    second = SECDiscoveryRequest(cik="2", forms=("8-K",))

    assert queue.enqueue(first) is True
    assert queue.enqueue(first) is False
    assert queue.enqueue(second) is False
    assert queue.take(limit=1) == (first,)
    assert queue.enqueue(second) is True
