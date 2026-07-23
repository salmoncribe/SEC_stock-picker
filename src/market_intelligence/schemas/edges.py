"""Typed relationship edges: the connection graph the propagation hypothesis needs.

An edge is a claim, extracted from one company's filing, that it stands in a
named business relationship to another company: it sells to them, buys from
them, competes with them, or partners with them. Each edge is a *hypothesis*,
not a fact -- the LLM proposes it and the event-study harness decides whether an
event at one end actually moves the other. That division is what makes an
imperfect extractor safe: a wrong edge fails validation and never fires an
alert.

**The source is always the filer; the target is whoever the filing names.** A
10-K by NVIDIA that names Microsoft as a customer produces one edge
``NVDA --customer--> MSFT``. The target starts as a raw string (``target_name``,
exactly as written) and is resolved to a ``target_cik``/``target_ticker`` only
when it matches a company we track; foreign names and private firms stay
unresolved, kept in the graph for context but excluded from validation because
they have no price series.

**Dedup is by the relationship, not the filing.** ``edge_key`` is
``(source_cik, target, edge_type)``, so an edge asserted in four consecutive
10-Ks is one row whose ``times_asserted`` counts to four -- the "don't get
bloated with the same info added four times" requirement, enforced at the key.
``first_seen_time`` / ``last_seen_time`` bound when the claim was observed;
``report_date`` points at the most recent filing that asserted it.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from market_intelligence.schemas.common import ProvenanceModel, Source


class EdgeType(StrEnum):
    """The four relationship kinds the extractor is allowed to assert.

    Kept deliberately small and economically distinct. ``CUSTOMER`` (source
    sells to target) and ``SUPPLIER`` (source buys from target) are the two the
    economic-links literature finds most predictive; ``COMPETITOR`` and
    ``PARTNER`` are weaker but cheap to carry and let the gate decide.
    """

    CUSTOMER = "customer"
    SUPPLIER = "supplier"
    COMPETITOR = "competitor"
    PARTNER = "partner"


class ResolutionStatus(StrEnum):
    """Whether the target name was matched to a company we track.

    ``RESOLVED`` -- matched to a CIK with enough confidence to validate.
    ``UNRESOLVED`` -- no confident match (foreign, private, or unknown); the
                      edge is kept for the graph but cannot be validated.
    ``AMBIGUOUS`` -- more than one plausible match; deliberately left unresolved
                     rather than guessing, since a wrong resolution invents a
                     relationship between two real companies.
    """

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"


class CompanyEdgeRecord(ProvenanceModel):
    """One relationship edge (a row of ``company_edges``).

    ``edge_key`` is the natural key ``(source_cik, target, edge_type)`` used for
    idempotent upserts; ``edge_id`` is its content hash and the primary key.
    ``target`` is the deduplication token -- the resolved ticker when known, else
    the normalized target name -- so the same company named two slightly
    different ways still collapses once resolution succeeds.
    """

    edge_id: str
    edge_key: str
    source_cik: str
    source_ticker: str | None = None
    source_company_id: str | None = None
    target: str
    target_name: str
    target_cik: str | None = None
    target_ticker: str | None = None
    edge_type: str
    resolution_status: str = ResolutionStatus.UNRESOLVED.value
    resolution_confidence: float | None = None
    evidence: str | None = None
    extraction_confidence: float | None = None
    extraction_method: str | None = None
    extraction_model: str | None = None
    accession_number: str | None = None
    filing_id: str | None = None
    report_date: date | None = None
    times_asserted: int = 1
    first_seen_time: datetime | None = None
    last_seen_time: datetime | None = None
    source: Source = Source.DERIVED

    @property
    def is_validatable(self) -> bool:
        """True when both ends have a CIK, so propagation can be measured."""
        return self.resolution_status == ResolutionStatus.RESOLVED.value and bool(self.target_cik)

    def to_row(self) -> dict[str, Any]:
        return super().to_row()


__all__ = [
    "CompanyEdgeRecord",
    "EdgeType",
    "ResolutionStatus",
]
