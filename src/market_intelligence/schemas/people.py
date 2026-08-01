"""Typed people and roles: who runs and sits on the board of a company.

Companion to ``schemas/edges.py``, following the same shape. A :class:`PersonRecord`
is one insider's persistent identity, keyed on their SEC ``reporting_owner_cik``
(see ``docs/specs/2026-07-25-people-insider-graph-design.md`` Decision A) -- not
on name, because names are not unique and are not even consistently formatted
across a single person's own filings. A :class:`RoleMembershipRecord` is one
person's relationship to one company, re-projected from the REPORTINGOWNER rows
``collectors/insider.py`` already fetched and stored: no new network fetch, no
new LLM call.

**Flags are OR'd across filings, never replaced.** A person who filed once as
a director and later as an officer keeps both flags true, since both were real
at different times and this table is not attempting to reconstruct tenure
(see the design doc's Data quality note on title/role staleness).
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from pydantic import Field

from market_intelligence.schemas.common import ProvenanceModel, Source


class PersonRecord(ProvenanceModel):
    """One insider's persistent identity (a row of ``people``).

    ``reporting_owner_cik`` is the natural key. ``canonical_name`` is the most
    recently observed filer name for this CIK (Decision G: SEC's filer names
    are not consistently formatted across filings/years, so the most recent
    is kept as canonical rather than attempting fuzzy-dedup logic now).
    ``name_variants`` keeps every distinct spelling observed, so a stale or
    oddly formatted name is never silently lost.
    """

    person_id: str
    reporting_owner_cik: str
    canonical_name: str | None = None
    name_variants: list[str] = Field(default_factory=list)
    first_seen_filing_date: date | None = None
    last_seen_filing_date: date | None = None
    source: Source = Source.SEC

    def to_row(self) -> dict[str, Any]:
        data = super().to_row()
        data["name_variants"] = json.dumps(sorted(set(self.name_variants)), sort_keys=True)
        return data


class RoleMembershipRecord(ProvenanceModel):
    """One person's relationship to one company (a row of ``role_memberships``).

    Natural key is ``(person_id, company_id)``. ``latest_officer_title`` is
    exactly that -- latest, not historical; Phase 3 does not attempt tenure
    reconstruction (e.g. "was CFO 2019-2022, now CEO").
    """

    role_id: str
    person_id: str
    company_id: str
    is_officer: bool = False
    is_director: bool = False
    is_ten_pct_owner: bool = False
    latest_officer_title: str | None = None
    first_seen: date | None = None
    last_seen: date | None = None
    source_filing_count: int = 0
    source: Source = Source.SEC


__all__ = ["PersonRecord", "RoleMembershipRecord"]
