"""Interlocking directorates: shared board members as graph edges.

Stage 3b of ``docs/specs/2026-07-25-people-insider-graph-design.md``, Decision
C. A self-join of ``role_memberships`` on ``person_id`` where ``company_id``
differs: if the same director sits on two boards, that is a channel
information can travel across. No new table -- these candidates are written
as ordinary rows into ``company_edges`` (the table the base signal-graph
design calls ``graph_edges``) with ``edge_type = SHARED_BOARD_MEMBER``, so the
edge inherits the existing candidate lifecycle, dedup, and Obsidian projection
for free (see ``autopilot/graph.py``).

Only ``is_director`` memberships count -- an officer or a 10%+ owner is not a
"board member," and folding them in would name a relationship this edge type
does not claim to have found.

**The noise this must not produce.** A handful of professional board-sitters
(mutual fund trustees who sit on dozens of boards) would otherwise flood the
graph with a fully-connected, low-information hub: 30 shared trusteeships
produce 30*29/2 = 435 edges from one person alone. Persons whose director
membership count exceeds ``interlock_max_companies_per_person`` are skipped
entirely -- their memberships are real, but the interlock they'd assert is
overwhelmingly incidental co-membership, not a meaningful information channel
(see the design's §5 "Interlock noise" note).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.schemas.common import Source, utcnow
from market_intelligence.schemas.edges import ResolutionStatus
from market_intelligence.storage import duckdb as duckdb_store

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config

#: Not part of ``schemas.edges.EdgeType`` -- that enum is deliberately "the
#: four relationship kinds the extractor [the LLM] is allowed to assert."
#: This edge type is never LLM-proposed; it is a pure derivation over
#: ``role_memberships`` and lives here as its producer's own constant,
#: mirroring how ``signals/impact.py`` defines ``PLACEBO_EDGE`` locally rather
#: than extending that enum.
SHARED_BOARD_MEMBER_EDGE_TYPE = "shared_board_member"

EXTRACTION_METHOD = "role_memberships_interlock_v1"


@dataclass(frozen=True)
class DirectorMembership:
    """One director's seat at one company -- the self-join's raw input."""

    person_id: str
    person_name: str | None
    company_id: str
    cik: str
    ticker: str | None
    company_name: str | None


@dataclass(frozen=True)
class InterlockCandidate:
    """One shared-board-member relationship between two companies.

    ``source``/``target`` are ordered by CIK (smaller first) so the same pair
    is never emitted twice in opposite directions across two different
    shared directors -- the ordering is arbitrary but must be stable, which
    "smaller CIK first" trivially is.
    """

    source_company_id: str
    source_cik: str
    source_ticker: str
    target_company_id: str
    target_cik: str
    target_ticker: str
    target_company_name: str
    shared_person_ids: tuple[str, ...]
    shared_person_names: tuple[str, ...]


def compute_interlocks(
    memberships: list[DirectorMembership],
    *,
    max_companies_per_person: int,
) -> tuple[list[InterlockCandidate], list[str]]:
    """Self-join director memberships into candidate interlock edges.

    Pure: takes plain rows, returns plain candidates plus the list of
    ``person_id``s skipped for exceeding ``max_companies_per_person`` (so a
    caller can report the cap without a second pass over the data). Only
    memberships with both a ``cik`` and a ``ticker`` can become a validatable
    edge -- an unpriced or unticked company has no note to link to and no
    return series to ever measure propagation against.
    """
    by_person: dict[str, list[DirectorMembership]] = {}
    for m in memberships:
        if m.ticker and m.cik:
            by_person.setdefault(m.person_id, []).append(m)

    capped: list[str] = []
    pairs: dict[tuple[str, str], dict[str, Any]] = {}

    for person_id, rows in by_person.items():
        # Distinct companies only: the same person appearing twice for one
        # company (should not happen after collectors/roles.py's own
        # (person_id, company_id) dedup, but never assume a caller obeyed it).
        by_company = {r.company_id: r for r in rows}
        if len(by_company) < 2:
            continue
        if len(by_company) > max_companies_per_person:
            capped.append(person_id)
            continue

        companies = sorted(by_company.values(), key=lambda r: r.cik)
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                a, b = companies[i], companies[j]
                source, target = (a, b) if a.cik <= b.cik else (b, a)
                key = (source.cik, target.cik)
                entry = pairs.setdefault(
                    key,
                    {
                        "source": source,
                        "target": target,
                        "person_ids": [],
                        "person_names": [],
                    },
                )
                entry["person_ids"].append(person_id)
                entry["person_names"].append(source.person_name or target.person_name)

    candidates: list[InterlockCandidate] = []
    for entry in pairs.values():
        pair_source: DirectorMembership = entry["source"]
        pair_target: DirectorMembership = entry["target"]
        # Narrows ticker: str | None -> str for mypy; both are guaranteed
        # truthy by the `if m.ticker and m.cik` filter when `by_person` was built.
        assert pair_source.ticker is not None
        assert pair_target.ticker is not None
        candidates.append(
            InterlockCandidate(
                source_company_id=pair_source.company_id,
                source_cik=pair_source.cik,
                source_ticker=pair_source.ticker,
                target_company_id=pair_target.company_id,
                target_cik=pair_target.cik,
                target_ticker=pair_target.ticker,
                target_company_name=pair_target.company_name or pair_target.ticker,
                shared_person_ids=tuple(entry["person_ids"]),
                shared_person_names=tuple(name for name in entry["person_names"] if name),
            )
        )
    return candidates, capped


def _load_memberships(con: duckdb.DuckDBPyConnection) -> list[DirectorMembership]:
    rows = con.execute(
        """
        SELECT r.person_id, p.canonical_name, r.company_id, c.cik, c.ticker, c.company_name
        FROM role_memberships r
        JOIN companies c ON c.company_id = r.company_id
        LEFT JOIN people p ON p.person_id = r.person_id
        WHERE r.is_director = TRUE
        """
    ).fetchall()
    return [
        DirectorMembership(
            person_id=str(row[0]),
            person_name=row[1],
            company_id=str(row[2]),
            cik=str(row[3]),
            ticker=row[4],
            company_name=row[5],
        )
        for row in rows
    ]


def _edge_row(candidate: InterlockCandidate, *, schema_version: str) -> dict[str, Any]:
    edge_key = f"{candidate.source_cik}|{candidate.target_ticker}|{SHARED_BOARD_MEMBER_EDGE_TYPE}"
    now = utcnow()
    names = ", ".join(dict.fromkeys(candidate.shared_person_names))  # de-dup, keep order
    evidence = f"Shared director(s): {names}" if names else None
    return {
        "edge_id": hashing.content_hash("edge", edge_key),
        "edge_key": edge_key,
        "source_cik": candidate.source_cik,
        "source_ticker": candidate.source_ticker,
        "source_company_id": candidate.source_company_id,
        "target": candidate.target_ticker,
        "target_name": candidate.target_company_name,
        "target_cik": candidate.target_cik,
        "target_ticker": candidate.target_ticker,
        "edge_type": SHARED_BOARD_MEMBER_EDGE_TYPE,
        "resolution_status": ResolutionStatus.RESOLVED.value,
        "resolution_confidence": 1.0,
        "evidence": evidence,
        "extraction_confidence": 1.0,
        "extraction_method": EXTRACTION_METHOD,
        "extraction_model": None,
        "accession_number": None,
        "filing_id": None,
        "report_date": None,
        # A full recomputation each run, not an incremental "+1" -- unlike the
        # LLM extractor (which sees one filing at a time and accumulates
        # times_asserted across runs), this reads the *entire* current
        # role_memberships table every time, so re-deriving the count fresh
        # is both correct and still idempotent: the same inputs always
        # produce the same count, never a drifting one.
        "times_asserted": len(candidate.shared_person_ids),
        "first_seen_time": now,
        "last_seen_time": now,
        "source": Source.DERIVED.value,
        "content_hash": hashing.content_hash(edge_key, tuple(sorted(candidate.shared_person_ids))),
        "schema_version": schema_version,
        "validation_status": "valid",
        "validation_errors": "[]",
        "collected_time": now,
    }


def interlocking_person_ids(
    con: duckdb.DuckDBPyConnection, *, max_companies_per_person: int
) -> set[str]:
    """Person ids that produce at least one shared_board_member candidate.

    Reuses :func:`compute_interlocks` verbatim -- the same derivation
    ``project`` uses to write real ``company_edges`` rows -- so a vault
    inclusion decision (docs/specs/2026-07-25-people-insider-graph-design.md
    §9.1(a)) can never diverge from what actually produced an interlock edge.
    ``company_edges`` itself carries no ``person_id`` column (only a
    human-readable ``evidence`` string), so this recomputes from
    ``role_memberships`` rather than trying to parse that string back apart --
    the design's §9 amendment is explicitly a projection-only change, no new
    schema. Applies the same noise cap (§5): a person capped out of
    ``compute_interlocks`` here does not qualify for a vault node on
    interlock grounds alone either.
    """
    memberships = _load_memberships(con)
    candidates, _capped = compute_interlocks(
        memberships, max_companies_per_person=max_companies_per_person
    )
    person_ids: set[str] = set()
    for candidate in candidates:
        person_ids.update(candidate.shared_person_ids)
    return person_ids


def project(config: Config, *, max_companies_per_person: int | None = None) -> RunSummary:
    """Derive ``shared_board_member`` candidate edges from ``role_memberships``.

    Writes into ``company_edges`` alongside the LLM-extracted edge types, so
    the existing graph lifecycle, validation gate, and Obsidian projection
    apply unchanged. ``max_companies_per_person`` overrides the configured
    cap for one run (mainly for tests); defaults to
    ``config.settings.people_graph.interlock_max_companies_per_person``.
    """
    people_graph_settings = config.settings.people_graph
    cap = max_companies_per_person or people_graph_settings.interlock_max_companies_per_person

    with pipeline_run(config, "people.project-interlocks") as (con, summary):
        schema_version = config.settings.app.schema_version
        memberships = _load_memberships(con)
        summary.bump("director_memberships", len(memberships))

        candidates, capped_persons = compute_interlocks(memberships, max_companies_per_person=cap)
        summary.bump("capped_persons", len(capped_persons))
        if capped_persons:
            summary.note(
                f"{len(capped_persons)} person(s) exceeded {cap} director memberships "
                "and were skipped as professional board-sitters"
            )

        rows = [_edge_row(candidate, schema_version=schema_version) for candidate in candidates]
        summary.collected = len(rows)
        if rows:
            result = duckdb_store.upsert_company_edges(con, rows)
            summary.inserted = result.inserted
            summary.updated = result.updated
            summary.deduped = result.deduped

    return summary


__all__ = [
    "EXTRACTION_METHOD",
    "SHARED_BOARD_MEMBER_EDGE_TYPE",
    "DirectorMembership",
    "InterlockCandidate",
    "compute_interlocks",
    "interlocking_person_ids",
    "project",
]
