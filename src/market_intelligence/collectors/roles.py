"""Role-membership collector: re-project REPORTINGOWNER data into people + roles.

Stage 3a of ``docs/specs/2026-07-25-people-insider-graph-design.md``. Adds no
new network fetch: ``collectors/insider.py`` already parses every Form 3/4/5
REPORTINGOWNER row and carries ``owner_cik``/``owner_name``/
``owner_relationship``/``owner_title`` verbatim in each ``insider_transaction``
event's ``payload``. This collector re-reads that payload out of the
already-stored ``events`` table and projects it into two new tables:
``people`` (one row per persistent SEC reporting-owner CIK -- Decision A) and
``role_memberships`` (one row per (person, company) relationship -- Decision
B), following Decision G for name handling.

Idempotent on both natural keys: re-running recomputes the same aggregates
from the same underlying events and upserts, never duplicates. Row counts
reconcile against distinct ``reportingOwnerCIK``s already present in
``insider_transaction`` events, per the design's stage-3a gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.entities import person_id_for_cik
from market_intelligence.schemas.common import utcnow
from market_intelligence.schemas.events import EventType
from market_intelligence.schemas.people import PersonRecord, RoleMembershipRecord
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet
from market_intelligence.validators.people import validate_person, validate_role_membership

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config


@dataclass(frozen=True)
class FilingRoleObservation:
    """One (owner, company) sighting -- one row per filing, not per transaction.

    Read with ``SELECT DISTINCT`` over the owner payload fields plus
    ``company_id``/``accession_number``/``available_time``, which collapses
    the several ``NONDERIV_TRANS`` rows one Form 4 filing produces (they all
    share the same REPORTINGOWNER row) down to one observation -- mirroring
    how ``clients/insider.py::parse_quarter_zip`` attributed a whole filing to
    one owner row in the first place.
    """

    owner_cik: str
    owner_name: str | None
    owner_relationship: str | None
    owner_title: str | None
    company_id: str
    accession_number: str
    available_time: datetime


@dataclass
class RoleAggregate:
    """Accumulated (person, company) relationship, before validation."""

    person_id: str
    company_id: str
    is_officer: bool = False
    is_director: bool = False
    is_ten_pct_owner: bool = False
    latest_title: str | None = None
    _latest_time: datetime | None = field(default=None, repr=False)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    accessions: set[str] = field(default_factory=set)


@dataclass
class PersonAggregate:
    """Accumulated person identity, before validation."""

    reporting_owner_cik: str
    canonical_name: str | None = None
    _latest_time: datetime | None = field(default=None, repr=False)
    name_variants: set[str] = field(default_factory=set)
    first_seen: datetime | None = None
    last_seen: datetime | None = None


def parse_relationship(text: str | None) -> tuple[bool, bool, bool]:
    """Classify SEC's free-text relationship field into three flags.

    Returns ``(is_officer, is_director, is_ten_pct_owner)``. SEC ships this
    field inconsistently across filings/years -- sometimes a single word
    ("Officer"), sometimes several joined by commas/slashes ("Officer,
    Director"). Case-insensitive substring matching on each of the three
    known categories is deliberately looser than an exact-value match: a
    false negative here silently *drops* a real role rather than inventing
    one, so the asymmetry that makes ``CompanyIndex`` refuse ambiguous
    matches does not apply the same way in this direction -- under-matching
    is the worse failure, not over-matching.
    """
    lowered = (text or "").lower()
    is_officer = "officer" in lowered
    is_director = "director" in lowered
    is_ten_pct_owner = "10%" in lowered or "ten percent" in lowered or "ten-percent" in lowered
    return is_officer, is_director, is_ten_pct_owner


def _load_observations(con: duckdb.DuckDBPyConnection) -> list[FilingRoleObservation]:
    """One row per distinct (owner, company, accession) sighting.

    ``SELECT DISTINCT`` collapses the several transactions a single filing
    reports (they all share the same REPORTINGOWNER row) so the aggregation
    below scans filings, not the much larger multiple of individual
    transactions.
    """
    rows = con.execute(
        """
        SELECT DISTINCT
            json_extract_string(payload, '$.owner_cik')          AS owner_cik,
            json_extract_string(payload, '$.owner_name')         AS owner_name,
            json_extract_string(payload, '$.owner_relationship') AS owner_relationship,
            json_extract_string(payload, '$.owner_title')        AS owner_title,
            company_id,
            accession_number,
            available_time
        FROM events
        WHERE event_type = ?
          AND company_id IS NOT NULL
          AND accession_number IS NOT NULL
          AND available_time IS NOT NULL
          AND json_extract_string(payload, '$.owner_cik') IS NOT NULL
          AND json_extract_string(payload, '$.owner_cik') <> ''
        """,
        [EventType.INSIDER_TRANSACTION],
    ).fetchall()
    return [
        FilingRoleObservation(
            owner_cik=str(r[0]),
            owner_name=r[1],
            owner_relationship=r[2],
            owner_title=r[3],
            company_id=str(r[4]),
            accession_number=str(r[5]),
            available_time=r[6],
        )
        for r in rows
    ]


def aggregate(
    observations: list[FilingRoleObservation],
) -> tuple[list[PersonAggregate], list[RoleAggregate]]:
    """Pure two-level aggregation: per-CIK (people) and per-(CIK, company) (roles).

    Kept separate from the DB read so it is directly unit-testable against
    hand-built observations. ``canonical_name`` / ``latest_title`` come from
    whichever observation has the latest ``available_time`` -- the filing
    date, never the transaction date, keeping this collector on the same
    point-in-time clock as everything else in the platform even though
    lookahead is not really a risk for an identity table.
    """
    people: dict[str, PersonAggregate] = {}
    roles: dict[tuple[str, str], RoleAggregate] = {}

    for obs in observations:
        person = people.setdefault(
            obs.owner_cik, PersonAggregate(reporting_owner_cik=obs.owner_cik)
        )
        if obs.owner_name:
            person.name_variants.add(obs.owner_name)
        if person.first_seen is None or obs.available_time < person.first_seen:
            person.first_seen = obs.available_time
        if person.last_seen is None or obs.available_time > person.last_seen:
            person.last_seen = obs.available_time
        if person._latest_time is None or obs.available_time >= person._latest_time:
            person._latest_time = obs.available_time
            if obs.owner_name:
                person.canonical_name = obs.owner_name

        role = roles.setdefault(
            (obs.owner_cik, obs.company_id),
            RoleAggregate(person_id=person_id_for_cik(obs.owner_cik), company_id=obs.company_id),
        )
        is_officer, is_director, is_ten_pct_owner = parse_relationship(obs.owner_relationship)
        role.is_officer = role.is_officer or is_officer
        role.is_director = role.is_director or is_director
        role.is_ten_pct_owner = role.is_ten_pct_owner or is_ten_pct_owner
        role.accessions.add(obs.accession_number)
        if role.first_seen is None or obs.available_time < role.first_seen:
            role.first_seen = obs.available_time
        if role.last_seen is None or obs.available_time > role.last_seen:
            role.last_seen = obs.available_time
        if role._latest_time is None or obs.available_time >= role._latest_time:
            role._latest_time = obs.available_time
            if obs.owner_title:
                role.latest_title = obs.owner_title

    return list(people.values()), list(roles.values())


def _person_row(agg: PersonAggregate, *, schema_version: str) -> tuple[dict[str, Any], bool]:
    variants = sorted(agg.name_variants)
    record = PersonRecord(
        person_id=person_id_for_cik(agg.reporting_owner_cik),
        reporting_owner_cik=agg.reporting_owner_cik,
        canonical_name=agg.canonical_name,
        name_variants=variants,
        first_seen_filing_date=agg.first_seen.date() if agg.first_seen else None,
        last_seen_filing_date=agg.last_seen.date() if agg.last_seen else None,
        content_hash=hashing.content_hash(agg.reporting_owner_cik, agg.canonical_name, variants),
        schema_version=schema_version,
        collected_time=utcnow(),
    )
    validate_person(record)
    return record.to_row(), record.is_rejected


def _role_row(agg: RoleAggregate, *, schema_version: str) -> tuple[dict[str, Any], bool]:
    record = RoleMembershipRecord(
        role_id=hashing.content_hash("role", agg.person_id, agg.company_id),
        person_id=agg.person_id,
        company_id=agg.company_id,
        is_officer=agg.is_officer,
        is_director=agg.is_director,
        is_ten_pct_owner=agg.is_ten_pct_owner,
        latest_officer_title=agg.latest_title,
        first_seen=agg.first_seen.date() if agg.first_seen else None,
        last_seen=agg.last_seen.date() if agg.last_seen else None,
        source_filing_count=len(agg.accessions),
        content_hash=hashing.content_hash(
            agg.person_id,
            agg.company_id,
            agg.is_officer,
            agg.is_director,
            agg.is_ten_pct_owner,
            agg.latest_title,
            len(agg.accessions),
        ),
        schema_version=schema_version,
        collected_time=utcnow(),
    )
    validate_role_membership(record)
    return record.to_row(), record.is_rejected


def sync(config: Config) -> RunSummary:
    """Re-project ``people`` + ``role_memberships`` from stored insider events."""
    with pipeline_run(config, "people.sync-roles") as (con, summary):
        schema_version = config.settings.app.schema_version
        observations = _load_observations(con)
        summary.bump("filing_observations", len(observations))

        people, roles = aggregate(observations)

        person_rows: list[dict[str, Any]] = []
        for person_agg in people:
            row, rejected = _person_row(person_agg, schema_version=schema_version)
            summary.collected += 1
            if rejected:
                summary.rejected += 1
            else:
                person_rows.append(row)

        role_rows: list[dict[str, Any]] = []
        for role_agg in roles:
            row, rejected = _role_row(role_agg, schema_version=schema_version)
            summary.collected += 1
            if rejected:
                summary.rejected += 1
            else:
                role_rows.append(row)

        if person_rows:
            result = duckdb_store.upsert_people(con, person_rows)
            summary.inserted += result.inserted
            summary.updated += result.updated
            parquet.write_records(
                config.paths.parquet_dir, "people", person_rows, ["reporting_owner_cik"]
            )

        if role_rows:
            result = duckdb_store.upsert_role_memberships(con, role_rows)
            summary.inserted += result.inserted
            summary.updated += result.updated
            parquet.write_records(
                config.paths.parquet_dir,
                "role_memberships",
                role_rows,
                ["person_id", "company_id"],
            )

        summary.bump("people", len(person_rows))
        summary.bump("role_memberships", len(role_rows))

    return summary


__all__ = [
    "FilingRoleObservation",
    "PersonAggregate",
    "RoleAggregate",
    "aggregate",
    "parse_relationship",
    "sync",
]
