"""Tests for the role-membership collector (stage 3a of the people-insider-
graph design): the free-text relationship parser, the pure two-level
aggregation, and the end-to-end collector against a seeded ``events`` table.

No test touches the network -- ``collectors/roles.py`` reads only what
``collectors/insider.py`` already stores, so every fixture here seeds
``insider_transaction`` events directly via ``duckdb_store.upsert_events``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from market_intelligence import database, hashing
from market_intelligence.collectors import roles as roles_collector
from market_intelligence.config import Config
from market_intelligence.entities import person_id_for_cik
from market_intelligence.schemas.common import Source, utcnow
from market_intelligence.schemas.events import EventRecord, EventType
from market_intelligence.storage import duckdb as duckdb_store

CIK_A = "0002000001"
CIK_B = "0002000002"
COMPANY_X = "company-x"
COMPANY_Y = "company-y"


def _dt(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=UTC)


def _insider_event(
    *,
    event_key: str,
    company_id: str | None,
    cik: str = "0000000001",
    ticker: str = "AAA",
    accession: str,
    available_time: datetime,
    owner_cik: str | None,
    owner_name: str | None = "Jane Doe",
    owner_relationship: str | None = "Officer",
    owner_title: str | None = "CFO",
) -> dict[str, Any]:
    record = EventRecord(
        event_id=hashing.content_hash("event", EventType.INSIDER_TRANSACTION, event_key),
        event_type=EventType.INSIDER_TRANSACTION,
        event_key=event_key,
        company_id=company_id,
        cik=cik,
        ticker=ticker,
        event_subtype="P",
        accession_number=accession,
        event_time=available_time,
        available_time=available_time,
        magnitude=1000.0,
        direction=1,
        payload={
            "owner_cik": owner_cik,
            "owner_name": owner_name,
            "owner_relationship": owner_relationship,
            "owner_title": owner_title,
            "shares": 100,
            "price_per_share": 10.0,
        },
        source=Source.SEC,
        schema_version="1.0.0",
        collected_time=utcnow(),
    )
    return record.to_row()


def _seed(config: Config, rows: list[dict[str, Any]]) -> None:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_events(con, rows)


def _people(config: Config) -> list[dict[str, Any]]:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        rows = con.execute(
            "SELECT reporting_owner_cik, canonical_name, name_variants, "
            "first_seen_filing_date, last_seen_filing_date FROM people"
        ).fetchall()
    cols = ("reporting_owner_cik", "canonical_name", "name_variants", "first_seen", "last_seen")
    return [dict(zip(cols, r, strict=True)) for r in rows]


def _roles(config: Config) -> list[dict[str, Any]]:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        rows = con.execute(
            "SELECT person_id, company_id, is_officer, is_director, is_ten_pct_owner, "
            "latest_officer_title, source_filing_count FROM role_memberships"
        ).fetchall()
    cols = (
        "person_id", "company_id", "is_officer", "is_director",
        "is_ten_pct_owner", "latest_officer_title", "source_filing_count",
    )  # fmt: skip
    return [dict(zip(cols, r, strict=True)) for r in rows]


# --------------------------------------------------------------------------- #
# pure: relationship-flag parsing                                             #
# --------------------------------------------------------------------------- #
def test_parse_relationship_recognizes_officer() -> None:
    assert roles_collector.parse_relationship("Officer") == (True, False, False)


def test_parse_relationship_recognizes_director() -> None:
    assert roles_collector.parse_relationship("Director") == (False, True, False)


def test_parse_relationship_recognizes_combined_text() -> None:
    assert roles_collector.parse_relationship("Officer, Director") == (True, True, False)


def test_parse_relationship_recognizes_ten_pct_owner_variants() -> None:
    assert roles_collector.parse_relationship("10% Owner")[2] is True
    assert roles_collector.parse_relationship("Ten Percent Owner")[2] is True


def test_parse_relationship_is_case_insensitive() -> None:
    assert roles_collector.parse_relationship("OFFICER") == (True, False, False)


def test_parse_relationship_unknown_text_is_all_false() -> None:
    assert roles_collector.parse_relationship("Something Else") == (False, False, False)


def test_parse_relationship_none_is_all_false() -> None:
    assert roles_collector.parse_relationship(None) == (False, False, False)


# --------------------------------------------------------------------------- #
# pure: two-level aggregation                                                 #
# --------------------------------------------------------------------------- #
def _obs(**overrides: Any) -> roles_collector.FilingRoleObservation:
    base: dict[str, Any] = {
        "owner_cik": CIK_A,
        "owner_name": "Jane Doe",
        "owner_relationship": "Officer",
        "owner_title": "CFO",
        "company_id": COMPANY_X,
        "accession_number": "acc-1",
        "available_time": _dt("2024-01-15"),
    }
    base.update(overrides)
    return roles_collector.FilingRoleObservation(**base)


def test_aggregate_ors_flags_across_filings() -> None:
    observations = [
        _obs(
            accession_number="acc-1",
            owner_relationship="Director",
            available_time=_dt("2024-01-01"),
        ),
        _obs(
            accession_number="acc-2",
            owner_relationship="Officer",
            available_time=_dt("2024-06-01"),
        ),
    ]
    _people_agg, roles = roles_collector.aggregate(observations)
    assert len(roles) == 1
    role = roles[0]
    assert role.is_director is True
    assert role.is_officer is True
    assert role.is_ten_pct_owner is False


def test_aggregate_latest_title_wins() -> None:
    observations = [
        _obs(accession_number="acc-1", owner_title="VP", available_time=_dt("2024-01-01")),
        _obs(accession_number="acc-2", owner_title="CFO", available_time=_dt("2024-06-01")),
    ]
    _people_agg, roles = roles_collector.aggregate(observations)
    assert roles[0].latest_title == "CFO"


def test_aggregate_canonical_name_is_most_recent() -> None:
    observations = [
        _obs(accession_number="acc-1", owner_name="J. Doe", available_time=_dt("2024-01-01")),
        _obs(accession_number="acc-2", owner_name="Jane Doe", available_time=_dt("2024-06-01")),
    ]
    people, _roles = roles_collector.aggregate(observations)
    assert len(people) == 1
    assert people[0].canonical_name == "Jane Doe"
    assert people[0].name_variants == {"J. Doe", "Jane Doe"}


def test_aggregate_separates_roles_per_company() -> None:
    observations = [
        _obs(company_id=COMPANY_X, accession_number="acc-1"),
        _obs(company_id=COMPANY_Y, accession_number="acc-2"),
    ]
    people, roles = roles_collector.aggregate(observations)
    assert len(people) == 1  # same person
    assert len(roles) == 2  # two distinct (person, company) relationships
    assert {r.company_id for r in roles} == {COMPANY_X, COMPANY_Y}


def test_aggregate_source_filing_count_counts_distinct_accessions() -> None:
    observations = [
        _obs(accession_number="acc-1"),
        _obs(accession_number="acc-2"),
        _obs(accession_number="acc-1"),  # duplicate accession, must not double-count
    ]
    _people_agg, roles = roles_collector.aggregate(observations)
    assert len(roles[0].accessions) == 2


def test_aggregate_first_and_last_seen_span_the_filings() -> None:
    observations = [
        _obs(accession_number="acc-1", available_time=_dt("2024-01-01")),
        _obs(accession_number="acc-2", available_time=_dt("2024-06-01")),
    ]
    people, roles = roles_collector.aggregate(observations)
    assert people[0].first_seen == _dt("2024-01-01")
    assert people[0].last_seen == _dt("2024-06-01")
    assert roles[0].first_seen == _dt("2024-01-01")
    assert roles[0].last_seen == _dt("2024-06-01")


# --------------------------------------------------------------------------- #
# collector: end to end against seeded events                                 #
# --------------------------------------------------------------------------- #
def test_sync_projects_people_and_role_memberships(tmp_config: Config) -> None:
    rows = [
        _insider_event(
            event_key="SK-1",
            company_id=COMPANY_X,
            accession="acc-1",
            available_time=_dt("2024-01-15"),
            owner_cik=CIK_A,
            owner_name="Jane Doe",
            owner_relationship="Officer, Director",
            owner_title="CFO",
        )
    ]
    _seed(tmp_config, rows)

    summary = roles_collector.sync(tmp_config)

    assert summary.status == "success"
    people = _people(tmp_config)
    assert len(people) == 1
    assert people[0]["reporting_owner_cik"] == CIK_A
    assert people[0]["canonical_name"] == "Jane Doe"

    memberships = _roles(tmp_config)
    assert len(memberships) == 1
    assert memberships[0]["person_id"] == person_id_for_cik(CIK_A)
    assert memberships[0]["company_id"] == COMPANY_X
    assert memberships[0]["is_officer"] is True
    assert memberships[0]["is_director"] is True
    assert memberships[0]["latest_officer_title"] == "CFO"
    assert memberships[0]["source_filing_count"] == 1


def test_sync_reconciles_against_distinct_reporting_owner_ciks(tmp_config: Config) -> None:
    """Stage 3a's own gate: person row count matches distinct owner CIKs."""
    rows = [
        _insider_event(
            event_key="SK-1", company_id=COMPANY_X, accession="acc-1",
            available_time=_dt("2024-01-15"), owner_cik=CIK_A,
        ),
        _insider_event(
            event_key="SK-2", company_id=COMPANY_X, accession="acc-2",
            available_time=_dt("2024-02-15"), owner_cik=CIK_B, owner_name="John Roe",
        ),
    ]  # fmt: skip
    _seed(tmp_config, rows)

    roles_collector.sync(tmp_config)

    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        distinct_ciks = con.execute(
            """
            SELECT count(DISTINCT json_extract_string(payload, '$.owner_cik'))
            FROM events WHERE event_type = ?
            """,
            [EventType.INSIDER_TRANSACTION],
        ).fetchone()[0]
        person_count = con.execute("SELECT count(*) FROM people").fetchone()[0]

    assert person_count == distinct_ciks == 2


def test_sync_skips_events_with_no_company_id(tmp_config: Config) -> None:
    rows = [
        _insider_event(
            event_key="SK-1", company_id=None, accession="acc-1",
            available_time=_dt("2024-01-15"), owner_cik=CIK_A,
        ),
    ]  # fmt: skip
    _seed(tmp_config, rows)

    summary = roles_collector.sync(tmp_config)

    assert summary.status == "success"
    assert _people(tmp_config) == []
    assert _roles(tmp_config) == []


def test_sync_is_idempotent(tmp_config: Config) -> None:
    rows = [
        _insider_event(
            event_key="SK-1", company_id=COMPANY_X, accession="acc-1",
            available_time=_dt("2024-01-15"), owner_cik=CIK_A,
        ),
    ]  # fmt: skip
    _seed(tmp_config, rows)

    first = roles_collector.sync(tmp_config)
    second = roles_collector.sync(tmp_config)

    assert (first.inserted, first.updated) == (2, 0)  # 1 person + 1 role
    assert (second.inserted, second.updated) == (0, 2)  # re-derives, updates in place
    assert len(_people(tmp_config)) == 1
    assert len(_roles(tmp_config)) == 1


def test_sync_multiple_transactions_same_filing_do_not_inflate_filing_count(
    tmp_config: Config,
) -> None:
    """Several NONDERIV_TRANS rows under one accession share one owner sighting."""
    rows = [
        _insider_event(
            event_key="SK-1", company_id=COMPANY_X, accession="acc-1",
            available_time=_dt("2024-01-15"), owner_cik=CIK_A,
        ),
        _insider_event(
            event_key="SK-2", company_id=COMPANY_X, accession="acc-1",
            available_time=_dt("2024-01-15"), owner_cik=CIK_A,
        ),
    ]  # fmt: skip
    _seed(tmp_config, rows)

    roles_collector.sync(tmp_config)

    memberships = _roles(tmp_config)
    assert len(memberships) == 1
    assert memberships[0]["source_filing_count"] == 1
