"""Tests for interlocking-directorate detection (stage 3b): the pure self-join
over director memberships, the noise cap, and the end-to-end projection into
``company_edges``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from market_intelligence import database
from market_intelligence.analytics import interlocks
from market_intelligence.config import Config
from market_intelligence.storage import duckdb as duckdb_store

CIK_A = "0000000001"  # smaller -- always the interlock "source" against B/C
CIK_B = "0000000002"
CIK_C = "0000000003"


def _membership(**overrides: Any) -> interlocks.DirectorMembership:
    base: dict[str, Any] = {
        "person_id": "person-1",
        "person_name": "Jane Doe",
        "company_id": "company-a",
        "cik": CIK_A,
        "ticker": "AAA",
        "company_name": "Alpha Corp",
    }
    base.update(overrides)
    return interlocks.DirectorMembership(**base)


# --------------------------------------------------------------------------- #
# pure: compute_interlocks                                                    #
# --------------------------------------------------------------------------- #
def test_two_shared_boards_produce_one_edge() -> None:
    memberships = [
        _membership(company_id="company-a", cik=CIK_A, ticker="AAA"),
        _membership(company_id="company-b", cik=CIK_B, ticker="BBB", company_name="Beta Corp"),
    ]
    candidates, capped = interlocks.compute_interlocks(memberships, max_companies_per_person=15)

    assert capped == []
    assert len(candidates) == 1
    edge = candidates[0]
    assert edge.source_cik == CIK_A
    assert edge.target_cik == CIK_B
    assert edge.target_ticker == "BBB"
    assert edge.shared_person_ids == ("person-1",)


def test_single_company_membership_produces_no_edge() -> None:
    memberships = [_membership(company_id="company-a")]
    candidates, capped = interlocks.compute_interlocks(memberships, max_companies_per_person=15)
    assert candidates == []
    assert capped == []


def test_three_shared_boards_produce_three_pairwise_edges() -> None:
    memberships = [
        _membership(company_id="company-a", cik=CIK_A, ticker="AAA"),
        _membership(company_id="company-b", cik=CIK_B, ticker="BBB"),
        _membership(company_id="company-c", cik=CIK_C, ticker="CCC"),
    ]
    candidates, _capped = interlocks.compute_interlocks(memberships, max_companies_per_person=15)
    pairs = {(c.source_cik, c.target_cik) for c in candidates}
    assert pairs == {(CIK_A, CIK_B), (CIK_A, CIK_C), (CIK_B, CIK_C)}


def test_edge_ordering_is_stable_by_cik_regardless_of_input_order() -> None:
    forward = [
        _membership(company_id="company-a", cik=CIK_A, ticker="AAA"),
        _membership(company_id="company-b", cik=CIK_B, ticker="BBB"),
    ]
    reversed_input = list(reversed(forward))

    candidates_fwd, _ = interlocks.compute_interlocks(forward, max_companies_per_person=15)
    candidates_rev, _ = interlocks.compute_interlocks(reversed_input, max_companies_per_person=15)

    assert candidates_fwd[0].source_cik == candidates_rev[0].source_cik == CIK_A
    assert candidates_fwd[0].target_cik == candidates_rev[0].target_cik == CIK_B


def test_two_distinct_directors_sharing_two_companies_produce_one_edge_with_two_names() -> None:
    memberships = [
        _membership(person_id="p1", person_name="Jane Doe", company_id="company-a", cik=CIK_A),
        _membership(person_id="p1", person_name="Jane Doe", company_id="company-b", cik=CIK_B),
        _membership(person_id="p2", person_name="John Roe", company_id="company-a", cik=CIK_A),
        _membership(person_id="p2", person_name="John Roe", company_id="company-b", cik=CIK_B),
    ]
    candidates, _capped = interlocks.compute_interlocks(memberships, max_companies_per_person=15)

    assert len(candidates) == 1
    edge = candidates[0]
    assert set(edge.shared_person_ids) == {"p1", "p2"}
    assert set(edge.shared_person_names) == {"Jane Doe", "John Roe"}


def test_person_above_cap_is_skipped_and_reported() -> None:
    memberships = [
        _membership(company_id=f"company-{i}", cik=f"{i:010d}", ticker=f"T{i}")
        for i in range(1, 5)
    ]
    candidates, capped = interlocks.compute_interlocks(memberships, max_companies_per_person=3)

    assert candidates == []
    assert capped == ["person-1"]


def test_membership_missing_ticker_is_excluded() -> None:
    memberships = [
        _membership(company_id="company-a", cik=CIK_A, ticker="AAA"),
        _membership(company_id="company-b", cik=CIK_B, ticker=None),
    ]
    candidates, _capped = interlocks.compute_interlocks(memberships, max_companies_per_person=15)
    assert candidates == []


# --------------------------------------------------------------------------- #
# collector: project() end to end against seeded role_memberships             #
# --------------------------------------------------------------------------- #
def _seed_companies(config: Config) -> None:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(
            con,
            [
                {
                    "company_id": "company-a", "cik": CIK_A, "ticker": "AAA",
                    "company_name": "Alpha Corp", "source": "sec", "validation_status": "valid",
                },
                {
                    "company_id": "company-b", "cik": CIK_B, "ticker": "BBB",
                    "company_name": "Beta Corp", "source": "sec", "validation_status": "valid",
                },
            ],
        )


def _role_row(person_id: str, company_id: str, *, is_director: bool = True) -> dict[str, Any]:
    return {
        "role_id": f"role-{person_id}-{company_id}",
        "person_id": person_id,
        "company_id": company_id,
        "is_officer": False,
        "is_director": is_director,
        "is_ten_pct_owner": False,
        "latest_officer_title": None,
        "first_seen": date(2024, 1, 1),
        "last_seen": date(2024, 6, 1),
        "source_filing_count": 1,
        "source": "sec",
        "validation_status": "valid",
        "validation_errors": "[]",
    }


def _seed_roles(config: Config, rows: list[dict[str, Any]]) -> None:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_role_memberships(con, rows)


def _seed_person(config: Config, person_id: str, canonical_name: str) -> None:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_people(
            con,
            [
                {
                    "person_id": person_id,
                    "reporting_owner_cik": "9999999999",
                    "canonical_name": canonical_name,
                    "name_variants": "[]",
                    "source": "sec",
                    "validation_status": "valid",
                }
            ],
        )


def _company_edges(config: Config) -> list[dict[str, Any]]:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        rows = con.execute(
            "SELECT edge_key, source_ticker, target_ticker, edge_type, times_asserted, evidence "
            "FROM company_edges"
        ).fetchall()
    cols = ("edge_key", "source_ticker", "target_ticker", "edge_type", "times_asserted", "evidence")
    return [dict(zip(cols, r, strict=True)) for r in rows]


def test_project_writes_shared_board_member_edge(tmp_config: Config) -> None:
    _seed_companies(tmp_config)
    _seed_person(tmp_config, "person-1", "Jane Doe")
    _seed_roles(
        tmp_config,
        [_role_row("person-1", "company-a"), _role_row("person-1", "company-b")],
    )

    summary = interlocks.project(tmp_config)

    assert summary.status == "success"
    edges = _company_edges(tmp_config)
    assert len(edges) == 1
    assert edges[0]["edge_type"] == interlocks.SHARED_BOARD_MEMBER_EDGE_TYPE
    assert edges[0]["source_ticker"] == "AAA"
    assert edges[0]["target_ticker"] == "BBB"
    assert edges[0]["times_asserted"] == 1
    assert "Jane Doe" in (edges[0]["evidence"] or "")


def test_project_ignores_officer_only_memberships(tmp_config: Config) -> None:
    """An officer at two companies is not a 'shared board member'."""
    _seed_companies(tmp_config)
    _seed_roles(
        tmp_config,
        [
            _role_row("person-1", "company-a", is_director=False),
            _role_row("person-1", "company-b", is_director=False),
        ],
    )

    interlocks.project(tmp_config)

    assert _company_edges(tmp_config) == []


def test_project_is_idempotent(tmp_config: Config) -> None:
    _seed_companies(tmp_config)
    _seed_roles(
        tmp_config,
        [_role_row("person-1", "company-a"), _role_row("person-1", "company-b")],
    )

    first = interlocks.project(tmp_config)
    second = interlocks.project(tmp_config)

    assert (first.inserted, first.updated) == (1, 0)
    assert (second.inserted, second.updated) == (0, 1)
    assert len(_company_edges(tmp_config)) == 1


def test_project_respects_configured_cap(tmp_config: Config) -> None:
    _seed_companies(tmp_config)
    _seed_roles(
        tmp_config,
        [_role_row("person-1", "company-a"), _role_row("person-1", "company-b")],
    )

    summary = interlocks.project(tmp_config, max_companies_per_person=1)

    assert _company_edges(tmp_config) == []
    assert summary.stage["capped_persons"] == 1
