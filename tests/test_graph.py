"""Tests for the Obsidian projection of the company relationship graph.

The reader-facing contract that would quietly break the graph view if it
regressed: a resolved counterparty must become a ``[[TICKER]]`` wiki-link so
Obsidian draws an edge, an unresolved target must stay plain text (there is no
note to link to), an inbound edge must link back to whoever named the company,
and the projection must be regenerable -- one note per company, overwritten in
place. ``render_company_note``/``render_person_note`` are pure, so most of this
is tested without a database; ``project_graph``/``project_person_notes`` are
exercised end to end against a temp DuckDB.

The second half of this file covers docs/specs/2026-07-25-people-insider-
graph-design.md §9 (the 2026-07-26 amendment): a person only earns a dedicated
note if they interlock or are tied to a currently-live event (never a pure
10%-owner, regardless), non-qualifying roster facts stay out of the AI-facing
vault, cluster-buy facts render split into "Watching"/"Resolved", and the
projector deletes stale notes for entities that no longer qualify. ``today`` is
always passed explicitly in these tests -- live/resolved classification must
never depend on real wall-clock time.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any

from market_intelligence import database, hashing
from market_intelligence.autopilot import graph
from market_intelligence.config import Config
from market_intelligence.schemas.common import Source, utcnow
from market_intelligence.schemas.edges import ResolutionStatus
from market_intelligence.schemas.events import EventRecord, EventType
from market_intelligence.schemas.samples import SELF_EDGE, EventSampleRecord
from market_intelligence.signals import dataset
from market_intelligence.signals.promotion import LadderStatus
from market_intelligence.storage import duckdb as duckdb_store

NVDA_CIK = "0001045810"
MSFT_CIK = "0000789019"


def _edge(**overrides: Any) -> dict[str, Any]:
    """A resolved NVDA --customer--> MSFT edge; override any field per test."""
    base: dict[str, Any] = {
        "source_ticker": "NVDA",
        "source_cik": NVDA_CIK,
        "target_ticker": "MSFT",
        "target_name": "Microsoft Corporation",
        "target_cik": MSFT_CIK,
        "edge_type": "customer",
        "resolution_status": ResolutionStatus.RESOLVED.value,
        "evidence": "sells GPUs to Microsoft",
        "extraction_confidence": 0.9,
        "report_date": date(2026, 1, 15),
        "times_asserted": 1,
    }
    base.update(overrides)
    return base


def _edge_row(edge_key: str, **overrides: Any) -> dict[str, Any]:
    """A full ``company_edges`` row for ``upsert_company_edges`` to store."""
    edge = _edge(**overrides)
    target = edge["target_ticker"] or "?"
    return {"edge_id": edge_key, "edge_key": edge_key, "target": target, **edge}


def _signal_status_row(
    edge_type: str, *, status: str = LadderStatus.ACTIVE.value
) -> dict[str, Any]:
    return {
        "signal_id": f"sig-insider-p-{edge_type}-20",
        "event_type": "insider_transaction",
        "event_subtype": "P",
        "edge_type": edge_type,
        "horizon_days": 20,
        "status": status,
        "confirm_streak": 2,
        "fail_streak": 0,
        "mean_car": 0.01,
        "hit_rate": 0.58,
        "n_clusters": 50,
        "direction": 1,
        "last_reason": "seeded active signal",
    }


def _seed_active_edge_types(config: Config, *edge_types: str) -> None:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_signal_status(
            con, [_signal_status_row(edge_type) for edge_type in edge_types]
        )


# --------------------------------------------------------------------------- #
# render_company_note -- pure, no I/O                                           #
# --------------------------------------------------------------------------- #
def test_resolved_outgoing_edge_produces_a_wiki_link() -> None:
    note = graph.render_company_note("NVDA", [_edge()], [])

    assert note.startswith("---\n")
    assert "ticker: NVDA" in note
    assert "  - company" in note
    assert "  - edge/customer" in note
    assert "# NVDA" in note
    assert "## Sells to / Customers" in note
    assert "[[MSFT]]" in note
    assert "sells GPUs to Microsoft" in note
    assert "(conf 90%)" in note


def test_unresolved_edge_shows_plain_name_and_no_wiki_link() -> None:
    edge = _edge(
        edge_type="competitor",
        target_ticker=None,
        target_cik=None,
        target_name="adidas AG",
        resolution_status=ResolutionStatus.UNRESOLVED.value,
        evidence="compete with adidas internationally",
    )

    note = graph.render_company_note("NKE", [edge], [])

    assert "## Competitors" in note
    assert "adidas AG" in note
    assert "[[adidas AG]]" not in note
    assert "[[" not in note  # no wiki-link at all for an unresolved-only note


def test_incoming_edge_links_back_to_the_naming_company() -> None:
    # MSFT was named as a customer by NVDA: its note should link back to [[NVDA]].
    note = graph.render_company_note("MSFT", [], [_edge()])

    assert "## Inbound (named by)" in note
    assert "### Named as a customer by" in note
    assert "[[NVDA]]" in note
    # A company with only incoming edges renders no outgoing sections.
    assert "## Sells to / Customers" not in note


def test_note_with_no_evidence_or_confidence_still_renders_cleanly() -> None:
    edge = _edge(evidence=None, extraction_confidence=None)

    note = graph.render_company_note("NVDA", [edge], [])

    assert note.endswith("\n")
    assert "[[MSFT]]" in note
    assert "conf" not in note  # no confidence parenthetical
    assert " -- " not in note.split("## Sells to / Customers", 1)[1]  # no evidence dash
    assert "- [[MSFT]]\n" in note or note.rstrip().endswith("- [[MSFT]]")


def test_note_groups_outgoing_edges_by_kind() -> None:
    note = graph.render_company_note(
        "NVDA",
        [
            _edge(edge_type="customer", target_ticker="MSFT", target_name="Microsoft"),
            _edge(
                edge_type="supplier",
                target_ticker="TSM",
                target_name="TSMC",
                target_cik="0001046179",
            ),
        ],
        [],
    )

    assert "## Sells to / Customers" in note
    assert "## Buys from / Suppliers" in note
    assert "  - edge/customer" in note
    assert "  - edge/supplier" in note
    # Customer group leads supplier group (edges.py priority order).
    assert note.index("Sells to") < note.index("Buys from")


# --------------------------------------------------------------------------- #
# project_graph -- reads the DB, writes the vault                              #
# --------------------------------------------------------------------------- #
def _seed_edges(config: Config) -> None:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_company_edges(
            con,
            [
                _edge_row("NVDA|MSFT|customer"),
                _edge_row(
                    "NVDA|adidas|competitor",
                    edge_type="competitor",
                    target_ticker=None,
                    target_cik=None,
                    target_name="adidas AG",
                    target="adidas ag",
                    resolution_status=ResolutionStatus.UNRESOLVED.value,
                    evidence="compete with adidas",
                    extraction_confidence=None,
                ),
            ],
        )
        duckdb_store.upsert_signal_status(
            con,
            [
                _signal_status_row("customer"),
                _signal_status_row("competitor"),
            ],
        )


def test_project_graph_writes_one_note_per_company_with_links(tmp_config: Config) -> None:
    _seed_edges(tmp_config)
    vault = tmp_config.paths.obsidian_vault_dir

    with database.connection(tmp_config.paths.database_path) as con:
        result = graph.project_graph(con, vault)

    # NVDA (source) and MSFT (resolved target) get notes; adidas (unresolved) does not.
    assert result.written == 2
    assert result.deleted == 0
    companies = vault / "companies"
    names = sorted(p.name for p in companies.glob("*.md"))
    assert names == ["MSFT.md", "NVDA.md"]

    nvda = (companies / "NVDA.md").read_text(encoding="utf-8")
    assert "[[MSFT]]" in nvda  # resolved target is linked
    assert "adidas AG" in nvda  # unresolved target shown by name
    assert "[[adidas" not in nvda  # ...but never linked

    msft = (companies / "MSFT.md").read_text(encoding="utf-8")
    assert "## Inbound (named by)" in msft
    assert "[[NVDA]]" in msft  # links back to the company that named it


def test_project_graph_is_idempotent(tmp_config: Config) -> None:
    """Regenerating overwrites the same files rather than appending copies."""
    _seed_edges(tmp_config)
    vault = tmp_config.paths.obsidian_vault_dir

    with database.connection(tmp_config.paths.database_path) as con:
        first = graph.project_graph(con, vault)
        first_text = (vault / "companies" / "NVDA.md").read_text(encoding="utf-8")
        second = graph.project_graph(con, vault)
        second_text = (vault / "companies" / "NVDA.md").read_text(encoding="utf-8")

    assert first.written == second.written == 2
    assert first_text == second_text
    assert len(list((vault / "companies").glob("*.md"))) == 2


def test_project_graph_writes_company_notes_for_open_trade_alerts(tmp_config: Config) -> None:
    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_company_edges(
            con,
            [
                _edge_row(
                    "AAA|BBB|customer",
                    source_ticker="AAA",
                    source_cik=AAA_CIK,
                    target_ticker="BBB",
                    target_cik=BBB_CIK,
                    target_name="Beta",
                    edge_type="customer",
                    evidence="sells into Beta channel",
                )
            ],
        )
        duckdb_store.insert_new_trade_alerts(
            con,
            [
                {
                    "alert_id": "alert-aaa",
                    "kind": "price_gap",
                    "ticker": "AAA",
                    "trigger_key": "gap:AAA:2026-07-24",
                    "fired_at": datetime(2026, 7, 24, 14, 30, tzinfo=UTC),
                    "direction": -1,
                    "entry_ref": 100.0,
                    "stop": 106.0,
                    "target": 88.0,
                    "time_exit_date": date(2026, 7, 31),
                    "confidence": 50,
                    "evidence": json.dumps({"gap_pct": -0.05, "note": "no catalyst"}),
                    "outcome": "open",
                    "source": "derived",
                    "validation_status": "valid",
                }
            ],
        )

    vault = tmp_config.paths.obsidian_vault_dir
    with database.connection(tmp_config.paths.database_path) as con:
        result = graph.project_graph(con, vault)

    assert result.written == 2
    note = (vault / "companies" / "AAA.md").read_text(encoding="utf-8")
    assert "## Trade alerts" in note
    assert "**price_gap** bearish, confidence 50" in note
    assert "basis: -5.0% gap; no catalyst" in note
    assert "## Sells to / Customers" in note
    assert "[[BBB]]" in note

    bbb = (vault / "companies" / "BBB.md").read_text(encoding="utf-8")
    assert "## Inbound (named by)" in bbb
    assert "[[AAA]]" in bbb


def test_anchor_shared_board_edges_are_capped_before_rendering() -> None:
    edges = [
        _edge(
            edge_id=f"edge-{i}",
            source_ticker="AAA",
            target_ticker=f"B{i}",
            target_name=f"Beta {i}",
            edge_type=graph.SHARED_BOARD_MEMBER_EDGE_TYPE,
            times_asserted=i,
        )
        for i in range(8)
    ]

    selected = graph._selected_anchor_board_edge_ids(edges, {"AAA"})

    assert len(selected) == graph._MAX_ANCHOR_BOARD_EDGES_PER_SIDE
    assert "edge-7" in selected
    assert "edge-0" not in selected


# --------------------------------------------------------------------------- #
# render_company_note -- People (officers & directors) section                #
# --------------------------------------------------------------------------- #
def _role(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "canonical_name": "Jane Doe",
        "person_link": "Jane Doe",
        "is_officer": True,
        "is_director": False,
        "is_ten_pct_owner": False,
        "latest_officer_title": "CFO",
    }
    base.update(overrides)
    return base


def test_render_company_note_with_roles_adds_people_section() -> None:
    note = graph.render_company_note("NVDA", [], [], [_role()])

    assert "## People to watch" in note
    assert "[[Jane Doe]]" in note
    assert "Officer (CFO)" in note


def test_render_company_note_without_roles_omits_people_section() -> None:
    note = graph.render_company_note("NVDA", [_edge()], [])
    assert "## People" not in note


def test_render_company_note_role_label_combines_flags() -> None:
    note = graph.render_company_note(
        "NVDA", [], [], [_role(is_officer=True, is_director=True, latest_officer_title="CEO")]
    )
    assert "Director, Officer (CEO)" in note


def test_render_company_note_roles_only_suppresses_no_relationships_message() -> None:
    note = graph.render_company_note("NVDA", [], [], [_role()])
    assert "_No relationships recorded._" not in note


def test_render_company_note_non_qualifying_role_renders_as_plain_text() -> None:
    """§9.1: a role whose person did not clear the inclusion bar has no ``person_link``.

    ``render_company_note`` never decides qualification itself -- it only
    trusts whether the caller (``_group_by_ticker``) set ``person_link`` --
    so a falsy link must fall back to plain text with no wiki-brackets at all.
    """
    role = _role(person_link=None)
    note = graph.render_company_note("NVDA", [], [], [role])

    assert "Jane Doe" in note
    assert "[[Jane Doe]]" not in note
    assert "[[" not in note


# --------------------------------------------------------------------------- #
# render_company_note / render_person_note -- insider cluster buys (§9.2)      #
# --------------------------------------------------------------------------- #
def _cluster_buy_fact(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event_id": "event-1",
        "ticker": "AAA",
        "window_start": date(2026, 1, 3),
        "window_end": date(2026, 1, 15),
        "distinct_insiders": 3,
        "total_dollar_value": 125_000.0,
        "person_ids": ("person-1",),
        "live": True,
        "outcome": None,
        "forward_abnormal_return": None,
    }
    base.update(overrides)
    return base


def _trade_alert_fact(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "price_gap",
        "ticker": "AAA",
        "fired_at": datetime(2026, 7, 24, 14, 30, tzinfo=UTC),
        "direction": 1,
        "entry_ref": 100.0,
        "stop": 94.0,
        "target": 112.0,
        "time_exit_date": date(2026, 7, 31),
        "confidence": 50,
        "evidence": json.dumps(
            {"gap_pct": 0.061, "catalyst": "filing_item/2.02", "note": "no track record yet"}
        ),
        "outcome": "open",
    }
    base.update(overrides)
    return base


def test_render_company_note_shows_open_trade_alert_as_reasoning_context() -> None:
    note = graph.render_company_note("AAA", [], [], [], [], [_trade_alert_fact()])

    assert "## Trade alerts" in note
    assert "**price_gap** bullish, confidence 50" in note
    assert "entry $100.00, stop $94.00, target $112.00, exit 2026-07-31" in note
    assert "basis: +6.1% gap; filing_item/2.02; no track record yet" in note
    assert "  - alert/trade" in note


def test_render_company_note_shows_live_cluster_buy_under_watching() -> None:
    note = graph.render_company_note("AAA", [], [], [], [_cluster_buy_fact()])

    assert "## Insider cluster buys" in note
    assert "### Watching" in note
    assert "### Resolved" not in note
    assert "3 insiders, $125,000 (2026-01-03 to 2026-01-15)" in note
    assert "  - event/insider_cluster_buy" in note


def test_render_company_note_shows_resolved_hit_with_car() -> None:
    fact = _cluster_buy_fact(live=False, outcome="hit", forward_abnormal_return=0.042)
    note = graph.render_company_note("AAA", [], [], [], [fact])

    assert "### Watching" not in note
    assert "### Resolved" in note
    assert "hit (+4.2% CAR, 20d)" in note


def test_render_company_note_shows_resolved_miss() -> None:
    fact = _cluster_buy_fact(live=False, outcome="miss", forward_abnormal_return=-0.01)
    note = graph.render_company_note("AAA", [], [], [], [fact])

    assert "miss (-1.0% CAR, 20d)" in note


def test_render_company_note_shows_resolved_not_yet_graded() -> None:
    """The horizon closed but ``signals build-dataset`` hasn't re-run since."""
    fact = _cluster_buy_fact(live=False, outcome=None, forward_abnormal_return=None)
    note = graph.render_company_note("AAA", [], [], [], [fact])

    assert "### Resolved" in note
    assert "not yet graded" in note


def test_render_company_note_orders_watching_before_resolved() -> None:
    resolved = _cluster_buy_fact(
        event_id="e2", live=False, outcome="hit", forward_abnormal_return=0.01
    )
    watching = _cluster_buy_fact(event_id="e1", live=True)
    note = graph.render_company_note("AAA", [], [], [], [resolved, watching])

    assert note.index("### Watching") < note.index("### Resolved")


def test_render_company_note_cluster_buy_alone_suppresses_no_relationships_message() -> None:
    note = graph.render_company_note("AAA", [], [], [], [_cluster_buy_fact()])
    assert "_No relationships recorded._" not in note


def test_render_person_note_links_cluster_buy_back_to_company() -> None:
    note = graph.render_person_note("Jane Doe", [], [_cluster_buy_fact(ticker="AAA")])

    assert "## Insider cluster buys" in note
    assert "[[AAA]] -- 3 insiders" in note
    assert "  - event/insider_cluster_buy" in note


def test_render_person_note_cluster_buy_alone_suppresses_no_roles_message() -> None:
    note = graph.render_person_note("Jane Doe", [], [_cluster_buy_fact()])
    assert "_No company roles recorded._" not in note


# --------------------------------------------------------------------------- #
# render_person_note -- pure, no I/O                                          #
# --------------------------------------------------------------------------- #
def test_render_person_note_with_roles_links_to_companies() -> None:
    note = graph.render_person_note(
        "Jane Doe",
        [
            {
                "ticker": "NVDA",
                "company_name": "NVIDIA Corporation",
                "is_officer": True,
                "is_director": False,
                "is_ten_pct_owner": False,
                "latest_officer_title": "CFO",
            }
        ],
    )
    assert "# Jane Doe" in note
    assert "## Roles" in note
    assert "[[NVDA]]" in note
    assert "Officer (CFO)" in note


def test_render_person_note_with_no_roles_says_so() -> None:
    note = graph.render_person_note("Jane Doe", [])
    assert "_No company roles recorded._" in note


def test_render_person_note_frontmatter_tags_reflect_roles_held() -> None:
    note = graph.render_person_note(
        "Jane Doe",
        [
            {
                "ticker": "NVDA", "company_name": "NVIDIA",
                "is_officer": False, "is_director": True, "is_ten_pct_owner": False,
                "latest_officer_title": None,
            }
        ],
    )  # fmt: skip
    assert "  - role/director" in note
    assert "  - role/officer" not in note


# --------------------------------------------------------------------------- #
# project_person_notes -- reads the DB, writes the vault                      #
#                                                                              #
# Jane Doe here is a director at BOTH NVDA and MSFT -- an interlock -- so she #
# clears the §9.1 bar and is the "happy path" for a qualifying person.        #
# --------------------------------------------------------------------------- #
def _seed_role_graph(config: Config) -> None:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(
            con,
            [
                {
                    "company_id": "company-nvda", "cik": NVDA_CIK, "ticker": "NVDA",
                    "company_name": "NVIDIA Corporation", "source": "sec",
                    "validation_status": "valid",
                },
                {
                    "company_id": "company-msft", "cik": MSFT_CIK, "ticker": "MSFT",
                    "company_name": "Microsoft Corporation", "source": "sec",
                    "validation_status": "valid",
                },
            ],
        )
        duckdb_store.upsert_signal_status(
            con, [_signal_status_row(graph.SHARED_BOARD_MEMBER_EDGE_TYPE)]
        )
        duckdb_store.upsert_people(
            con,
            [
                {
                    "person_id": "person-jane", "reporting_owner_cik": "9000000001",
                    "canonical_name": "Jane Doe", "name_variants": "[]",
                    "source": "sec", "validation_status": "valid",
                },
            ],
        )
        duckdb_store.upsert_role_memberships(
            con,
            [
                {
                    "role_id": "role-1", "person_id": "person-jane",
                    "company_id": "company-nvda", "is_officer": True, "is_director": True,
                    "is_ten_pct_owner": False, "latest_officer_title": "CFO",
                    "source_filing_count": 1, "source": "sec", "validation_status": "valid",
                },
                {
                    "role_id": "role-2", "person_id": "person-jane",
                    "company_id": "company-msft", "is_officer": False, "is_director": True,
                    "is_ten_pct_owner": False, "latest_officer_title": None,
                    "source_filing_count": 1, "source": "sec", "validation_status": "valid",
                },
            ],
        )


def test_project_person_notes_writes_one_note_per_person(tmp_config: Config) -> None:
    _seed_role_graph(tmp_config)
    vault = tmp_config.paths.obsidian_vault_dir

    with database.connection(tmp_config.paths.database_path) as con:
        result = graph.project_person_notes(con, vault)

    assert result.written == 1
    assert result.deleted == 0
    note = (vault / "people" / "Jane Doe.md").read_text(encoding="utf-8")
    assert "[[NVDA]]" in note
    assert "[[MSFT]]" in note
    assert "Officer (CFO)" in note
    assert "Director" in note


def test_project_graph_also_covers_companies_with_only_roles(tmp_config: Config) -> None:
    """A company with qualifying officers/directors but no company_edges still gets a note."""
    _seed_role_graph(tmp_config)
    vault = tmp_config.paths.obsidian_vault_dir

    with database.connection(tmp_config.paths.database_path) as con:
        result = graph.project_graph(con, vault)

    assert result.written == 2
    nvda = (vault / "companies" / "NVDA.md").read_text(encoding="utf-8")
    assert "## People to watch" in nvda
    assert "[[Jane Doe]]" in nvda


def test_project_person_notes_disambiguates_name_collisions(tmp_config: Config) -> None:
    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(
            con,
            [
                {
                    "company_id": "company-nvda", "cik": NVDA_CIK, "ticker": "NVDA",
                    "company_name": "NVIDIA", "source": "sec", "validation_status": "valid",
                },
                {
                    "company_id": "company-msft", "cik": MSFT_CIK, "ticker": "MSFT",
                    "company_name": "Microsoft", "source": "sec", "validation_status": "valid",
                },
            ],
        )
        duckdb_store.upsert_people(
            con,
            [
                {
                    "person_id": "person-a", "reporting_owner_cik": "9000000001",
                    "canonical_name": "Jane Doe", "name_variants": "[]",
                    "source": "sec", "validation_status": "valid",
                },
                {
                    "person_id": "person-b", "reporting_owner_cik": "9000000002",
                    "canonical_name": "Jane Doe", "name_variants": "[]",
                    "source": "sec", "validation_status": "valid",
                },
            ],
        )
        # Both are directors at both companies -- both interlock independently
        # (the point under test is filename disambiguation, not the interlock
        # graph's shape), so both clear the §9.1 bar.
        duckdb_store.upsert_role_memberships(
            con,
            [
                {
                    "role_id": "role-a1", "person_id": "person-a", "company_id": "company-nvda",
                    "is_officer": False, "is_director": True, "is_ten_pct_owner": False,
                    "latest_officer_title": None, "source_filing_count": 1,
                    "source": "sec", "validation_status": "valid",
                },
                {
                    "role_id": "role-a2", "person_id": "person-a", "company_id": "company-msft",
                    "is_officer": False, "is_director": True, "is_ten_pct_owner": False,
                    "latest_officer_title": None, "source_filing_count": 1,
                    "source": "sec", "validation_status": "valid",
                },
                {
                    "role_id": "role-b1", "person_id": "person-b", "company_id": "company-nvda",
                    "is_officer": False, "is_director": True, "is_ten_pct_owner": False,
                    "latest_officer_title": None, "source_filing_count": 1,
                    "source": "sec", "validation_status": "valid",
                },
                {
                    "role_id": "role-b2", "person_id": "person-b", "company_id": "company-msft",
                    "is_officer": False, "is_director": True, "is_ten_pct_owner": False,
                    "latest_officer_title": None, "source_filing_count": 1,
                    "source": "sec", "validation_status": "valid",
                },
            ],
        )
        duckdb_store.upsert_signal_status(
            con, [_signal_status_row(graph.SHARED_BOARD_MEMBER_EDGE_TYPE)]
        )

    vault = tmp_config.paths.obsidian_vault_dir
    with database.connection(tmp_config.paths.database_path) as con:
        result = graph.project_person_notes(con, vault)

    assert result.written == 2
    names = sorted(p.name for p in (vault / "people").glob("*.md"))
    assert names == ["Jane Doe (person-a).md", "Jane Doe (person-b).md"]


def test_project_person_notes_disambiguates_case_insensitive_collisions(
    tmp_config: Config,
) -> None:
    """Two names differing only by case still collide on a case-insensitive
    filesystem (macOS/APFS default) -- found against the real vault
    (2026-07-26), where 9 of 10,472 qualifying persons silently overwrote each
    other before this was fixed. Both must get a suffixed filename, not just
    whichever pair happens to match with identical casing.
    """
    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(
            con,
            [
                {
                    "company_id": "company-nvda", "cik": NVDA_CIK, "ticker": "NVDA",
                    "company_name": "NVIDIA", "source": "sec", "validation_status": "valid",
                },
                {
                    "company_id": "company-msft", "cik": MSFT_CIK, "ticker": "MSFT",
                    "company_name": "Microsoft", "source": "sec", "validation_status": "valid",
                },
            ],
        )
        duckdb_store.upsert_people(
            con,
            [
                {
                    "person_id": "person-a", "reporting_owner_cik": "9000000001",
                    "canonical_name": "Jane Doe", "name_variants": "[]",
                    "source": "sec", "validation_status": "valid",
                },
                {
                    "person_id": "person-b", "reporting_owner_cik": "9000000002",
                    "canonical_name": "JANE DOE", "name_variants": "[]",
                    "source": "sec", "validation_status": "valid",
                },
            ],
        )
        duckdb_store.upsert_role_memberships(
            con,
            [
                _role_membership_row("role-a1", "person-a", "company-nvda", is_director=True),
                _role_membership_row("role-a2", "person-a", "company-msft", is_director=True),
                _role_membership_row("role-b1", "person-b", "company-nvda", is_director=True),
                _role_membership_row("role-b2", "person-b", "company-msft", is_director=True),
            ],
        )
        duckdb_store.upsert_signal_status(
            con, [_signal_status_row(graph.SHARED_BOARD_MEMBER_EDGE_TYPE)]
        )

    vault = tmp_config.paths.obsidian_vault_dir
    with database.connection(tmp_config.paths.database_path) as con:
        result = graph.project_person_notes(con, vault)

    assert result.written == 2
    names = sorted(p.name for p in (vault / "people").glob("*.md"))
    assert names == ["JANE DOE (person-b).md", "Jane Doe (person-a).md"]


def test_project_person_notes_is_idempotent(tmp_config: Config) -> None:
    _seed_role_graph(tmp_config)
    vault = tmp_config.paths.obsidian_vault_dir

    with database.connection(tmp_config.paths.database_path) as con:
        first = graph.project_person_notes(con, vault)
        first_text = (vault / "people" / "Jane Doe.md").read_text(encoding="utf-8")
        second = graph.project_person_notes(con, vault)
        second_text = (vault / "people" / "Jane Doe.md").read_text(encoding="utf-8")

    assert first.written == second.written == 1
    assert first_text == second_text


def test_project_person_notes_respects_configured_interlock_cap(tmp_config: Config) -> None:
    """A director capped out of ``compute_interlocks`` does not qualify either."""
    _seed_role_graph(tmp_config)
    vault = tmp_config.paths.obsidian_vault_dir

    with database.connection(tmp_config.paths.database_path) as con:
        result = graph.project_person_notes(con, vault, max_companies_per_person=1)

    assert result.written == 0
    assert list((vault / "people").glob("*.md")) == []


# --------------------------------------------------------------------------- #
# §9.1 -- vault inclusion is earned, not assumed                               #
# --------------------------------------------------------------------------- #
AAA_CIK = "0000000101"
BBB_CIK = "0000000102"
CCC_CIK = "0000000103"
DDD_CIK = "0000000104"


def _company_row(company_id: str, cik: str, ticker: str, name: str) -> dict[str, Any]:
    return {
        "company_id": company_id, "cik": cik, "ticker": ticker,
        "company_name": name, "source": "sec", "validation_status": "valid",
    }  # fmt: skip


def _person_row(person_id: str, cik: str, name: str) -> dict[str, Any]:
    return {
        "person_id": person_id, "reporting_owner_cik": cik, "canonical_name": name,
        "name_variants": "[]", "source": "sec", "validation_status": "valid",
    }  # fmt: skip


def _role_membership_row(
    role_id: str,
    person_id: str,
    company_id: str,
    *,
    is_officer: bool = False,
    is_director: bool = False,
    is_ten_pct_owner: bool = False,
    latest_officer_title: str | None = None,
) -> dict[str, Any]:
    return {
        "role_id": role_id, "person_id": person_id, "company_id": company_id,
        "is_officer": is_officer, "is_director": is_director,
        "is_ten_pct_owner": is_ten_pct_owner, "latest_officer_title": latest_officer_title,
        "source_filing_count": 1, "source": "sec", "validation_status": "valid",
    }  # fmt: skip


def _cluster_buy_event_id(event_key: str) -> str:
    return hashing.content_hash("event", EventType.INSIDER_CLUSTER_BUY, event_key)


def _cluster_buy_event_row(
    *,
    event_key: str,
    company_id: str,
    cik: str,
    ticker: str,
    available_time: datetime,
    person_ids: list[str],
    total_dollar_value: float = 10_000.0,
) -> dict[str, Any]:
    payload = {
        "owner_ciks": [],
        "person_ids": person_ids,
        "total_shares": 100.0,
        "distinct_insiders": len(person_ids),
        "window_days": 5,
        "window_start": available_time.isoformat(),
        "accession_numbers": [],
        "magnitude_unit": "usd",
    }
    record = EventRecord(
        event_id=_cluster_buy_event_id(event_key),
        event_type=EventType.INSIDER_CLUSTER_BUY,
        event_key=event_key,
        company_id=company_id,
        cik=cik,
        ticker=ticker,
        event_time=available_time,
        available_time=available_time,
        magnitude=total_dollar_value,
        direction=1,
        payload=payload,
        source=Source.DERIVED,
        schema_version="1.0.0",
        collected_time=utcnow(),
    )
    return record.to_row()


def _seed_event_sample(
    config: Config, *, event_id: str, horizon_days: int, forward_abnormal_return: float
) -> None:
    record = EventSampleRecord(
        sample_id=hashing.content_hash("sample", event_id, SELF_EDGE, horizon_days),
        event_id=event_id,
        edge_id=SELF_EDGE,
        horizon_days=horizon_days,
        forward_abnormal_return=forward_abnormal_return,
        source=Source.DERIVED,
    )
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_event_samples(con, [record.to_row()])


def test_longest_evaluated_horizon_matches_the_real_dataset_builder() -> None:
    """The design doc cites "20 days" -- verified against the source, not assumed."""
    assert graph.LONGEST_EVALUATED_HORIZON_DAYS == max(dataset.DEFAULT_HORIZONS) == 20


def test_officer_only_no_interlock_no_event_gets_no_note_but_stays_plain_text(
    tmp_config: Config,
) -> None:
    """The exact bug this amendment fixes: a title alone is not Obsidian-worthy."""
    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(con, [_company_row("company-aaa", AAA_CIK, "AAA", "Alpha")])
        duckdb_store.upsert_people(con, [_person_row("person-1", "9990000001", "Jane Doe")])
        duckdb_store.upsert_role_memberships(
            con,
            [
                _role_membership_row(
                    "role-1", "person-1", "company-aaa",
                    is_officer=True, latest_officer_title="CFO",
                )
            ],
        )  # fmt: skip

    vault = tmp_config.paths.obsidian_vault_dir
    today = date(2026, 1, 1)
    with database.connection(tmp_config.paths.database_path) as con:
        people_result = graph.project_person_notes(con, vault, today=today)
        company_result = graph.project_graph(con, vault, today=today)

    assert people_result.written == 0
    assert list((vault / "people").glob("*.md")) == []

    assert company_result.written == 0
    assert list((vault / "companies").glob("*.md")) == []


def test_pure_ten_pct_owner_never_qualifies_even_with_live_event(tmp_config: Config) -> None:
    """§9.1: an institutional 10%-owner is ownership-flow data, never a node."""
    today = date(2026, 6, 1)
    live_available_time = datetime(2026, 5, 28, tzinfo=UTC)  # window_end+20d >= today: live

    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(con, [_company_row("company-bbb", BBB_CIK, "BBB", "Beta")])
        duckdb_store.upsert_people(con, [_person_row("person-fund", "9990000002", "Big Fund LLC")])
        duckdb_store.upsert_role_memberships(
            con,
            [
                _role_membership_row(
                    "role-fund", "person-fund", "company-bbb", is_ten_pct_owner=True
                )
            ],
        )
        duckdb_store.upsert_events(
            con,
            [
                _cluster_buy_event_row(
                    event_key="CB-FUND",
                    company_id="company-bbb",
                    cik=BBB_CIK,
                    ticker="BBB",
                    available_time=live_available_time,
                    person_ids=["person-fund"],
                )
            ],
        )

    vault = tmp_config.paths.obsidian_vault_dir
    with database.connection(tmp_config.paths.database_path) as con:
        people_result = graph.project_person_notes(con, vault, today=today)
        company_result = graph.project_graph(con, vault, today=today)

    assert people_result.written == 0
    assert list((vault / "people").glob("*.md")) == []

    bbb = (vault / "companies" / "BBB.md").read_text(encoding="utf-8")
    assert "Big Fund LLC" not in bbb
    assert "[[Big Fund LLC]]" not in bbb
    assert "### Watching" in bbb  # the event itself still renders on the company note
    assert company_result.written == 1


def test_officer_qualifies_via_live_cluster_buy_event(tmp_config: Config) -> None:
    """§9.1(b): an officer with no interlock still earns a note while the event is live."""
    today = date(2026, 6, 1)
    live_available_time = datetime(2026, 5, 20, tzinfo=UTC)  # window_end+20d = 2026-06-09 >= today

    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(con, [_company_row("company-ccc", CCC_CIK, "CCC", "Gamma")])
        duckdb_store.upsert_people(con, [_person_row("person-officer", "9990000003", "Sam Roe")])
        duckdb_store.upsert_role_memberships(
            con,
            [
                _role_membership_row(
                    "role-officer", "person-officer", "company-ccc", is_officer=True
                )
            ],
        )
        duckdb_store.upsert_events(
            con,
            [
                _cluster_buy_event_row(
                    event_key="CB-LIVE",
                    company_id="company-ccc",
                    cik=CCC_CIK,
                    ticker="CCC",
                    available_time=live_available_time,
                    person_ids=["person-officer"],
                )
            ],
        )

    vault = tmp_config.paths.obsidian_vault_dir
    with database.connection(tmp_config.paths.database_path) as con:
        people_result = graph.project_person_notes(con, vault, today=today)
        company_result = graph.project_graph(con, vault, today=today)

    assert people_result.written == 1
    note = (vault / "people" / "Sam Roe.md").read_text(encoding="utf-8")
    assert "## Insider cluster buys" in note
    assert "### Watching" in note
    assert "### Resolved" not in note
    assert "[[CCC]]" in note

    ccc = (vault / "companies" / "CCC.md").read_text(encoding="utf-8")
    assert "[[Sam Roe]]" in ccc
    assert "### Watching" in ccc
    assert company_result.written == 1


def test_interlocking_director_qualifies_via_role_alone(tmp_config: Config) -> None:
    """§9.1(a): a director on 2+ tracked boards qualifies even with zero events."""
    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(
            con,
            [
                _company_row("company-ccc", CCC_CIK, "CCC", "Gamma"),
                _company_row("company-ddd", DDD_CIK, "DDD", "Delta"),
            ],
        )
        duckdb_store.upsert_people(con, [_person_row("person-dir", "9990000004", "Pat Lee")])
        duckdb_store.upsert_role_memberships(
            con,
            [
                _role_membership_row("role-dir-1", "person-dir", "company-ccc", is_director=True),
                _role_membership_row("role-dir-2", "person-dir", "company-ddd", is_director=True),
            ],
        )
        duckdb_store.upsert_signal_status(
            con, [_signal_status_row(graph.SHARED_BOARD_MEMBER_EDGE_TYPE)]
        )

    vault = tmp_config.paths.obsidian_vault_dir
    with database.connection(tmp_config.paths.database_path) as con:
        result = graph.project_person_notes(con, vault, today=date(2026, 1, 1))

    assert result.written == 1
    note = (vault / "people" / "Pat Lee.md").read_text(encoding="utf-8")
    assert "[[CCC]]" in note
    assert "[[DDD]]" in note
    assert "## Insider cluster buys" not in note


def test_resolved_cluster_buy_shows_graded_hit_and_prunes_the_now_stale_person_note(
    tmp_config: Config,
) -> None:
    """A live event that qualifies a person stops qualifying once resolved (§9.2/§9.3)."""
    event_key = "CB-HIT"
    available_time = datetime(2026, 5, 20, tzinfo=UTC)
    live_today = date(2026, 6, 1)  # window_end+20d = 2026-06-09 >= today: live
    resolved_today = date(2026, 7, 1)  # now well past the 20d horizon: resolved

    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(con, [_company_row("company-ddd", DDD_CIK, "DDD", "Delta")])
        duckdb_store.upsert_people(con, [_person_row("person-officer", "9990000005", "Alex Kim")])
        duckdb_store.upsert_role_memberships(
            con,
            [
                _role_membership_row(
                    "role-officer", "person-officer", "company-ddd", is_officer=True
                )
            ],
        )
        duckdb_store.upsert_events(
            con,
            [
                _cluster_buy_event_row(
                    event_key=event_key,
                    company_id="company-ddd",
                    cik=DDD_CIK,
                    ticker="DDD",
                    available_time=available_time,
                    person_ids=["person-officer"],
                )
            ],
        )

    vault = tmp_config.paths.obsidian_vault_dir
    with database.connection(tmp_config.paths.database_path) as con:
        first = graph.project_person_notes(con, vault, today=live_today)

    assert first.written == 1
    assert (vault / "people" / "Alex Kim.md").exists()

    # The horizon has now closed and the dataset was rebuilt: seed the graded
    # outcome and re-project as of a later date.
    _seed_event_sample(
        tmp_config,
        event_id=_cluster_buy_event_id(event_key),
        horizon_days=20,
        forward_abnormal_return=0.05,
    )
    with database.connection(tmp_config.paths.database_path) as con:
        second = graph.project_person_notes(con, vault, today=resolved_today)
        company_result = graph.project_graph(con, vault, today=resolved_today)

    assert second.written == 0
    assert second.deleted == 1
    assert not (vault / "people" / "Alex Kim.md").exists()

    ddd = (vault / "companies" / "DDD.md").read_text(encoding="utf-8")
    assert "### Resolved" in ddd
    assert "hit (+5.0% CAR, 20d)" in ddd
    assert "[[Alex Kim]]" not in ddd  # no longer qualifies -- back to plain text
    assert "Alex Kim" not in ddd
    assert company_result.written == 1


def test_resolved_cluster_buy_without_a_sample_is_omitted_from_the_vault(
    tmp_config: Config,
) -> None:
    """The horizon closed but no grade exists, so this is not AI-facing context."""
    available_time = datetime(2026, 1, 1, tzinfo=UTC)
    today = date(2026, 6, 1)  # well past the 20d horizon; no event_samples row exists

    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(con, [_company_row("company-ccc", CCC_CIK, "CCC", "Gamma")])
        duckdb_store.upsert_people(con, [_person_row("person-officer", "9990000006", "Ari Vance")])
        duckdb_store.upsert_role_memberships(
            con,
            [
                _role_membership_row(
                    "role-officer", "person-officer", "company-ccc", is_officer=True
                )
            ],
        )
        duckdb_store.upsert_events(
            con,
            [
                _cluster_buy_event_row(
                    event_key="CB-UNGRADED",
                    company_id="company-ccc",
                    cik=CCC_CIK,
                    ticker="CCC",
                    available_time=available_time,
                    person_ids=["person-officer"],
                )
            ],
        )

    vault = tmp_config.paths.obsidian_vault_dir
    with database.connection(tmp_config.paths.database_path) as con:
        people_result = graph.project_person_notes(con, vault, today=today)
        company_result = graph.project_graph(con, vault, today=today)

    assert people_result.written == 0  # resolved, not live -- no longer an inclusion reason
    assert company_result.written == 0
    assert not (vault / "companies" / "CCC.md").exists()


# --------------------------------------------------------------------------- #
# §9.3 -- the vault writer deletes, not just adds                             #
# --------------------------------------------------------------------------- #
def test_project_graph_prunes_a_stale_company_note(tmp_config: Config) -> None:
    _seed_edges(tmp_config)
    vault = tmp_config.paths.obsidian_vault_dir
    (vault / "companies").mkdir(parents=True, exist_ok=True)
    (vault / "companies" / "ZZZZ.md").write_text("stale", encoding="utf-8")

    with database.connection(tmp_config.paths.database_path) as con:
        result = graph.project_graph(con, vault)

    assert result.deleted == 1
    assert not (vault / "companies" / "ZZZZ.md").exists()
    assert (vault / "companies" / "NVDA.md").exists()


def test_project_person_notes_prunes_a_stale_person_note(tmp_config: Config) -> None:
    _seed_role_graph(tmp_config)
    vault = tmp_config.paths.obsidian_vault_dir
    (vault / "people").mkdir(parents=True, exist_ok=True)
    (vault / "people" / "Ghost Person.md").write_text("stale", encoding="utf-8")

    with database.connection(tmp_config.paths.database_path) as con:
        result = graph.project_person_notes(con, vault)

    assert result.deleted == 1
    assert not (vault / "people" / "Ghost Person.md").exists()
    assert (vault / "people" / "Jane Doe.md").exists()


def test_project_graph_never_touches_files_outside_its_own_subdirectory(
    tmp_config: Config,
) -> None:
    _seed_edges(tmp_config)
    vault = tmp_config.paths.obsidian_vault_dir
    (vault / "briefings").mkdir(parents=True, exist_ok=True)
    (vault / "briefings" / "2026-01-01.md").write_text("briefing", encoding="utf-8")
    (vault / ".obsidian").mkdir(parents=True, exist_ok=True)
    (vault / ".obsidian" / "config").write_text("{}", encoding="utf-8")

    with database.connection(tmp_config.paths.database_path) as con:
        graph.project_graph(con, vault)
        graph.project_person_notes(con, vault)

    assert (vault / "briefings" / "2026-01-01.md").exists()
    assert (vault / ".obsidian" / "config").exists()
