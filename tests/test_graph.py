"""Tests for the Obsidian projection of the company relationship graph.

The reader-facing contract that would quietly break the graph view if it
regressed: a resolved counterparty must become a ``[[TICKER]]`` wiki-link so
Obsidian draws an edge, an unresolved target must stay plain text (there is no
note to link to), an inbound edge must link back to whoever named the company,
and the projection must be regenerable -- one note per company, overwritten in
place. ``render_company_note`` is pure, so most of this is tested without a
database; ``project_graph`` is exercised end to end against a temp DuckDB.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from market_intelligence import database
from market_intelligence.autopilot import graph
from market_intelligence.config import Config
from market_intelligence.schemas.edges import ResolutionStatus
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


def test_project_graph_writes_one_note_per_company_with_links(tmp_config: Config) -> None:
    _seed_edges(tmp_config)
    vault = tmp_config.paths.obsidian_vault_dir

    with database.connection(tmp_config.paths.database_path) as con:
        written = graph.project_graph(con, vault)

    # NVDA (source) and MSFT (resolved target) get notes; adidas (unresolved) does not.
    assert written == 2
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

    assert first == second == 2
    assert first_text == second_text
    assert len(list((vault / "companies").glob("*.md"))) == 2
