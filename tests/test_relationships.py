"""Tests for the relationship collector: extract, filter self-edges, resolve, store.

The behaviours that would silently corrupt the graph if they broke: a self-edge
must never be stored, an unresolved target must be kept but marked
non-validatable, and the same relationship across filings must dedupe to one row.
The LLM is stubbed, so these run offline and deterministically.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from market_intelligence import database
from market_intelligence.collectors import relationships
from market_intelligence.config import Config
from market_intelligence.schemas.edges import ResolutionStatus
from market_intelligence.storage import duckdb as duckdb_store

SOURCE_CIK = "0001045810"  # NVDA
MSFT_CIK = "0000789019"


class StubProvider:
    """Returns a fixed edge set, so extraction is deterministic and offline."""

    model = "stub-model"

    def __init__(self, edges: list[dict[str, Any]]) -> None:
        self._edges = edges
        self.calls = 0

    def complete_json(self, *, system: str, user: str, schema: Any, temperature: float = 0.0):
        self.calls += 1
        return {"edges": self._edges}


def _seed(config: Config, text: str) -> str:
    """Seed NVDA + MSFT as priced companies and one NVDA Item 1 section on disk."""
    section_path = Path(config.paths.normalized_dir) / "nvda_item1.txt"
    section_path.parent.mkdir(parents=True, exist_ok=True)
    section_path.write_text(text, encoding="utf-8")

    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(
            con,
            [
                {
                    "company_id": "c-nvda",
                    "cik": SOURCE_CIK,
                    "ticker": "NVDA",
                    "company_name": "NVIDIA Corporation",
                    "source": "sec",
                },
                {
                    "company_id": "c-msft",
                    "cik": MSFT_CIK,
                    "ticker": "MSFT",
                    "company_name": "Microsoft Corporation",
                    "source": "sec",
                },
            ],
        )
        # Both companies need a return series to survive priced_only.
        duckdb_store.upsert_daily_returns(
            con,
            [
                {
                    "return_id": "r-nvda",
                    "symbol": "NVDA",
                    "price_date": date(2026, 1, 2),
                    "total_return": 0.0,
                    "source": "market",
                },
                {
                    "return_id": "r-msft",
                    "symbol": "MSFT",
                    "price_date": date(2026, 1, 2),
                    "total_return": 0.0,
                    "source": "market",
                },
            ],
        )
        con.execute(
            """
            INSERT INTO filing_sections
              (section_id, cik, accession_number, form, item_code, report_date, text_path, source)
            VALUES (?, ?, ?, '10-K', '1', ?, ?, 'sec')
            """,
            ["sec-1", SOURCE_CIK, "0001045810-26-000001", date(2026, 1, 15), str(section_path)],
        )
    return str(section_path)


def _run(config: Config, edges: list[dict[str, Any]], **kwargs: Any):
    return relationships.sync(config, provider=StubProvider(edges), **kwargs)


def test_resolvable_edge_is_stored_and_validatable(tmp_config: Config) -> None:
    _seed(tmp_config, "NVIDIA sells GPUs to Microsoft for its data centers.")

    summary = _run(
        tmp_config,
        [
            {
                "target": "Microsoft Corporation",
                "type": "customer",
                "evidence": "sells to Microsoft",
                "confidence": 0.9,
            }
        ],
    )

    assert summary.collected == 1
    assert summary.stage["validatable_edges"] == 1
    with database.connection(tmp_config.paths.database_path) as con:
        row = con.execute(
            "SELECT source_ticker, target_ticker, edge_type, resolution_status FROM company_edges"
        ).fetchone()
    assert row == ("NVDA", "MSFT", "customer", ResolutionStatus.RESOLVED.value)


def test_self_edge_is_dropped(tmp_config: Config) -> None:
    """The 'unnamed counterparty filled in with the filer' hallucination."""
    _seed(tmp_config, "Our largest supplier accounted for 11% of purchases.")

    summary = _run(
        tmp_config,
        [
            {
                "target": "NVIDIA Corporation",
                "type": "supplier",
                "evidence": "our largest supplier",
                "confidence": 0.8,
            }
        ],
    )

    assert summary.collected == 0
    assert summary.stage["self_edges_dropped"] == 1
    with database.connection(tmp_config.paths.database_path) as con:
        assert con.execute("SELECT count(*) FROM company_edges").fetchone()[0] == 0


def test_unresolved_target_is_kept_but_not_validatable(tmp_config: Config) -> None:
    _seed(tmp_config, "We compete with adidas internationally.")

    summary = _run(
        tmp_config,
        [
            {
                "target": "adidas AG",
                "type": "competitor",
                "evidence": "compete with adidas",
                "confidence": 1.0,
            }
        ],
    )

    assert summary.collected == 1
    assert summary.stage["unresolved_edges"] == 1
    with database.connection(tmp_config.paths.database_path) as con:
        status, cik = con.execute(
            "SELECT resolution_status, target_cik FROM company_edges"
        ).fetchone()
    assert status != ResolutionStatus.RESOLVED.value
    assert cik is None


def test_unknown_edge_type_is_skipped(tmp_config: Config) -> None:
    _seed(tmp_config, "Some text.")

    summary = _run(
        tmp_config,
        [
            {
                "target": "Microsoft Corporation",
                "type": "frenemy",
                "evidence": "x",
                "confidence": 0.5,
            }
        ],
    )

    assert summary.collected == 0


def test_rerun_does_not_reprocess_or_duplicate(tmp_config: Config) -> None:
    """The resumability guard: an already-extracted filing is skipped."""
    _seed(tmp_config, "NVIDIA sells to Microsoft.")
    edges = [
        {"target": "Microsoft Corporation", "type": "customer", "evidence": "e", "confidence": 0.9}
    ]

    first = _run(tmp_config, edges)
    provider = StubProvider(edges)
    second = relationships.sync(tmp_config, provider=provider)

    assert first.collected == 1
    assert second.stage["candidate_sections"] == 0
    assert provider.calls == 0  # no section fed to the model on the second run
    with database.connection(tmp_config.paths.database_path) as con:
        assert con.execute("SELECT count(*) FROM company_edges").fetchone()[0] == 1
