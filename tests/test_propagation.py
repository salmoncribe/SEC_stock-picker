"""Tests for the propagation dataset builder.

The one behaviour that matters: a source's event must be scored against the
*target's* return series, tagged with the relationship type, so the existing
gate judges it as a propagation cell. Getting the source/target wiring backwards
would measure the wrong company and quietly invalidate every propagation result.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from market_intelligence import database
from market_intelligence.config import Config
from market_intelligence.signals import dataset
from market_intelligence.storage import duckdb as duckdb_store

NKE_CIK = "0000320187"


def _returns(symbol: str, start: date, values: list[float]) -> list[dict]:
    """A daily abnormal-return series on consecutive calendar days."""
    from datetime import timedelta

    rows = []
    for i, v in enumerate(values):
        d = start + timedelta(days=i)
        rows.append(
            {
                "return_id": f"r-{symbol}-{d}",
                "symbol": symbol,
                "price_date": d,
                "total_return": v,
                "abnormal_return": v,
                "source": "market",
            }
        )
    return rows


def _seed(config: Config, *, edge_type: str = "competitor", resolution: str = "resolved") -> None:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        # ONON (target) has a clean return series in the discovery era; NKE
        # (source) needs one only to pass the validatable-edge existence check.
        duckdb_store.upsert_daily_returns(
            con, _returns("ONON", date(2020, 3, 2), [0.01, 0.02, -0.01, 0.03, 0.00, 0.04])
        )
        duckdb_store.upsert_daily_returns(con, _returns("NKE", date(2020, 3, 2), [0.0, 0.0, 0.0]))
        duckdb_store.upsert_events(
            con,
            [
                {
                    "event_id": "evt-nke-1",
                    "event_type": "insider_transaction",
                    "event_key": "nke-1",
                    "event_subtype": "P",
                    "ticker": "NKE",
                    "available_time": datetime(2020, 3, 2, tzinfo=UTC),
                    "magnitude": 100000.0,
                    "direction": 1,
                }
            ],
        )
        duckdb_store.upsert_company_edges(
            con,
            [
                {
                    "edge_id": "e1",
                    "edge_key": f"{NKE_CIK}|ONON|{edge_type}",
                    "source_cik": NKE_CIK,
                    "source_ticker": "NKE",
                    "target": "ONON",
                    "target_name": "On Holding AG",
                    "target_cik": "0001858681",
                    "target_ticker": "ONON",
                    "edge_type": edge_type,
                    "resolution_status": resolution,
                    "times_asserted": 1,
                    "source": "derived",
                }
            ],
        )


def test_source_event_is_scored_against_the_target_returns(tmp_config: Config) -> None:
    _seed(tmp_config)

    summary = dataset.build_propagation(tmp_config, horizons=(1,))

    assert summary.inserted == 1
    with database.connection(tmp_config.paths.database_path) as con:
        row = con.execute(
            """
            SELECT edge_id, source_ticker, target_ticker, forward_abnormal_return, split
            FROM event_samples
            """
        ).fetchone()
    edge_id, source, target, fwd, split = row
    assert (edge_id, source, target) == ("competitor", "NKE", "ONON")
    # Horizon-1 window opens the trading day strictly after 2020-03-02, i.e.
    # ONON's 2020-03-03 abnormal return of +0.02 -- the target's move, not NKE's.
    assert fwd == pytest.approx(0.02)
    assert split == "discovery"


def test_unresolved_edge_is_not_measured(tmp_config: Config) -> None:
    _seed(tmp_config, resolution="unresolved")

    summary = dataset.build_propagation(tmp_config, horizons=(1,))

    assert summary.inserted == 0


def test_self_pointing_edge_is_skipped(tmp_config: Config) -> None:
    """A resolved edge whose target equals its source is not propagation."""
    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_daily_returns(
            con, _returns("NKE", date(2020, 3, 2), [0.01, 0.02, 0.03])
        )
        duckdb_store.upsert_events(
            con,
            [
                {
                    "event_id": "evt-nke-1",
                    "event_type": "insider_transaction",
                    "event_key": "nke-1",
                    "event_subtype": "P",
                    "ticker": "NKE",
                    "available_time": datetime(2020, 3, 2, tzinfo=UTC),
                }
            ],
        )
        duckdb_store.upsert_company_edges(
            con,
            [
                {
                    "edge_id": "e-self",
                    "edge_key": f"{NKE_CIK}|NKE|competitor",
                    "source_cik": NKE_CIK,
                    "source_ticker": "NKE",
                    "target": "NKE",
                    "target_name": "NIKE, Inc.",
                    "target_cik": NKE_CIK,
                    "target_ticker": "NKE",
                    "edge_type": "competitor",
                    "resolution_status": "resolved",
                    "times_asserted": 1,
                    "source": "derived",
                }
            ],
        )

    summary = dataset.build_propagation(tmp_config, horizons=(1,))

    assert summary.inserted == 0


def test_one_event_fans_out_to_multiple_targets(tmp_config: Config) -> None:
    """A single source event points at two targets -> two distinct samples."""
    _seed(tmp_config)
    with database.connection(tmp_config.paths.database_path) as con:
        duckdb_store.upsert_daily_returns(
            con, _returns("LULU", date(2020, 3, 2), [0.01, -0.02, 0.01, 0.0])
        )
        duckdb_store.upsert_company_edges(
            con,
            [
                {
                    "edge_id": "e2",
                    "edge_key": f"{NKE_CIK}|LULU|competitor",
                    "source_cik": NKE_CIK,
                    "source_ticker": "NKE",
                    "target": "LULU",
                    "target_name": "lululemon athletica inc.",
                    "target_cik": "0001397187",
                    "target_ticker": "LULU",
                    "edge_type": "competitor",
                    "resolution_status": "resolved",
                    "times_asserted": 1,
                    "source": "derived",
                }
            ],
        )

    summary = dataset.build_propagation(tmp_config, horizons=(1,))

    assert summary.inserted == 2
    with database.connection(tmp_config.paths.database_path) as con:
        targets = {
            r[0] for r in con.execute("SELECT DISTINCT target_ticker FROM event_samples").fetchall()
        }
    assert targets == {"ONON", "LULU"}
