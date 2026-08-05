"""The allowlist gate: an ACTIVE ladder cell is necessary but not sufficient.

`event_alerts` must only fire for a cell that is BOTH `signal_status.status
== 'active'` AND listed in `signals.approved_signals.APPROVED_CELLS`. These
tests seed two otherwise-identical active cells -- one approved, one not --
and check only the approved one produces an alert, for both the self-edge
and the propagation join.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from market_intelligence import database
from market_intelligence.autopilot import briefing as briefing_builder
from market_intelligence.config import Config
from market_intelligence.signals.approved_signals import APPROVED_CELLS, is_approved
from market_intelligence.signals.promotion import LadderStatus
from market_intelligence.storage import duckdb as duckdb_store

AS_OF = date(2026, 7, 22)


def _status_row(
    *,
    event_subtype: str,
    edge_type: str = "self",
    horizon_days: int = 60,
) -> dict[str, Any]:
    return {
        "signal_id": f"sig-{event_subtype}-{edge_type}-{horizon_days}",
        "event_type": "insider_transaction",
        "event_subtype": event_subtype,
        "edge_type": edge_type,
        "horizon_days": horizon_days,
        "status": LadderStatus.ACTIVE.value,
        "confirm_streak": 2,
        "fail_streak": 0,
        "mean_car": 0.03,
        "hit_rate": 0.6,
        "n_clusters": 500,
        "direction": 1,
        "last_reason": "seeded",
    }


def _event_row(
    event_id: str, ticker: str, subtype: str, *, cik: str | None = None
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_type": "insider_transaction",
        "event_key": f"{ticker}:{event_id}",
        "event_subtype": subtype,
        "ticker": ticker,
        "cik": cik,
        "available_time": datetime(2026, 7, 22, tzinfo=UTC),
        "magnitude": 100_000.0,
        "direction": 1,
        "extraction_confidence": 0.9,
    }


class TestApprovedSignalsModule:
    def test_a_validated_cell_is_approved(self):
        assert is_approved("insider_transaction", "P", "self", 60)

    def test_an_unvalidated_cell_is_not_approved(self):
        """A/F/M looked good in-discovery and reversed on the real holdout --
        must never be in the allowlist regardless of what the ladder says."""
        assert not is_approved("insider_transaction", "A", "self", 60)
        assert not is_approved("insider_transaction", "F", "self", 120)
        assert not is_approved("insider_transaction", "M", "self", 90)

    def test_cross_company_propagation_is_never_approved(self):
        """No edge_type other than 'self' should ever appear here as of
        2026-08-04 -- the propagation graph isn't dated/complete yet."""
        assert all(cell[2] == "self" for cell in APPROVED_CELLS)

    def test_right_code_wrong_horizon_is_not_approved(self):
        """P is only validated at 60d -- other horizons must not slip through."""
        assert not is_approved("insider_transaction", "P", "self", 20)
        assert not is_approved("insider_transaction", "P", "self", 120)


class TestEventAlertsRespectsTheAllowlist:
    def test_active_but_unapproved_cell_fires_nothing(self, tmp_config: Config) -> None:
        with database.connection(tmp_config.paths.database_path) as con:
            database.init_db(con)
            duckdb_store.upsert_signal_status(con, [_status_row(event_subtype="A")])
            duckdb_store.upsert_events(con, [_event_row("evt-a-1", "ACME", "A")])

        with database.connection(tmp_config.paths.database_path) as con:
            alerts = briefing_builder.event_alerts(con, as_of=AS_OF, lookback_days=4)

        assert alerts == []

    def test_active_and_approved_cell_fires(self, tmp_config: Config) -> None:
        with database.connection(tmp_config.paths.database_path) as con:
            database.init_db(con)
            duckdb_store.upsert_signal_status(con, [_status_row(event_subtype="P")])
            duckdb_store.upsert_events(con, [_event_row("evt-p-1", "ACME", "P")])

        with database.connection(tmp_config.paths.database_path) as con:
            alerts = briefing_builder.event_alerts(con, as_of=AS_OF, lookback_days=4)

        assert [a.ticker for a in alerts] == ["ACME"]

    def test_both_active_only_the_approved_one_fires(self, tmp_config: Config) -> None:
        """The realistic case: several cells are active at once, only some approved."""
        with database.connection(tmp_config.paths.database_path) as con:
            database.init_db(con)
            duckdb_store.upsert_signal_status(
                con,
                [
                    _status_row(event_subtype="A"),
                    _status_row(event_subtype="P"),
                    _status_row(event_subtype="D", horizon_days=20),
                ],
            )
            duckdb_store.upsert_events(
                con,
                [
                    _event_row("evt-a-2", "AAAA", "A"),
                    _event_row("evt-p-2", "PPPP", "P"),
                    _event_row("evt-d-2", "DDDD", "D"),
                ],
            )

        with database.connection(tmp_config.paths.database_path) as con:
            alerts = briefing_builder.event_alerts(con, as_of=AS_OF, lookback_days=4)

        assert sorted(a.ticker for a in alerts) == ["DDDD", "PPPP"]
