"""Trade-alert assembly: EventAlert + price history -> persisted ledger rows.

Offline throughout: ``memory_db`` is an in-memory DuckDB with the schema
initialized, ``tmp_config`` carries the real ``trading`` defaults from
``config/settings.yaml`` (account_equity=10000, atr_period=14,
min_confidence=60, ...).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import duckdb
import pytest

from market_intelligence.autopilot.types import EventAlert
from market_intelligence.config import Config
from market_intelligence.signals import trade_alerts
from market_intelligence.signals.confidence import ConfidenceInputs
from market_intelligence.storage import duckdb as duckdb_store

AS_OF = date(2026, 7, 20)


def _price_row(symbol: str, price_date: date, close: float) -> dict[str, Any]:
    return {
        "price_id": f"{symbol}-{price_date.isoformat()}",
        "symbol": symbol,
        "price_date": price_date,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "adj_close": close,
        "volume": 5_000_000,
        "provider": "yfinance",
        "validation_status": "valid",
        "source": "market",
    }


def _seed_prices(
    con: duckdb.DuckDBPyConnection, symbol: str, n: int = 40, start_close: float = 100.0
) -> float:
    """n bars for ``symbol`` ending on AS_OF, oscillating around start_close.

    Returns the last close.
    """
    rows = []
    close = start_close
    first_day = AS_OF - timedelta(days=n - 1)
    for i in range(n):
        close += 0.4 if i % 2 == 0 else -0.3
        rows.append(_price_row(symbol, first_day + timedelta(days=i), close))
    duckdb_store.upsert_daily_prices(con, rows)
    return rows[-1]["close"]


def _seed_nke(con: duckdb.DuckDBPyConnection, n: int = 40, start_close: float = 100.0) -> float:
    return _seed_prices(con, "NKE", n=n, start_close=start_close)


def _alert(**overrides: Any) -> EventAlert:
    fields: dict[str, Any] = {
        "ticker": "NKE",
        "event_type": "insider_transaction",
        "event_subtype": "P",
        "available_on": AS_OF,
        "horizon_days": 20,
        "direction": 1,
        "predicted_car": 0.03,
        "basis": "active, holdout hit 65.0%",
        "event_id": "ev1",
        "hit_rate": 0.65,
        "n_clusters": 40,
        "extraction_confidence": 0.9,
    }
    fields.update(overrides)
    return EventAlert(**fields)


class TestBuildRecords:
    def test_self_edge_alert_produces_one_record(
        self, memory_db: duckdb.DuckDBPyConnection, tmp_config: Config
    ) -> None:
        last_close = _seed_nke(memory_db)
        alert = _alert()

        records, notes = trade_alerts.build_records(memory_db, [alert], tmp_config, AS_OF)

        assert notes == []
        assert len(records) == 1
        record = records[0]
        assert record.kind == "reaction_lag"
        assert record.ticker == "NKE"
        assert record.trigger_key == "ev1:self:20"
        assert record.edge_id == "self"
        assert record.event_id == "ev1"
        assert record.direction == 1
        assert record.plan.entry_ref == pytest.approx(last_close, rel=0.02)
        assert 0 <= record.confidence <= 100

        # evidence must be JSON-serializable and name the cell stats.
        payload = json.loads(json.dumps(record.evidence))
        assert payload["event_type"] == "insider_transaction"
        assert payload["hit_rate"] == 0.65
        assert payload["n_clusters"] == 40

    def test_ticker_with_no_price_history_is_skipped_with_note(
        self, memory_db: duckdb.DuckDBPyConnection, tmp_config: Config
    ) -> None:
        alert = _alert(ticker="ZZZZ")

        records, notes = trade_alerts.build_records(memory_db, [alert], tmp_config, AS_OF)

        assert records == []
        assert len(notes) == 1
        assert "ZZZZ" in notes[0]

    def test_null_price_rows_are_skipped_not_crashing(
        self, memory_db: duckdb.DuckDBPyConnection, tmp_config: Config
    ) -> None:
        _seed_nke(memory_db)
        # A row with a NULL close cannot become a Bar; it must be dropped,
        # not raise, and the remaining valid bars still produce a plan.
        duckdb_store.upsert_daily_prices(
            memory_db,
            [
                {
                    "price_id": "NKE-null",
                    "symbol": "NKE",
                    "price_date": AS_OF - timedelta(days=100),
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "adj_close": None,
                    "volume": None,
                    "provider": "yfinance",
                    "validation_status": "valid",
                    "source": "market",
                }
            ],
        )
        alert = _alert()

        records, notes = trade_alerts.build_records(memory_db, [alert], tmp_config, AS_OF)

        assert notes == []
        assert len(records) == 1


class TestPersistNew:
    def test_inserts_then_dedups_on_second_call(
        self, memory_db: duckdb.DuckDBPyConnection, tmp_config: Config
    ) -> None:
        _seed_nke(memory_db)
        records, _ = trade_alerts.build_records(memory_db, [_alert()], tmp_config, AS_OF)

        inserted = trade_alerts.persist_new(memory_db, records)
        assert len(inserted) == 1
        assert inserted[0].alert_id == records[0].alert_id

        again = trade_alerts.persist_new(memory_db, records)
        assert again == []

        n = memory_db.execute("SELECT count(*) FROM trade_alerts").fetchone()[0]
        assert n == 1

    def test_persisted_evidence_round_trips_from_json(
        self, memory_db: duckdb.DuckDBPyConnection, tmp_config: Config
    ) -> None:
        _seed_nke(memory_db)
        records, _ = trade_alerts.build_records(memory_db, [_alert()], tmp_config, AS_OF)
        trade_alerts.persist_new(memory_db, records)

        row = memory_db.execute(
            "SELECT evidence FROM trade_alerts WHERE alert_id = ?",
            [records[0].alert_id],
        ).fetchone()
        payload = json.loads(row[0])
        assert payload == records[0].evidence


class TestGating:
    def test_sendable_and_gated_split_and_mark_gated(
        self, memory_db: duckdb.DuckDBPyConnection, tmp_config: Config
    ) -> None:
        _seed_nke(memory_db)
        high_conf = _alert(
            event_id="ev-hi", hit_rate=0.95, n_clusters=500, extraction_confidence=1.0
        )
        low_conf = _alert(
            event_id="ev-lo", hit_rate=0.5, n_clusters=0, extraction_confidence=0.0
        )
        records, _ = trade_alerts.build_records(
            memory_db, [high_conf, low_conf], tmp_config, AS_OF
        )
        inserted = trade_alerts.persist_new(memory_db, records)
        min_conf = tmp_config.settings.trading.min_confidence

        send = trade_alerts.sendable(inserted, min_conf)
        gate = trade_alerts.gated(inserted, min_conf)

        assert len(send) + len(gate) == len(inserted)
        assert send and gate  # this fixture is built to straddle the floor
        assert all(r.confidence >= min_conf for r in send)
        assert all(r.confidence < min_conf for r in gate)

        trade_alerts.mark_gated(memory_db, gate)
        for record in gate:
            row = memory_db.execute(
                "SELECT delivery_note FROM trade_alerts WHERE alert_id = ?",
                [record.alert_id],
            ).fetchone()
            assert row[0] == "gated_below_min_confidence"
        for record in send:
            row = memory_db.execute(
                "SELECT delivery_note FROM trade_alerts WHERE alert_id = ?",
                [record.alert_id],
            ).fetchone()
            assert row[0] is None


class TestMarkDelivered:
    def test_delivered_true_flips_the_flag(
        self, memory_db: duckdb.DuckDBPyConnection, tmp_config: Config
    ) -> None:
        _seed_nke(memory_db)
        records, _ = trade_alerts.build_records(memory_db, [_alert()], tmp_config, AS_OF)
        inserted = trade_alerts.persist_new(memory_db, records)

        trade_alerts.mark_delivered(memory_db, inserted, delivered=True)

        row = memory_db.execute(
            "SELECT delivered, delivery_note FROM trade_alerts WHERE alert_id = ?",
            [inserted[0].alert_id],
        ).fetchone()
        assert row[0] is True
        assert row[1] is None

    def test_delivered_false_sets_telegram_failed_note(
        self, memory_db: duckdb.DuckDBPyConnection, tmp_config: Config
    ) -> None:
        _seed_nke(memory_db)
        records, _ = trade_alerts.build_records(memory_db, [_alert()], tmp_config, AS_OF)
        inserted = trade_alerts.persist_new(memory_db, records)

        trade_alerts.mark_delivered(memory_db, inserted, delivered=False)

        row = memory_db.execute(
            "SELECT delivered, delivery_note FROM trade_alerts WHERE alert_id = ?",
            [inserted[0].alert_id],
        ).fetchone()
        assert row[0] is False
        assert row[1] == "telegram_failed"


class TestFailureIsolation:
    def test_one_bad_ticker_does_not_lose_the_rest(
        self,
        memory_db: duckdb.DuckDBPyConnection,
        tmp_config: Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed_prices(memory_db, "NKE")
        _seed_prices(memory_db, "AAPL", start_close=150.0)

        good = _alert(ticker="NKE", event_id="ev-good", horizon_days=20)
        bad = _alert(ticker="AAPL", event_id="ev-bad", horizon_days=25)

        real_build_trade_plan = trade_alerts.build_trade_plan

        def _boom(**kwargs: Any) -> Any:
            if kwargs["horizon_days"] == 25:
                raise ValueError("simulated plan blow-up")
            return real_build_trade_plan(**kwargs)

        monkeypatch.setattr(trade_alerts, "build_trade_plan", _boom)

        records, notes = trade_alerts.build_records(memory_db, [good, bad], tmp_config, AS_OF)

        assert len(records) == 1
        assert records[0].ticker == "NKE"
        assert any("AAPL" in note and "ValueError" in note for note in notes)


class TestConfidenceWiring:
    def test_higher_hit_rate_yields_strictly_higher_confidence(
        self, memory_db: duckdb.DuckDBPyConnection, tmp_config: Config
    ) -> None:
        _seed_nke(memory_db)
        low = _alert(event_id="ev-low", hit_rate=0.52, n_clusters=100)
        high = _alert(event_id="ev-high", hit_rate=0.95, n_clusters=100)

        records, notes = trade_alerts.build_records(memory_db, [low, high], tmp_config, AS_OF)

        assert notes == []
        by_id = {record.event_id: record for record in records}
        assert by_id["ev-high"].confidence > by_id["ev-low"].confidence

    def test_extraction_confidence_wired_verbatim_and_moves_the_score(
        self,
        memory_db: duckdb.DuckDBPyConnection,
        tmp_config: Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed_nke(memory_db)
        none_conf = _alert(
            event_id="ev-none", hit_rate=0.6, n_clusters=25, extraction_confidence=None
        )
        high_conf = _alert(
            event_id="ev-hi", hit_rate=0.6, n_clusters=25, extraction_confidence=0.95
        )

        captured: dict[str, ConfidenceInputs] = {}
        real_score = trade_alerts.score

        def _capture(inputs: ConfidenceInputs) -> int:
            captured[str(inputs.extraction_confidence)] = inputs
            return real_score(inputs)

        monkeypatch.setattr(trade_alerts, "score", _capture)

        records, notes = trade_alerts.build_records(
            memory_db, [none_conf, high_conf], tmp_config, AS_OF
        )

        assert notes == []
        by_id = {record.event_id: record for record in records}
        assert by_id["ev-hi"].confidence >= by_id["ev-none"].confidence

        # Verbatim wiring: ConfidenceInputs actually received the alert's own
        # hit_rate/n_clusters/extraction_confidence, not some derived value --
        # capture-based so this survives Michael retuning the blend weights.
        assert captured["None"].hit_rate == 0.6
        assert captured["None"].n_clusters == 25
        assert captured["None"].extraction_confidence is None
        assert captured["0.95"].hit_rate == 0.6
        assert captured["0.95"].n_clusters == 25
        assert captured["0.95"].extraction_confidence == 0.95


class TestHorizons:
    def test_same_ticker_same_event_two_horizons_produce_distinct_records(
        self, memory_db: duckdb.DuckDBPyConnection, tmp_config: Config
    ) -> None:
        _seed_nke(memory_db)
        short = _alert(event_id="ev1", horizon_days=5)
        long_horizon = _alert(event_id="ev1", horizon_days=20)

        records, notes = trade_alerts.build_records(
            memory_db, [short, long_horizon], tmp_config, AS_OF
        )

        assert notes == []
        assert len(records) == 2
        trigger_keys = {record.trigger_key for record in records}
        assert trigger_keys == {"ev1:self:5", "ev1:self:20"}
        exit_dates = {record.plan.time_exit_date for record in records}
        assert len(exit_dates) == 2
