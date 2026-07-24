"""Tests for the daily loop: the diff, the alerts, and partial-failure handling.

Two things here would corrupt the product silently if they broke: the diff must
name the right transition for each cell, and the orchestrator must always
produce a briefing -- a loop that goes quiet on failure looks exactly like one
that had nothing to report.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from market_intelligence import database
from market_intelligence.autopilot import briefing as briefing_builder
from market_intelligence.autopilot import notify, orchestrator
from market_intelligence.autopilot.orchestrator import Step
from market_intelligence.autopilot.types import ChangeKind, RunStatus
from market_intelligence.collectors import RunSummary
from market_intelligence.config import Config
from market_intelligence.signals import trade_alerts as trade_alerts_builder
from market_intelligence.signals.promotion import LadderStatus
from market_intelligence.storage import duckdb as duckdb_store

AS_OF = date(2026, 7, 22)


def _seed_status(
    config: Config,
    *,
    subtype: str,
    horizon: int,
    status: str,
    edge: str = "self",
    mean_car: float = 0.01,
    hit_rate: float = 0.58,
    direction: int = 1,
) -> None:
    row = {
        "signal_id": f"sig-{subtype}-{edge}-{horizon}",
        "event_type": "insider_transaction",
        "event_subtype": subtype,
        "edge_type": edge,
        "horizon_days": horizon,
        "status": status,
        "confirm_streak": 2,
        "fail_streak": 0,
        "mean_car": mean_car,
        "hit_rate": hit_rate,
        "n_clusters": 500,
        "direction": direction,
        "last_reason": "seeded",
    }
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_signal_status(con, [row])


def _seed_event(
    config: Config,
    *,
    ticker: str,
    subtype: str,
    available: datetime,
    extraction_confidence: float = 0.9,
) -> None:
    row = {
        "event_id": f"evt-{ticker}-{subtype}-{available.date()}",
        "event_type": "insider_transaction",
        "event_key": f"{ticker}:{subtype}:{available.date()}",
        "event_subtype": subtype,
        "ticker": ticker,
        "available_time": available,
        "extraction_confidence": extraction_confidence,
    }
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_events(con, [row])


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
    config: Config,
    symbol: str,
    *,
    as_of: date,
    n: int = 40,
    start_close: float = 100.0,
) -> None:
    """n bars for ``symbol`` ending on/before ``as_of`` (see test_trade_alerts.py)."""
    rows = []
    close = start_close
    first_day = as_of - timedelta(days=n - 1)
    for i in range(n):
        close += 0.4 if i % 2 == 0 else -0.3
        rows.append(_price_row(symbol, first_day + timedelta(days=i), close))
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_daily_prices(con, rows)


def _snapshot(config: Config) -> dict[briefing_builder.CellKey, str]:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        return briefing_builder.snapshot_statuses(con)


# --------------------------------------------------------------------------- #
# the diff                                                                    #
# --------------------------------------------------------------------------- #
def test_activation_is_reported(tmp_config: Config) -> None:
    _seed_status(tmp_config, subtype="P", horizon=20, status=LadderStatus.CANDIDATE.value)
    before = _snapshot(tmp_config)
    _seed_status(tmp_config, subtype="P", horizon=20, status=LadderStatus.ACTIVE.value)

    with database.connection(tmp_config.paths.database_path) as con:
        changes = briefing_builder.detect_changes(con, before)

    assert len(changes) == 1
    assert changes[0].kind == ChangeKind.ACTIVATED
    assert changes[0].label == "insider_transaction/P/self/20d"


def test_demotion_and_retirement_are_distinguished(tmp_config: Config) -> None:
    _seed_status(tmp_config, subtype="P", horizon=5, status=LadderStatus.ACTIVE.value)
    _seed_status(tmp_config, subtype="S", horizon=5, status=LadderStatus.ACTIVE.value)
    before = _snapshot(tmp_config)
    _seed_status(tmp_config, subtype="P", horizon=5, status=LadderStatus.DORMANT.value)
    _seed_status(tmp_config, subtype="S", horizon=5, status=LadderStatus.RETIRED.value)

    with database.connection(tmp_config.paths.database_path) as con:
        changes = {c.event_subtype: c.kind for c in briefing_builder.detect_changes(con, before)}

    assert changes["P"] == ChangeKind.DEMOTED
    assert changes["S"] == ChangeKind.RETIRED


def test_a_brand_new_candidate_is_reported(tmp_config: Config) -> None:
    before = _snapshot(tmp_config)  # empty
    _seed_status(tmp_config, subtype="P", horizon=1, status=LadderStatus.CANDIDATE.value)

    with database.connection(tmp_config.paths.database_path) as con:
        changes = briefing_builder.detect_changes(con, before)

    assert [c.kind for c in changes] == [ChangeKind.NEW_CANDIDATE]


def test_unchanged_cells_produce_no_line(tmp_config: Config) -> None:
    _seed_status(tmp_config, subtype="P", horizon=20, status=LadderStatus.ACTIVE.value)
    before = _snapshot(tmp_config)  # active

    with database.connection(tmp_config.paths.database_path) as con:
        assert briefing_builder.detect_changes(con, before) == []


# --------------------------------------------------------------------------- #
# alerts                                                                      #
# --------------------------------------------------------------------------- #
def test_recent_event_through_active_cell_fires_an_alert(tmp_config: Config) -> None:
    _seed_status(
        tmp_config,
        subtype="P",
        horizon=20,
        status=LadderStatus.ACTIVE.value,
        direction=1,
        hit_rate=0.58,
    )
    _seed_event(
        tmp_config,
        ticker="AMD",
        subtype="P",
        available=datetime(2026, 7, 22, tzinfo=UTC),
        extraction_confidence=0.87,
    )

    with database.connection(tmp_config.paths.database_path) as con:
        alerts = briefing_builder.event_alerts(con, as_of=AS_OF, lookback_days=4)

    assert len(alerts) == 1
    assert alerts[0].ticker == "AMD"
    assert alerts[0].direction == 1
    assert alerts[0].predicted_car == 0.01
    assert alerts[0].horizon_days == 20
    assert alerts[0].event_id == "evt-AMD-P-2026-07-22"
    assert alerts[0].hit_rate == 0.58
    assert alerts[0].n_clusters == 500
    assert alerts[0].extraction_confidence == 0.87


def test_event_through_a_non_active_cell_does_not_fire(tmp_config: Config) -> None:
    _seed_status(tmp_config, subtype="P", horizon=20, status=LadderStatus.CANDIDATE.value)
    _seed_event(tmp_config, ticker="AMD", subtype="P", available=datetime(2026, 7, 22, tzinfo=UTC))

    with database.connection(tmp_config.paths.database_path) as con:
        assert briefing_builder.event_alerts(con, as_of=AS_OF, lookback_days=4) == []


def test_event_outside_the_lookback_window_does_not_fire(tmp_config: Config) -> None:
    _seed_status(tmp_config, subtype="P", horizon=20, status=LadderStatus.ACTIVE.value)
    _seed_event(tmp_config, ticker="AMD", subtype="P", available=datetime(2026, 7, 1, tzinfo=UTC))

    with database.connection(tmp_config.paths.database_path) as con:
        assert briefing_builder.event_alerts(con, as_of=AS_OF, lookback_days=4) == []


# --------------------------------------------------------------------------- #
# orchestration: a briefing is always produced                               #
# --------------------------------------------------------------------------- #
def ok_step(name: str) -> Step:
    return Step(name, lambda _c: RunSummary(pipeline_name=name, status="success", inserted=1))


def boom_step(name: str, *, critical: bool = False) -> Step:
    def run(_c: Config) -> RunSummary:
        raise RuntimeError("kaboom")

    return Step(name, run, critical=critical)


def test_all_steps_succeed_is_success_and_writes_a_note(tmp_config: Config) -> None:
    briefing = orchestrator.run(tmp_config, as_of=AS_OF, steps=[ok_step("a"), ok_step("b")])

    assert briefing.run_status == RunStatus.SUCCESS
    note = tmp_config.paths.obsidian_vault_dir / "briefings" / f"{AS_OF.isoformat()}.md"
    assert note.exists()


def test_a_failing_data_step_degrades_to_partial_not_fatal(tmp_config: Config) -> None:
    briefing = orchestrator.run(
        tmp_config, as_of=AS_OF, steps=[boom_step("sync-prices"), ok_step("evaluate")]
    )

    assert briefing.run_status == RunStatus.PARTIAL
    assert any("sync-prices FAILED" in n for n in briefing.notes)
    # The briefing still exists despite the failure.
    assert (tmp_config.paths.obsidian_vault_dir / "briefings" / f"{AS_OF.isoformat()}.md").exists()


def test_a_failing_critical_step_is_a_failed_run(tmp_config: Config) -> None:
    briefing = orchestrator.run(
        tmp_config,
        as_of=AS_OF,
        steps=[ok_step("sync-prices"), boom_step("evaluate", critical=True)],
    )

    assert briefing.run_status == RunStatus.FAILED
    assert any("evaluate FAILED" in n for n in briefing.notes)


def test_a_step_reporting_nonsuccess_status_degrades_to_partial(tmp_config: Config) -> None:
    def degraded(_c: Config) -> RunSummary:
        return RunSummary(pipeline_name="x", status="failed")

    briefing = orchestrator.run(tmp_config, as_of=AS_OF, steps=[Step("x", degraded)])

    assert briefing.run_status == RunStatus.PARTIAL


def test_the_run_diffs_this_runs_transitions(tmp_config: Config) -> None:
    """A step that promotes a cell must show up as an ACTIVATED change."""
    _seed_status(tmp_config, subtype="P", horizon=20, status=LadderStatus.CANDIDATE.value)

    def promote(_c: Config) -> RunSummary:
        _seed_status(tmp_config, subtype="P", horizon=20, status=LadderStatus.ACTIVE.value)
        return RunSummary(pipeline_name="evaluate", status="success")

    briefing = orchestrator.run(
        tmp_config, as_of=AS_OF, steps=[Step("evaluate", promote, critical=True)]
    )

    assert briefing.run_status == RunStatus.SUCCESS
    assert [c.kind for c in briefing.changes] == [ChangeKind.ACTIVATED]


# --------------------------------------------------------------------------- #
# trade alerts: the daily loop persists and sends them                       #
# --------------------------------------------------------------------------- #
def _seed_alertable_cell(
    tmp_config: Config,
    *,
    ticker: str = "AMD",
    hit_rate: float = 0.58,
    extraction_confidence: float = 0.9,
) -> None:
    """A fired self-edge alert with enough price history to plan a trade."""
    _seed_status(
        tmp_config,
        subtype="P",
        horizon=20,
        status=LadderStatus.ACTIVE.value,
        direction=1,
        hit_rate=hit_rate,
    )
    _seed_event(
        tmp_config,
        ticker=ticker,
        subtype="P",
        available=datetime(2026, 7, 22, tzinfo=UTC),
        extraction_confidence=extraction_confidence,
    )
    _seed_prices(tmp_config, ticker, as_of=AS_OF)


def _trade_alert_rows(config: Config) -> list[tuple[Any, ...]]:
    with database.connection(config.paths.database_path) as con:
        return con.execute(
            "SELECT alert_id, confidence, delivered, delivery_note FROM trade_alerts"
        ).fetchall()


def test_a_fired_alert_persists_one_row_and_dedups_on_rerun(tmp_config: Config) -> None:
    _seed_alertable_cell(tmp_config)

    orchestrator.run(tmp_config, as_of=AS_OF, steps=[])

    rows = _trade_alert_rows(tmp_config)
    assert len(rows) == 1
    first_alert_id, first_confidence = rows[0][0], rows[0][1]

    orchestrator.run(tmp_config, as_of=AS_OF, steps=[])

    rows_again = _trade_alert_rows(tmp_config)
    assert len(rows_again) == 1
    assert rows_again[0][0] == first_alert_id
    assert rows_again[0][1] == first_confidence


def test_a_broken_trade_alert_build_leaves_a_note_and_does_not_alter_run_status(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_alertable_cell(tmp_config)

    def boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("build blew up")

    monkeypatch.setattr(trade_alerts_builder, "build_records", boom)

    briefing = orchestrator.run(tmp_config, as_of=AS_OF, steps=[])

    # run_status reflects only what the step loop produced (empty here ->
    # SUCCESS); the trade-alert failure is reported solely via the note.
    assert briefing.run_status == RunStatus.SUCCESS
    assert any("trade-alerts FAILED" in n for n in briefing.notes)
    assert _trade_alert_rows(tmp_config) == []


def test_a_sent_alert_is_stamped_delivered(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_alertable_cell(tmp_config)
    captured: dict[str, Any] = {}

    def fake_send(_config: Config, records: Any) -> bool:
        captured["records"] = list(records)
        return True

    monkeypatch.setattr(notify, "send_trade_alerts", fake_send)

    orchestrator.run(tmp_config, as_of=AS_OF, steps=[])

    assert len(captured["records"]) == 1

    rows = _trade_alert_rows(tmp_config)
    assert len(rows) == 1
    assert rows[0][2] is True  # delivered


def test_a_gated_alert_is_never_sent(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_alertable_cell(tmp_config)
    # Push the confidence floor above anything this fixture can score, so the
    # one alert built here is gated regardless of the confidence blend.
    tmp_config.settings.trading.min_confidence = 99

    def fail_if_called(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("send_trade_alerts must not be called for a gated record")

    monkeypatch.setattr(notify, "send_trade_alerts", fail_if_called)

    orchestrator.run(tmp_config, as_of=AS_OF, steps=[])

    rows = _trade_alert_rows(tmp_config)
    assert len(rows) == 1
    assert rows[0][2] is False  # delivered
    assert rows[0][3] == "gated_below_min_confidence"
