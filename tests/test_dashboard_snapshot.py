from __future__ import annotations

import json
from datetime import UTC, date, datetime

from market_intelligence.autopilot.types import Briefing, RunStatus
from market_intelligence.dashboard_snapshot import (
    build_snapshot,
    refresh_snapshot,
    snapshot_path,
    write_snapshot,
)


def test_snapshot_tracks_open_prediction_against_latest_price(memory_db, tmp_config) -> None:
    memory_db.execute(
        """
        INSERT INTO daily_prices (price_id, symbol, price_date, close, adj_close)
        VALUES ('p0', 'NVDA', '2026-07-20', 100, 100),
               ('p1', 'NVDA', '2026-07-22', 105, 105)
        """
    )
    memory_db.execute(
        """
        INSERT INTO trade_alerts (
          alert_id, kind, ticker, trigger_key, fired_at, direction, entry_ref,
          stop, target, time_exit_date, confidence, outcome
        ) VALUES (
          'a1', 'reaction_lag', 'NVDA', 'event:self:20', '2026-07-20T12:00:00+00:00',
          1, 100, 95, 110, '2026-08-10', 72, 'open'
        )
        """
    )
    memory_db.execute(
        """
        INSERT INTO signal_status (
          signal_id, event_type, event_subtype, edge_type, horizon_days, status,
          confirm_streak, fail_streak, mean_car, hit_rate, n_clusters, direction
        ) VALUES ('s1', 'insider_transaction', 'P', 'self', 20, 'active', 2, 0, .1, .6, 40, 1)
        """
    )
    briefing = Briefing(as_of=date(2026, 7, 22), run_status=RunStatus.SUCCESS)

    payload = build_snapshot(
        memory_db, tmp_config, briefing, now=datetime(2026, 7, 22, tzinfo=UTC)
    )

    assert payload["metrics"]["open_alerts"] == 1
    assert payload["metrics"]["active_signals"] == 1
    alert = payload["open_alerts"][0]
    assert alert["last_price"] == 105.0
    assert alert["actual_return"] == 0.05
    assert alert["expected_return"] == 0.1
    assert alert["progress_to_target"] == 0.5
    assert alert["vault_url"].startswith("obsidian://open?")


def test_snapshot_writes_valid_json_atomically(memory_db, tmp_config) -> None:
    payload = build_snapshot(
        memory_db,
        tmp_config,
        Briefing(as_of=date(2026, 7, 22), run_status=RunStatus.PARTIAL),
        now=datetime(2026, 7, 22, tzinfo=UTC),
    )

    path = write_snapshot(tmp_config, payload)

    assert path == snapshot_path(tmp_config)
    assert json.loads(path.read_text(encoding="utf-8"))["briefing"]["status"] == "partial"


def test_refresh_snapshot_uses_the_latest_autopilot_run(tmp_config) -> None:
    from market_intelligence import database

    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        con.execute(
            """
            INSERT INTO pipeline_runs (run_id, pipeline_name, started_time, completed_time, status)
            VALUES ('run', 'autopilot', '2026-07-22T12:00:00+00:00',
                    '2026-07-22T12:30:00+00:00', 'success')
            """
        )

    path = refresh_snapshot(tmp_config, now=datetime(2026, 7, 23, tzinfo=UTC))

    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot["briefing"] == {
        "as_of": "2026-07-22",
        "status": "success",
        "notes": [],
        "ingest": {},
    }
