"""Tests for the autopilot dead-man's-switch watchdog (``scripts/watch_autopilot.py``).

Three behaviors matter and would let a hang go unnoticed again if they broke:
a live process past the hang threshold must be killed and alerted; a day with
no process and no briefing note past the deadline must be alerted (once, then
throttled); and a normal completed run must never trigger either. Telegram is
driven through ``httpx.MockTransport`` throughout, same contract as
``test_notify.py``: never touch the network. Process discovery/killing is
injected (``find_pids``/``kill``) rather than mocked at the ``subprocess``
level, so these tests never spawn or signal a real process.

``scripts/`` is not a package, so the module is loaded by file path rather
than a normal import. Its own log/state files are redirected into ``tmp_path``
by monkeypatching ``PROJECT_ROOT`` -- see the module's docstring for why it
deliberately uses a local path rather than ``config.paths.logs_dir``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from market_intelligence.config import Config

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "watch_autopilot.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("watch_autopilot", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wd = _load_module()


@pytest.fixture(autouse=True)
def _local_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the watchdog's own log/state files under ``tmp_path``.

    Without this, every test run would read and write the real
    ``~/quant/logs/autopilot_watchdog_state.json`` -- polluting the repo and,
    worse, leaking throttle state (``last_missing_run_alert_at`` etc.) across
    unrelated test runs.
    """
    monkeypatch.setattr(wd, "PROJECT_ROOT", tmp_path)


def _args(**overrides: object) -> argparse.Namespace:
    base = argparse.Namespace(
        loop=False,
        interval_seconds=900,
        hang_threshold_minutes=wd.DEFAULT_HANG_THRESHOLD_MINUTES,
        deadline_hour=wd.DEFAULT_DEADLINE_HOUR,
        repeat_minutes=wd.DEFAULT_MISSING_RUN_REPEAT_MINUTES,
        memory_critical_free_pct=wd.DEFAULT_MEMORY_CRITICAL_FREE_PCT,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _healthy_free_pct() -> float:
    return 50.0


def _with_creds(config: Config, *, token: str = "bot-token", chat_id: str = "12345") -> Config:
    config.env.telegram_bot_token = token
    config.env.telegram_chat_id = chat_id
    return config


def _recording_transport() -> tuple[httpx.MockTransport, list[dict[str, object]]]:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler), captured


def _write_note(config: Config) -> None:
    note_dir = config.paths.obsidian_vault_dir / "briefings"
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / f"{wd._today_utc_iso()}.md").write_text("# briefing\n", encoding="utf-8")


def _before_deadline(args: argparse.Namespace) -> datetime:
    return datetime.now().replace(hour=max(args.deadline_hour - 1, 0), minute=0, second=0)


def _past_deadline(args: argparse.Namespace) -> datetime:
    return datetime.now().replace(hour=min(args.deadline_hour + 1, 23), minute=0, second=0)


# --------------------------------------------------------------------------- #
# _parse_etime_seconds -- pure function, the ps(1) output parser             #
# --------------------------------------------------------------------------- #
def test_parse_etime_seconds_handles_mm_ss() -> None:
    assert wd._parse_etime_seconds("17:36") == 17 * 60 + 36


def test_parse_etime_seconds_handles_hh_mm_ss() -> None:
    assert wd._parse_etime_seconds("01:02:03") == 3723


def test_parse_etime_seconds_handles_day_prefix() -> None:
    assert wd._parse_etime_seconds("2-01:02:03") == 2 * 86400 + 3723


def test_parse_etime_seconds_returns_none_on_garbage() -> None:
    assert wd._parse_etime_seconds("not-a-time") is None
    assert wd._parse_etime_seconds("") is None


# --------------------------------------------------------------------------- #
# Hung run: alive past the hang threshold -> killed + alerted                #
# --------------------------------------------------------------------------- #
def test_hung_process_is_killed_and_alerted(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _with_creds(tmp_config)
    monkeypatch.setattr(wd, "_elapsed_seconds", lambda pid: 250 * 60)  # 250m > 240m default

    transport, captured = _recording_transport()
    kill_calls: list[bool] = []

    def fake_kill() -> bool:
        kill_calls.append(True)
        return True

    wd._run_once(
        config,
        _args(),
        transport=transport,
        find_pids=lambda: [999, 1000],
        kill=fake_kill,
        read_free_pct=_healthy_free_pct,
    )

    assert kill_calls == [True]
    assert len(captured) == 1
    text = captured[0]["body"]["text"]  # type: ignore[index]
    assert "killed a hung run" in text
    assert "250 minutes" in text


def test_process_alive_under_threshold_takes_no_action(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _with_creds(tmp_config)
    monkeypatch.setattr(wd, "_elapsed_seconds", lambda pid: 10 * 60)  # well under 240m

    transport, captured = _recording_transport()
    kill_calls: list[bool] = []

    def fake_kill() -> bool:
        kill_calls.append(True)
        return True

    wd._run_once(
        config,
        _args(),
        transport=transport,
        find_pids=lambda: [999],
        kill=fake_kill,
        read_free_pct=_healthy_free_pct,
    )

    assert kill_calls == []
    assert captured == []


# --------------------------------------------------------------------------- #
# Memory pressure: fast blowup, independent of the hang threshold            #
# --------------------------------------------------------------------------- #
def test_critical_memory_pressure_kills_and_alerts_before_hang_threshold(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Must fire even when the process is nowhere near the hang threshold --
    the 2026-07-30 incident hit critical memory in under 10 minutes."""
    config = _with_creds(tmp_config)
    monkeypatch.setattr(wd, "_elapsed_seconds", lambda pid: 5 * 60)  # 5m, far under any hang

    transport, captured = _recording_transport()
    kill_calls: list[bool] = []

    def fake_kill() -> bool:
        kill_calls.append(True)
        return True

    wd._run_once(
        config,
        _args(memory_critical_free_pct=8.0),
        transport=transport,
        find_pids=lambda: [999],
        kill=fake_kill,
        read_free_pct=lambda: 3.0,  # well below the 8% floor
    )

    assert kill_calls == [True]
    assert len(captured) == 1
    text = captured[0]["body"]["text"]  # type: ignore[index]
    assert "critical memory pressure" in text
    assert "3.0%" in text


def test_memory_above_floor_takes_no_action(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _with_creds(tmp_config)
    monkeypatch.setattr(wd, "_elapsed_seconds", lambda pid: 5 * 60)

    transport, captured = _recording_transport()
    kill_calls: list[bool] = []

    def fake_kill() -> bool:
        kill_calls.append(True)
        return True

    wd._run_once(
        config,
        _args(memory_critical_free_pct=8.0),
        transport=transport,
        find_pids=lambda: [999],
        kill=fake_kill,
        read_free_pct=lambda: 25.0,  # healthy
    )

    assert kill_calls == []
    assert captured == []


# --------------------------------------------------------------------------- #
# Missing run: no process, no note, past deadline -> alerted (then throttled) #
# --------------------------------------------------------------------------- #
def test_missing_run_before_deadline_is_normal_no_alert(tmp_config: Config) -> None:
    config = _with_creds(tmp_config)
    args = _args()
    transport, captured = _recording_transport()

    wd._run_once(
        config, args, now=_before_deadline(args), transport=transport, find_pids=lambda: []
    )

    assert captured == []


def test_missing_run_past_deadline_alerts(tmp_config: Config) -> None:
    config = _with_creds(tmp_config)
    args = _args()
    transport, captured = _recording_transport()

    wd._run_once(config, args, now=_past_deadline(args), transport=transport, find_pids=lambda: [])

    assert len(captured) == 1
    text = captured[0]["body"]["text"]  # type: ignore[index]
    assert "never happened" in text


def test_missing_run_alert_is_throttled_on_repeat_calls(tmp_config: Config) -> None:
    config = _with_creds(tmp_config)
    args = _args(repeat_minutes=180)
    transport, captured = _recording_transport()
    now = _past_deadline(args)

    wd._run_once(config, args, now=now, transport=transport, find_pids=lambda: [])
    wd._run_once(config, args, now=now, transport=transport, find_pids=lambda: [])

    assert len(captured) == 1  # second call within the repeat window sends nothing


# --------------------------------------------------------------------------- #
# Completed run: note exists -> no alert, no action, regardless of the hour  #
# --------------------------------------------------------------------------- #
def test_completed_run_with_note_never_alerts(tmp_config: Config) -> None:
    config = _with_creds(tmp_config)
    _write_note(config)
    args = _args()
    transport, captured = _recording_transport()

    wd._run_once(config, args, now=_past_deadline(args), transport=transport, find_pids=lambda: [])

    assert captured == []


# --------------------------------------------------------------------------- #
# Mount unavailable: a distinct alert, not the generic "missing run" one     #
# --------------------------------------------------------------------------- #
def test_mount_unavailable_sends_a_distinct_alert(tmp_config: Config) -> None:
    config = _with_creds(tmp_config)
    config.paths.home = config.paths.home / "not-actually-mounted"
    args = _args()
    transport, captured = _recording_transport()

    wd._run_once(config, args, now=_past_deadline(args), transport=transport, find_pids=lambda: [])

    assert len(captured) == 1
    text = captured[0]["body"]["text"]  # type: ignore[index]
    assert "external data drive isn't mounted" in text


def test_mount_unavailable_before_deadline_stays_quiet(tmp_config: Config) -> None:
    config = _with_creds(tmp_config)
    config.paths.home = config.paths.home / "not-actually-mounted"
    args = _args()
    transport, captured = _recording_transport()

    wd._run_once(
        config, args, now=_before_deadline(args), transport=transport, find_pids=lambda: []
    )

    assert captured == []


# --------------------------------------------------------------------------- #
# Telegram graceful degradation -- matches notify.py's own contract          #
# --------------------------------------------------------------------------- #
def test_missing_credentials_makes_no_network_call_and_does_not_raise(
    tmp_config: Config,
) -> None:
    config = _with_creds(tmp_config, token="", chat_id="")
    args = _args()
    transport, captured = _recording_transport()

    wd._run_once(config, args, now=_past_deadline(args), transport=transport, find_pids=lambda: [])

    assert captured == []
