#!/usr/bin/env python3
"""Dead-man's-switch watchdog for the daily autopilot loop.

The 2026-07 incident this exists to prevent: ``autopilot`` hung for 5+ hours a
day for several days straight on a pathological query, and nothing ever told
Michael -- the run just silently never reached ``notify.send``, day after day,
until he happened to notice. ``ai.quant.autopilot.plist`` now wraps the real
command in ``gtimeout`` so a hang gets SIGTERM'd automatically instead of
blocking the DuckDB write lock for hours (see that plist's comments), but a
`timeout` wrapper is one mechanism, not a guarantee -- launchd oddities, a
`gtimeout` that itself never got installed on some future machine, or a
process in an unusual state could all defeat it. This script is the second,
independent layer: it runs on its own schedule (``StartInterval``, mirroring
``ai.quant.graph-watchdog.plist``/``watch_relationship_graph.py``) and checks,
from *outside* the pipeline entirely, whether today's run is alive, hung, or
never happened at all.

Deliberately does **not** read ``pipeline_runs`` (or any other DuckDB table)
to make that determination, even though ``orchestrator.run`` now bookkeeps a
``pipeline_name="autopilot"`` row for exactly this purpose. The reason is
mechanical, not stylistic: DuckDB is single-writer, and a hung step holds its
connection open for the entire hang -- which means a watchdog `connect()`
attempt made *during* the very hang it is trying to detect will itself fail
with ``IOException: could not set lock``. Polling the DB to answer "is the DB
stuck" is exactly backwards. Instead this script asks three questions, none of
which touch the database:

* **Is the process still alive, and for how long?** (``pgrep``/``ps -o etime``
  against the same command line the plist launches.) A process alive past the
  hang threshold is killed with SIGTERM -- never SIGKILL, so DuckDB gets the
  same clean chance to flush its WAL that the `gtimeout` wrapper already gives
  it (confirmed empirically: this exact process has survived SIGTERM cleanly
  for days).
* **Is the machine running out of memory right now?** (``ram_guard.read_free_pct``,
  the same ``memory_pressure``-backed reading ``orchestrator._check_ram``
  already uses once per run.) Added 2026-07-30 after a real incident: a bulk
  ``UPDATE`` over ~1.5M rows drove this 16GB mini from healthy to under 1%
  free in under 10 minutes -- fast enough that no wall-clock threshold this
  watchdog would reasonably use could have caught it before the machine
  became unusable. ``ram_guard`` itself never kills the live autopilot
  process (it explicitly protects the invoking process and its ancestors --
  its job is stopping *duplicate* waste, not the one real run), so this is a
  genuine gap only this watchdog can close. Checked first, ahead of the hang
  threshold, since a memory crisis is urgent regardless of how long the run
  has been going.
* **Did today's run ever produce a briefing?** (does
  ``<vault>/briefings/<UTC-date>.md`` exist.) ``orchestrator._deliver`` writes
  that note on every exit path -- success, partial, and even the DB-busy
  failure escape hatch -- so its absence past a generous deadline means the
  loop never got that far, or never started at all (e.g. the Mac was asleep
  through 06:00 and stayed asleep).

One more deliberate departure from ``watch_relationship_graph.py``: this
watchdog's own lock/state/log files live under the local project directory
(``PROJECT_ROOT/logs``), not ``config.paths.logs_dir``. ``config.paths.home``
resolves onto the external "Extreme" SSD (``MARKET_INTELLIGENCE_HOME``), and
part of this script's job is to notice and report when that mount is asleep
or missing -- it cannot depend on writing to the very volume it is checking
for. ``graph_watchdog`` doesn't need this property (it only bookkeeps a
process that itself already depends on that mount, so degrading alongside it
is fine); this one does, so it deviates on purpose.

Unlike ``graph-watchdog`` (``--quiet`` by convention -- Michael only wants
Telegram for trade alerts, that watchdog's job is purely local), this
watchdog's entire purpose is alerting Michael, so it never suppresses
Telegram. There is no ``--quiet`` flag here at all.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from market_intelligence.autopilot import ram_guard
from market_intelligence.config import Config, load_config

# Matches the wall-clock ceiling the `gtimeout` wrapper in
# ai.quant.autopilot.plist enforces -- this is defense in depth for the case
# that wrapper somehow doesn't fire (a launchd oddity, a machine missing
# gtimeout), not the primary kill mechanism. Raised from 120 to 240 on
# 2026-07-30, same day and same reason as the plist's ceiling: the first live
# run under 120m was killed mid-`build-dataset` while still legitimately
# progressing (bounded memory, no errors) -- 120m was too tight for a real
# catch-up-sized run, not just for a hang.
DEFAULT_HANG_THRESHOLD_MINUTES = 240

# Local hour (the same clock StartCalendarInterval's Hour=6 trigger uses)
# past which "no run at all today" becomes worth paging about. Four hours
# past the 06:00 trigger is the same generous margin as the hang threshold.
DEFAULT_DEADLINE_HOUR = 10

# Don't re-alert "still missing" more often than this once the deadline has
# passed -- Michael should hear about it promptly, then be reminded
# periodically, not paged every ~20 minutes for the rest of the day.
DEFAULT_MISSING_RUN_REPEAT_MINUTES = 180

# Free-memory floor (macOS's own "free percentage", same figure ram_guard.py
# already reads via `memory_pressure`), below which a live autopilot process
# is killed regardless of how long it has been running. 2026-07-30 incident:
# a single-statement bulk UPDATE over ~1.5M rows drove this 16GB mini to under
# 1% free in under 10 minutes, well inside any wall-clock hang threshold --
# the hang check alone would not have caught it. Lower (more permissive) than
# ram_guard's own DEFAULT_ACT_BELOW_FREE_PCT (20%): that guard proactively
# frees waste on a merely-busy machine, this one only fires once the machine
# is in genuine crisis territory, since killing the one live pipeline run is
# a much bigger deal than killing an accidental duplicate.
DEFAULT_MEMORY_CRITICAL_FREE_PCT = 8.0

# Matches the command line launchd actually runs (see ai.quant.autopilot.plist):
# `gtimeout ... uv run market-intelligence autopilot run`. `pgrep -f` matches
# full command lines, so this catches the gtimeout wrapper, the uv shim, and
# the python child alike -- any match means "today's run is alive".
AUTOPILOT_PATTERN = "market-intelligence autopilot run"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = _parse_args()
    config = load_config()
    # Deliberately no config.paths.ensure() here: that creates directories on
    # config.paths.home, which is the external mount this script must be able
    # to report on as *missing* -- creating directories on it would either
    # fail loudly on a genuinely absent mount or (worse) mask the condition.

    lock_path = _local_path("autopilot_watchdog.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _log(config, "another watchdog pass is already running")
            return 0

        if args.loop:
            while True:
                _run_once(config, args)
                time.sleep(args.interval_seconds)
        _run_once(config, args)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run forever, sleeping between checks. Launchd normally omits this.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=900,
        help="Sleep interval for --loop mode (manual testing only).",
    )
    parser.add_argument(
        "--hang-threshold-minutes",
        type=int,
        default=DEFAULT_HANG_THRESHOLD_MINUTES,
        help="Kill a live autopilot process older than this, wall-clock.",
    )
    parser.add_argument(
        "--deadline-hour",
        type=int,
        default=DEFAULT_DEADLINE_HOUR,
        help="Local hour past which a missing run is alertable.",
    )
    parser.add_argument(
        "--repeat-minutes",
        type=int,
        default=DEFAULT_MISSING_RUN_REPEAT_MINUTES,
        help="Minimum gap between repeated 'still missing' alerts.",
    )
    parser.add_argument(
        "--memory-critical-free-pct",
        type=float,
        default=DEFAULT_MEMORY_CRITICAL_FREE_PCT,
        help="Kill a live autopilot process immediately if free memory drops below this.",
    )
    return parser.parse_args()


def _run_once(
    config: Config,
    args: argparse.Namespace,
    *,
    now: datetime | None = None,
    transport: httpx.BaseTransport | None = None,
    find_pids: Callable[[], list[int]] = lambda: _find_autopilot_pids(),
    kill: Callable[[], bool] = lambda: _kill_autopilot(),
    read_free_pct: Callable[[], float] = ram_guard.read_free_pct,
) -> None:
    """One watchdog pass. ``find_pids``/``kill`` are injectable so tests can
    drive the hang/kill branch without spawning or signaling real processes;
    ``transport`` is injectable the same way ``notify.send`` does it, so the
    whole suite can run through ``httpx.MockTransport`` and never touch the
    network (see module docstring for why this diverges from
    ``watch_relationship_graph.py``, which has no test coverage of its own).
    ``now`` is injectable the same way ``orchestrator.run`` injects its clock,
    so the deadline-hour branch is testable without waiting for a real one.
    ``read_free_pct`` defaults to ``ram_guard.read_free_pct`` -- the same
    ``memory_pressure``-backed primitive ``orchestrator._check_ram`` already
    uses -- reused rather than re-implemented; see
    ``DEFAULT_MEMORY_CRITICAL_FREE_PCT`` for why this needs its own, lower
    threshold rather than sharing ``ram_guard``'s.
    """
    state = _read_state(config)
    now_local = now or datetime.now()
    pids = find_pids()

    if pids:
        elapsed_minutes = _max_elapsed_minutes(pids)
        free_pct = read_free_pct()
        if free_pct < args.memory_critical_free_pct:
            _log(
                config,
                f"autopilot pids={pids} free_pct={free_pct:.1f}% "
                f"< critical={args.memory_critical_free_pct}%; killing",
            )
            killed = kill()
            _send_text(
                config,
                "Quant autopilot watchdog: killed a run for critical memory "
                "pressure.\n"
                f"Free memory: {free_pct:.1f}% "
                f"(critical floor {args.memory_critical_free_pct}%).\n"
                f"pid(s): {pids}\n"
                f"Sent SIGTERM: {killed}\n"
                "Distinct from the hang-threshold kill: this fired because the "
                "machine was running out of RAM, not because the run had been "
                "going too long -- see the 2026-07-30 bulk-UPDATE-memory "
                "incident. A wall-clock timeout alone would not have caught "
                "this.",
                transport=transport,
            )
        elif elapsed_minutes is not None and elapsed_minutes >= args.hang_threshold_minutes:
            _log(
                config,
                f"autopilot pids={pids} elapsed={elapsed_minutes:.0f}m "
                f">= threshold={args.hang_threshold_minutes}m; killing",
            )
            killed = kill()
            _send_text(
                config,
                "Quant autopilot watchdog: killed a hung run.\n"
                f"Alive for {elapsed_minutes:.0f} minutes "
                f"(threshold {args.hang_threshold_minutes}m).\n"
                f"pid(s): {pids}\n"
                f"Sent SIGTERM: {killed}\n"
                "This should already have been caught by the gtimeout wrapper "
                "in ai.quant.autopilot.plist -- if you're seeing this, that "
                "layer didn't fire and is worth a look.",
                transport=transport,
            )
        else:
            _log(
                config,
                f"autopilot alive pids={pids} elapsed={elapsed_minutes}m "
                f"free_pct={free_pct:.1f}%, ok",
            )
        state["last_checked_at"] = _now_iso()
        _write_state(config, state)
        return

    # No live process. Either it finished (note should exist) or it never ran.
    if not _mount_available(config):
        if now_local.hour >= args.deadline_hour and _should_warn(
            state, "last_mount_alert_at", args.repeat_minutes
        ):
            _send_text(
                config,
                "Quant autopilot watchdog: the external data drive isn't "
                "mounted.\n"
                f"Expected home: {config.paths.home}\n"
                "Autopilot cannot run or write its briefing until this volume "
                "is back.",
                transport=transport,
            )
            state["last_mount_alert_at"] = _now_iso()
        _log(config, "mount unavailable, no autopilot process")
        state["last_checked_at"] = _now_iso()
        _write_state(config, state)
        return

    if _note_exists_for_today(config):
        _log(config, "no live process, today's briefing note exists -- run completed")
        state["last_checked_at"] = _now_iso()
        state["last_seen_note_date"] = _today_utc_iso()
        _write_state(config, state)
        return

    if now_local.hour < args.deadline_hour:
        _log(config, "no live process, no note yet, before deadline -- normal")
        state["last_checked_at"] = _now_iso()
        _write_state(config, state)
        return

    if _should_warn(state, "last_missing_run_alert_at", args.repeat_minutes):
        _send_text(
            config,
            "Quant autopilot watchdog: today's run never happened.\n"
            f"No autopilot process is running and no briefing note exists for "
            f"{_today_utc_iso()}.\n"
            "Likely the Mac was asleep through 06:00 (RunAtLoad now catches "
            "up on wake/login) or the LaunchAgent didn't fire. Run it "
            "manually: cd ~/quant && uv run market-intelligence autopilot run",
            transport=transport,
        )
        state["last_missing_run_alert_at"] = _now_iso()
    _log(config, "no live process, no note, past deadline -- alerted (or throttled)")
    state["last_checked_at"] = _now_iso()
    _write_state(config, state)


# --------------------------------------------------------------------------- #
# Process inspection -- no DB access, see module docstring for why           #
# --------------------------------------------------------------------------- #
def _find_autopilot_pids() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", AUTOPILOT_PATTERN],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _max_elapsed_minutes(pids: list[int]) -> float | None:
    """Wall-clock minutes since the oldest matching pid started, or ``None``."""
    best: float | None = None
    for pid in pids:
        seconds = _elapsed_seconds(pid)
        if seconds is None:
            continue
        minutes = seconds / 60.0
        if best is None or minutes > best:
            best = minutes
    return best


def _elapsed_seconds(pid: int) -> int | None:
    result = subprocess.run(
        ["ps", "-o", "etime=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return _parse_etime_seconds(result.stdout.strip())


def _parse_etime_seconds(etime: str) -> int | None:
    """Parse macOS/BSD ``ps -o etime`` -- ``[[dd-]hh:]mm:ss``."""
    etime = etime.strip()
    if not etime:
        return None
    days = 0
    if "-" in etime:
        day_part, etime = etime.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = etime.split(":")
    try:
        values = [int(p) for p in parts]
    except ValueError:
        return None
    if len(values) == 2:
        hours, (minutes, seconds) = 0, values
    elif len(values) == 3:
        hours, minutes, seconds = values
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _kill_autopilot() -> bool:
    """Send SIGTERM (never SIGKILL) to every matching process; True if any matched."""
    result = subprocess.run(
        ["pkill", "-TERM", "-f", AUTOPILOT_PATTERN],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


# --------------------------------------------------------------------------- #
# Filesystem checks -- also no DB access                                     #
# --------------------------------------------------------------------------- #
def _mount_available(config: Config) -> bool:
    try:
        return config.paths.home.exists()
    except OSError:
        return False


def _note_exists_for_today(config: Config) -> bool:
    try:
        note = config.paths.obsidian_vault_dir / "briefings" / f"{_today_utc_iso()}.md"
        return note.exists()
    except OSError:
        return False


def _today_utc_iso() -> str:
    # Matches orchestrator.run's own `briefing_date` default exactly
    # (`datetime.now(tz=UTC).date()`), since that's what names the note file.
    return datetime.now(UTC).date().isoformat()


# --------------------------------------------------------------------------- #
# Telegram -- a minimal, standalone equivalent of notify._post_text          #
# (mirrors watch_relationship_graph.py's own _send_text rather than reaching #
# into notify.py's private helper across modules).                          #
# --------------------------------------------------------------------------- #
def _send_text(config: Config, text: str, *, transport: httpx.BaseTransport | None = None) -> bool:
    token = (config.env.telegram_bot_token or "").strip()
    chat_id = (config.env.telegram_chat_id or "").strip()
    if not token or not chat_id:
        _log(config, "telegram skipped: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    timeout = httpx.Timeout(
        config.settings.http.timeout_seconds,
        connect=config.settings.http.connect_timeout_seconds,
    )
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(url, json=payload)
    except httpx.HTTPError as exc:
        _log(config, f"telegram failed: {exc}")
        return False
    if not response.is_success:
        _log(config, f"telegram rejected: status={response.status_code} body={response.text[:200]}")
        return False
    return True


# --------------------------------------------------------------------------- #
# Local (not config.paths.logs_dir -- see module docstring) state/log/lock   #
# --------------------------------------------------------------------------- #
def _local_path(name: str) -> Path:
    return PROJECT_ROOT / "logs" / name


def _read_state(config: Config) -> dict[str, Any]:
    path = _local_path("autopilot_watchdog_state.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(config: Config, state: dict[str, Any]) -> None:
    try:
        path = _local_path("autopilot_watchdog_state.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass


def _should_warn(state: dict[str, Any], key: str, minutes: int) -> bool:
    last = state.get(key)
    if not isinstance(last, str):
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except ValueError:
        return True
    return (datetime.now(UTC) - last_dt).total_seconds() >= minutes * 60


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log(config: Config, message: str) -> None:
    try:
        line = f"{_now_iso()} {message}\n"
        path = _local_path("autopilot_watchdog.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
