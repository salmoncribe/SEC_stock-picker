#!/usr/bin/env python3
"""Watch relationship extraction and refresh the Obsidian graph projection.

This is intentionally small and operational: launchd can run it every few
minutes, and each pass checks that the extractor is alive, projects whatever
the database currently exposes, and sends Telegram nudges when the generated
company-note count grows or the pipeline looks stuck.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import httpx

from market_intelligence.autopilot import graph
from market_intelligence.config import Config, load_config

DEFAULT_INTERVAL_SECONDS = 180
DEFAULT_HEARTBEAT_MINUTES = 30
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_PATTERN = "extract_relationships.sh|signals extract-relationships"


@dataclass
class Metrics:
    edges: int
    resolved_edges: int
    source_companies: int
    projectable_notes: int
    remaining_sections: int


def main() -> int:
    args = _parse_args()
    config = load_config()
    config.paths.ensure()

    lock_path = config.paths.logs_dir / "graph_watchdog.lock"
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
        default=DEFAULT_INTERVAL_SECONDS,
        help="Sleep interval for --loop mode.",
    )
    parser.add_argument(
        "--heartbeat-minutes",
        type=int,
        default=DEFAULT_HEARTBEAT_MINUTES,
        help="Send a no-growth alive heartbeat at most this often.",
    )
    parser.add_argument(
        "--notify-start",
        action="store_true",
        help="Send a Telegram message even if no new notes were built.",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=16,
        help="Relationship sections to extract before each projection attempt.",
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Do not run an extraction batch when no extractor is active.",
    )
    return parser.parse_args()


def _run_once(config: Config, args: argparse.Namespace) -> None:
    state = _read_state(config)
    now = _now()
    extractor_alive = _extractor_alive()
    batch_ran = False
    batch_ok = True

    if not extractor_alive and not args.no_restart:
        batch_ok = _run_extraction_batch(config, args.batch_limit)
        batch_ran = True
        extractor_alive = _extractor_alive()

    try:
        metrics = _read_metrics(config)
        written = _project_notes(config)
    except Exception as exc:
        state["projection_failures"] = int(state.get("projection_failures", 0)) + 1
        _log(config, f"projection skipped: {type(exc).__name__}: {exc}")
        if _should_warn(state, "last_failure_ping_at", args.heartbeat_minutes):
            _send_text(
                config,
                "Quant graph watchdog: projection could not run.\n"
                f"Reason: {type(exc).__name__}: {exc}\n"
                f"Extractor alive: {extractor_alive}",
            )
            state["last_failure_ping_at"] = now
        state["last_checked_at"] = now
        _write_state(config, state)
        return

    state["projection_failures"] = 0
    previous_notes = int(state.get("last_note_count", 0))
    previous_edges = int(state.get("last_edge_count", 0))
    grew = written > previous_notes
    edges_grew = metrics.edges > previous_edges

    if grew:
        _send_text(
            config,
            "Quant Obsidian graph grew.\n"
            f"Company notes: {previous_notes} -> {written}\n"
            f"Edges: {previous_edges} -> {metrics.edges}\n"
            f"Resolved edges: {metrics.resolved_edges}\n"
            f"Sources processed: {metrics.source_companies}\n"
            f"Remaining sections: {metrics.remaining_sections}\n"
            f"Vault: {config.paths.obsidian_vault_dir}",
        )
        state["last_growth_ping_at"] = now
    elif args.notify_start or _should_warn(state, "last_heartbeat_ping_at", args.heartbeat_minutes):
        status = "active elsewhere" if extractor_alive else "idle"
        action = "batch ran" if batch_ran else "checked"
        if batch_ran and not batch_ok:
            action = "batch failed"
        _send_text(
            config,
            "Quant graph watchdog heartbeat.\n"
            f"Extractor: {status} ({action})\n"
            f"Company notes: {written}\n"
            f"Edges: {metrics.edges}"
            + (" (new edges waiting for links)" if edges_grew else "")
            + f"\nRemaining sections: {metrics.remaining_sections}",
        )
        state["last_heartbeat_ping_at"] = now

    state.update(
        {
            "last_checked_at": now,
            "last_note_count": written,
            "last_edge_count": metrics.edges,
            "last_resolved_edges": metrics.resolved_edges,
            "last_source_companies": metrics.source_companies,
            "last_remaining_sections": metrics.remaining_sections,
            "extractor_alive": extractor_alive,
            "extraction_batch_ran": batch_ran,
            "extraction_batch_ok": batch_ok,
        }
    )
    _write_state(config, state)
    _log(
        config,
        "watchdog ok "
        f"notes={written} edges={metrics.edges} resolved={metrics.resolved_edges} "
        f"remaining={metrics.remaining_sections} extractor_alive={extractor_alive}",
    )


def _read_metrics(config: Config) -> Metrics:
    con = duckdb.connect(str(config.paths.database_path), read_only=True)
    try:
        con.execute("SET TimeZone='UTC'")
        edges = _scalar(con, "SELECT count(*) FROM company_edges")
        resolved = _scalar(
            con,
            "SELECT count(*) FROM company_edges WHERE resolution_status = 'resolved'",
        )
        sources = _scalar(con, "SELECT count(DISTINCT source_ticker) FROM company_edges")
        projectable = _scalar(
            con,
            """
            SELECT count(DISTINCT ticker)
            FROM (
                SELECT source_ticker AS ticker FROM company_edges
                UNION
                SELECT target_ticker AS ticker FROM company_edges WHERE target_ticker IS NOT NULL
            )
            """,
        )
        remaining = _scalar(con, _remaining_sql())
        return Metrics(edges, resolved, sources, projectable, remaining)
    finally:
        con.close()


def _project_notes(config: Config) -> int:
    con = duckdb.connect(str(config.paths.database_path), read_only=True)
    try:
        con.execute("SET TimeZone='UTC'")
        return graph.project_graph(con, config.paths.obsidian_vault_dir)
    finally:
        con.close()


def _remaining_sql() -> str:
    return """
        SELECT count(*) FROM filing_sections s
        JOIN companies c ON c.cik = s.cik
        WHERE s.item_code = '1'
          AND c.ticker IN (SELECT DISTINCT symbol FROM daily_returns)
          AND s.text_path IS NOT NULL
          AND s.accession_number = (
              SELECT s2.accession_number FROM filing_sections s2
              WHERE s2.cik = s.cik AND s2.item_code = '1'
              ORDER BY s2.report_date DESC NULLS LAST, s2.accession_number LIMIT 1
          )
          AND NOT EXISTS (
              SELECT 1 FROM company_edges e WHERE e.accession_number = s.accession_number
          )
          AND NOT EXISTS (
              SELECT 1 FROM processed_relationship_sections p
              WHERE p.accession_number = s.accession_number
                AND p.item_code = s.item_code
          )
    """


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    value = con.execute(sql).fetchone()[0]
    return int(value or 0)


def _extractor_alive() -> bool:
    result = subprocess.run(
        ["pgrep", "-fl", EXTRACT_PATTERN],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _run_extraction_batch(config: Config, batch_limit: int) -> bool:
    _log(config, f"extractor idle; running batch size={batch_limit}")
    env = dict(**_clean_env(), RELATIONSHIP_BATCH_LIMIT=str(batch_limit))
    result = subprocess.run(
        ["./scripts/extract_relationships.sh", "1"],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        _log(config, f"extraction batch failed: status={result.returncode}")
        return False
    return True


def _clean_env() -> dict[str, str]:
    import os

    return dict(os.environ)


def _send_text(config: Config, text: str) -> bool:
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
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
    except httpx.HTTPError as exc:
        _log(config, f"telegram failed: {exc}")
        return False
    if not response.is_success:
        _log(config, f"telegram rejected: status={response.status_code} body={response.text[:200]}")
        return False
    return True


def _read_state(config: Config) -> dict[str, Any]:
    path = _state_path(config)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(config: Config, state: dict[str, Any]) -> None:
    path = _state_path(config)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _state_path(config: Config) -> Path:
    return config.paths.logs_dir / "graph_watchdog_state.json"


def _should_warn(state: dict[str, Any], key: str, minutes: int) -> bool:
    last = state.get(key)
    if not isinstance(last, str):
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except ValueError:
        return True
    return (datetime.now(UTC) - last_dt).total_seconds() >= minutes * 60


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log(config: Config, message: str) -> None:
    line = f"{_now()} {message}\n"
    path = config.paths.logs_dir / "graph_watchdog.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.open("a", encoding="utf-8").write(line)


if __name__ == "__main__":
    sys.exit(main())
