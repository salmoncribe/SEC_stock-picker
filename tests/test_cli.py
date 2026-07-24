"""CLI smoke tests (offline).

Exercises the commands that do not require network access: help, version,
init-db, status, validate, plus the credential-gate behaviour (collectors must
exit 2 before making any request when credentials are missing).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from market_intelligence.cli import app
from market_intelligence.collectors import RunSummary
from market_intelligence.collectors import gaps as gaps_collector
from market_intelligence.config import reset_config_cache

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the app at a throwaway home with valid test credentials."""
    monkeypatch.setenv("MARKET_INTELLIGENCE_HOME", str(tmp_path))
    monkeypatch.setenv("SEC_USER_AGENT", "TestSuite/0.1 (tests@market-intel.test)")
    monkeypatch.setenv("FRED_API_KEY", "test-fred-key")
    reset_config_cache()
    yield
    reset_config_cache()


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "market-intelligence" in result.output.lower()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "market-intelligence" in result.output


def test_init_db_creates_database(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init-db"])
    assert result.exit_code == 0
    assert (tmp_path / "data" / "database" / "market_intelligence.duckdb").exists()


def test_status_runs() -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "configuration" in result.output.lower()


def test_validate_runs() -> None:
    runner.invoke(app, ["init-db"])
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0


def test_sec_and_fred_help() -> None:
    assert runner.invoke(app, ["sec", "--help"]).exit_code == 0
    assert runner.invoke(app, ["fred", "--help"]).exit_code == 0


def test_collect_filings_without_sec_user_agent_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "MarketIntelligence/0.1 (you@example.com)")
    reset_config_cache()
    result = runner.invoke(app, ["sec", "collect-filings", "--ticker", "NVDA"])
    assert result.exit_code == 2


def test_sync_insider_help() -> None:
    result = runner.invoke(app, ["sec", "sync-insider", "--help"])
    assert result.exit_code == 0
    assert "--start-quarter" in result.output
    assert "--end-quarter" in result.output


def test_sync_insider_without_sec_user_agent_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "MarketIntelligence/0.1 (you@example.com)")
    reset_config_cache()
    result = runner.invoke(app, ["sec", "sync-insider"])
    assert result.exit_code == 2


def test_fred_sync_without_key_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    reset_config_cache()
    result = runner.invoke(app, ["fred", "sync"])
    assert result.exit_code == 2


def test_scan_gaps_invokes_scan_and_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_scan(config: object, **kwargs: object) -> RunSummary:
        captured.update(kwargs)
        return RunSummary(pipeline_name="gaps.scan", status="success")

    monkeypatch.setattr(gaps_collector, "scan", fake_scan)
    result = runner.invoke(app, ["market", "scan-gaps"])

    assert result.exit_code == 0
    assert captured["notify"] is True


def test_scan_gaps_dry_run_passes_notify_false(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_scan(config: object, **kwargs: object) -> RunSummary:
        captured.update(kwargs)
        return RunSummary(pipeline_name="gaps.scan", status="success")

    monkeypatch.setattr(gaps_collector, "scan", fake_scan)
    result = runner.invoke(app, ["market", "scan-gaps", "--dry-run"])

    assert result.exit_code == 0
    assert captured["notify"] is False


def test_scan_gaps_skipped_status_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_scan(config: object, **kwargs: object) -> RunSummary:
        return RunSummary(pipeline_name="gaps.scan", status="skipped")

    monkeypatch.setattr(gaps_collector, "scan", fake_scan)
    result = runner.invoke(app, ["market", "scan-gaps"])

    assert result.exit_code == 0


def test_scan_gaps_raising_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_scan(config: object, **kwargs: object) -> RunSummary:
        raise RuntimeError("provider blew up")

    monkeypatch.setattr(gaps_collector, "scan", fake_scan)
    result = runner.invoke(app, ["market", "scan-gaps"])

    assert result.exit_code == 1
