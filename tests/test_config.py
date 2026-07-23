"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from market_intelligence.config import (
    CompanyEntry,
    Config,
    ConfigError,
    FredDefaults,
    GapScannerConfig,
    TradingConfig,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loads_all_yaml_sections() -> None:
    config = load_config(project_root=PROJECT_ROOT)
    assert config.settings.sec.base_url.startswith("https://")
    assert config.settings.fred.base_url.startswith("https://")
    assert len(config.fred_series.series) >= 6
    assert any(entry.ticker == "NVDA" for entry in config.companies.companies)
    assert "10-K" in config.forms.supported
    assert "S-1" in config.forms.ipo


def test_paths_are_rooted_at_home(tmp_path: Path) -> None:
    config = load_config(project_root=PROJECT_ROOT, home=tmp_path)
    assert (
        config.paths.database_path == tmp_path / "data" / "database" / "market_intelligence.duckdb"
    )
    assert config.paths.raw_dir == tmp_path / "data" / "raw"
    config.paths.ensure()
    assert config.paths.raw_dir.is_dir()
    assert config.paths.logs_dir.is_dir()


def test_require_sec_user_agent_rejects_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "MarketIntelligence/0.1 (you@example.com)")
    config = load_config(project_root=PROJECT_ROOT)
    with pytest.raises(ConfigError, match="SEC_USER_AGENT"):
        config.require_sec_user_agent()


def test_require_sec_user_agent_accepts_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "MarketIntelligence/0.1 (real@company.test)")
    config = load_config(project_root=PROJECT_ROOT)
    assert config.require_sec_user_agent() == "MarketIntelligence/0.1 (real@company.test)"


def test_require_sec_user_agent_rejects_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "   ")
    config = load_config(project_root=PROJECT_ROOT)
    with pytest.raises(ConfigError):
        config.require_sec_user_agent()


def test_require_fred_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    config = load_config(project_root=PROJECT_ROOT)
    with pytest.raises(ConfigError, match="FRED_API_KEY"):
        config.require_fred_api_key()


def test_require_fred_api_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "abc123")
    config = load_config(project_root=PROJECT_ROOT)
    assert config.require_fred_api_key() == "abc123"


def test_fred_defaults_empty_string_becomes_none() -> None:
    defaults = FredDefaults(observation_start="2015-01-01", observation_end="")
    assert defaults.observation_start == "2015-01-01"
    assert defaults.observation_end is None


def test_company_entry_ticker_is_uppercased() -> None:
    assert CompanyEntry(ticker="nvda").ticker == "NVDA"


def test_missing_config_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Missing configuration file"):
        load_config(project_root=tmp_path)


def test_fingerprint_is_stable_and_serializable() -> None:
    config = load_config(project_root=PROJECT_ROOT)
    first = config.fingerprint()
    second = load_config(project_root=PROJECT_ROOT).fingerprint()
    assert first == second
    assert "series" in first and "forms" in first


class TestTradingConfig:
    def test_model_defaults_without_yaml_entry(self) -> None:
        """Constructed directly (no settings.yaml involved) — pins the actual defaults."""
        t = TradingConfig()
        assert t.account_equity == 10_000.0
        assert t.risk_pct_per_trade == 1.0
        assert t.atr_period == 14
        assert t.atr_stop_multiple == 2.0
        assert t.max_position_pct == 20.0
        assert t.min_confidence == 60

    def test_yaml_values_load(self, tmp_config: Config) -> None:
        t = tmp_config.settings.trading
        assert t.account_equity > 0
        assert 0 < t.risk_pct_per_trade <= 100
        assert t.atr_period >= 2
        assert t.atr_stop_multiple > 0
        assert 0 < t.max_position_pct <= 100
        assert 0 <= t.min_confidence <= 100


class TestGapScannerConfig:
    def test_model_defaults_without_yaml_entry(self) -> None:
        """Constructed directly (no settings.yaml involved) — pins the actual defaults."""
        g = GapScannerConfig()
        assert g.min_gap_pct == 3.0
        assert g.min_avg_dollar_volume == 5_000_000.0
        assert g.gap_target_r_multiple == 2.0
        assert g.gap_max_hold_days == 5
        assert g.gap_catalyst_lookback_days == 7
        assert g.require_catalyst is False

    def test_yaml_values_load(self, tmp_config: Config) -> None:
        g = tmp_config.settings.gap_scanner
        assert g.min_gap_pct > 0
        assert g.min_avg_dollar_volume > 0
        assert g.gap_target_r_multiple > 0
        assert g.gap_max_hold_days >= 1
        assert g.gap_catalyst_lookback_days >= 1
        assert g.require_catalyst in (True, False)
