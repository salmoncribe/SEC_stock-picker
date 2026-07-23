"""Typed configuration: ``.env`` secrets + YAML settings.

Loading is validated end to end with Pydantic. Secrets come from the
environment (``.env`` via python-dotenv); everything else comes from the
YAML files under ``config/``.

``get_config()`` is cached; ``load_config()`` is the uncached loader that
tests parameterize with a temp project root / home.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid, with remediation text."""


# --------------------------------------------------------------------------- #
# Environment (.env) — secrets only                                           #
# --------------------------------------------------------------------------- #
class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    sec_user_agent: str | None = None
    fred_api_key: str | None = None
    market_intelligence_home: str | None = None
    log_level: str = "INFO"
    # Autopilot notification secrets, shared with the existing Telegram bot.
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None


# --------------------------------------------------------------------------- #
# YAML: settings.yaml                                                          #
# --------------------------------------------------------------------------- #
class AppInfo(BaseModel):
    name: str = "market-intelligence"
    schema_version: str = "1.0.0"


class HttpConfig(BaseModel):
    timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 10.0


class RetryConfig(BaseModel):
    max_attempts: int = Field(default=5, ge=1)
    initial_backoff_seconds: float = Field(default=1.0, ge=0.0)
    max_backoff_seconds: float = Field(default=30.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    jitter_seconds: float = Field(default=0.5, ge=0.0)


class SourcePacing(BaseModel):
    requests_per_second: float = Field(default=5.0, gt=0.0)


class PacingConfig(BaseModel):
    sec: SourcePacing = SourcePacing(requests_per_second=5.0)
    fred: SourcePacing = SourcePacing(requests_per_second=8.0)


class SecConfig(BaseModel):
    base_url: str
    data_base_url: str
    company_tickers_path: str
    submissions_path_template: str
    company_facts_path_template: str


class FredConfig(BaseModel):
    base_url: str
    series_observations_path: str
    series_path: str
    file_type: str = "json"


class AutopilotConfig(BaseModel):
    """Tuning for the self-checking loop and the promotion ladder.

    ``discovery_boundary`` freezes the discovery/holdout split so re-running the
    gate cannot mint new admissions from a growing pool; new data only ever
    grows holdout. ``promotion_streak`` (K) is how many consecutive holdout
    confirmations a cell needs before it is trusted to fire alerts.
    ``retire_after`` is how many consecutive failing runs retire a dormant cell.
    """

    enabled: bool = True
    discovery_boundary: str = "2023-01-01"
    promotion_streak: int = Field(default=2, ge=1)
    retire_after: int = Field(default=3, ge=1)
    obsidian_vault_subdir: str = "obsidian"


class LLMConfig(BaseModel):
    """Local LLM backend for relationship extraction (the connection engine).

    Defaults target an ``ollama`` server on the loopback interface serving
    ``qwen2.5:7b-instruct``. ``num_ctx`` is large enough for a full 10-K Item 1
    section; a small model on a 16 GB machine is the reason the 7B (not 14B) is
    the default. Everything is a plain default so no YAML entry is required.
    """

    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    timeout_seconds: float = Field(default=180.0, gt=0.0)
    num_ctx: int = Field(default=16384, ge=2048)


class TradingConfig(BaseModel):
    """User-owned risk parameters for rendered trade plans.

    Every number here belongs to Michael, not the system: the alert layer only
    does arithmetic with them. ``account_equity`` ships as an obvious
    placeholder so a rendered size is never mistaken for advice about a real
    account until he sets it.
    """

    account_equity: float = Field(default=10_000.0, gt=0.0)
    risk_pct_per_trade: float = Field(default=1.0, gt=0.0, le=100.0)
    atr_period: int = Field(default=14, ge=2)
    atr_stop_multiple: float = Field(default=2.0, gt=0.0)
    max_position_pct: float = Field(default=20.0, gt=0.0, le=100.0)
    min_confidence: int = Field(default=60, ge=0, le=100)


class GapScannerConfig(BaseModel):
    """Morning price-gap scan thresholds (see trade-alerts design §8).

    ``gap_target_r_multiple`` sets the profit target as a multiple of the stop
    distance: target = entry +/- (gap_target_r_multiple * R), where R is the
    entry-to-stop distance. ``require_catalyst`` defaults to False so the
    scanner also surfaces unexplained gaps (no confirmed news event) rather
    than only catalyst-backed ones — Michael reviews those manually.
    """

    min_gap_pct: float = Field(default=3.0, gt=0.0)
    min_avg_dollar_volume: float = Field(default=5_000_000.0, gt=0.0)
    gap_target_r_multiple: float = Field(default=2.0, gt=0.0)
    gap_max_hold_days: int = Field(default=5, ge=1)
    gap_catalyst_lookback_days: int = Field(default=7, ge=1)
    require_catalyst: bool = False


class SettingsFile(BaseModel):
    app: AppInfo = AppInfo()
    http: HttpConfig = HttpConfig()
    retry: RetryConfig = RetryConfig()
    pacing: PacingConfig = PacingConfig()
    autopilot: AutopilotConfig = AutopilotConfig()
    llm: LLMConfig = LLMConfig()
    trading: TradingConfig = TradingConfig()
    gap_scanner: GapScannerConfig = GapScannerConfig()
    sec: SecConfig
    fred: FredConfig


# --------------------------------------------------------------------------- #
# YAML: fred_series.yaml                                                       #
# --------------------------------------------------------------------------- #
class FredSeriesEntry(BaseModel):
    id: str
    alias: str | None = None
    note: str | None = None


class FredDefaults(BaseModel):
    observation_start: str | None = None
    observation_end: str | None = None

    @field_validator("observation_start", "observation_end", mode="before")
    @classmethod
    def _empty_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return None
        return v


class FredSeriesFile(BaseModel):
    series: list[FredSeriesEntry] = Field(default_factory=list)
    defaults: FredDefaults = FredDefaults()


# --------------------------------------------------------------------------- #
# YAML: companies.yaml                                                         #
# --------------------------------------------------------------------------- #
class CompanyEntry(BaseModel):
    ticker: str
    cik: str | None = None
    name: str | None = None

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class CompaniesFile(BaseModel):
    companies: list[CompanyEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# YAML: forms.yaml                                                             #
# --------------------------------------------------------------------------- #
class FormsFile(BaseModel):
    supported: list[str] = Field(default_factory=list)
    ipo: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Resolved paths                                                              #
# --------------------------------------------------------------------------- #
class Paths(BaseModel):
    home: Path
    project_root: Path
    config_dir: Path
    data_dir: Path
    raw_dir: Path
    normalized_dir: Path
    parquet_dir: Path
    database_dir: Path
    database_path: Path
    logs_dir: Path
    log_file: Path
    obsidian_vault_dir: Path

    def ensure(self) -> Paths:
        """Create every data/log directory if missing. Safe to call repeatedly."""
        for directory in (
            self.data_dir,
            self.raw_dir,
            self.normalized_dir,
            self.parquet_dir,
            self.database_dir,
            self.logs_dir,
            self.obsidian_vault_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------- #
# Aggregate config object                                                      #
# --------------------------------------------------------------------------- #
class Config:
    def __init__(
        self,
        *,
        env: EnvSettings,
        settings: SettingsFile,
        fred_series: FredSeriesFile,
        companies: CompaniesFile,
        forms: FormsFile,
        paths: Paths,
    ) -> None:
        self.env = env
        self.settings = settings
        self.fred_series = fred_series
        self.companies = companies
        self.forms = forms
        self.paths = paths

    # -- credential gates --------------------------------------------------- #
    _PLACEHOLDER_EMAILS = ("you@example.com", "your-email@example.com")

    def require_sec_user_agent(self) -> str:
        ua = (self.env.sec_user_agent or "").strip()
        if not ua or any(ph in ua for ph in self._PLACEHOLDER_EMAILS):
            raise ConfigError(
                "SEC_USER_AGENT is not configured with a real contact email. "
                "SEC EDGAR requires a descriptive User-Agent identifying your "
                "application and a contact address, e.g. "
                '"MarketIntelligence/0.1 (you@yourdomain.com)". '
                "Set SEC_USER_AGENT in your .env (see .env.example)."
            )
        return ua

    def require_fred_api_key(self) -> str:
        key = (self.env.fred_api_key or "").strip()
        if not key:
            raise ConfigError(
                "FRED_API_KEY is not configured. Get a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html and set "
                "FRED_API_KEY in your .env (see .env.example)."
            )
        return key

    def fingerprint(self) -> dict[str, Any]:
        """Config subset that meaningfully affects collection output.

        Hashed into ``pipeline_runs.config_hash`` so a run is traceable to the
        configuration that produced it.
        """
        return {
            "app": self.settings.app.model_dump(),
            "sec": self.settings.sec.model_dump(),
            "fred": self.settings.fred.model_dump(),
            "pacing": self.settings.pacing.model_dump(),
            "retry": self.settings.retry.model_dump(),
            "http": self.settings.http.model_dump(),
            "series": [s.model_dump() for s in self.fred_series.series],
            "fred_defaults": self.fred_series.defaults.model_dump(),
            "companies": [c.model_dump() for c in self.companies.companies],
            "forms": self.forms.model_dump(),
        }


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def _project_root() -> Path:
    # src/market_intelligence/config.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Missing configuration file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} must contain a mapping at the top level.")
    return data


def _build_paths(home: Path, project_root: Path, vault_subdir: str) -> Paths:
    data_dir = home / "data"
    database_dir = data_dir / "database"
    logs_dir = home / "logs"
    return Paths(
        home=home,
        project_root=project_root,
        config_dir=project_root / "config",
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        normalized_dir=data_dir / "normalized",
        parquet_dir=data_dir / "parquet",
        database_dir=database_dir,
        database_path=database_dir / "market_intelligence.duckdb",
        logs_dir=logs_dir,
        log_file=logs_dir / "market_intelligence.log",
        obsidian_vault_dir=home / vault_subdir,
    )


def load_config(
    project_root: str | Path | None = None,
    home: str | Path | None = None,
) -> Config:
    """Load and validate configuration. Uncached (see ``get_config``)."""
    root = Path(project_root).expanduser().resolve() if project_root else _project_root()
    load_dotenv(root / ".env")  # populates os.environ; no-op if the file is absent
    env = EnvSettings()

    if home is not None:
        resolved_home = Path(home).expanduser().resolve()
    elif env.market_intelligence_home:
        resolved_home = Path(env.market_intelligence_home).expanduser().resolve()
    else:
        resolved_home = root

    config_dir = root / "config"
    settings = SettingsFile(**_load_yaml(config_dir / "settings.yaml"))
    fred_series = FredSeriesFile(**_load_yaml(config_dir / "fred_series.yaml"))
    companies = CompaniesFile(**_load_yaml(config_dir / "companies.yaml"))
    forms = FormsFile(**_load_yaml(config_dir / "forms.yaml"))

    return Config(
        env=env,
        settings=settings,
        fred_series=fred_series,
        companies=companies,
        forms=forms,
        paths=_build_paths(resolved_home, root, settings.autopilot.obsidian_vault_subdir),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached configuration."""
    return load_config()


def reset_config_cache() -> None:
    """Clear the cached config (used by tests)."""
    get_config.cache_clear()
