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


class SettingsFile(BaseModel):
    app: AppInfo = AppInfo()
    http: HttpConfig = HttpConfig()
    retry: RetryConfig = RetryConfig()
    pacing: PacingConfig = PacingConfig()
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

    def ensure(self) -> Paths:
        """Create every data/log directory if missing. Safe to call repeatedly."""
        for directory in (
            self.data_dir,
            self.raw_dir,
            self.normalized_dir,
            self.parquet_dir,
            self.database_dir,
            self.logs_dir,
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


def _build_paths(home: Path, project_root: Path) -> Paths:
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
        paths=_build_paths(resolved_home, root),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached configuration."""
    return load_config()


def reset_config_cache() -> None:
    """Clear the cached config (used by tests)."""
    get_config.cache_clear()
