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
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
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

    ``min_confidence`` is separate from ``trading.min_confidence`` on purpose:
    gap alerts carry no track record yet and cap at confidence 50 (see
    ``signals.confidence`` / ``collectors.gaps``), so gating them against the
    daily path's floor (60 by default) would mean they never text. This floor
    lets them text anyway — raise it to 60+ to silence gap alerts entirely
    until the ledger matures enough to lift the cap.
    """

    min_gap_pct: float = Field(default=3.0, gt=0.0)
    min_avg_dollar_volume: float = Field(default=5_000_000.0, gt=0.0)
    gap_target_r_multiple: float = Field(default=2.0, gt=0.0)
    gap_max_hold_days: int = Field(default=5, ge=1)
    gap_catalyst_lookback_days: int = Field(default=7, ge=1)
    require_catalyst: bool = False
    min_confidence: int = Field(default=40, ge=0, le=100)


class PeopleGraphConfig(BaseModel):
    """Tuning for the people/insider graph (see docs/specs/2026-07-25-people-
    insider-graph-design.md, §5 "Data quality notes specific to people").

    ``interlock_max_companies_per_person`` caps how many ``role_memberships``
    a single director may contribute interlock edges from before being
    treated as a professional board-sitter / mutual-fund trustee and skipped
    entirely -- without this cap a handful of serial board members would
    flood the graph with a fully-connected, low-information hub.

    ``cluster_buy_window_days`` / ``cluster_buy_min_insiders`` define what
    counts as a cluster: open-market purchases (Form 4 code ``P``) by at
    least this many distinct insiders at the same company within this many
    days of each other. First-cut defaults, meant to be tuned against the
    discovery split like every other threshold in this platform, not chosen
    by inspection.
    """

    interlock_max_companies_per_person: int = Field(default=15, ge=2)
    cluster_buy_window_days: int = Field(default=5, ge=1)
    cluster_buy_min_insiders: int = Field(default=2, ge=2)


class RelationshipDecisionConfig(BaseModel):
    """Fail-closed policy for the SEC relationship decision workflow.

    This is deliberately a research policy, not trading configuration.  A
    configured live-quality first-public source, market source, and broker
    source are each required before a decision can become notify-eligible.
    Keeping that fact in the config model makes an omitted provider an explicit
    suppression rather than an implicit assumption.
    """

    enabled: bool = False
    strategy_version: str = "relationship-decision-v1"
    evidence_score_threshold: int = Field(default=8, ge=0)
    minimum_independent_clusters: int = Field(default=30, ge=1)
    net_target_pct: float = Field(default=2.0, gt=0.0)
    min_reward_to_risk: float = Field(default=2.0, gt=0.0)
    max_quote_age_seconds: float = Field(default=5.0, gt=0.0)
    max_spread_bps: float = Field(default=50.0, gt=0.0)
    max_participation_pct: float = Field(default=10.0, gt=0.0, le=100.0)
    regular_hours_only: bool = True
    # Provider ids remain null until the operator has selected and configured
    # an audited source. Null means research-only; it never means "use daily
    # yfinance as a live substitute".
    first_public_source: str | None = None
    realtime_market_source: str | None = None
    broker_read_only_source: str | None = None
    short_candidates_enabled: bool = False
    kill_switch_enabled: bool = True


class PortfolioConfig(BaseModel):
    """Account-level portfolio construction for the paper-trading simulator.

    Distinct from ``TradingConfig`` on purpose. ``TradingConfig`` sizes one
    rendered alert against Michael's real account; this configures a *simulated*
    account that the brain reasons about as a whole -- covariance, views,
    optimizer, risk budget. Nothing here reaches a brokerage: the simulator
    proposes, it never executes (repo design decision G).

    Geometry note: the validated backtest geometry (8x ATR disaster-brake stops,
    sizing decoupled from stop width, horizon exits in trading days) lives in
    the simulator, not here, and deliberately diverges from the live
    ``trading.atr_stop_multiple`` of 2.0 that the alert path still ships. That
    divergence is documented, not silently reconciled -- changing the live value
    is Michael's call, not a side effect of building the simulator.

    ``graph_return_views_enabled`` ships False and stays False until a
    propagation cell actually passes the measurement gate (ADMITTED via
    impact.judge, then the beta-hedged >=15bps tradeability screen). No
    propagation coefficient has ever been measured; an empty gate is a
    legitimate outcome that the report states plainly rather than papering over.

    ``hwm_decay_halflife_days`` is the half-life of the *gap* between the
    account's high-water mark and its current equity, and it exists because a
    governor with an exit rule and no re-entry rule is an absorbing state. The
    replay measured that: equity peaked at $15,942 on 2020-02-14, fell to
    $12,917 by 2020-03-12 (18.97%), and the governor flattened to cash -- after
    which equity moved on 0 of the next 1,603 sessions and the governor returned
    gross on 0 of 1,604, because a flat account cannot make the new high the
    ratcheting mark needed. 63 sessions is one quarter: long enough that a real
    crash is still respected for months, short enough that six years of forgone
    recovery cannot happen again. Setting it very large restores the old
    absorbing behavior; there is deliberately no way to switch it off.

    ``vol_estimate_lambda`` is the decay of the covariance the *volatility
    scaler* reads, and it is deliberately separate from ``ewma_lambda`` because
    the two estimates have different jobs. The optimizer wants a stable
    covariance -- a jumpy one churns the book, and turnover already exceeds its
    cap -- so it keeps the ``lw_blend`` mixture. The scaler wants a fast one.
    Sharing one estimate between them measured as a risk control that does not
    act: over the 2,518-session replay ``vol_scale`` bound on 37 days (1.5%) and
    returned exactly 1.000 through the entire COVID crash, including 2020-03-12
    when SPY's realized volatility was ~50%/yr against the 12% target. The
    blended estimate was not wrong, it was too slow to turn: in late February
    2020 its trailing year was eleven months of calm.

    0.94 is the RiskMetrics (1996) short-horizon standard -- a published
    constant with a ~11-session half-life, chosen *before* looking at what it
    does to any particular window. **It must not be tuned against the COVID
    window.** We already know COVID happened, so a value fitted to it is fitted
    to one event, and the resulting backtest would be measuring hindsight
    rather than a risk control. Change it only for a reason that would have
    been sayable in 1996.

    ``bl_engine`` selects the Black-Litterman implementation. Verified
    2026-07-31 on pandas 3.0.3 / numpy 2.5.1 / Python 3.14.6: PyPortfolioOpt's
    posterior agrees with the hand-rolled closed form to 1.4e-17, so pypfopt is
    the default and "numpy" is a tested substitute rather than a fallback we
    expect to need. pypfopt never sits in the optimizer solve path either way.
    """

    # -- account ---------------------------------------------------------- #
    starting_cash: float = Field(default=10_000.0, gt=0.0)
    fractional_shares: bool = True
    long_only: bool = True
    simulation_version: str = "portfolio-sim-v1"

    # -- risk limits ------------------------------------------------------ #
    max_position: float = Field(default=0.15, gt=0.0, le=1.0)
    max_cluster: float = Field(default=0.30, gt=0.0, le=1.0)
    max_gross: float = Field(default=1.0, gt=0.0, le=1.0)
    target_vol: float = Field(default=0.12, gt=0.0)
    max_single_name_risk_share: float = Field(default=0.35, gt=0.0, le=1.0)

    # -- drawdown governor (Michael's moderate risk budget) --------------- #
    # Halve gross at -10%, flatten at -15%. ``governor_bypasses_band`` is a
    # decision, not a tuning knob: when the governor de-risks, the no-trade
    # band must not suppress the resulting sells, or the account stays more
    # exposed than the governor intended precisely during a drawdown. The band
    # continues to suppress ordinary rebalance churn.
    governor_halve_drawdown: float = Field(default=0.10, gt=0.0, lt=1.0)
    governor_halve_scale: float = Field(default=0.5, ge=0.0, le=1.0)
    governor_flatten_drawdown: float = Field(default=0.15, gt=0.0, lt=1.0)
    governor_bypasses_band: bool = True
    hwm_decay_halflife_days: int = Field(default=63, ge=1)

    # -- covariance ------------------------------------------------------- #
    # ``ewma_lambda`` / ``lw_blend`` build the *optimizer's* covariance;
    # ``vol_estimate_lambda`` is the volatility scaler's own, faster one. See
    # the class docstring for why they are two numbers and not one.
    cov_lookback: int = Field(default=252, ge=20)
    ewma_lambda: float = Field(default=0.97, gt=0.0, lt=1.0)
    lw_blend: float = Field(default=0.5, ge=0.0, le=1.0)
    vol_estimate_lambda: float = Field(default=0.94, gt=0.0, lt=1.0)

    # -- views / Black-Litterman ------------------------------------------ #
    tau: float = Field(default=0.05, gt=0.0)
    bl_engine: Literal["pypfopt", "numpy"] = "pypfopt"
    graph_return_views_enabled: bool = False

    # -- optimizer -------------------------------------------------------- #
    objective: Literal["max_sharpe", "min_cvar"] = "max_sharpe"
    risk_aversion: float = Field(default=2.5, gt=0.0)
    turnover_penalty_bps: float = Field(default=25.0, ge=0.0)
    no_trade_band: float = Field(default=0.01, ge=0.0)
    rebalance_days: int = Field(default=5, ge=1)
    #: **Declared, reported against, and enforced by nothing.** No code reads
    #: this field; the replay reports ``turnover_annual`` and no constraint,
    #: penalty or veto anywhere consults the number. Read it as a preregistered
    #: criterion, not as a limit the simulator can hold.
    #:
    #: It is also unreachable as written, and the arithmetic is short. Turnover
    #: on ``cli._turnover_series``' definition is roughly
    #: ``2 * gross * 252 / holding_days``: every position is bought once and
    #: sold once, and it is held for its signal's horizon. The admitted cells
    #: are 20-session, and the ten-year replay ran at 0.74 mean gross, so
    #: ``2 * 0.74 * 252/20 = 18.6``x/yr of round trips before anything damps
    #: them -- the replay measured 14.0x. Reaching 4.0x needs either a
    #: ~70-session average hold (holding a position long past the window its
    #: edge was measured over) or ~20% gross. Cutting gross does not help on
    #: net: alpha and cost both scale with exposure, so a 3.5x cut takes
    #: 2.12%/yr of gross alpha and 0.62%/yr of slippage down together and
    #: leaves 0.43%/yr net where there was 1.50%/yr. Measured on the replay's
    #: own price history: gross 0.75 -> 0.48 -> 0.25 gave turnover
    #: 12.4x -> 8.0x -> 4.8x and Sharpe 0.74 -> 0.63 -> 0.40.
    #:
    #: Note also that this definition counts BOTH sides of a round trip, which
    #: is twice the SEC N-1A convention. That halves the reported figure but
    #: does not rescue the criterion, and the metric is not being redefined to
    #: pass it.
    max_turnover_annual: float = Field(default=4.0, gt=0.0)
    cvar_alpha: float = Field(default=0.95, gt=0.0, lt=1.0)

    # -- costs (5bps default = parity with analytics/backtest.py) --------- #
    slippage_bps: float = Field(default=5.0, ge=0.0)
    commission_per_share: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check_governor_ladder(self) -> PortfolioConfig:
        """Flatten must sit strictly below halve, or the ladder is unreachable."""
        if self.governor_flatten_drawdown <= self.governor_halve_drawdown:
            raise ValueError(
                "governor_flatten_drawdown must exceed governor_halve_drawdown "
                f"(got flatten={self.governor_flatten_drawdown}, "
                f"halve={self.governor_halve_drawdown})"
            )
        return self

    @model_validator(mode="after")
    def _check_position_cluster_ladder(self) -> PortfolioConfig:
        """A cluster cap below the position cap would never bind."""
        if self.max_cluster < self.max_position:
            raise ValueError(
                "max_cluster must be >= max_position "
                f"(got cluster={self.max_cluster}, position={self.max_position})"
            )
        return self


class SettingsFile(BaseModel):
    app: AppInfo = AppInfo()
    http: HttpConfig = HttpConfig()
    retry: RetryConfig = RetryConfig()
    pacing: PacingConfig = PacingConfig()
    autopilot: AutopilotConfig = AutopilotConfig()
    llm: LLMConfig = LLMConfig()
    trading: TradingConfig = TradingConfig()
    gap_scanner: GapScannerConfig = GapScannerConfig()
    people_graph: PeopleGraphConfig = PeopleGraphConfig()
    relationship_decision: RelationshipDecisionConfig = RelationshipDecisionConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
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

    def relationship_decision_readiness(self) -> tuple[bool, tuple[str, ...]]:
        """Return whether a candidate workflow may leave research-only mode.

        This intentionally does not declare an edge proven or send anything.
        It only makes missing operator decisions observable as stable reasons,
        rather than allowing daily prices or EDGAR acceptance time to stand in
        for live-quality market/first-public/borrow evidence.
        """
        policy = self.settings.relationship_decision
        reasons: list[str] = []
        if not policy.enabled:
            reasons.append("relationship_decision_disabled")
        if not policy.first_public_source:
            reasons.append("first_public_source_unconfigured")
        if not policy.realtime_market_source:
            reasons.append("realtime_market_source_unconfigured")
        if not policy.broker_read_only_source:
            reasons.append("broker_read_only_source_unconfigured")
        # Contracts alone are not a selected, audited live adapter or a
        # completed shadow-period review. Keep candidate notification closed.
        reasons.append("live_candidate_mode_not_implemented")
        return not reasons, tuple(reasons)

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
            "relationship_decision": self.settings.relationship_decision.model_dump(),
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
