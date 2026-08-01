"""Command-line interface.

Commands::

    market-intelligence init-db
    market-intelligence sec sync-companies
    market-intelligence sec collect-filings --ticker NVDA [--forms 10-K,10-Q,8-K] [--limit N]
    market-intelligence sec collect-ipos [--forms S-1,S-1/A,F-1,F-1/A]
    market-intelligence sec ingest-documents [--ticker NVDA] [--limit N]
    market-intelligence sec sync-insider [--start-quarter 2016q1] [--end-quarter 2024q4]
    market-intelligence sec sync-filing-events [--forms 8-K] [--ticker NVDA]
    market-intelligence fred sync [--series DGS10,UNRATE]
    market-intelligence market sync-constituents
    market-intelligence market sync-prices [--symbols NVDA,MU] [--start YYYY-MM-DD]
    market-intelligence market compute-returns [--benchmark SPY] [--method market_model]
    market-intelligence market scan-gaps [--dry-run]
    market-intelligence signals build-dataset [--subtypes P,S] [--horizons 1,5,20]
    market-intelligence signals evaluate [--no-placebo]
    market-intelligence signals backtest [--holdout] [--all-cells] [--stop 8]
    market-intelligence signals extract-relationships [--ticker NKE] [--limit N]
    market-intelligence signals project-graph
    market-intelligence people sync-roles
    market-intelligence people project-interlocks
    market-intelligence people detect-cluster-buys
    market-intelligence portfolio replay [--split discovery|holdout] [--write]
    market-intelligence portfolio stress [--scenarios bootstrap,vol_x2,corr_one,covid]
    market-intelligence portfolio step --as-of YYYY-MM-DD
    market-intelligence portfolio status
    market-intelligence portfolio sweep [--register]
    market-intelligence portfolio study-graph
    market-intelligence portfolio holdout-read
    market-intelligence autopilot run
    market-intelligence ram-guard [--dry-run]
    market-intelligence validate
    market-intelligence reconcile [--no-verify-hashes]
    market-intelligence status

Exit codes: 0 success, 1 pipeline/runtime failure, 2 configuration error.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from market_intelligence import __version__, database, reconciliation
from market_intelligence.analytics import backtest as backtest_engine
from market_intelligence.analytics import backtest_data
from market_intelligence.analytics import cluster_buy as cluster_buy_analytics
from market_intelligence.analytics import interlocks as interlocks_analytics
from market_intelligence.analytics.experiment_registry import ExperimentRecord, ExperimentStage
from market_intelligence.analytics.impact import AdmissionThresholds
from market_intelligence.analytics.returns import AbnormalReturnMethod
from market_intelligence.autopilot import graph, notify, ram_guard
from market_intelligence.autopilot import orchestrator as autopilot
from market_intelligence.autopilot.types import RunStatus
from market_intelligence.collectors import RunSummary
from market_intelligence.collectors import constituents as constituents_collector
from market_intelligence.collectors import documents as document_collector
from market_intelligence.collectors import filing_events as filing_events_collector
from market_intelligence.collectors import fred as fred_collector
from market_intelligence.collectors import gaps as gaps_collector
from market_intelligence.collectors import insider as insider_collector
from market_intelligence.collectors import prices as price_collector
from market_intelligence.collectors import relationships as relationships_collector
from market_intelligence.collectors import returns as returns_collector
from market_intelligence.collectors import roles as roles_collector
from market_intelligence.collectors import sec as sec_collector
from market_intelligence.config import Config, ConfigError, PortfolioConfig, get_config
from market_intelligence.hashing import sha256_json
from market_intelligence.logging_config import configure_logging, get_logger
from market_intelligence.portfolio import account as portfolio_account
from market_intelligence.portfolio import feed as portfolio_feed
from market_intelligence.portfolio import metrics as portfolio_metrics
from market_intelligence.portfolio import report as portfolio_report
from market_intelligence.portfolio import risk as portfolio_risk
from market_intelligence.portfolio import simulator as portfolio_simulator
from market_intelligence.portfolio import store as portfolio_store
from market_intelligence.portfolio.costs import CostModel
from market_intelligence.signals import dataset as dataset_builder
from market_intelligence.signals import impact as impact_gate

app = typer.Typer(
    help="Local market-intelligence data platform (baseline).",
    no_args_is_help=True,
    add_completion=False,
)
sec_app = typer.Typer(help="SEC EDGAR collection.", no_args_is_help=True)
fred_app = typer.Typer(help="FRED economic-data collection.", no_args_is_help=True)
market_app = typer.Typer(
    help="Index membership, prices, and derived returns.", no_args_is_help=True
)
signals_app = typer.Typer(help="Event-study datasets and measured effects.", no_args_is_help=True)
people_app = typer.Typer(
    help="People, boards, and the insider graph (roles, interlocks, cluster buys).",
    no_args_is_help=True,
)
autopilot_app = typer.Typer(help="The self-checking daily loop.", no_args_is_help=True)
portfolio_app = typer.Typer(
    help="Paper-trading portfolio simulator. Proposes; never executes.",
    no_args_is_help=True,
)
app.add_typer(sec_app, name="sec")
app.add_typer(fred_app, name="fred")
app.add_typer(market_app, name="market")
app.add_typer(signals_app, name="signals")
app.add_typer(people_app, name="people")
app.add_typer(autopilot_app, name="autopilot")
app.add_typer(portfolio_app, name="portfolio")

console = Console()
err_console = Console(stderr=True)
log = get_logger("cli")

# Tables that carry per-record validation status.
_VALIDATED_TABLES = (
    "companies",
    "filings",
    "filing_documents",
    "filing_sections",
    "economic_series",
    "economic_observations",
    "index_constituents",
    "daily_prices",
    "daily_returns",
    "events",
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _load() -> Config:
    """Load config, or print a clear error and exit(2)."""
    try:
        config = get_config()
    except ConfigError as exc:
        err_console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    config.paths.ensure()
    configure_logging(config.env.log_level, log_file=config.paths.log_file)
    return config


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    return items or None


def _split_csv_tuple(value: str | None) -> tuple[str, ...] | None:
    items = _split_csv(value)
    return tuple(items) if items else None


def _emit_summary(summary: RunSummary) -> None:
    table = Table(title=f"pipeline: {summary.pipeline_name}", show_edge=True)
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("collected", str(summary.collected))
    # Network/raw-store counters are only meaningful for pipelines that fetch
    # payloads; suppress the zero rows elsewhere to keep the table readable.
    if summary.downloaded or summary.stored or summary.skipped:
        table.add_row("downloaded", str(summary.downloaded))
        table.add_row("stored", str(summary.stored))
        table.add_row("skipped", str(summary.skipped))
    table.add_row("inserted", str(summary.inserted))
    table.add_row("updated", str(summary.updated))
    if summary.deduped:
        table.add_row("deduped", str(summary.deduped))
    table.add_row("rejected", str(summary.rejected))
    for key in sorted(summary.stage):
        table.add_row(f"[dim]{key}[/dim]", str(summary.stage[key]))
    status_color = "green" if summary.status == "success" else "red"
    table.add_row("status", f"[{status_color}]{summary.status}[/{status_color}]")
    console.print(table)
    for note in summary.notes[:25]:
        console.print(f"  • {note}")
    if len(summary.notes) > 25:
        console.print(f"  … and {len(summary.notes) - 25} more notes")


def _run(action: Callable[[], RunSummary]) -> None:
    """Execute a collector, render its summary, and set the exit code."""
    try:
        summary = action()
    except ConfigError as exc:
        err_console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except Exception as exc:  # surfaced to the user, not swallowed
        log.error("pipeline_failed", error=str(exc))
        err_console.print(f"[red]Pipeline failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _emit_summary(summary)
    if summary.status != "success":
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# Top-level commands                                                           #
# --------------------------------------------------------------------------- #
@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"market-intelligence {__version__}")


@app.command("init-db")
def init_db() -> None:
    """Create the DuckDB database and all tables (idempotent)."""
    config = _load()
    with database.connection(config.paths.database_path) as con:
        tables = database.init_db(con)
        counts = database.table_counts(con)
    console.print(f"[green]Initialized[/green] {config.paths.database_path}")
    for table in tables:
        console.print(f"  • {table}: {counts[table]} rows")


@app.command()
def status() -> None:
    """Show configuration readiness, table row counts, and recent runs."""
    config = _load()

    sec_ua = (config.env.sec_user_agent or "").strip()
    sec_ready = bool(sec_ua) and "example.com" not in sec_ua
    fred_ready = bool((config.env.fred_api_key or "").strip())

    cfg_table = Table(title="configuration", show_edge=True)
    cfg_table.add_column("item", style="bold")
    cfg_table.add_column("value")
    cfg_table.add_row("home", str(config.paths.home))
    cfg_table.add_row("database", str(config.paths.database_path))
    cfg_table.add_row("SEC_USER_AGENT", "[green]set[/green]" if sec_ready else "[red]missing[/red]")
    cfg_table.add_row("FRED_API_KEY", "[green]set[/green]" if fred_ready else "[red]missing[/red]")
    console.print(cfg_table)

    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        counts = database.table_counts(con)
        runs = con.execute(
            """
            SELECT pipeline_name, started_time, status,
                   records_inserted, records_updated, records_rejected
            FROM pipeline_runs
            ORDER BY started_time DESC
            LIMIT 10
            """
        ).fetchall()

    count_table = Table(title="tables", show_edge=True)
    count_table.add_column("table", style="bold")
    count_table.add_column("rows", justify="right")
    for table, count in counts.items():
        count_table.add_row(table, str(count))
    console.print(count_table)

    run_table = Table(title="recent pipeline runs", show_edge=True)
    for column in ("pipeline", "started", "status", "ins", "upd", "rej"):
        run_table.add_column(column)
    if not runs:
        run_table.add_row("—", "—", "—", "—", "—", "—")
    for name, started, run_status, ins, upd, rej in runs:
        run_table.add_row(str(name), str(started), str(run_status), str(ins), str(upd), str(rej))
    console.print(run_table)


@app.command()
def validate() -> None:
    """Audit stored records: validation-status counts and known anomalies."""
    config = _load()
    supported = set(config.forms.supported)

    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        status_table = Table(title="validation status by table", show_edge=True)
        for column in ("table", "valid", "warning", "rejected"):
            status_table.add_column(column)
        for table in _VALIDATED_TABLES:
            rows = con.execute(
                f'SELECT validation_status, count(*) FROM "{table}" GROUP BY 1'
            ).fetchall()
            by_status = {str(status_value): int(count) for status_value, count in rows}
            status_table.add_row(
                table,
                str(by_status.get("valid", 0)),
                str(by_status.get("warning", 0)),
                str(by_status.get("rejected", 0)),
            )
        console.print(status_table)

        anomalies = Table(title="anomalies", show_edge=True)
        anomalies.add_column("check", style="bold")
        anomalies.add_column("count", justify="right")

        if supported:
            placeholders = ", ".join(["?"] * len(supported))
            unsupported = con.execute(
                f"SELECT count(*) FROM filings WHERE form NOT IN ({placeholders})",
                list(supported),
            ).fetchone()
            anomalies.add_row(
                "filings with unsupported form", str(unsupported[0] if unsupported else 0)
            )

        missing_values = con.execute(
            "SELECT count(*) FROM economic_observations WHERE value IS NULL"
        ).fetchone()
        anomalies.add_row(
            "observations with missing value", str(missing_values[0] if missing_values else 0)
        )

        missing_urls = con.execute(
            "SELECT count(*) FROM filings WHERE source_url IS NULL AND filing_url IS NULL"
        ).fetchone()
        anomalies.add_row("filings missing source url", str(missing_urls[0] if missing_urls else 0))

    console.print(anomalies)


# --------------------------------------------------------------------------- #
# SEC subcommands                                                              #
# --------------------------------------------------------------------------- #
@sec_app.command("sync-companies")
def sec_sync_companies() -> None:
    """Download the SEC ticker->CIK map and store the company universe."""
    config = _load()
    _run(lambda: sec_collector.sync_companies(config))


def _priced_universe(config: Config) -> list[str]:
    """Every ticker that was ever an index member *and* has price history.

    Price history is the join condition, not an afterthought: a company with
    filings but no prices contributes events with no label to measure them
    against, so it adds collection cost and no signal. Membership is taken over
    all time, not current members only -- excluding the companies that were
    acquired or delisted is precisely the survivorship bias the point-in-time
    universe exists to prevent.
    """
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        rows = con.execute(
            """
            SELECT DISTINCT ic.ticker
            FROM index_constituents ic
            JOIN daily_prices p ON p.symbol = ic.ticker
            ORDER BY ic.ticker
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


@sec_app.command("collect-filings")
def sec_collect_filings(
    ticker: str | None = typer.Option(None, "--ticker", help="Single ticker, e.g. NVDA."),
    universe: bool = typer.Option(
        False, "--universe", help="Every ever-index-member ticker that has price history."
    ),
    forms: str | None = typer.Option(None, "--forms", help="Comma-separated form types."),
    limit: int | None = typer.Option(None, "--limit", help="Max filings per ticker."),
) -> None:
    """Collect filings for a ticker, the priced universe, or the configured companies."""
    config = _load()
    if ticker and universe:
        err_console.print("[red]--ticker and --universe are mutually exclusive.[/red]")
        raise typer.Exit(code=2)

    tickers: list[str] | None
    if universe:
        tickers = _priced_universe(config)
        if not tickers:
            err_console.print(
                "[red]No priced universe found.[/red] Run 'market sync-constituents' "
                "and 'market sync-prices' first."
            )
            raise typer.Exit(code=2)
        console.print(f"Collecting filings for [bold]{len(tickers)}[/bold] tickers.")
    else:
        tickers = [ticker] if ticker else None

    _run(
        lambda: sec_collector.collect_filings(
            config, tickers=tickers, forms=_split_csv(forms), limit=limit
        )
    )


@sec_app.command("collect-ipos")
def sec_collect_ipos(
    forms: str | None = typer.Option(None, "--forms", help="Comma-separated IPO form types."),
    ticker: str | None = typer.Option(None, "--ticker", help="Restrict to a single ticker."),
    limit: int | None = typer.Option(None, "--limit", help="Max filings per ticker."),
) -> None:
    """Scan configured companies for IPO/registration filings (S-1/F-1 family).

    Note: the SEC provides no global "all recent IPOs" endpoint in the required
    form; this scans configured companies' submissions. See the README for the
    documented extension point.
    """
    config = _load()
    tickers = [ticker] if ticker else None
    _run(
        lambda: sec_collector.collect_ipos(
            config, forms=_split_csv(forms), tickers=tickers, limit=limit
        )
    )


@sec_app.command("ingest-documents")
def sec_ingest_documents(
    ticker: str | None = typer.Option(None, "--ticker", help="Restrict to a single ticker."),
    forms: str | None = typer.Option(
        None, "--forms", help="Comma-separated forms; default = 10-K/10-Q (and amendments)."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Max filings to ingest."),
    reingest: bool = typer.Option(
        False, "--reingest", help="Re-fetch filings already ingested (default: skip them)."
    ),
) -> None:
    """Download filing documents, preserve the bytes, and extract Item sections.

    Operates on filings already recorded by ``collect-filings``. Resumable:
    filings already in ``filing_documents`` are skipped, results are written
    every few filings, and an interrupted run keeps everything it finished. Run
    it repeatedly until the candidate count reaches zero.
    """
    config = _load()
    tickers = [ticker] if ticker else None
    _run(
        lambda: document_collector.ingest_documents(
            config,
            tickers=tickers,
            forms=_split_csv(forms),
            limit=limit,
            skip_ingested=not reingest,
        )
    )


@sec_app.command("sync-insider")
def sec_sync_insider(
    start_quarter: str | None = typer.Option(
        None, "--start-quarter", help="Earliest quarter id, e.g. 2016q1 (default: 2006q1)."
    ),
    end_quarter: str | None = typer.Option(
        None, "--end-quarter", help="Latest quarter id, e.g. 2024q4 (default: the current quarter)."
    ),
) -> None:
    """Collect SEC's pre-parsed quarterly Form 3/4/5 insider-transaction datasets.

    Emits one event per non-derivative transaction for issuers in the
    collected universe (companies/filings CIKs, falling back to priced
    tickers). Resumable: a quarter already stored is skipped, and a run
    interrupted partway through leaves every earlier quarter's events durably
    persisted. A 404 for the most recent quarter means SEC has not published
    it yet, not a failure.
    """
    config = _load()
    _run(
        lambda: insider_collector.sync(config, start_quarter=start_quarter, end_quarter=end_quarter)
    )


@sec_app.command("sync-filing-events")
def sec_sync_filing_events(
    ticker: str | None = typer.Option(None, "--ticker", help="Restrict to a single ticker."),
    forms: str | None = typer.Option(
        None, "--forms", help="Comma-separated forms; default = 8-K (amendments excluded)."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Max filings to process."),
) -> None:
    """Emit one event per (8-K, item code) from preserved submissions data.

    Reads the raw store rather than the network, so it is fast and offline.
    Each item code becomes its own event with a neutral direction; the sign of
    each code's effect is left for ``signals evaluate`` to measure.
    """
    config = _load()
    tickers = [ticker] if ticker else None
    _run(
        lambda: filing_events_collector.sync(
            config, forms=_split_csv_tuple(forms), tickers=tickers, limit=limit
        )
    )


@autopilot_app.command("run")
def autopilot_run() -> None:
    """Run the daily loop: refresh data, mature samples, re-gate, and brief.

    Sequences the pipeline, advances the promotion ladder, writes the Obsidian
    briefing note, and pushes the Telegram nudge. Designed to run unattended
    from launchd; a failed step degrades the run rather than aborting it, and a
    briefing is always produced. Exit code is non-zero only when the run failed
    outright (the gate itself could not run).
    """
    config = _load()
    briefing = autopilot.run(config)

    active = sum(1 for s in briefing.active_signals)
    console.print(
        f"[bold]Autopilot {briefing.run_status}[/bold] for {briefing.as_of}: "
        f"{len(briefing.changes)} change(s), {len(briefing.alerts)} alert(s), "
        f"{active} active signal(s)."
    )
    for note in briefing.notes:
        err_console.print(f"[yellow]{note}[/yellow]")
    if briefing.run_status == RunStatus.FAILED:
        raise typer.Exit(code=1)


@app.command("ram-guard")
def ram_guard_cmd(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would be stopped without killing anything."
    ),
) -> None:
    """Check memory pressure and stop duplicate processes if the mini is under it.

    Only acts once macOS itself reports low free memory (default: below 20%);
    on a healthy machine this only reports what it sees. The only things it
    ever stops are exact duplicates of a known singleton command (a second
    `ollama serve`, a collector or backfill script launched twice) -- never the
    process holding the DuckDB lock, never this session or its ancestors. Safe
    to run by hand before a long LLM run, or on a schedule (see
    config/launchd/ai.quant.ramguard.plist).
    """
    config = _load()
    result = ram_guard.run_guard(database_path=config.paths.database_path, dry_run=dry_run)

    color = "yellow" if (result.killed or result.kill_candidates) else "green"
    console.print(f"[{color}]{result.summary_line}[/{color}]")

    if result.zombies:
        console.print(
            f"[dim]{len(result.zombies)} zombie process(es) seen (parent must reap):[/dim]"
        )
        for z in result.zombies:
            console.print(f"  • pid {z.pid} (ppid {z.ppid}): {z.command[:80]}")

    if result.high_memory:
        table = Table(title="other high-memory processes (report only)", show_edge=True)
        table.add_column("pid", justify="right")
        table.add_column("rss (MB)", justify="right")
        table.add_column("command")
        for proc in result.high_memory:
            table.add_row(str(proc.pid), f"{proc.rss_mb:.0f}", proc.command[:100])
        console.print(table)

    for note in result.notes:
        err_console.print(f"[yellow]{note}[/yellow]")


@app.command()
def reconcile(
    verify_hashes: bool = typer.Option(
        True, "--verify-hashes/--no-verify-hashes", help="Re-hash section text files."
    ),
) -> None:
    """Reconcile pipeline counts and explain any differences."""
    config = _load()

    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        run = reconciliation.latest_run(con, reconciliation.INGEST_PIPELINE)
        storage, anomalies = reconciliation.storage_checks(con, verify_hashes=verify_hashes)
        breakdown = reconciliation.integrity_breakdown(con)
        coverage = reconciliation.section_coverage(con)

    failures = 0

    if run is None:
        console.print(
            f"[yellow]No {reconciliation.INGEST_PIPELINE} run recorded yet.[/yellow] "
            "Run `market-intelligence sec ingest-documents` first."
        )
    else:
        counts = Table(title=f"run counters — {run['run_id'][:12]} ({run['status']})")
        counts.add_column("counter", style="bold")
        counts.add_column("value", justify="right")
        for key in ("collected", "downloaded", "stored", "skipped",
                    "inserted", "updated", "deduped", "rejected"):  # fmt: skip
            counts.add_row(key, str(run[key]))
        for key in sorted(run["stage"]):
            counts.add_row(f"[dim]{key}[/dim]", str(run["stage"][key]))
        console.print(counts)

        failures += _emit_checks("run identities", reconciliation.run_checks(run))

    failures += _emit_checks("storage agreement", storage)

    integrity = Table(title="document integrity status", show_edge=True)
    integrity.add_column("status", style="bold")
    integrity.add_column("documents", justify="right")
    for status_value, count in breakdown or [("—", 0)]:
        integrity.add_row(status_value, str(count))
    console.print(integrity)

    if coverage:
        cov = Table(title="section coverage by item", show_edge=True)
        cov.add_column("item", style="bold")
        cov.add_column("sections", justify="right")
        cov.add_column("filings", justify="right")
        for code, sections, filings in coverage:
            cov.add_row(code, str(sections), str(filings))
        console.print(cov)

    totals = ", ".join(f"{key}={value}" for key, value in anomalies.items())
    console.print(f"[dim]{totals}[/dim]")

    if failures:
        err_console.print(f"[red]{failures} reconciliation check(s) failed.[/red]")
        raise typer.Exit(code=1)
    console.print("[green]All reconciliation checks passed.[/green]")


def _emit_checks(title: str, checks: list[reconciliation.Check]) -> int:
    """Render a check table; return how many failed."""
    table = Table(title=title, show_edge=True)
    table.add_column("check", style="bold")
    table.add_column("identity")
    table.add_column("left", justify="right")
    table.add_column("right", justify="right")
    table.add_column("result")
    failures = 0
    for check in checks:
        if check.ok:
            result = "[green]ok[/green]"
        else:
            failures += 1
            result = f"[red]off by {check.residual:+d}[/red]"
        table.add_row(check.name, check.identity, str(check.left), str(check.right), result)
    console.print(table)
    for check in checks:
        if not check.ok:
            console.print(f"  [yellow]{check.name}[/yellow]: {check.explanation}")
    return failures


# --------------------------------------------------------------------------- #
# FRED subcommands                                                             #
# --------------------------------------------------------------------------- #
@fred_app.command("sync")
def fred_sync(
    series: str | None = typer.Option(
        None, "--series", help="Comma-separated series ids; default = config/fred_series.yaml."
    ),
) -> None:
    """Download configured FRED series metadata and observations."""
    config = _load()
    _run(lambda: fred_collector.sync(config, series_ids=_split_csv(series)))


# --------------------------------------------------------------------------- #
# Market subcommands                                                           #
# --------------------------------------------------------------------------- #
@market_app.command("sync-constituents")
def market_sync_constituents() -> None:
    """Collect point-in-time S&P 500 index membership.

    Windows whose added_date is an approximation rather than an attested date
    are stored with validation_status='warning' and the reason in
    validation_errors, so a point-in-time query can exclude them.
    """
    config = _load()
    _run(lambda: constituents_collector.sync(config))


@market_app.command("sync-prices")
def market_sync_prices(
    symbols: str | None = typer.Option(
        None, "--symbols", help="Comma-separated tickers; default = the collected index universe."
    ),
    start: datetime | None = typer.Option(
        None, "--start", formats=["%Y-%m-%d"], help="Earliest bar to request (YYYY-MM-DD)."
    ),
    end: datetime | None = typer.Option(
        None, "--end", formats=["%Y-%m-%d"], help="Latest bar to request (YYYY-MM-DD)."
    ),
) -> None:
    """Download daily OHLCV bars. Resumable: re-running fetches only what is new."""
    config = _load()
    _run(
        lambda: price_collector.sync(
            config,
            symbols=_split_csv(symbols),
            start=start.date() if start else None,
            end=end.date() if end else None,
        )
    )


@market_app.command("compute-returns")
def market_compute_returns(
    symbols: str | None = typer.Option(
        None, "--symbols", help="Comma-separated tickers; default = everything with prices."
    ),
    benchmark: str = typer.Option(
        returns_collector.DEFAULT_BENCHMARK, "--benchmark", help="Benchmark symbol."
    ),
    method: str = typer.Option(
        AbnormalReturnMethod.MARKET_MODEL.value,
        "--method",
        help="market_model | market_adjusted | sector_adjusted",
    ),
) -> None:
    """Derive abnormal returns from already-collected prices. Reads, never refetches."""
    config = _load()
    try:
        chosen = AbnormalReturnMethod(method)
    except ValueError as exc:
        valid = ", ".join(m.value for m in AbnormalReturnMethod)
        err_console.print(f"[red]Unknown method[/red] {method!r}. Expected one of: {valid}")
        raise typer.Exit(code=2) from exc
    _run(
        lambda: returns_collector.compute(
            config,
            symbols=_split_csv(symbols),
            benchmark=benchmark,
            method=chosen,
        )
    )


@market_app.command("scan-gaps")
def market_scan_gaps(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Build and persist candidates, but skip the Telegram send.",
    ),
) -> None:
    """Morning gap scan: overnight price gaps -> the trade-alert ledger.

    Reads survivors, fetches live quotes, plans candidates, persists new
    rows, and (unless --dry-run) sends the sendable ones to Telegram. See
    ``collectors/gaps.py`` for the three-phase DB-lock-safe design and the
    LaunchAgent at ``config/launchd/ai.quant.morning-scan.plist``.

    Exit codes deliberately diverge from the usual success/failure mapping:
    a busy DuckDB lock makes ``scan`` return a "skipped" summary rather than
    raising -- that is the designed outcome (another collector or the
    relationship extractor held the write lock), not a crash, so it exits 0
    like a normal success. A "partial" summary (a mostly-throttled quote
    provider) still completed the run and exits 0 too; only an actual
    exception, or a status this scan never produces in practice, exits 1.
    """
    config = _load()
    try:
        summary = gaps_collector.scan(config, notify=not dry_run)
    except ConfigError as exc:
        err_console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except Exception as exc:  # surfaced to the user, not swallowed
        log.error("pipeline_failed", error=str(exc))
        err_console.print(f"[red]Pipeline failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _emit_summary(summary)
    if summary.status not in ("success", "skipped", "partial"):
        raise typer.Exit(code=1)


@signals_app.command("build-dataset")
def signals_build_dataset(
    event_types: str | None = typer.Option(
        None, "--event-types", help="Comma-separated event types; default = all."
    ),
    subtypes: str | None = typer.Option(
        None, "--subtypes", help="Comma-separated event subtypes, e.g. P,S for insider trades."
    ),
    horizons: str = typer.Option(
        "1,5,20", "--horizons", help="Comma-separated trading-day horizons."
    ),
    split_date: datetime = typer.Option(
        dataset_builder.DEFAULT_SPLIT_DATE.isoformat(),
        "--split-date",
        formats=["%Y-%m-%d"],
        help="Chronological discovery/holdout boundary.",
    ),
) -> None:
    """Join events to forward abnormal returns. Reads stored data; fetches nothing."""
    config = _load()
    try:
        parsed_horizons = tuple(int(h) for h in (_split_csv(horizons) or []))
    except ValueError as exc:
        err_console.print(f"[red]--horizons must be integers, got[/red] {horizons!r}")
        raise typer.Exit(code=2) from exc
    if not parsed_horizons or any(h < 1 for h in parsed_horizons):
        err_console.print("[red]--horizons must be one or more positive integers.[/red]")
        raise typer.Exit(code=2)

    _run(
        lambda: dataset_builder.build(
            config,
            event_types=_split_csv(event_types),
            subtypes=_split_csv(subtypes),
            horizons=parsed_horizons,
            split_date=split_date.date(),
        )
    )


@signals_app.command("evaluate")
def signals_evaluate(
    placebo: bool = typer.Option(
        True, "--placebo/--no-placebo", help="Also measure randomly-paired control cells."
    ),
    min_clusters: int = typer.Option(
        200, "--min-clusters", help="Independent observations required."
    ),
    min_abs_t: float = typer.Option(3.0, "--min-t", help="Minimum |t| on the discovery split."),
    min_car: float = typer.Option(
        0.002, "--min-car", help="Minimum |mean CAR| to count as economically real."
    ),
) -> None:
    """Measure every cell and decide which may fire alerts."""
    config = _load()
    thresholds = AdmissionThresholds(
        min_clusters=min_clusters, min_abs_t=min_abs_t, min_abs_mean_car=min_car
    )
    _run(lambda: impact_gate.evaluate(config, thresholds=thresholds, with_placebo=placebo))


@signals_app.command("backtest")
def signals_backtest(
    equity: float = typer.Option(
        1_000_000.0, "--equity", help="Starting equity for the replay."
    ),
    hedge: str = typer.Option("SPY", "--hedge", help="Beta-hedge instrument; '' disables it."),
    stop_multiple: float = typer.Option(
        8.0, "--stop", help="ATR multiple for the disaster stop."
    ),
    risk_pct: float = typer.Option(0.5, "--risk", help="Risk budget per trade, percent."),
    max_positions: int = typer.Option(1000, "--positions", help="Concurrent position cap."),
    gross_pct: float = typer.Option(
        200.0, "--gross", help="Gross notional cap, percent of equity."
    ),
    all_cells: bool = typer.Option(
        False,
        "--all-cells",
        help="Replay every admitted cell, including those whose edge does not survive hedging.",
    ),
    holdout: bool = typer.Option(
        False,
        "--holdout",
        help="Also replay the holdout. Every parameter must be fixed on discovery first.",
    ),
) -> None:
    """Replay the admitted cells as a portfolio. Research only; places no orders.

    Defaults encode what the 2026-07-29 study found (docs/specs/
    2026-07-29-backtest-findings.md): hold each cell's calibrated horizon in
    trading days, hedge beta against an index, keep the stop wide enough not to
    be noise-triggered, and drop cells whose edge does not survive hedging.

    ``--holdout`` is opt-in on purpose. Reading the holdout while still choosing
    parameters turns its track record into a description of the choices, which is
    the one number in this system that has to stay uncontaminated.
    """
    config = _load()
    splits = ("discovery", "holdout") if holdout else ("discovery",)
    replay_config = backtest_engine.BacktestConfig(
        starting_equity=equity,
        risk_pct_per_trade=risk_pct,
        atr_stop_multiple=stop_multiple,
        max_positions=max_positions,
        max_gross_exposure_pct=gross_pct,
        hedge_symbol=hedge or None,
        max_position_pct=5.0,
    )

    with database.connection(config.paths.database_path) as con:
        # A replay writes nothing, but it still needs the schema to be present
        # to read a fresh or partially-built database without a catalog error.
        database.init_db(con)
        inputs = backtest_data.load_replay_inputs(
            con,
            hedge_symbol=hedge or None,
            split_date=dataset_builder.DEFAULT_SPLIT_DATE,
            tradeable_only=not all_cells,
            splits=splits,
        )
        if not inputs.cells:
            console.print(
                "[yellow]No tradeable cells.[/yellow] Run 'signals evaluate' first, or pass "
                "--all-cells to replay admitted cells whose edge does not survive hedging."
            )
            return

        cell_table = Table(title="Cells (beta-hedged edge measured on discovery)")
        for column in ("cell", "dir", "label CAR", "hedged edge", "verdict"):
            cell_table.add_column(column)
        for cell, kept in [(c, True) for c in inputs.cells] + [
            (c, False) for c in inputs.dropped
        ]:
            edge = inputs.hedged_edges.get(cell.key)
            cell_table.add_row(
                f"{cell.event_type}/{cell.event_subtype or '-'}/{cell.horizon_days}d",
                f"{cell.direction:+d}",
                f"{cell.mean_car:+.3%}",
                "n/a" if edge is None else f"{edge:+.3%}",
                "[green]trade[/green]" if kept else "[red]drop[/red]",
            )
        console.print(cell_table)

        results = Table(title="Portfolio replay")
        for column in ("split", "trades", "return", "max DD", "Sharpe", "win", "profit factor"):
            results.add_column(column)
        for split in splits:
            result = backtest_engine.run_backtest(
                inputs.signals[split], inputs.bars, replay_config, betas=inputs.betas
            )
            results.add_row(
                split,
                str(len(result.trades)),
                f"{result.total_return:+.2%}",
                f"{result.max_drawdown:.2%}",
                f"{result.sharpe:.2f}",
                f"{result.win_rate:.1%}",
                f"{result.profit_factor:.3f}",
            )
        console.print(results)


@signals_app.command("build-propagation")
def signals_build_propagation(
    event_types: str | None = typer.Option(
        None, "--event-types", help="Comma-separated event types; default = all."
    ),
    subtypes: str | None = typer.Option(
        None, "--subtypes", help="Comma-separated event subtypes, e.g. P,S."
    ),
    horizons: str = typer.Option("1,5,20", "--horizons", help="Trading-day horizons."),
) -> None:
    """Build propagation samples from the relationship graph.

    For every validatable edge, scores the source's events against the target's
    forward returns into ``event_samples`` with the relationship type as the
    edge. Then ``signals evaluate`` judges the propagation cells alongside the
    self-control, through the same gate and promotion ladder.
    """
    config = _load()
    _run(
        lambda: dataset_builder.build_propagation(
            config,
            event_types=_split_csv(event_types),
            subtypes=_split_csv(subtypes),
            horizons=tuple(int(h) for h in horizons.split(",") if h.strip()),
        )
    )


@signals_app.command("extract-relationships")
def signals_extract_relationships(
    ticker: str | None = typer.Option(None, "--ticker", help="Restrict to a single ticker."),
    items: str | None = typer.Option(
        "1", "--items", help="Comma-separated 10-K item codes (default: 1 = Business)."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Max sections to process."),
    priced_only: bool = typer.Option(
        True, "--priced-only/--all", help="Only extract from companies with a return series."
    ),
    latest_only: bool = typer.Option(
        True,
        "--latest-only/--all-filings",
        help="Only each company's most recent filing (current relationships).",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Retry sections previously marked as failed model calls.",
    ),
) -> None:
    """Read 10-K sections and extract typed company relationship edges via the LLM.

    Needs a running ollama server (see the connection-engine setup). Resumable:
    filings already in ``company_edges`` are skipped and results flush per batch,
    so a full-universe run can be re-run until the candidate count is zero.
    Defaults to Item 1 of each company's most recent filing -- the current
    graph, not every historical restatement of it.
    """
    config = _load()
    tickers = [ticker] if ticker else None
    _run(
        lambda: relationships_collector.sync(
            config,
            tickers=tickers,
            items=_split_csv_tuple(items) or relationships_collector.SECTION_ITEMS,
            limit=limit,
            priced_only=priced_only,
            latest_only=latest_only,
            retry_failed=retry_failed,
        )
    )


@signals_app.command("project-graph")
def signals_project_graph() -> None:
    """Regenerate the Obsidian vault from ``company_edges``/``role_memberships``.

    Writes one interlinked note per company and one per qualifying person
    (docs/specs/2026-07-25-people-insider-graph-design.md §9.1) into the
    vault, so the graph is navigable in Obsidian's graph view. The vault is a
    projection of the database and is overwritten each run -- the database is
    the source of truth. Also prunes any ``companies/``/``people/`` note on
    disk for an entity that no longer qualifies (§9.3).
    """
    config = _load()
    config.paths.ensure()
    cap = config.settings.people_graph.interlock_max_companies_per_person
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        companies = graph.project_graph(
            con, config.paths.obsidian_vault_dir, max_companies_per_person=cap
        )
        people = graph.project_person_notes(
            con, config.paths.obsidian_vault_dir, max_companies_per_person=cap
        )
    console.print(
        f"[green]Wrote[/green] {companies.written} company note(s) and {people.written} "
        f"person note(s) to {config.paths.obsidian_vault_dir} "
        f"[yellow](pruned {companies.deleted} stale company note(s) and "
        f"{people.deleted} stale person note(s))[/yellow]"
    )


# --------------------------------------------------------------------------- #
# People subcommands (docs/specs/2026-07-25-people-insider-graph-design.md)    #
# --------------------------------------------------------------------------- #
@people_app.command("sync-roles")
def people_sync_roles() -> None:
    """Re-project ``people``/``role_memberships`` from stored insider events.

    Adds no new network fetch: re-reads the owner fields
    ``collectors/insider.py`` already stored in each ``insider_transaction``
    event's payload. Idempotent -- re-running recomputes the same aggregates
    and upserts, never duplicates.
    """
    config = _load()
    _run(lambda: roles_collector.sync(config))


@people_app.command("project-interlocks")
def people_project_interlocks() -> None:
    """Derive ``shared_board_member`` candidate edges from ``role_memberships``.

    A self-join on ``person_id`` where ``company_id`` differs: two companies
    sharing a director become one edge in ``company_edges``, entering the
    graph lifecycle at ``candidate`` like any other edge type. Run
    ``people sync-roles`` first.
    """
    config = _load()
    _run(lambda: interlocks_analytics.project(config))


@people_app.command("detect-cluster-buys")
def people_detect_cluster_buys() -> None:
    """Derive ``insider_cluster_buy`` candidate events from Form 4 purchases.

    Groups open-market purchases (code P) by company into non-overlapping
    windows and emits one event per (company, window) with at least the
    configured minimum number of distinct insiders. Writes into ``events``
    alongside every other producer, so ``signals build-dataset`` /
    ``signals evaluate`` measure this cell through the existing gate
    unchanged.
    """
    config = _load()
    _run(lambda: cluster_buy_analytics.detect(config))


# --------------------------------------------------------------------------- #
# Portfolio subcommands (the paper-trading simulator)                          #
# --------------------------------------------------------------------------- #
#: The chronological splits the event dataset is cut into. ``holdout`` is
#: sealed: every entry point that would read it goes through
#: ``_refuse_sealed_read`` first.
_PORTFOLIO_SPLITS = ("discovery", "holdout")

#: The preregistered stress scenarios ``simulator.run_stress_suite`` knows.
_STRESS_SCENARIOS = ("bootstrap", "vol_x2", "corr_one", "covid")

#: Feed read budget. ``feed.load_feed`` is the one bounded read of a 7.2 GB
#: single-writer database that the graph watchdog holds roughly 8 minutes in
#: 10, so this has to outlast a whole watchdog cycle rather than use the 3x30s
#: default that suits a short collector write.
_FEED_RETRY_ATTEMPTS = 20
_FEED_RETRY_SECONDS = 45.0

#: Write budget for ``portfolio step``. Deliberately shorter: the read phase
#: has already waited out any long lock holder, and a step that still cannot
#: write should fail loudly rather than sit on an increasingly stale decision.
_WRITE_RETRY_ATTEMPTS = 5
_WRITE_RETRY_SECONDS = 30.0

#: The declared parameter grid. It lives in code, not on the command line, on
#: purpose: a grid chosen after seeing results is not a grid, it is a search,
#: and the deflated Sharpe ratio only deflates trials it was told about.
_SWEEP_GRID: tuple[tuple[str, dict[str, float]], ...] = (
    ("baseline", {}),
    ("risk_aversion=1.5", {"risk_aversion": 1.5}),
    ("risk_aversion=4.0", {"risk_aversion": 4.0}),
    ("turnover_penalty=10bps", {"turnover_penalty_bps": 10.0}),
    ("turnover_penalty=50bps", {"turnover_penalty_bps": 50.0}),
)

#: CSCV block ceiling -- ``metrics.probability_of_backtest_overfitting``
#: enumerates C(S, S/2) partitions and refuses more than 20 blocks.
_PBO_MAX_BLOCKS = 16

#: ``ExperimentRecord`` requires a registration to be written strictly before
#: the window it seals. The holdout's real split date is years in the past, so
#: what is actually sealed at registration time is "every holdout read from
#: this moment on" -- one second later is the smallest honest expression of it.
_SEAL_DELAY = timedelta(seconds=1)


def _portfolio_settings(config: Config) -> PortfolioConfig:
    """Every knob, all defaulted, from ``config/settings.yaml``."""
    return config.settings.portfolio


def _propagation_admitted(con: duckdb.DuckDBPyConnection) -> int:
    """Admitted cells whose edge is a graph relationship rather than ``self``.

    Zero is the expected answer and a publishable one: ``GateReport`` renders
    it as "propagation gate: 0 admitted" rather than omitting the line. A
    report that mentions the graph only when the graph worked overstates.
    """
    row = con.execute(
        "SELECT count(*) FROM signal_status "
        "WHERE last_verdict = 'admitted' AND coalesce(edge_type, 'self') <> 'self'"
    ).fetchone()
    return int(row[0]) if row else 0


def _feed_read(
    config: Config, *, split: str | None = None, as_of: date | None = None
) -> tuple[portfolio_feed.FeedBundle, int]:
    """THE bounded read: the feed bundle and the propagation tally, one cycle.

    Both come out of a single connect-use-close so the simulator computes with
    zero open connections, and so the honesty line in the report is measured
    against the same database state the replay actually ran on.
    """

    def _once() -> tuple[portfolio_feed.FeedBundle, int]:
        with database.connection(config.paths.database_path) as con:
            # A replay writes nothing, but a fresh or partially-built database
            # would raise a catalog error without the schema present.
            database.init_db(con)
            return portfolio_feed.load_feed(con, split=split, as_of=as_of), _propagation_admitted(
                con
            )

    return database.with_db_retry(
        _once,
        sleep=time.sleep,
        max_attempts=_FEED_RETRY_ATTEMPTS,
        retry_seconds=_FEED_RETRY_SECONDS,
    )


def _require_tradeable_feed(bundle: portfolio_feed.FeedBundle, *, split: str) -> None:
    """Refuse to replay a feed that cannot produce a day. Exits 1 with the fix."""
    if bundle.panel.calendar.size < 2:
        err_console.print(
            f"[red]No price history for the {split} split.[/red] Run "
            "'market sync-prices' and 'signals build-dataset' first."
        )
        raise typer.Exit(code=1)
    if not bundle.cells:
        err_console.print(
            "[yellow]No tradeable cells.[/yellow] Run 'signals evaluate' first -- "
            "a replay with no admitted cell holds cash and measures nothing."
        )
        raise typer.Exit(code=1)


def _artifact_dir(config: Config, settings: PortfolioConfig, name: str) -> Path:
    """``<home>/data/parquet/portfolio/<simulation_version>/<name>``.

    ``simulation_version`` is in the path because that is what isolates
    experiments: a scratch grid must never overwrite the artifacts belonging to
    the account of record.
    """
    return config.paths.parquet_dir / "portfolio" / settings.simulation_version / name


def _turnover_series(result: portfolio_simulator.ReplayResult) -> np.ndarray:
    """Per-day traded notional as a fraction of that day's marked equity."""
    traded: dict[date, float] = defaultdict(float)
    for fill in result.fills:
        traded[fill.as_of] += abs(fill.shares * fill.price)
    out = np.zeros(result.calendar.size, dtype=np.float64)
    for index in range(result.calendar.size):
        equity = float(result.equity_curve[index])
        if equity > 0.0:
            out[index] = traded.get(result.calendar[index].astype(object), 0.0) / equity
    return out


def _replay_metrics(
    result: portfolio_simulator.ReplayResult,
) -> portfolio_metrics.PortfolioMetrics:
    return portfolio_metrics.compute_metrics(
        result.daily_returns,
        result.equity_curve,
        turnover=_turnover_series(result),
        benchmark_returns=result.benchmark_returns,
    )


def _metrics_table(
    metrics: portfolio_metrics.PortfolioMetrics, *, title: str, ending_equity: float
) -> Table:
    """The metrics table. States dollars first -- this is a $10,000 account."""
    table = Table(title=title, show_edge=True)
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("ending equity", f"${ending_equity:,.2f}")
    table.add_row("total return", f"{metrics.total_return:+.2%}")
    table.add_row("annualized return", f"{metrics.annualized_return:+.2%}")
    table.add_row("annualized vol", f"{metrics.annualized_vol:.2%}")
    table.add_row("Sharpe", f"{metrics.sharpe:.3f}")
    table.add_row("Sortino", f"{metrics.sortino:.3f}")
    table.add_row("Calmar", f"{metrics.calmar:.3f}")
    table.add_row("max drawdown", f"{metrics.max_drawdown:.2%}")
    table.add_row("CVaR 95", f"{metrics.cvar_95:.2%}")
    table.add_row("turnover (annual)", f"{metrics.turnover_annual:.2f}x")
    table.add_row("days", str(metrics.n_days))
    for name, value in (
        ("alpha", metrics.alpha),
        ("beta", metrics.beta),
        ("information ratio", metrics.information_ratio),
    ):
        table.add_row(name, "not measured" if value is None else f"{value:+.3f}")
    return table


def _stress_table(outcomes: tuple[portfolio_simulator.StressOutcome, ...]) -> Table:
    table = Table(title="stress suite (risk controls, never profit)", show_edge=True)
    for column in ("scenario", "realized vol", "max DD", "governor fired", "verdict"):
        table.add_column(column)
    for outcome in outcomes:
        table.add_row(
            outcome.scenario,
            f"{outcome.realized_vol:.2%}" + ("" if outcome.vol_within_target else " [red]![/red]"),
            f"{outcome.max_drawdown:.2%}"
            + ("" if outcome.drawdown_within_halt else " [red]![/red]"),
            _yes_no(outcome.governor_fired),
            "[green]pass[/green]" if outcome.passed else "[red]fail[/red]",
        )
    return table


def _yes_no(value: bool) -> str:
    return "[green]yes[/green]" if value else "[red]no[/red]"


def _daily_sharpe(returns: np.ndarray) -> float:
    """Per-observation Sharpe. PSR/DSR require day units, never annualized."""
    if returns.size < 2:
        return 0.0
    deviation = float(np.std(returns, ddof=1))
    return float(np.mean(returns)) / deviation if deviation > 0.0 else 0.0


def _moments(returns: np.ndarray) -> tuple[float, float]:
    """Sample skewness and the RAW fourth moment (3.0 for a normal, not excess)."""
    centered = returns - float(np.mean(returns))
    variance = float(np.mean(np.square(centered)))
    if variance <= 0.0:
        return 0.0, 3.0
    skew = float(np.mean(centered**3)) / variance**1.5
    kurtosis = float(np.mean(centered**4)) / variance**2
    return skew, max(kurtosis, 1.0)


def _pbo_blocks(n_periods: int) -> int:
    """Largest even block count <= 16 leaving >= 2 rows per block; 0 if none fits."""
    for blocks in range(_PBO_MAX_BLOCKS, 1, -2):
        if n_periods >= blocks * 2:
            return blocks
    return 0


def _gate_report(
    *,
    propagation_admitted: int,
    dsr: float | None = None,
    pbo: float | None = None,
    n_trials: int = 0,
    stress_outcomes: tuple[portfolio_simulator.StressOutcome, ...] = (),
    holdout_unsealed: bool = False,
) -> portfolio_report.GateReport:
    return portfolio_report.GateReport(
        dsr=dsr,
        pbo=pbo,
        n_trials=n_trials,
        propagation_cells_admitted=propagation_admitted,
        stress_outcomes=stress_outcomes,
        holdout_unsealed=holdout_unsealed,
    )


def _sealed_read_refusals(config: Config, *, family: str) -> list[str]:
    """Every reason the sealed holdout may not be read. Empty means it may.

    Refusal is the default and the common case. The verdict is read off the
    experiment chain rather than recomputed here, because the chain is what a
    preregistration *is*: a claim recorded before the holdout was touched. A
    gate re-derived at read time could be re-derived until it passed.
    """

    def _once() -> tuple[ExperimentRecord, ...]:
        with database.connection(config.paths.database_path) as con:
            database.init_db(con)
            return portfolio_store.load_experiment_chain(con, family=family)

    chain = database.with_db_retry(
        _once,
        sleep=time.sleep,
        max_attempts=_WRITE_RETRY_ATTEMPTS,
        retry_seconds=_WRITE_RETRY_SECONDS,
    )
    if not chain:
        return [
            f"no preregistered experiment chain for strategy family {family!r} "
            f"(run 'portfolio sweep --register' first)"
        ]
    if not any(record.stage is ExperimentStage.REGISTERED for record in chain):
        return [f"the chain for {family!r} holds no REGISTERED preregistration record"]

    judged = [record for record in chain if "gate_passed" in record.metrics]
    if not judged:
        return [
            f"the chain for {family!r} records no gate verdict "
            f"(no sweep has judged DSR, PBO and stress)"
        ]

    metrics = judged[-1].metrics
    reasons: list[str] = []
    dsr = metrics.get("dsr")
    pbo = metrics.get("pbo")
    if dsr is None:
        reasons.append("DSR was never measured")
    elif float(dsr) < portfolio_report.DSR_THRESHOLD:
        reasons.append(f"DSR {float(dsr):.3f} is below {portfolio_report.DSR_THRESHOLD:.2f}")
    if pbo is None:
        reasons.append("PBO was never measured")
    elif float(pbo) > portfolio_report.PBO_THRESHOLD:
        reasons.append(f"PBO {float(pbo):.3f} is above {portfolio_report.PBO_THRESHOLD:.2f}")
    if not bool(metrics.get("stress_passed")):
        reasons.append("the stress suite did not pass every scenario")
    if not reasons and not bool(metrics.get("gate_passed")):
        reasons.append("the recorded gate verdict is FAILED")
    return reasons


def _refuse_sealed_read(config: Config, *, family: str) -> None:
    """Print why the holdout stays sealed and exit 1, or return and let it open."""
    reasons = _sealed_read_refusals(config, family=family)
    if not reasons:
        return
    err_console.print("[red]Refusing to read the sealed holdout.[/red]")
    for reason in reasons:
        err_console.print(f"  • {reason}")
    err_console.print(
        "[dim]The holdout is read once. Reading it while parameters are still "
        "being chosen turns its track record into a description of those choices.[/dim]"
    )
    raise typer.Exit(code=1)


def _ledger_is_empty(con: duckdb.DuckDBPyConnection, *, simulation_version: str) -> bool:
    """True when no fill exists for this version. Mirrors ``store``'s own join.

    ``paper_fills`` has no version column, so a fill belongs to a version only
    through its parent order -- the same join ``load_account`` folds.
    """
    row = con.execute(
        """
        SELECT count(*) FROM paper_fills f
        JOIN paper_orders o ON o.paper_order_id = f.paper_order_id
        WHERE o.simulation_version = ?
        """,
        [simulation_version],
    ).fetchone()
    return not row or int(row[0]) == 0


def _persist_replay(
    config: Config,
    settings: PortfolioConfig,
    result: portfolio_simulator.ReplayResult,
    *,
    opened_on: date,
) -> tuple[int, int] | None:
    """Write genesis, then orders, then fills. ``None`` means the ledger exists.

    Orders strictly before fills: ``paper_fills`` carries no
    ``simulation_version`` column, so ``store.write_fills`` refuses any fill
    whose parent order is absent -- an orphan would be written successfully and
    then vanish from the account.

    Refuses outright once a ledger exists under this ``simulation_version``. A
    replay opens its account on its own first calendar day, so materializing
    one on top of a ledger opened elsewhere would leave two DEPOSIT rows and an
    account holding twice the capital it was funded with -- and ``load_account``
    only ever checks the first of them. The remedy is a new
    ``simulation_version``, never a reconciliation.

    The closing mark is written too. Without it a replay-created account loads
    with the high-water mark the fold produces -- the opening deposit, forever,
    because the ratchet lives in ``account.mark`` and a fill log carries no
    prices. The live drawdown governor would then measure every future decline
    from $10,000 and never fire. One row is enough: the governor needs the
    account's *current* peak, not the history of how it got there, and the
    per-day marks are already on the SSD in ``equity.parquet``.
    """
    orders = tuple(order for decision in result.decisions for order in decision.orders)
    final = result.final_state
    closing_equity = float(result.equity_curve[-1])
    peak = float(final.high_water_mark)
    closing_drawdown = max(0.0, 1.0 - closing_equity / peak) if peak > 0 else 0.0

    def _once() -> tuple[int, int] | None:
        with database.connection(config.paths.database_path) as con:
            database.init_db(con)
            if not _ledger_is_empty(con, simulation_version=settings.simulation_version):
                return None
            portfolio_store.write_genesis(
                con,
                cash=settings.starting_cash,
                as_of=opened_on,
                simulation_version=settings.simulation_version,
            )
            written_orders = portfolio_store.write_orders(
                con, orders, simulation_version=settings.simulation_version
            )
            written_fills = portfolio_store.write_fills(
                con, result.fills, simulation_version=settings.simulation_version
            )
            portfolio_store.write_account_mark(
                con,
                simulation_version=settings.simulation_version,
                as_of=final.as_of,
                equity=closing_equity,
                cash=float(final.cash),
                high_water_mark=peak,
                drawdown=closing_drawdown,
            )
            return written_orders, written_fills

    return database.with_db_retry(
        _once,
        sleep=time.sleep,
        max_attempts=_WRITE_RETRY_ATTEMPTS,
        retry_seconds=_WRITE_RETRY_SECONDS,
    )


@portfolio_app.command("replay")
def portfolio_replay(
    split: str = typer.Option("discovery", "--split", help="discovery | holdout."),
    write: bool = typer.Option(
        False,
        "--write/--no-write",
        help="Persist orders and fills to the account of record. OFF by default.",
    ),
    as_of: datetime | None = typer.Option(
        None,
        "--as-of",
        formats=["%Y-%m-%d"],
        help="Point-in-time bound on fitted betas and known graph edges.",
    ),
    notify_telegram: bool = typer.Option(
        False, "--notify/--no-notify", help="Also push the report card to Telegram."
    ),
) -> None:
    """Replay the simulated account over a split, then print and file its metrics.

    One bounded read (``feed.load_feed`` under ``with_db_retry``), then the
    replay and the metrics compute with zero open connections, then artifacts
    land as Parquet under ``MARKET_INTELLIGENCE_HOME`` -- never back into the
    7.2 GB database.

    **Writes nothing to the database unless ``--write`` is passed.** A replay is
    research; the account of record is written by ``portfolio step``. Passing
    ``--write`` here rebuilds a whole ledger under the configured
    ``simulation_version``, so point it at a scratch version, not at the real
    account.

    ``--split holdout`` is not a shortcut past the seal: it goes through the
    same preregistration-and-gate check as ``portfolio holdout-read``.
    """
    config = _load()
    settings = _portfolio_settings(config)
    if split not in _PORTFOLIO_SPLITS:
        err_console.print(
            f"[red]Unknown split[/red] {split!r}. Expected one of: "
            f"{', '.join(_PORTFOLIO_SPLITS)}"
        )
        raise typer.Exit(code=2)
    if split == "holdout":
        _refuse_sealed_read(config, family=settings.simulation_version)

    bundle, propagation = _feed_read(
        config, split=split, as_of=as_of.date() if as_of else None
    )
    _require_tradeable_feed(bundle, split=split)

    result = portfolio_simulator.replay(bundle, settings)
    metrics = _replay_metrics(result)
    ending_equity = float(result.equity_curve[-1])
    gate = _gate_report(
        propagation_admitted=propagation, holdout_unsealed=split == "holdout"
    )

    out_dir = _artifact_dir(config, settings, f"replay-{split}")
    written = portfolio_report.write_artifacts(result, metrics, out_dir=out_dir)

    console.print(
        _metrics_table(
            metrics,
            title=f"portfolio replay — {split} ({settings.simulation_version})",
            ending_equity=ending_equity,
        )
    )
    console.print(f"[dim]artifacts: {out_dir}[/dim]")
    for path in written:
        console.print(f"  • {path.name}")

    card = portfolio_report.render_telegram(
        metrics,
        gate,
        title=f"Portfolio replay — {split}",
        ending_equity=ending_equity,
    )
    console.print(card)
    if notify_telegram:
        delivered = notify._post_text(config, card)
        console.print(f"telegram: {_yes_no(delivered)}")

    if not write:
        console.print(
            "[dim]--no-write is the default: nothing was written to the database.[/dim]"
        )
        return

    written_rows = _persist_replay(
        config, settings, result, opened_on=result.calendar[0].astype(object)
    )
    if written_rows is None:
        err_console.print(
            f"[red]A ledger already exists under simulation_version "
            f"{settings.simulation_version!r}.[/red] A replay opens its own genesis "
            f"deposit, so writing on top of one would fund the account twice. Bump "
            f"portfolio.simulation_version in config/settings.yaml and re-run."
        )
        raise typer.Exit(code=1)
    orders, fills = written_rows
    console.print(
        f"[yellow]Wrote[/yellow] {orders} order row(s) and {fills} fill row(s) "
        f"under simulation_version={settings.simulation_version!r}."
    )


@portfolio_app.command("stress")
def portfolio_stress(
    scenarios: str = typer.Option(
        ",".join(_STRESS_SCENARIOS),
        "--scenarios",
        help="Comma-separated: bootstrap, vol_x2, corr_one, covid.",
    ),
    seed: int = typer.Option(0, "--seed", help="Bootstrap seed; the suite is deterministic."),
) -> None:
    """Re-run the replay under adverse regimes and report the risk controls.

    Read-only, and it never tunes anything. The pass criteria are preregistered
    in ``simulator``: realized vol within 1.25x target, drawdown inside the
    flatten threshold plus 5 points, and the governor actually observed firing.
    A scenario cannot be passed by relaxing what it was measured against --
    the criteria are read off the same config the replay ran under.
    """
    config = _load()
    settings = _portfolio_settings(config)
    chosen = tuple(_split_csv(scenarios) or ())
    unknown = [name for name in chosen if name not in _STRESS_SCENARIOS]
    if not chosen or unknown:
        err_console.print(
            f"[red]Unknown scenario(s)[/red] {', '.join(unknown) or '(none given)'}. "
            f"Expected one or more of: {', '.join(_STRESS_SCENARIOS)}"
        )
        raise typer.Exit(code=2)

    bundle, _ = _feed_read(config, split="discovery")
    _require_tradeable_feed(bundle, split="discovery")

    outcomes = portfolio_simulator.run_stress_suite(
        bundle, settings, scenarios=chosen, seed=seed
    )
    console.print(_stress_table(outcomes))
    passed = sum(1 for outcome in outcomes if outcome.passed)
    color = "green" if passed == len(outcomes) else "red"
    console.print(f"[{color}]{passed}/{len(outcomes)} scenario(s) passed.[/{color}]")


def _step_lines(
    *,
    as_of: date,
    settings: PortfolioConfig,
    decision: portfolio_simulator.DayDecision,
    fills: tuple[portfolio_account.Fill, ...],
    equity: float,
) -> list[str]:
    """The step report, in dollars. Shared by the Telegram nudge and the note."""
    lines = [
        f"📒 Paper account — {as_of.isoformat()}",
        f"Equity: ${equity:,.2f} (opened at ${settings.starting_cash:,.2f})",
        f"Decision: {decision.optimizer_status}, "
        f"governor x{decision.governor_scale:.2f}, vol x{decision.vol_scale:.2f}",
        f"Orders: {len(decision.orders)} · Fills: {len(fills)}",
    ]
    lines.extend(
        f"  {fill.side} {fill.shares:.4f} {fill.symbol} @ ${fill.price:,.4f}" for fill in fills
    )
    if not fills:
        lines.append("  (no fills — the band or the cash cap suppressed every trade)")
    lines.append(f"simulation_version: {settings.simulation_version}")
    return lines


@portfolio_app.command("step")
def portfolio_step(
    as_of: datetime = typer.Option(
        ..., "--as-of", formats=["%Y-%m-%d"], help="The session to step (a trading day)."
    ),
    open_account: bool = typer.Option(
        False,
        "--open-account",
        help="Write the opening deposit when the ledger is empty. Refuses otherwise.",
    ),
    force_rebalance: bool = typer.Option(
        False, "--force-rebalance", help="Rebalance regardless of the scheduled cadence."
    ),
    notify_telegram: bool = typer.Option(
        True, "--notify/--no-notify", help="Push the step report to Telegram."
    ),
) -> None:
    """Step the account of record forward one session. THE ONLY writer of it.

    Checks the kill switch first and fails closed if it is engaged, or if the
    switch cannot be read at all -- a safety switch whose state is unknown is
    not a safety switch.

    Then: one bounded read (kill switch, feed, stored fills), one
    ``simulator.step_day`` with zero open connections, then one bounded write
    that persists **orders before fills** because ``store.write_fills`` refuses
    a fill whose parent order is absent. Finally the Telegram/Obsidian report.
    """
    config = _load()
    settings = _portfolio_settings(config)
    day = as_of.date()

    def _read() -> tuple[bool, portfolio_feed.FeedBundle, portfolio_account.AccountState | None]:
        with database.connection(config.paths.database_path) as con:
            database.init_db(con)
            if portfolio_store.kill_switch_active(con):
                return True, portfolio_feed.load_feed(con, as_of=day), None
            bundle = portfolio_feed.load_feed(con, as_of=day)
            if _ledger_is_empty(con, simulation_version=settings.simulation_version):
                return False, bundle, None
            return (
                False,
                bundle,
                portfolio_store.load_account(
                    con,
                    simulation_version=settings.simulation_version,
                    expected_starting_cash=settings.starting_cash,
                    fractional=settings.fractional_shares,
                ),
            )

    try:
        engaged, bundle, stored = database.with_db_retry(
            _read,
            sleep=time.sleep,
            max_attempts=_FEED_RETRY_ATTEMPTS,
            retry_seconds=_FEED_RETRY_SECONDS,
        )
    except portfolio_store.GenesisMismatchError as exc:
        err_console.print(f"[red]Account of record refuses to load:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if engaged:
        err_console.print(
            "[red]Kill switch engaged — refusing to step the account.[/red] "
            "Resolve the kill_switch_events row, then re-run."
        )
        raise typer.Exit(code=1)

    opening = stored is None
    if opening and not open_account:
        err_console.print(
            f"[red]No account of record for simulation_version "
            f"{settings.simulation_version!r}.[/red] Re-run with --open-account to write "
            f"the opening ${settings.starting_cash:,.2f} deposit."
        )
        raise typer.Exit(code=1)

    panel = bundle.panel
    try:
        panel.index_of(day)
    except KeyError as exc:
        err_console.print(f"[red]{day.isoformat()} is not a trading day in the panel.[/red]")
        raise typer.Exit(code=2) from exc

    state = stored if stored is not None else portfolio_account.genesis(settings.starting_cash, day)

    # The governor reads equity through t-1 only. The stored ledger carries the
    # high-water mark but no equity curve, so the curve it needs is exactly two
    # points: the peak it has ever marked, and yesterday's close.
    previous = day - timedelta(days=1)
    prior_prices = {
        symbol: price
        for symbol in state.positions
        if (price := panel.last_known_price(symbol, as_of=previous)) is not None
    }
    prior_equity = float(portfolio_account.mark(state, prior_prices, stale_ok=True).equity)
    equity_history = np.array([state.high_water_mark, prior_equity], dtype=np.float64)

    # ``step_day`` is the published signature but projects the day's fills away,
    # and the account of record is a fold over fills -- persisting orders alone
    # would leave a ledger that reconstructs to the wrong account. This is the
    # one place that needs the unprojected result.
    decision, state, fills, equity = portfolio_simulator.execute_day(
        state,
        bundle,
        settings,
        as_of=day,
        equity_history=equity_history,
        costs=CostModel(
            slippage_bps=settings.slippage_bps,
            commission_per_share=settings.commission_per_share,
        ),
        force_rebalance=force_rebalance,
    )

    # ``execute_day`` returns the state whose mark the day's own ``account.mark``
    # already ratcheted, so this is that mark -- not a second, independent one.
    # The drawdown is recomputed from it by ``mark``'s own formula rather than
    # threaded out of the simulator, so the two cannot disagree.
    high_water = float(state.high_water_mark)
    drawdown = max((high_water - float(equity)) / high_water, 0.0) if high_water > 0 else 0.0

    def _write() -> tuple[int, int]:
        with database.connection(config.paths.database_path) as con:
            database.init_db(con)
            if opening:
                portfolio_store.write_genesis(
                    con,
                    cash=settings.starting_cash,
                    as_of=day,
                    simulation_version=settings.simulation_version,
                )
            written_orders = portfolio_store.write_orders(
                con, decision.orders, simulation_version=settings.simulation_version
            )
            written_fills = portfolio_store.write_fills(
                con, fills, simulation_version=settings.simulation_version
            )
            # Last, and unconditional. The high-water mark moves on *marking*
            # days, not fill days, so a session that traded nothing still has a
            # mark to record -- and the account rebuild has no other way to
            # recover it (see ``store.load_account``).
            portfolio_store.write_account_mark(
                con,
                simulation_version=settings.simulation_version,
                as_of=day,
                equity=float(equity),
                cash=float(state.cash),
                high_water_mark=high_water,
                drawdown=drawdown,
            )
            return written_orders, written_fills

    orders_written, fills_written = database.with_db_retry(
        _write,
        sleep=time.sleep,
        max_attempts=_WRITE_RETRY_ATTEMPTS,
        retry_seconds=_WRITE_RETRY_SECONDS,
    )

    lines = _step_lines(
        as_of=day, settings=settings, decision=decision, fills=fills, equity=float(equity)
    )
    message = "\n".join(lines)
    for line in lines:
        console.print(line)
    console.print(
        f"[green]Persisted[/green] {orders_written} order row(s), then "
        f"{fills_written} fill row(s), then the {day.isoformat()} mark "
        f"(high-water ${high_water:,.2f}, drawdown {drawdown:.2%})."
    )

    note_dir = config.paths.obsidian_vault_dir / "portfolio"
    note_dir.mkdir(parents=True, exist_ok=True)
    note = note_dir / f"{day.isoformat()}.md"
    note.write_text(message + "\n", encoding="utf-8")
    console.print(f"[dim]note: {note}[/dim]")

    if notify_telegram:
        # ``_post_text`` is the shared Telegram primitive behind every sender in
        # ``autopilot.notify``; it never raises, so a down notifier degrades the
        # report rather than the step that already wrote the ledger.
        console.print(f"telegram: {_yes_no(notify._post_text(config, message))}")


def _last_known_closes(
    con: duckdb.DuckDBPyConnection, symbols: tuple[str, ...]
) -> dict[str, float]:
    """Latest stored close per held symbol. Bounded by the account's own names."""
    if not symbols:
        return {}
    placeholders = ", ".join("?" * len(symbols))
    rows = con.execute(
        f"""
        SELECT symbol, last(close ORDER BY price_date) FROM daily_prices
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
        """,
        list(symbols),
    ).fetchall()
    return {str(row[0]): float(row[1]) for row in rows if row[1] is not None}


@portfolio_app.command("status")
def portfolio_status() -> None:
    """Rebuild the account from ``paper_fills`` and print it. Read-only.

    The account is a fold over fills, never a stored balance, so this is the
    account -- not a cached view of it. An unwritten ledger and an unfunded
    account are the same picture, and this says so rather than printing a
    zero account that looks tradeable.

    The high-water mark is the one number here the fold cannot produce; it is
    restored from ``paper_account_marks`` by ``load_account``, and the session
    it was last marked on is printed beside it so a stale mark is visible
    rather than merely absent.
    """
    config = _load()
    settings = _portfolio_settings(config)

    def _once() -> tuple[
        portfolio_account.AccountState | None,
        dict[str, float],
        str | None,
        date | None,
    ]:
        with database.connection(config.paths.database_path) as con:
            database.init_db(con)
            try:
                state = portfolio_store.load_account(
                    con,
                    simulation_version=settings.simulation_version,
                    expected_starting_cash=settings.starting_cash,
                    fractional=settings.fractional_shares,
                )
            except portfolio_store.GenesisMismatchError as exc:
                return None, {}, str(exc), None
            latest = portfolio_store.load_latest_mark(
                con, simulation_version=settings.simulation_version
            )
            return (
                state,
                _last_known_closes(con, tuple(sorted(state.positions))),
                None,
                None if latest is None else latest[0],
            )

    state, prices, problem, marked_on = database.with_db_retry(
        _once,
        sleep=time.sleep,
        max_attempts=_WRITE_RETRY_ATTEMPTS,
        retry_seconds=_WRITE_RETRY_SECONDS,
    )

    if state is None:
        console.print(
            f"[yellow]No account of record for simulation_version "
            f"{settings.simulation_version!r}.[/yellow]"
        )
        console.print(f"  {problem}")
        console.print(
            "  Run 'market-intelligence portfolio step --as-of <date> --open-account' "
            "to write the opening deposit."
        )
        return

    marked = portfolio_account.mark(state, prices, stale_ok=True)
    summary = Table(title=f"paper account — {settings.simulation_version}", show_edge=True)
    summary.add_column("item", style="bold")
    summary.add_column("value", justify="right")
    summary.add_row("as of", state.as_of.isoformat())
    summary.add_row("equity", f"${marked.equity:,.2f}")
    summary.add_row("cash", f"${state.cash:,.2f}")
    # ``state.high_water_mark`` is the restored one, so this is the peak the
    # governor will actually measure against -- not ``marked``'s, which has
    # already been ratcheted up by today's equity and would hide a drawdown.
    summary.add_row("high-water mark", f"${state.high_water_mark:,.2f}")
    summary.add_row(
        "mark as of",
        marked_on.isoformat() if marked_on is not None else "never marked (folded)",
    )
    summary.add_row("drawdown", f"{marked.drawdown:.2%}")
    summary.add_row("positions", str(len(state.positions)))
    console.print(summary)

    if not state.positions:
        console.print("[dim]No open positions — the account is entirely in cash.[/dim]")
        return

    holdings = Table(title="holdings", show_edge=True)
    for column in ("symbol", "shares", "cost basis", "mark", "value"):
        holdings.add_column(column, justify="right")
    for symbol in sorted(state.positions):
        position = state.positions[symbol]
        price = marked.prices[symbol]
        holdings.add_row(
            symbol,
            f"{position.shares:.4f}",
            f"${position.cost_basis:,.2f}",
            f"${price:,.4f}" + ("" if symbol in prices else " [yellow](at cost)[/yellow]"),
            f"${position.shares * price:,.2f}",
        )
    console.print(holdings)


def _trial_config(settings: PortfolioConfig, name: str, knobs: dict[str, float]) -> PortfolioConfig:
    """One grid point, on its own ``simulation_version``.

    The version suffix is the isolation: a scratch trial must never be able to
    write, or be mistaken for, the account of record.
    """
    payload = settings.model_dump()
    payload.update(knobs)
    payload["simulation_version"] = f"{settings.simulation_version}-sweep-{name}"
    return PortfolioConfig(**payload)


def _sweep_records(
    settings: PortfolioConfig,
    bundle: portfolio_feed.FeedBundle,
    *,
    family: str,
    trial_names: tuple[str, ...],
    metrics: dict[str, object],
) -> tuple[ExperimentRecord, ExperimentRecord]:
    """A registration and its calibrated child. The chain, not a loose result.

    ``ExperimentStage.REGISTERED`` forbids metrics, so the grid is declared in
    one record and judged in a second that names it as parent. Both ids carry
    the recording timestamp: the registry is append-only and immutable, so a
    later sweep appends a new link rather than rewriting an existing one.
    """
    recorded_at = datetime.now(UTC)
    stamp = recorded_at.isoformat()
    configuration_hash = sha256_json(settings.model_dump())
    data_snapshot_hash = sha256_json(
        {
            "days": int(bundle.panel.calendar.size),
            "symbols": list(bundle.panel.symbols),
            "cells": [list(cell.key) for cell in bundle.cells],
            "signal_days": sorted(day.isoformat() for day in bundle.signals_by_day),
        }
    )
    common = {
        "experiment_id": f"{family}:{stamp}",
        "strategy_family": family,
        "hypothesis": (
            "The declared parameter grid contains a configuration whose edge "
            "survives deflation for the size of the search and the stress suite."
        ),
        "code_hash": sha256_json({"package_version": __version__, "grid": list(trial_names)}),
        "configuration_hash": configuration_hash,
        "data_snapshot_hash": data_snapshot_hash,
        "cost_model_version": (
            f"slippage_bps={settings.slippage_bps};"
            f"commission_per_share={settings.commission_per_share}"
        ),
        "feature_version": settings.simulation_version,
    }
    registration = ExperimentRecord(
        record_id=f"{family}:registration:{stamp}",
        stage=ExperimentStage.REGISTERED,
        recorded_at=recorded_at,
        parameters={"grid": list(trial_names), "portfolio": settings.model_dump()},
        sealed_window_start=recorded_at + _SEAL_DELAY,
        **common,
    )
    calibration = ExperimentRecord(
        record_id=f"{family}:calibration:{stamp}",
        stage=ExperimentStage.CALIBRATED,
        recorded_at=recorded_at,
        parameters={"grid": list(trial_names)},
        metrics=metrics,
        parent_record_hash=registration.record_hash,
        **common,
    )
    return registration, calibration


@portfolio_app.command("sweep")
def portfolio_sweep(
    register: bool = typer.Option(
        False, "--register", help="Persist the preregistration and the gate verdict."
    ),
    stress: bool = typer.Option(
        True, "--stress/--no-stress", help="Run the stress suite on the winning trial."
    ),
    seed: int = typer.Option(0, "--seed", help="Stress seed; the suite is deterministic."),
) -> None:
    """Run the declared grid on discovery and judge it: DSR, PBO, and stress.

    Every trial is replayed, including the losers, because dropping them is the
    precise act of self-deception the deflated Sharpe ratio exists to prevent:
    both the number of trials and the spread of their Sharpes push the bar up.
    PBO comes from CSCV over the same trial matrix.

    ``--register`` writes the chain -- a registration declaring the grid, and a
    calibrated child carrying the verdict. ``portfolio holdout-read`` reads that
    verdict and refuses without it.
    """
    config = _load()
    settings = _portfolio_settings(config)
    bundle, propagation = _feed_read(config, split="discovery")
    _require_tradeable_feed(bundle, split="discovery")

    names: list[str] = []
    sharpes: list[float] = []
    columns: list[np.ndarray] = []
    trials: list[tuple[str, PortfolioConfig, portfolio_simulator.ReplayResult]] = []
    for name, knobs in _SWEEP_GRID:
        trial = _trial_config(settings, name, knobs)
        result = portfolio_simulator.replay(bundle, trial)
        names.append(name)
        sharpes.append(_daily_sharpe(result.daily_returns))
        columns.append(result.daily_returns)
        trials.append((name, trial, result))

    table = Table(title="sweep — declared grid (discovery)", show_edge=True)
    for column in ("trial", "simulation_version", "total return", "daily Sharpe"):
        table.add_column(column)
    for (name, trial, result), sharpe in zip(trials, sharpes, strict=True):
        growth = float(np.prod(1.0 + result.daily_returns)) - 1.0
        table.add_row(name, trial.simulation_version, f"{growth:+.2%}", f"{sharpe:.4f}")
    console.print(table)

    best = int(np.argmax(np.asarray(sharpes, dtype=np.float64)))
    matrix = np.column_stack(columns)
    winner = trials[best][2]

    dsr: float | None = None
    try:
        skewness, kurtosis = _moments(winner.daily_returns)
        dsr = portfolio_metrics.deflated_sharpe_ratio(
            sharpes[best],
            np.asarray(sharpes, dtype=np.float64),
            n_observations=int(winner.daily_returns.size),
            skewness=skewness,
            kurtosis=kurtosis,
        )
    except ValueError as exc:
        err_console.print(f"[yellow]DSR not measured:[/yellow] {exc}")

    pbo: float | None = None
    blocks = _pbo_blocks(matrix.shape[0])
    if blocks >= 2 and matrix.shape[1] >= 2:
        try:
            pbo = portfolio_metrics.probability_of_backtest_overfitting(matrix, n_blocks=blocks)
        except ValueError as exc:
            err_console.print(f"[yellow]PBO not measured:[/yellow] {exc}")
    else:
        err_console.print(
            f"[yellow]PBO not measured:[/yellow] {matrix.shape[0]} day(s) and "
            f"{matrix.shape[1]} trial(s) cannot fill an even CSCV partition."
        )

    outcomes: tuple[portfolio_simulator.StressOutcome, ...] = ()
    if stress:
        outcomes = portfolio_simulator.run_stress_suite(bundle, trials[best][1], seed=seed)
        console.print(_stress_table(outcomes))

    gate = _gate_report(
        propagation_admitted=propagation,
        dsr=dsr,
        pbo=pbo,
        n_trials=len(names),
        stress_outcomes=outcomes,
    )
    verdict = Table(title="anti-overfitting gate", show_edge=True)
    verdict.add_column("leg", style="bold")
    verdict.add_column("value", justify="right")
    verdict.add_column("bar", justify="right")
    verdict.add_row("DSR", "not measured" if dsr is None else f"{dsr:.3f}", ">= 0.95")
    verdict.add_row("PBO", "not measured" if pbo is None else f"{pbo:.3f}", "<= 0.20")
    verdict.add_row("stress", f"{sum(1 for o in outcomes if o.passed)}/{len(outcomes)}", "all pass")
    verdict.add_row("trials", str(len(names)), "—")
    verdict.add_row("propagation admitted", str(propagation), "—")
    console.print(verdict)
    console.print(
        f"gate: {'[green]PASSED[/green]' if gate.gate_passed else '[red]FAILED[/red]'} "
        f"(winning trial: {names[best]})"
    )

    if not register:
        console.print("[dim]--register was not passed: nothing was recorded.[/dim]")
        return

    family = settings.simulation_version
    registration, calibration = _sweep_records(
        settings,
        bundle,
        family=family,
        trial_names=tuple(names),
        metrics={
            "dsr": dsr,
            "pbo": pbo,
            "n_trials": len(names),
            "best_trial": names[best],
            "trial_sharpes": [float(value) for value in sharpes],
            "stress_scenarios": [outcome.scenario for outcome in outcomes],
            "stress_passed": bool(outcomes) and all(o.passed for o in outcomes),
            "gate_passed": gate.gate_passed,
            "propagation_cells_admitted": propagation,
        },
    )

    def _write() -> None:
        with database.connection(config.paths.database_path) as con:
            database.init_db(con)
            portfolio_store.write_experiment_record(con, registration)
            portfolio_store.write_experiment_record(con, calibration)

    database.with_db_retry(
        _write,
        sleep=time.sleep,
        max_attempts=_WRITE_RETRY_ATTEMPTS,
        retry_seconds=_WRITE_RETRY_SECONDS,
    )
    console.print(
        f"[green]Recorded[/green] the chain for family {family!r}: "
        f"registration {registration.record_hash[:12]} -> "
        f"calibration {calibration.record_hash[:12]}."
    )


@portfolio_app.command("study-graph")
def portfolio_study_graph(
    bootstrap: int = typer.Option(1000, "--bootstrap", help="Dyadic bootstrap resamples."),
    seed: int = typer.Option(0, "--seed", help="Bootstrap seed; the study is deterministic."),
) -> None:
    """Do graph-linked pairs actually co-move more than unlinked ones?

    The honest test of whether the relationship graph earns its place in the
    covariance at all. If the confidence interval includes zero, the graph
    block target is decoration and this says so. Read-only.
    """
    config = _load()
    bundle, _ = _feed_read(config, split="discovery")
    panel = bundle.panel
    if panel.calendar.size < 4 or len(panel.symbols) < 2:
        err_console.print(
            f"[red]Not enough history for a correlation study:[/red] "
            f"{panel.calendar.size} day(s), {len(panel.symbols)} symbol(s)."
        )
        raise typer.Exit(code=1)

    last_day = panel.calendar[-1].astype(object)
    raw = panel.trailing_returns(
        end_exclusive=last_day + timedelta(days=1), lookback=int(panel.calendar.size)
    )
    # A missing day reads as no move -- the same convention the stress harness
    # uses. ``linked_pair_correlation_study`` refuses non-finite input outright.
    returns = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    clusters = portfolio_risk.commercial_clusters(panel.symbols, bundle.edges, as_of=last_day)
    study = portfolio_risk.linked_pair_correlation_study(
        returns, panel.symbols, clusters, n_bootstrap=bootstrap, seed=seed
    )

    table = Table(title="linked-pair correlation study", show_edge=True)
    table.add_column("item", style="bold")
    table.add_column("value", justify="right")
    table.add_row("linked mean corr", f"{study.linked_mean_corr:+.4f}")
    table.add_row("unlinked mean corr", f"{study.random_mean_corr:+.4f}")
    table.add_row("difference", f"{study.difference:+.4f}")
    table.add_row("95% CI", f"[{study.ci_low:+.4f}, {study.ci_high:+.4f}]")
    table.add_row("linked pairs", str(study.n_linked_pairs))
    table.add_row("unlinked pairs", str(study.n_random_pairs))
    console.print(table)
    if study.significant:
        console.print(
            "[green]The CI excludes zero: the graph is measurable risk structure.[/green]"
        )
    else:
        console.print(
            "[yellow]The CI includes zero: no evidence the graph is risk structure. "
            "No evidence is not evidence.[/yellow]"
        )


@portfolio_app.command("holdout-read")
def portfolio_holdout_read(
    family: str | None = typer.Option(
        None, "--family", help="Strategy family; default = the configured simulation_version."
    ),
    write: bool = typer.Option(
        False, "--write/--no-write", help="Persist the holdout replay. OFF by default."
    ),
) -> None:
    """Break the seal and read the holdout. Refuses by default.

    The one number in this system that has to stay uncontaminated. It opens
    only when a preregistered experiment chain exists **and** the verdict
    recorded in that chain passed every leg of the gate: DSR >= 0.95,
    PBO <= 0.20, and every stress scenario. Anything else -- no chain, no
    verdict, an unmeasured statistic, a failed leg -- is a refusal that names
    exactly what is missing.
    """
    config = _load()
    settings = _portfolio_settings(config)
    strategy_family = family or settings.simulation_version
    _refuse_sealed_read(config, family=strategy_family)

    bundle, propagation = _feed_read(config, split="holdout")
    _require_tradeable_feed(bundle, split="holdout")

    result = portfolio_simulator.replay(bundle, settings)
    metrics = _replay_metrics(result)
    ending_equity = float(result.equity_curve[-1])
    gate = _gate_report(propagation_admitted=propagation, holdout_unsealed=True)

    out_dir = _artifact_dir(config, settings, "holdout")
    portfolio_report.write_artifacts(result, metrics, out_dir=out_dir)
    console.print(
        _metrics_table(
            metrics,
            title=f"SEALED HOLDOUT — {strategy_family}",
            ending_equity=ending_equity,
        )
    )
    console.print(f"[dim]artifacts: {out_dir}[/dim]")
    console.print(
        portfolio_report.render_telegram(
            metrics, gate, title="Sealed holdout", ending_equity=ending_equity
        )
    )
    if write:
        written_rows = _persist_replay(
            config, settings, result, opened_on=result.calendar[0].astype(object)
        )
        if written_rows is None:
            err_console.print(
                f"[red]A ledger already exists under simulation_version "
                f"{settings.simulation_version!r}.[/red] Bump it and re-run."
            )
            raise typer.Exit(code=1)
        console.print(
            f"[yellow]Wrote[/yellow] {written_rows[0]} order(s) and {written_rows[1]} fill(s)."
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
