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
    market-intelligence signals extract-relationships [--ticker NKE] [--limit N]
    market-intelligence autopilot run
    market-intelligence ram-guard [--dry-run]
    market-intelligence validate
    market-intelligence reconcile [--no-verify-hashes]
    market-intelligence status

Exit codes: 0 success, 1 pipeline/runtime failure, 2 configuration error.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table

from market_intelligence import __version__, database, reconciliation
from market_intelligence.analytics.impact import AdmissionThresholds
from market_intelligence.analytics.returns import AbnormalReturnMethod
from market_intelligence.autopilot import graph, ram_guard
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
from market_intelligence.collectors import sec as sec_collector
from market_intelligence.config import Config, ConfigError, get_config
from market_intelligence.logging_config import configure_logging, get_logger
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
autopilot_app = typer.Typer(help="The self-checking daily loop.", no_args_is_help=True)
app.add_typer(sec_app, name="sec")
app.add_typer(fred_app, name="fred")
app.add_typer(market_app, name="market")
app.add_typer(signals_app, name="signals")
app.add_typer(autopilot_app, name="autopilot")

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
        )
    )


@signals_app.command("project-graph")
def signals_project_graph() -> None:
    """Regenerate the Obsidian relationship-graph vault from ``company_edges``.

    Writes one interlinked note per company into the vault, so the graph is
    navigable in Obsidian's graph view. The vault is a projection of the
    database and is overwritten each run -- the database is the source of truth.
    """
    config = _load()
    config.paths.ensure()
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        written = graph.project_graph(con, config.paths.obsidian_vault_dir)
    console.print(
        f"[green]Wrote[/green] {written} company note(s) to {config.paths.obsidian_vault_dir}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
