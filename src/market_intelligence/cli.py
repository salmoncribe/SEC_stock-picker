"""Command-line interface.

Commands::

    market-intelligence init-db
    market-intelligence sec sync-companies
    market-intelligence sec collect-filings --ticker NVDA [--forms 10-K,10-Q,8-K] [--limit N]
    market-intelligence sec collect-ipos [--forms S-1,S-1/A,F-1,F-1/A]
    market-intelligence sec ingest-documents [--ticker NVDA] [--limit N]
    market-intelligence fred sync [--series DGS10,UNRATE]
    market-intelligence validate
    market-intelligence reconcile [--no-verify-hashes]
    market-intelligence status

Exit codes: 0 success, 1 pipeline/runtime failure, 2 configuration error.
"""

from __future__ import annotations

from collections.abc import Callable

import typer
from rich.console import Console
from rich.table import Table

from market_intelligence import __version__, database, reconciliation
from market_intelligence.collectors import RunSummary
from market_intelligence.collectors import documents as document_collector
from market_intelligence.collectors import fred as fred_collector
from market_intelligence.collectors import sec as sec_collector
from market_intelligence.config import Config, ConfigError, get_config
from market_intelligence.logging_config import configure_logging, get_logger

app = typer.Typer(
    help="Local market-intelligence data platform (baseline).",
    no_args_is_help=True,
    add_completion=False,
)
sec_app = typer.Typer(help="SEC EDGAR collection.", no_args_is_help=True)
fred_app = typer.Typer(help="FRED economic-data collection.", no_args_is_help=True)
app.add_typer(sec_app, name="sec")
app.add_typer(fred_app, name="fred")

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


@sec_app.command("collect-filings")
def sec_collect_filings(
    ticker: str | None = typer.Option(None, "--ticker", help="Single ticker, e.g. NVDA."),
    forms: str | None = typer.Option(None, "--forms", help="Comma-separated form types."),
    limit: int | None = typer.Option(None, "--limit", help="Max filings per ticker."),
) -> None:
    """Collect filings for a ticker (or the configured companies)."""
    config = _load()
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
) -> None:
    """Download filing documents, preserve the bytes, and extract Item sections.

    Operates on filings already recorded by ``collect-filings``. Re-running is
    safe: unchanged documents are skipped at the raw store and rows are upserted
    rather than duplicated.
    """
    config = _load()
    tickers = [ticker] if ticker else None
    _run(
        lambda: document_collector.ingest_documents(
            config, tickers=tickers, forms=_split_csv(forms), limit=limit
        )
    )


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
