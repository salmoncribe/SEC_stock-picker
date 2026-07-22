"""Price collector: fetch OHLCV bars -> validate -> raw/Parquet/DuckDB.

``sync`` fetches daily bars for the configured (or explicitly requested)
symbol universe from ``YFinanceMarketDataProvider``, persists the raw payload
per symbol, validates the normalized records, and upserts the non-rejected
rows into DuckDB and the Parquet mirror. All bookkeeping (the
``pipeline_runs`` row) is handled by ``pipeline_run``.

**Resumability is the point.** This collector is meant to run over roughly
500 symbols and 10 years of history, and to be interrupted and restarted
freely. Before fetching a symbol it asks the database "what is the latest
date I already have?" and requests only what comes after that -- never the
whole history again. A re-run against an already-current universe makes zero
network calls and reports ``inserted=0``.

**One dead symbol must not sink the run.** A ``MarketDataUnavailable`` from
the provider (a delisted or mistyped symbol, an upstream outage) is caught
per symbol: the failure is counted and named in ``summary.notes``, and
collection continues with the rest of the universe. Any other exception is a
bug in this collector or its callers and is left to propagate.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.clients.market_yfinance import (
    PROVIDER_NAME,
    MarketDataUnavailable,
    YFinanceMarketDataProvider,
)
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.schemas.common import utcnow
from market_intelligence.schemas.market import DailyPriceRecord
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet, raw
from market_intelligence.validators.market import validate_price, validate_price_series

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config

# Every run includes these regardless of what universe was requested, so the
# returns collector always has a benchmark series to subtract.
BENCHMARK_SYMBOLS: tuple[str, ...] = ("SPY",)

# First backfill for a symbol with no prior rows. ~10 years, matching this
# platform's designed backtest horizon (docs/specs/2026-07-22-signal-graph
# -design.md, S1).
DEFAULT_LOOKBACK_DAYS = 365 * 10


def _default_universe(config: Config, con: duckdb.DuckDBPyConnection) -> set[str]:
    """Distinct constituent tickers if the index has been collected, else the
    configured watchlist.
    """
    rows = con.execute("SELECT DISTINCT ticker FROM index_constituents").fetchall()
    tickers = {str(r[0]).upper() for r in rows if r[0]}
    if tickers:
        return tickers
    return {c.ticker.upper() for c in config.companies.companies}


def _resolve_symbols(
    config: Config, con: duckdb.DuckDBPyConnection, symbols: list[str] | None
) -> list[str]:
    """The universe to collect: the requested (or default) symbols, plus the
    benchmark(s). A dict-as-ordered-set keeps the result deterministic, which
    matters for tests asserting on fetch order.
    """
    base = {s.upper() for s in symbols} if symbols is not None else _default_universe(config, con)
    merged = dict.fromkeys(sorted(base))
    merged.update(dict.fromkeys(BENCHMARK_SYMBOLS))
    return list(merged)


def _fetch_start(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    explicit_start: date | None,
    default_start: date,
) -> date:
    """Where to resume fetching ``symbol`` from.

    Three cases, in order:

    * **No history yet** -- start at ``explicit_start`` (or the default
      lookback) and back-fill from there.
    * **A caller asked for history older than what is stored** -- start at
      ``explicit_start``. Resuming from the newest bar would leave the
      requested earlier window permanently unreachable while the run still
      reported success, which is the worst combination available: a silent
      hole in the very history the backtest depends on. Re-requesting the
      overlap is safe -- the upsert is keyed on ``(symbol, price_date)``, so
      the already-stored bars become updates, not duplicates -- and costs only
      network time.
    * **Otherwise** -- resume the day after the newest stored bar, which is
      the ordinary daily-update path and makes no redundant requests.
    """
    result = con.execute(
        "SELECT min(price_date), max(price_date) FROM daily_prices WHERE symbol = ?",
        [symbol],
    ).fetchone()
    existing_min, existing_max = (result[0], result[1]) if result else (None, None)

    if existing_max is None:
        return explicit_start or default_start

    if explicit_start is not None and existing_min is not None and explicit_start < existing_min:
        return explicit_start

    return existing_max + timedelta(days=1)


def sync(
    config: Config,
    *,
    symbols: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    provider: YFinanceMarketDataProvider | None = None,
) -> RunSummary:
    """Collect daily OHLCV bars for ``symbols`` (default: the PIT universe)."""
    with pipeline_run(config, "prices.sync") as (con, summary):
        active_provider = provider or YFinanceMarketDataProvider()
        schema_version = config.settings.app.schema_version
        effective_end = end or date.today()
        default_start = effective_end - timedelta(days=DEFAULT_LOOKBACK_DAYS)

        universe = _resolve_symbols(config, con, symbols)

        collected = 0
        rejected = 0
        price_rows: list[dict[str, Any]] = []

        for symbol in universe:
            fetch_start = _fetch_start(con, symbol, start, default_start)
            if fetch_start > effective_end:
                # Already current: no network call, nothing to reconcile.
                summary.note(f"up_to_date:{symbol}")
                continue

            try:
                fetched = active_provider.fetch(symbol, fetch_start, effective_end)
            except MarketDataUnavailable as exc:
                summary.bump("price_fetch_failed")
                summary.note(f"price_fetch_failed:{symbol}:{exc}")
                continue

            summary.downloaded += 1
            saved = raw.save_raw(
                config.paths.raw_dir, "market", "daily_prices", symbol, fetched.raw_csv, ext="csv"
            )
            if saved.was_new:
                summary.stored += 1
            else:
                summary.skipped += 1

            records = [
                DailyPriceRecord(
                    price_id=hashing.content_hash("price", symbol, bar.date.isoformat()),
                    symbol=symbol,
                    price_date=bar.date,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    adj_close=bar.adj_close,
                    volume=bar.volume,
                    provider=PROVIDER_NAME,
                    source_url=fetched.source_url,
                    content_hash=hashing.content_hash(
                        symbol, bar.date.isoformat(), bar.close, bar.adj_close, bar.volume
                    ),
                    collected_time=utcnow(),
                    schema_version=schema_version,
                )
                for bar in fetched.bars
            ]
            for record in records:
                validate_price(record)
            validate_price_series(records)

            for record in records:
                collected += 1
                if record.is_rejected:
                    rejected += 1
                else:
                    price_rows.append(record.to_row())

            summary.note(f"{symbol}: {len(records)} bars fetched from {fetch_start}")

        result = duckdb_store.upsert_daily_prices(con, price_rows)
        parquet.write_records(
            config.paths.parquet_dir,
            "daily_prices",
            price_rows,
            ["price_id"],
            partition_col="symbol",
        )

        summary.collected = collected
        summary.inserted = result.inserted
        summary.updated = result.updated
        summary.rejected = rejected

    return summary


__all__ = ["BENCHMARK_SYMBOLS", "DEFAULT_LOOKBACK_DAYS", "sync"]
