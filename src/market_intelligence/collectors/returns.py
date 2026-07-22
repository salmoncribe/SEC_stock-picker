"""Returns collector: derive abnormal returns from already-stored prices.

``compute`` never fetches. It reads ``daily_prices`` rows already persisted by
``collectors.prices``, converts each symbol's adjusted-close series into
simple daily returns, and asks ``analytics.returns`` to subtract the
benchmark's (and, if a sector map is supplied, the sector's) expected return.

The output is the label the entire signal layer is graded against, so
lookahead is treated as a hard invariant rather than an ordinary data-quality
problem: ``analytics.returns.compute_abnormal_returns`` is contracted to never
fit an estimation window that reaches into the day it prices, so
``validate_return`` rejecting a row for exactly that reason means this
collector wired something up wrong, not that the input data was noisy. See
the lookahead branch below.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.analytics.returns import (
    AbnormalReturnMethod,
    compute_abnormal_returns,
    simple_returns,
)
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.logging_config import get_logger
from market_intelligence.schemas.common import utcnow
from market_intelligence.schemas.market import DailyReturnRecord
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet
from market_intelligence.validators.market import validate_return

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config

_log = get_logger("collectors.returns")

DEFAULT_BENCHMARK = "SPY"


def _resolve_symbols(
    con: duckdb.DuckDBPyConnection, symbols: list[str] | None, benchmark: str
) -> list[str]:
    """Default universe is every symbol with stored prices; the benchmark is
    always included, so it gets its own (degenerate, zero-abnormal) return
    row alongside everything measured against it.
    """
    if symbols is not None:
        base = [s.upper() for s in symbols]
    else:
        rows = con.execute("SELECT DISTINCT symbol FROM daily_prices").fetchall()
        base = sorted({str(r[0]).upper() for r in rows if r[0]})
    merged = dict.fromkeys(base)
    merged[benchmark.upper()] = None
    return list(merged)


def _load_adjusted_close(
    con: duckdb.DuckDBPyConnection, symbols: list[str]
) -> dict[str, list[tuple[date, float]]]:
    """Read each symbol's adjusted-close series, oldest first.

    Rejected bars are excluded: a bar the price collector refused to store as
    authoritative must not feed a return calculation either.
    """
    if not symbols:
        return {}
    placeholders = ", ".join(["?"] * len(symbols))
    rows = con.execute(
        f"""
        SELECT symbol, price_date, adj_close
        FROM daily_prices
        WHERE symbol IN ({placeholders})
          AND validation_status <> 'rejected'
          AND adj_close IS NOT NULL
        ORDER BY symbol, price_date
        """,
        symbols,
    ).fetchall()
    out: dict[str, list[tuple[date, float]]] = {}
    for symbol, price_date, adj_close in rows:
        out.setdefault(str(symbol), []).append((price_date, float(adj_close)))
    return out


def _sector_return_series(
    returns_by_symbol: dict[str, list[tuple[date, float]]],
    sector_map: dict[str, str],
) -> dict[str, list[tuple[date, float]]]:
    """Equal-weighted average daily return per sector, built from member returns.

    Computed once for the whole universe rather than per-symbol, so a
    sector's series is identical no matter which member consumes it.
    """
    by_sector_date: dict[str, dict[date, list[float]]] = {}
    for symbol, points in returns_by_symbol.items():
        sector = sector_map.get(symbol)
        if sector is None:
            continue
        bucket = by_sector_date.setdefault(sector, {})
        for day, ret in points:
            bucket.setdefault(day, []).append(ret)

    return {
        sector: sorted((day, sum(values) / len(values)) for day, values in by_date.items())
        for sector, by_date in by_sector_date.items()
    }


def compute(
    config: Config,
    *,
    symbols: list[str] | None = None,
    benchmark: str = DEFAULT_BENCHMARK,
    method: AbnormalReturnMethod = AbnormalReturnMethod.MARKET_MODEL,
    sector_map: dict[str, str] | None = None,
) -> RunSummary:
    """Derive and store abnormal returns from already-collected prices."""
    with pipeline_run(config, "returns.compute") as (con, summary):
        schema_version = config.settings.app.schema_version
        benchmark_symbol = benchmark.upper()

        universe = _resolve_symbols(con, symbols, benchmark_symbol)
        price_by_symbol = _load_adjusted_close(con, universe)
        returns_by_symbol = {
            symbol: simple_returns(bars) for symbol, bars in price_by_symbol.items()
        }

        market_returns = returns_by_symbol.get(benchmark_symbol, [])
        if not market_returns:
            summary.note(f"no_benchmark_history:{benchmark_symbol}")

        sector_series = _sector_return_series(returns_by_symbol, sector_map) if sector_map else {}

        collected = 0
        rejected = 0
        return_rows: list[dict[str, Any]] = []

        for symbol in universe:
            asset_returns = returns_by_symbol.get(symbol, [])
            if not asset_returns:
                summary.note(f"no_price_history:{symbol}")
                continue

            sector_returns = None
            if sector_map is not None:
                sector = sector_map.get(symbol)
                sector_returns = sector_series.get(sector) if sector else None

            points = compute_abnormal_returns(
                symbol,
                asset_returns,
                market_returns=market_returns,
                sector_returns=sector_returns,
                method=method,
            )

            for point in points:
                record = DailyReturnRecord(
                    return_id=hashing.content_hash("return", symbol, point.price_date.isoformat()),
                    symbol=symbol,
                    price_date=point.price_date,
                    total_return=point.total_return,
                    market_return=point.market_return,
                    sector_return=point.sector_return,
                    abnormal_return=point.abnormal_return,
                    beta=point.beta,
                    alpha=point.alpha,
                    method=point.method,
                    estimation_window_start=point.estimation_window_start,
                    content_hash=hashing.content_hash(
                        symbol, point.price_date.isoformat(), point.abnormal_return
                    ),
                    collected_time=utcnow(),
                    schema_version=schema_version,
                )
                validate_return(record)
                collected += 1
                if record.is_rejected:
                    rejected += 1
                    if any("lookahead" in err for err in record.validation_errors):
                        # analytics.returns guarantees a window ending strictly
                        # before price_date; landing here means this collector
                        # fed it something it should never see. Log it loudly
                        # rather than let the row disappear into `rejected`
                        # looking like ordinary bad data.
                        _log.error(
                            "lookahead_detected",
                            symbol=symbol,
                            price_date=point.price_date.isoformat(),
                            estimation_window_start=(
                                point.estimation_window_start.isoformat()
                                if point.estimation_window_start
                                else None
                            ),
                        )
                        summary.bump("lookahead_rejected")
                else:
                    return_rows.append(record.to_row())

        result = duckdb_store.upsert_daily_returns(con, return_rows)
        parquet.write_records(
            config.paths.parquet_dir,
            "daily_returns",
            return_rows,
            ["return_id"],
            partition_col="symbol",
        )

        summary.collected = collected
        summary.inserted = result.inserted
        summary.updated = result.updated
        summary.rejected = rejected

    return summary


__all__ = ["DEFAULT_BENCHMARK", "compute"]
