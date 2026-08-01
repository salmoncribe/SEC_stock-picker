"""The one bounded DB read, and the numpy panel everything downstream uses.

DuckDB is single-writer and the graph-watchdog contends for the lock every 600
seconds, holding it roughly 8 minutes in 10. So the simulator reads *once*,
under ``with_db_retry``, and then computes with zero open connections. Anything
bulky goes to Parquet on the SSD rather than back into the database.

``MarketPanel.trailing_returns(end_exclusive=...)`` is the no-lookahead choke
point for the entire system. Every covariance estimate, every volatility, every
scaling decision reads its history through that one method, so a single
off-by-one there is the only place a return from day *t* can leak into a
decision made on day *t*. Test it harder than anything else in this module.

Loaders from ``analytics/backtest_data.py`` are reused verbatim -- they are
already correct and already tested. This module assembles, it does not re-query.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import duckdb
import numpy as np

from market_intelligence.analytics.backtest import BacktestSignal
from market_intelligence.analytics.backtest_data import (
    DEFAULT_MIN_HEDGED_EDGE,
    AdmittedCell,
    filter_tradeable,
    load_admitted_cells,
    load_bars,
    load_betas,
    load_hedged_edges,
    load_signals,
)
from market_intelligence.signals.trade_plan import Bar

#: The benchmark column. Carried in the panel rather than fetched separately so
#: benchmark-relative metrics read the same calendar as everything else.
BENCHMARK_SYMBOL = "SPY"

#: Edge types asserting a measured commercial relationship. Mirrors
#: ``schemas.edges.EdgeType`` as literals so this layer stays SQL-and-numpy.
#: ``shared_board_member`` is excluded on purpose: two firms sharing a director
#: have a plausible information path but no measured return relationship, so an
#: interlock is a channel, not a signal, and never enters a return view or a
#: risk cluster.
COMMERCIAL_EDGE_TYPES = ("customer", "supplier", "competitor", "partner")


@dataclass(frozen=True)
class MarketPanel:
    """Prices as dense ``[T, N]`` float64 arrays, adjusted space plus raw close.

    Adjusted space (``OHLC * adj_close / close``) is what returns and ATR are
    computed in; ``raw_close`` is retained because an executed price is a raw
    price and the ledger must record what a broker would have charged.

    ``symbols[j]`` names column *j*; ``calendar[i]`` dates row *i*, strictly
    ascending. Missing observations are ``np.nan`` -- never forward-filled here,
    because a fill decision belongs to whoever knows why the bar is missing.
    """

    calendar: np.ndarray  # [T] of datetime64[D], strictly ascending
    symbols: tuple[str, ...]
    open: np.ndarray  # [T, N] adjusted
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    raw_close: np.ndarray  # [T, N] unadjusted, for ledger prices

    def index_of(self, day: date) -> int:
        """Row index for ``day``. Raises if absent -- no nearest-match guessing."""
        target = np.datetime64(day, "D")
        row = int(np.searchsorted(self.calendar, target, side="left"))
        if row >= self.calendar.size or self.calendar[row] != target:
            raise KeyError(f"{day.isoformat()} is not a trading day in this panel")
        return row

    def column_of(self, symbol: str) -> int:
        """Column index for ``symbol``. Raises if absent."""
        try:
            return self.symbols.index(symbol)
        except ValueError as exc:
            raise KeyError(f"{symbol} is not a symbol in this panel") from exc

    def trailing_returns(
        self, *, end_exclusive: date, lookback: int, symbols: tuple[str, ...] | None = None
    ) -> np.ndarray:
        """Simple returns over the ``lookback`` rows ending STRICTLY BEFORE ``end_exclusive``.

        The single most leak-prone function in the codebase. A decision made on
        day *t* may see returns through *t-1* and no further. Returns ``[L, K]``.

        The boundary is found by ``searchsorted(..., side="left")``, which counts
        the rows *strictly before* ``end_exclusive`` whether or not that day is
        itself a trading day. Deliberately looser than :meth:`index_of`: a
        decision date is a fact about the calendar the caller lives on, and a
        request for history before a holiday is well posed even though asking
        for that holiday's row is not.

        Returns of day *i* need closes on rows *i-1* and *i*, so the last row
        that may contribute is ``end_row - 1`` -- the price on day *t-1* is
        knowable on day *t*, the price on day *t* is not. Short history pads the
        leading rows with ``np.nan`` rather than returning a short array, so the
        shape is always ``[lookback, K]`` and a caller cannot silently estimate
        a covariance on fewer observations than it asked for.
        """
        if lookback < 1:
            raise ValueError(f"lookback must be at least 1, got {lookback}")
        columns = self._column_indices(symbols)
        out = np.full((lookback, columns.size), np.nan, dtype=np.float64)
        if columns.size == 0 or self.calendar.size == 0:
            return out

        end_row = int(np.searchsorted(self.calendar, np.datetime64(end_exclusive, "D"), "left"))
        if end_row < 2:  # fewer than two closes precede the day: no return exists
            return out
        start_row = max(0, end_row - lookback - 1)
        prices = self.close[start_row:end_row][:, columns]
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = prices[1:] / prices[:-1] - 1.0
        returns[~np.isfinite(returns)] = np.nan
        out[lookback - returns.shape[0] :] = returns
        return out

    def last_known_price(self, symbol: str, *, as_of: date) -> float | None:
        """Most recent non-NaN close at or before ``as_of``; never looks forward.

        Used to close out a symbol that has gone NaN-forever (delisting), so the
        account books a real exit instead of carrying a ghost.

        Reads ``raw_close``: the caller is booking a fill, and a fill is what a
        broker would have charged. ``raw_close`` and ``close`` are missing on
        exactly the same rows, so this cannot see a bar the adjusted series
        hides.
        """
        column = self.raw_close[:, self.column_of(symbol)]
        end_row = int(np.searchsorted(self.calendar, np.datetime64(as_of, "D"), "right"))
        known = np.flatnonzero(np.isfinite(column[:end_row]))
        if known.size == 0:
            return None
        return float(column[known[-1]])

    def _column_indices(self, symbols: tuple[str, ...] | None) -> np.ndarray:
        """Requested column order, or the panel's own when ``symbols`` is None."""
        if symbols is None:
            return np.arange(len(self.symbols), dtype=np.intp)
        return np.array([self.column_of(symbol) for symbol in symbols], dtype=np.intp)


@dataclass(frozen=True)
class FeedBundle:
    """Everything one replay needs, read once and then held in memory."""

    panel: MarketPanel
    signals_by_day: dict[date, list[BacktestSignal]]
    cells: list[AdmittedCell]
    hedged_edges: dict[tuple[str, str | None, int], float]
    betas: dict[str, float]
    edges: tuple[tuple[str, str, str, date], ...]  # (source, target, edge_type, report_date)
    spy_index: int | None  # column of SPY in the panel, for benchmark-relative metrics


def build_panel(bars_by_symbol: dict[str, list], symbols: tuple[str, ...]) -> MarketPanel:
    """Dense panel from the loader's per-symbol bar lists. Pure; no DB.

    ``symbols`` fixes the column order; a symbol with no bars still gets a
    column, entirely NaN, because dropping it would let a signal quietly refer
    to a name the panel cannot price. The calendar is the union of every date
    any requested symbol trades on, so a name that lists mid-history simply
    carries NaN before its first bar.
    """
    columns = tuple(symbols)
    per_symbol: list[list[Bar]] = [list(bars_by_symbol.get(symbol, ())) for symbol in columns]
    days = sorted({bar.date for bars in per_symbol for bar in bars})
    calendar = np.array(days, dtype="datetime64[D]")
    row_of = {day: row for row, day in enumerate(days)}

    shape = (len(days), len(columns))
    open_, high, low, close, raw_close = (
        np.full(shape, np.nan, dtype=np.float64) for _ in range(5)
    )
    for col, bars in enumerate(per_symbol):
        for bar in bars:
            row = row_of[bar.date]
            # Each bar carries its own split/dividend factor, so a corporate
            # action inside a window scales that window's history rather than
            # printing as a return. The latest bar's factor is ~1, which is why
            # adjusted prices stay comparable with a raw entry reference.
            factor = bar.adj_close / bar.close if bar.close > 0 else 1.0
            open_[row, col] = bar.open * factor
            high[row, col] = bar.high * factor
            low[row, col] = bar.low * factor
            close[row, col] = bar.adj_close if bar.close > 0 else bar.close
            raw_close[row, col] = bar.close
    return MarketPanel(
        calendar=calendar,
        symbols=columns,
        open=open_,
        high=high,
        low=low,
        close=close,
        raw_close=raw_close,
    )


def load_commercial_edges(
    con: duckdb.DuckDBPyConnection, *, as_of: date
) -> tuple[tuple[str, str, str, date], ...]:
    """Commercial edges with ``report_date <= as_of``. Board interlocks excluded.

    Interlocks are a channel, not a signal: two firms sharing a director have a
    plausible information path but no measured return relationship. They are
    excluded from return views entirely and, per the build plan, from risk
    clustering too.

    Own query, ~15 lines. Do NOT import ``signals.dataset._validatable_edges``
    -- it is private and carries different filtering semantics.

    ``report_date`` is the MIN of the qualifying filings rather than a DISTINCT
    row: ``company_edges`` is keyed on ``(source_cik, target, edge_type)``, so
    one ticker-level relationship can hold several rows -- two source CIKs
    behind one ticker, two spellings of one target -- and selecting the date
    without grouping would turn one relationship into several. An undated edge
    is dropped: ``risk.commercial_clusters`` compares ``report_date <= as_of``
    per day, and an edge that cannot answer "when was this knowable" cannot be
    used point-in-time.
    """
    placeholders = ", ".join("?" * len(COMMERCIAL_EDGE_TYPES))
    rows = con.execute(
        f"""
        SELECT source_ticker, target_ticker, lower(edge_type), MIN(report_date)
        FROM company_edges
        WHERE resolution_status = 'resolved'
          AND source_ticker IS NOT NULL AND source_ticker <> ''
          AND target_ticker IS NOT NULL AND target_ticker <> ''
          AND source_ticker <> target_ticker
          AND lower(edge_type) IN ({placeholders})
          AND report_date IS NOT NULL
          AND report_date <= ?
        GROUP BY source_ticker, target_ticker, lower(edge_type)
        ORDER BY source_ticker, target_ticker, lower(edge_type)
        """,
        [*COMMERCIAL_EDGE_TYPES, as_of],
    ).fetchall()
    return tuple((str(r[0]), str(r[1]), str(r[2]), r[3]) for r in rows)


def load_feed(
    con: duckdb.DuckDBPyConnection,
    *,
    split: str | None = None,
    as_of: date | None = None,
    min_hedged_edge: float | None = None,
) -> FeedBundle:
    """The one bounded read. Wrap in ``with_db_retry`` (20 attempts, 45s apart).

    Read-only. Opens no write transaction, holds no lock beyond the query, and
    must be the only DB access the replay performs.

    ``as_of`` is the point-in-time bound on what the bundle is allowed to know:
    betas are fitted strictly before it, edges are reported at or before it.
    Left None the two fall back differently, because they are undated in
    different ways. A beta is a single number with no row-level date, so its
    fallback is the earliest signal date -- a hedge ratio can then never be
    fitted on the era it hedges even when the caller forgets to say so. An edge
    carries its own ``report_date`` and ``risk.commercial_clusters`` re-filters
    it every day, so its fallback is unbounded; bounding edges at the first
    signal date would starve every later day of graph structure it legitimately
    knew about. Signals and prices likewise carry their own dates and are read
    in full, with the per-day discipline enforced by the simulator through
    ``trailing_returns`` and each signal's ``available_on``.

    Tradeability is measured on discovery only -- as in ``load_replay_inputs``,
    so running the holdout cannot feed back into which cells the strategy holds.
    """
    admitted = load_admitted_cells(con)
    hedged_edges = load_hedged_edges(con, admitted, split="discovery")
    cells = filter_tradeable(
        admitted,
        hedged_edges,
        min_hedged_edge=DEFAULT_MIN_HEDGED_EDGE if min_hedged_edge is None else min_hedged_edge,
    )
    signals = load_signals(con, cells, split=split)

    signals_by_day: dict[date, list[BacktestSignal]] = defaultdict(list)
    for signal in signals:
        signals_by_day[signal.available_on].append(signal)
    beta_cutoff = as_of if as_of is not None else min(signals_by_day, default=date.max)
    edge_cutoff = as_of if as_of is not None else date.max

    symbols = sorted({signal.symbol for signal in signals})
    requested = symbols if BENCHMARK_SYMBOL in symbols else [*symbols, BENCHMARK_SYMBOL]
    bars = load_bars(con, requested)
    # The benchmark earns a column only if it has bars. An all-NaN SPY column
    # would turn every benchmark-relative metric into NaN instead of reporting
    # that the benchmark is missing.
    benchmark = {BENCHMARK_SYMBOL} if bars.get(BENCHMARK_SYMBOL) else set()
    panel_symbols = tuple(sorted(set(symbols) | benchmark))

    return FeedBundle(
        panel=build_panel(bars, panel_symbols),
        signals_by_day=dict(signals_by_day),
        cells=cells,
        hedged_edges=hedged_edges,
        betas=load_betas(con, symbols, before=beta_cutoff),
        edges=load_commercial_edges(con, as_of=edge_cutoff),
        spy_index=panel_symbols.index(BENCHMARK_SYMBOL) if benchmark else None,
    )


__all__ = [
    "BENCHMARK_SYMBOL",
    "COMMERCIAL_EDGE_TYPES",
    "FeedBundle",
    "MarketPanel",
    "build_panel",
    "load_commercial_edges",
    "load_feed",
]
