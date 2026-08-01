"""Turn the stored validation verdicts into replayable backtest signals.

The gate (``signals.impact``) decides *which* cells predict; the promotion
ladder (``signals.promotion``) records that decision in ``signal_status``. This
module reads that decision and nothing else: a cell the ladder has not admitted
cannot produce a trade here, so the backtest replays the strategy the system
would actually have fired rather than a fresh search over the same data.

Two rules keep the replay honest:

* **The expected move comes from the cell, not the sample.** ``mean_car`` on
  ``signal_status`` is the *discovery* mean. Using it to set a holdout trade's
  target is legitimate — it was knowable before the holdout era began. Reading
  each sample's own ``forward_abnormal_return`` would be reading the answer.
* **One position per symbol-day.** Insider filings arrive in clumps (six
  officers selling the same morning is six events), and several admitted cells
  can fire on the same name at once. Left alone, the clump would consume the
  whole position budget on one name and inflate the apparent breadth of the
  strategy. Conflicting directions on the same name-day are dropped outright
  rather than resolved by magnitude: when the evidence points both ways, no
  trade is the answer the gate's own logic implies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from market_intelligence.analytics.backtest import BacktestSignal
from market_intelligence.signals.trade_plan import Bar

if TYPE_CHECKING:
    import duckdb

#: Only the issuer's own edge is replayed. Propagation edges carry their own
#: cells and are not admitted yet; mixing them in would silently change what
#: strategy the reported numbers describe.
SELF_EDGE = "self"

#: A cell is tradeable while it is climbing or holding the promotion ladder.
#: ``retired`` is the ladder's way of saying the edge failed out of sample.
TRADEABLE_STATUSES = ("candidate", "active")


@dataclass(frozen=True)
class AdmittedCell:
    """One admitted (event type, subtype, edge, horizon) cell and its measured edge."""

    event_type: str
    event_subtype: str | None
    horizon_days: int
    direction: int
    mean_car: float
    hit_rate: float
    n_clusters: int

    @property
    def key(self) -> tuple[str, str | None, int]:
        return (self.event_type, self.event_subtype, self.horizon_days)

    @property
    def predicted_move(self) -> float:
        """Expected absolute move; direction is carried separately."""
        return abs(self.mean_car)


def load_admitted_cells(con: duckdb.DuckDBPyConnection) -> list[AdmittedCell]:
    """Cells the promotion ladder currently admits, strongest edge first."""
    placeholders = ", ".join("?" * len(TRADEABLE_STATUSES))
    rows = con.execute(
        f"""
        SELECT event_type, event_subtype, horizon_days, direction,
               mean_car, hit_rate, n_clusters
        FROM signal_status
        WHERE status IN ({placeholders})
          AND last_verdict = 'admitted'
          AND edge_type = ?
          AND mean_car IS NOT NULL
          AND direction IS NOT NULL
        ORDER BY abs(mean_car) DESC, event_type, event_subtype, horizon_days
        """,
        [*TRADEABLE_STATUSES, SELF_EDGE],
    ).fetchall()
    return [
        AdmittedCell(
            event_type=str(row[0]),
            event_subtype=row[1],
            horizon_days=int(row[2]),
            direction=int(row[3]),
            mean_car=float(row[4]),
            hit_rate=float(row[5]) if row[5] is not None else 0.0,
            n_clusters=int(row[6]) if row[6] is not None else 0,
        )
        for row in rows
    ]


def signals_from_rows(
    rows: list[tuple[str, str | None, int, str, date]], cells: list[AdmittedCell]
) -> list[BacktestSignal]:
    """Collapse ``(event_type, subtype, horizon, ticker, available_on)`` rows to signals.

    One signal per symbol-day, taking the largest expected move among the cells
    that fired. A symbol-day whose cells disagree on direction produces nothing.
    """
    by_key = {cell.key: cell for cell in cells}
    best: dict[tuple[str, date], AdmittedCell] = {}
    conflicted: set[tuple[str, date]] = set()

    for event_type, subtype, horizon, ticker, available_on in rows:
        cell = by_key.get((event_type, subtype, horizon))
        if cell is None:
            continue
        slot = (ticker, available_on)
        incumbent = best.get(slot)
        if incumbent is None:
            best[slot] = cell
            continue
        if incumbent.direction != cell.direction:
            conflicted.add(slot)
            continue
        if cell.predicted_move > incumbent.predicted_move:
            best[slot] = cell

    return sorted(
        (
            BacktestSignal(
                symbol=ticker,
                available_on=available_on,
                direction=cell.direction,
                predicted_move=cell.predicted_move,
                horizon_days=cell.horizon_days,
            )
            for (ticker, available_on), cell in best.items()
            if (ticker, available_on) not in conflicted
        ),
        key=lambda signal: (signal.available_on, signal.symbol),
    )


def load_signals(
    con: duckdb.DuckDBPyConnection,
    cells: list[AdmittedCell],
    *,
    split: str | None = None,
) -> list[BacktestSignal]:
    """Every admitted-cell firing in ``split`` (or all splits), as signals."""
    if not cells:
        return []
    clauses = ["edge_id = ?", "available_on IS NOT NULL"]
    params: list[object] = [SELF_EDGE]
    if split is not None:
        clauses.append("split = ?")
        params.append(split)

    cell_sql = " OR ".join(
        "(event_type = ? AND event_subtype IS NOT DISTINCT FROM ? AND horizon_days = ?)"
        for _ in cells
    )
    for cell in cells:
        params.extend([cell.event_type, cell.event_subtype, cell.horizon_days])

    rows = con.execute(
        f"""
        SELECT DISTINCT event_type, event_subtype, horizon_days, target_ticker, available_on
        FROM event_samples
        WHERE {" AND ".join(clauses)} AND ({cell_sql})
        """,
        params,
    ).fetchall()
    return signals_from_rows(
        [(str(r[0]), r[1], int(r[2]), str(r[3]), r[4]) for r in rows], cells
    )


#: A cell has to clear the cost of trading it, not merely print a positive
#: number. Round-trip slippage at the replay's default 5 bps a side is 10 bps;
#: 15 bps leaves a little room for the hedge leg.
DEFAULT_MIN_HEDGED_EDGE = 0.0015


def load_hedged_edges(
    con: duckdb.DuckDBPyConnection,
    cells: list[AdmittedCell],
    *,
    split: str = "discovery",
) -> dict[tuple[str, str | None, int], float]:
    """Mean signed return per cell after hedging beta but *not* alpha.

    ``analytics.returns`` defines
    ``abnormal_return = total_return - (alpha + beta * market_return)``. Shorting
    an index removes the ``beta * market_return`` term; nothing removes ``alpha``,
    which is the name's own trailing drift over the window. So a cell's *label*
    edge can be large and entirely made of subtracted alpha -- insiders sell into
    strength, and the sale cells here do exactly that.

    This measures what is left once only the hedgeable term is hedged: the mean
    of ``direction * sum(total_return - beta * market_return)`` across the cell's
    samples. Computed on discovery, it is the honest answer to "would trading
    this cell have made money", and it is knowable before the holdout begins.
    """
    if not cells:
        return {}
    cell_sql = " OR ".join(
        "(es.event_type = ? AND es.event_subtype IS NOT DISTINCT FROM ? "
        "AND es.horizon_days = ?)"
        for _ in cells
    )
    params: list[object] = [SELF_EDGE, split]
    for cell in cells:
        params.extend([cell.event_type, cell.event_subtype, cell.horizon_days])

    rows = con.execute(
        f"""
        WITH matched AS (
            SELECT es.sample_id, es.event_type, es.event_subtype, es.horizon_days,
                   ss.direction, es.target_ticker, es.t0, es.window_end
            FROM event_samples es
            JOIN signal_status ss
              ON ss.event_type = es.event_type
             AND ss.event_subtype IS NOT DISTINCT FROM es.event_subtype
             AND ss.horizon_days = es.horizon_days
             AND ss.edge_type = es.edge_id
            WHERE es.edge_id = ? AND es.split = ?
              AND es.forward_abnormal_return IS NOT NULL
              AND ({cell_sql})
        ),
        per_sample AS (
            SELECT m.event_type, m.event_subtype, m.horizon_days, m.sample_id,
                   m.direction * sum(
                       dr.total_return - coalesce(dr.beta, 1.0) * dr.market_return
                   ) AS hedged
            FROM matched m
            JOIN daily_returns dr
              ON dr.symbol = m.target_ticker
             AND dr.price_date BETWEEN m.t0 AND m.window_end
             AND dr.total_return IS NOT NULL
             AND dr.market_return IS NOT NULL
            GROUP BY 1, 2, 3, 4, m.direction
        )
        SELECT event_type, event_subtype, horizon_days, avg(hedged)
        FROM per_sample
        GROUP BY 1, 2, 3
        """,
        params,
    ).fetchall()
    return {(str(r[0]), r[1], int(r[2])): float(r[3]) for r in rows if r[3] is not None}


def filter_tradeable(
    cells: list[AdmittedCell],
    hedged_edges: dict[tuple[str, str | None, int], float],
    *,
    min_hedged_edge: float = DEFAULT_MIN_HEDGED_EDGE,
) -> list[AdmittedCell]:
    """Keep only cells whose edge survives beta-hedging by a tradeable margin.

    A cell with no measured hedged edge is dropped rather than trusted: absence
    of the measurement is not evidence the edge is real.
    """
    return [
        cell
        for cell in cells
        if hedged_edges.get(cell.key) is not None
        and hedged_edges[cell.key] >= min_hedged_edge
    ]


def load_betas(
    con: duckdb.DuckDBPyConnection, symbols: list[str], *, before: date
) -> dict[str, float]:
    """Median market-model beta per symbol, fitted strictly before ``before``.

    The hedge ratio has to be knowable at decision time. Taking the median over
    the discovery era and holding it fixed does that for the holdout run, and
    keeps the ratio from chasing a single noisy daily fit. Symbols with no fitted
    beta are omitted so the caller's own fallback applies, rather than being
    handed a zero -- which would read as "needs no hedge".
    """
    if not symbols:
        return {}
    placeholders = ", ".join("?" * len(symbols))
    rows = con.execute(
        f"""
        SELECT symbol, median(beta)
        FROM daily_returns
        WHERE symbol IN ({placeholders})
          AND price_date < ?
          AND beta IS NOT NULL AND isfinite(beta)
        GROUP BY symbol
        """,
        [*symbols, before],
    ).fetchall()
    return {str(row[0]): float(row[1]) for row in rows if row[1] is not None}


def load_bars(
    con: duckdb.DuckDBPyConnection, symbols: list[str]
) -> dict[str, list[Bar]]:
    """Daily bars per symbol, oldest first, for the symbols the signals touch."""
    if not symbols:
        return {}
    placeholders = ", ".join("?" * len(symbols))
    rows = con.execute(
        f"""
        SELECT symbol, price_date, open, high, low, close, adj_close
        FROM daily_prices
        WHERE symbol IN ({placeholders})
          AND close IS NOT NULL AND close > 0
        ORDER BY symbol, price_date
        """,
        list(symbols),
    ).fetchall()
    bars: dict[str, list[Bar]] = defaultdict(list)
    for symbol, price_date, open_, high, low, close, adj_close in rows:
        bars[str(symbol)].append(
            Bar(
                date=price_date,
                open=float(open_ if open_ is not None else close),
                high=float(high if high is not None else close),
                low=float(low if low is not None else close),
                close=float(close),
                adj_close=float(adj_close if adj_close is not None else close),
            )
        )
    return dict(bars)


@dataclass(frozen=True)
class ReplayInputs:
    """Everything one replay needs, assembled without consulting the holdout."""

    cells: list[AdmittedCell]
    hedged_edges: dict[tuple[str, str | None, int], float]
    dropped: list[AdmittedCell]
    signals: dict[str, list[BacktestSignal]]
    bars: dict[str, list[Bar]]
    betas: dict[str, float]


def load_replay_inputs(
    con: duckdb.DuckDBPyConnection,
    *,
    hedge_symbol: str | None,
    split_date: date,
    tradeable_only: bool = True,
    min_hedged_edge: float = DEFAULT_MIN_HEDGED_EDGE,
    splits: tuple[str, ...] = ("discovery", "holdout"),
) -> ReplayInputs:
    """Assemble cells, signals, bars and hedge ratios for a replay.

    Both discretionary choices -- which cells are tradeable, and each symbol's
    hedge ratio -- are measured on discovery only, so running the holdout does
    not feed anything back into how the strategy was specified.
    """
    admitted = load_admitted_cells(con)
    hedged_edges = load_hedged_edges(con, admitted, split="discovery") if admitted else {}
    cells = (
        filter_tradeable(admitted, hedged_edges, min_hedged_edge=min_hedged_edge)
        if tradeable_only
        else admitted
    )
    signals = {split: load_signals(con, cells, split=split) for split in splits}
    symbols = sorted({signal.symbol for group in signals.values() for signal in group})
    bar_symbols = [*symbols, hedge_symbol] if hedge_symbol else list(symbols)
    return ReplayInputs(
        cells=cells,
        hedged_edges=hedged_edges,
        dropped=[cell for cell in admitted if cell not in cells],
        signals=signals,
        bars=load_bars(con, bar_symbols),
        betas=load_betas(con, symbols, before=split_date),
    )


__all__ = [
    "DEFAULT_MIN_HEDGED_EDGE",
    "SELF_EDGE",
    "TRADEABLE_STATUSES",
    "AdmittedCell",
    "ReplayInputs",
    "filter_tradeable",
    "load_admitted_cells",
    "load_bars",
    "load_betas",
    "load_hedged_edges",
    "load_replay_inputs",
    "load_signals",
    "signals_from_rows",
]
