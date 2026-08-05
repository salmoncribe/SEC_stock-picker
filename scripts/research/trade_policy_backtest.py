"""Trade-level exit-policy backtest: win rate, win size, and what 30%/yr needs.

Answers the portfolio-management question independently of signal quality:
GIVEN a signal, which exit rule turns it into the most money, how often do we
win, and how big do the wins have to be to cover the losses?

Deliberately scoped to the pre-COVID window. Entries 2016-07-25 (first bar in
the panel) through 2019-11-29, with every open trade force-closed at
2020-02-19 -- the S&P's pre-crash peak -- so no trade in this study can
experience the COVID drawdown.

Every policy is measured twice: once on all signals, and once de-concentrated
to ONE TRADE PER TICKER (the first). The gap between those two columns is the
part of the result that belongs to the signal feed rather than to the exit
rule. See docs/plans/2026-08-04-trading-strategy.md section 0.

Returns are computed both raw and SPY-hedged. Costs are applied as a
round-trip spread charged against the raw return.

Usage:
    uv run --with duckdb --with pandas --with numpy --with pyarrow \
        python scripts/research/trade_policy_backtest.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

PRICES = "/Volumes/Extreme/home-migrated/quant/data/parquet/daily_prices/*/*.parquet"
SAMPLES = "/Volumes/Extreme/home-migrated/quant/data/parquet/event_samples/**/*.parquet"

ENTRY_START = "2016-07-25"
ENTRY_END = "2019-11-29"
HARD_EXIT = "2020-02-19"  # S&P 500 pre-COVID closing peak
MAX_HOLD = 252            # ceiling on any policy's hold, in trading sessions
ROUND_TRIP_BPS = 27.0     # measured point-in-time mean spread on this population
PARKING = ("JNJ", "PG", "KO")


# ---------------------------------------------------------------- data loading

def load() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    con = duckdb.connect()
    con.execute("set memory_limit='8GB'; set threads=6;")

    events = con.execute(f"""
        select distinct target_ticker as sym, t0
        from read_parquet('{SAMPLES}', union_by_name=true)
        where event_type='insider_transaction' and event_subtype='P'
          and horizon_days=60
          and t0 >= date '{ENTRY_START}' and t0 <= date '{ENTRY_END}'
        order by t0, sym
    """).df()

    syms = tuple(sorted(set(events.sym) | {"SPY", *PARKING}))
    px = con.execute(f"""
        select symbol, price_date, adj_close
        from read_parquet('{PRICES}', hive_partitioning=true)
        where symbol in {syms} and adj_close > 0
          and price_date <= date '{HARD_EXIT}'
        order by symbol, price_date
    """).df()

    closes: dict[str, np.ndarray] = {}
    dates: dict[str, np.ndarray] = {}
    for sym, g in px.groupby("symbol", sort=False):
        closes[sym] = g.adj_close.to_numpy(dtype=float)
        dates[sym] = g.price_date.to_numpy()
    return events, closes, dates


# ------------------------------------------------------------- exit policies

def simulate(prices: np.ndarray, policy: dict) -> tuple[int, float]:
    """Walk one trade forward. Returns (bars_held, gross_return).

    ``prices[0]`` is the entry bar. Policies are evaluated on the close of each
    subsequent bar, so an exit signalled on bar i fills at bar i's close --
    the same 'a stop is a trigger, not a ceiling' convention the repo uses
    elsewhere. No intrabar fills are claimed.
    """
    entry = prices[0]
    horizon = min(policy.get("horizon", MAX_HOLD), len(prices) - 1)
    if horizon < 1:
        return 0, 0.0

    trail = policy.get("trail")           # fractional drop from running peak
    hard = policy.get("hard_stop")        # fractional drop from entry
    arm_at = policy.get("arm_at", 0.0)    # trailing only arms above this gain

    peak = entry
    for i in range(1, horizon + 1):
        p = prices[i]
        if p > peak:
            peak = p
        if hard is not None and p <= entry * (1.0 - hard):
            return i, p / entry - 1.0
        if trail is not None and peak >= entry * (1.0 + arm_at):
            if p <= peak * (1.0 - trail):
                return i, p / entry - 1.0
    return horizon, prices[horizon] / entry - 1.0


POLICIES: dict[str, dict] = {
    # Baseline: the horizon the signal was measured at.
    "fixed 60d": {"horizon": 60},
    "fixed 120d": {"horizon": 120},
    # Michael's rule: ride it until it fizzles.
    "trail 5%": {"trail": 0.05},
    "trail 8%": {"trail": 0.08},
    "trail 10%": {"trail": 0.10},
    "trail 15%": {"trail": 0.15},
    "trail 20%": {"trail": 0.20},
    # Ride it, but don't ride it forever.
    "trail 10% cap 120d": {"trail": 0.10, "horizon": 120},
    "trail 15% cap 120d": {"trail": 0.15, "horizon": 120},
    # Cut losers hard, let winners run free.
    "stop 8% + run": {"hard_stop": 0.08},
    "stop 15% + run": {"hard_stop": 0.15},
    # Let it prove itself first, then trail it.
    "arm +10% trail 10%": {"trail": 0.10, "arm_at": 0.10},
    "arm +10% trail 15%": {"trail": 0.15, "arm_at": 0.10},
    # Belt and braces.
    "stop 15% + trail 15%": {"hard_stop": 0.15, "trail": 0.15},
}


# ----------------------------------------------------------------- evaluation

def run(events, closes, dates, dedupe: bool) -> pd.DataFrame:
    if dedupe:
        events = events.sort_values(["sym", "t0"]).groupby("sym", as_index=False).first()

    spy_c, spy_d = closes["SPY"], dates["SPY"]
    rows = []

    for name, policy in POLICIES.items():
        rets, hedged, bars = [], [], []
        for sym, t0 in zip(events.sym, events.t0):
            c, d = closes.get(sym), dates.get(sym)
            if c is None:
                continue
            j = int(np.searchsorted(d, np.datetime64(t0)))
            if j >= len(c) - 1:
                continue
            held, gross = simulate(c[j:], policy)
            if held == 0:
                continue
            # SPY over the identical calendar span.
            i0 = int(np.searchsorted(spy_d, d[j]))
            i1 = int(np.searchsorted(spy_d, d[min(j + held, len(d) - 1)]))
            i1 = min(i1, len(spy_c) - 1)
            bench = spy_c[i1] / spy_c[i0] - 1.0 if i0 < len(spy_c) else 0.0
            net = gross - ROUND_TRIP_BPS / 10_000.0
            rets.append(net)
            hedged.append(net - bench)
            bars.append(held)

        r = np.asarray(rets)
        h = np.asarray(hedged)
        b = np.asarray(bars)
        if not len(r):
            continue
        wins, losses = r[r > 0], r[r <= 0]
        win_rate = len(wins) / len(r)
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = -losses.mean() if len(losses) else 0.0
        expectancy = r.mean()
        hold = b.mean()
        # Each slot recycles 252/hold times a year and compounds.
        ann = (1.0 + expectancy) ** (252.0 / hold) - 1.0 if hold > 0 else 0.0
        rows.append({
            "policy": name,
            "n": len(r),
            "win%": 100 * win_rate,
            "avg_win%": 100 * avg_win,
            "avg_loss%": 100 * avg_loss,
            "W/L": (avg_win / avg_loss) if avg_loss else np.inf,
            "expect%": 100 * expectancy,
            "hedged%": 100 * h.mean(),
            "hold_d": hold,
            "trades/yr": 252.0 / hold if hold else 0.0,
            "ann%": 100 * ann,
        })
    return pd.DataFrame(rows)


def main() -> None:
    events, closes, dates = load()
    print(f"signals in window: {len(events):,} distinct (ticker, t0) pairs")
    print(f"distinct tickers:  {events.sym.nunique():,}")
    print(f"window: {ENTRY_START} -> {ENTRY_END}, hard exit {HARD_EXIT}")
    print(f"cost applied: {ROUND_TRIP_BPS:.0f}bps round trip\n")

    for dedupe in (False, True):
        tag = "ONE TRADE PER TICKER (de-concentrated)" if dedupe else "ALL SIGNALS (raw)"
        df = run(events, closes, dates, dedupe).sort_values("ann%", ascending=False)
        print(f"{'=' * 100}\n{tag}\n{'=' * 100}")
        print(df.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
        print()

    # Benchmarks over the identical window.
    print("=" * 100)
    print("BENCHMARKS over the same window")
    print("=" * 100)
    spy_c, spy_d = closes["SPY"], dates["SPY"]
    i0 = int(np.searchsorted(spy_d, np.datetime64(ENTRY_START)))
    yrs = (pd.Timestamp(HARD_EXIT) - pd.Timestamp(ENTRY_START)).days / 365.25
    tot = spy_c[-1] / spy_c[i0] - 1.0
    print(f"  SPY buy & hold:  {100 * tot:6.2f}% total, "
          f"{100 * ((1 + tot) ** (1 / yrs) - 1):5.2f}%/yr over {yrs:.2f} years")
    for sym in PARKING:
        c, d = closes[sym], dates[sym]
        k = int(np.searchsorted(d, np.datetime64(ENTRY_START)))
        t = c[-1] / c[k] - 1.0
        print(f"  {sym:<4} buy & hold: {100 * t:6.2f}% total, "
              f"{100 * ((1 + t) ** (1 / yrs) - 1):5.2f}%/yr")

    print("\n" + "=" * 100)
    print("WHAT 30%/yr REQUIRES  (per-trade expectancy, net of cost)")
    print("=" * 100)
    print(f"{'hold (days)':>12} {'trades/yr':>10} {'need for 20%':>14} "
          f"{'need for 30%':>14} {'need for 40%':>14}")
    for hold in (20, 40, 60, 90, 120, 180, 252):
        n = 252 / hold
        row = "".join(
            f"{100 * ((1 + tgt) ** (1 / n) - 1):>13.2f}%" for tgt in (0.20, 0.30, 0.40)
        )
        print(f"{hold:>12} {n:>10.2f} {row}")
    print("\nRead with the tables above: find your policy's hold_d, then check")
    print("whether its expect% clears the number in this table.")


if __name__ == "__main__":
    main()
