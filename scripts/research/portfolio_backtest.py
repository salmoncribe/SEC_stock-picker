"""Account-level backtest of Michael's trading rules, pre-COVID window.

The per-trade study (trade_policy_backtest.py) annualizes expectancy as
(1+E)^(252/hold), which silently assumes a fresh trade is always waiting the
moment a slot frees. It never is. This simulates the actual account:

  * N equal-weight slots
  * a slot with no qualifying signal is PARKED in a dividend basket
    (equal-weight JNJ/PG/KO -- adj_close carries the dividends)
  * exits by trailing stop / fixed horizon / hard stop, per policy
  * 27bps round-trip cost on every equity leg, parking included
  * daily marking, so drawdown is real rather than trade-sampled

Deliberately pre-COVID: entries 2016-07-25..2019-11-29, everything liquidated
2020-02-19 at the S&P's pre-crash peak.

More signals arrive some days than there are free slots, so slot assignment is
seeded and the whole run is repeated across seeds. A strategy whose result
depends on WHICH of the same-day candidates it happened to pick is not a
strategy -- that exact bug (DuckDB row order) swung a prior backtest from
$25,559 to $66,109 on identical inputs.

Usage:
    uv run --with duckdb --with pandas --with numpy --with pyarrow \
        python scripts/research/portfolio_backtest.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

PRICES = "/Volumes/Extreme/home-migrated/quant/data/parquet/daily_prices/*/*.parquet"
SAMPLES = "/Volumes/Extreme/home-migrated/quant/data/parquet/event_samples/**/*.parquet"

START, ENTRY_END, HARD_EXIT = "2016-07-25", "2019-11-29", "2020-02-19"
COST = 27.0 / 10_000.0
PARKING = ("JNJ", "PG", "KO")
SEEDS = range(12)
START_EQUITY = 2_000.0

POLICIES = {
    "fixed 60d":           {"horizon": 60},
    "fixed 120d":          {"horizon": 120},
    "trail 8%":            {"trail": 0.08},
    "trail 10%":           {"trail": 0.10},
    "trail 10% cap 120d":  {"trail": 0.10, "horizon": 120},
    "trail 15%":           {"trail": 0.15},
    "trail 20%":           {"trail": 0.20},
    "stop 15% + run":      {"hard_stop": 0.15},
    "arm +10% trail 10%":  {"trail": 0.10, "arm_at": 0.10},
}


def load():
    con = duckdb.connect()
    con.execute("set memory_limit='8GB'; set threads=6;")
    ev = con.execute(f"""
        select distinct target_ticker as sym, t0
        from read_parquet('{SAMPLES}', union_by_name=true)
        where event_type='insider_transaction' and event_subtype='P'
          and horizon_days=60
          and t0 >= date '{START}' and t0 <= date '{ENTRY_END}'
    """).df()
    syms = tuple(sorted(set(ev.sym) | {"SPY", *PARKING}))
    px = con.execute(f"""
        select symbol, price_date, adj_close
        from read_parquet('{PRICES}', hive_partitioning=true)
        where symbol in {syms} and adj_close > 0 and price_date <= date '{HARD_EXIT}'
        order by price_date
    """).df()
    px["price_date"] = pd.to_datetime(px.price_date)
    cal = np.sort(px.price_date.unique()).astype("datetime64[ns]")
    wide = px.pivot_table(index="price_date", columns="symbol", values="adj_close")
    wide = wide.reindex(pd.DatetimeIndex(cal)).ffill()
    ev["t0"] = pd.to_datetime(ev.t0)
    return ev, wide, cal


def parking_index(wide: pd.DataFrame, cal) -> np.ndarray:
    """Equal-weight, daily-rebalanced JNJ/PG/KO total-return index."""
    cols = [c for c in PARKING if c in wide.columns]
    rets = wide[cols].pct_change().fillna(0.0).mean(axis=1).to_numpy()
    return np.cumprod(1.0 + rets)


def run_one(ev, wide, cal, policy: dict, slots: int, seed: int, park: bool) -> dict:
    rng = np.random.default_rng(seed)
    dt_index = {d: i for i, d in enumerate(cal)}
    park_idx = parking_index(wide, cal)

    # signals keyed by the session they become actionable (t0 + 1)
    by_day: dict[int, list[str]] = {}
    for sym, t0 in zip(ev.sym, ev.t0):
        i = int(np.searchsorted(cal, np.datetime64(t0)))
        if i + 1 < len(cal):
            by_day.setdefault(i + 1, []).append(sym)

    px = {s: wide[s].to_numpy() for s in wide.columns}
    cash = START_EQUITY
    open_pos: list[dict] = []
    parked = 0.0            # dollars in the parking basket
    parked_units = 0.0
    curve = np.zeros(len(cal))

    horizon = policy.get("horizon", 10_000)
    trail, hard, arm = policy.get("trail"), policy.get("hard_stop"), policy.get("arm_at", 0.0)

    for i in range(len(cal)):
        # ---- mark and evaluate exits
        still: list[dict] = []
        for p in open_pos:
            price = px[p["sym"]][i]
            if not np.isfinite(price):
                still.append(p)
                continue
            p["peak"] = max(p["peak"], price)
            held = i - p["i0"]
            exit_now = held >= horizon or i == len(cal) - 1
            if hard is not None and price <= p["entry"] * (1 - hard):
                exit_now = True
            if trail is not None and p["peak"] >= p["entry"] * (1 + arm):
                if price <= p["peak"] * (1 - trail):
                    exit_now = True
            if exit_now:
                cash += p["units"] * price * (1 - COST)
            else:
                still.append(p)
        open_pos = still

        # ---- open new positions into free slots
        free = slots - len(open_pos)
        cands = by_day.get(i, [])
        if free > 0 and cands:
            held_syms = {p["sym"] for p in open_pos}
            cands = [c for c in cands if c in px and c not in held_syms]
            rng.shuffle(cands)
            take = cands[:free]
            if take:
                # unpark what we need
                equity_now = cash + parked_units * park_idx[i] + sum(
                    p["units"] * px[p["sym"]][i] for p in open_pos
                    if np.isfinite(px[p["sym"]][i])
                )
                target = equity_now / slots
                for sym in take:
                    price = px[sym][i]
                    if not np.isfinite(price) or price <= 0:
                        continue
                    need = min(target, cash + parked_units * park_idx[i])
                    if need < 1.0:
                        break
                    if need > cash:                       # liquidate parking
                        short = need - cash
                        units = short / park_idx[i]
                        units = min(units, parked_units)
                        parked_units -= units
                        cash += units * park_idx[i] * (1 - COST)
                    spend = min(need, cash)
                    if spend < 1.0:
                        break
                    cash -= spend
                    open_pos.append({
                        "sym": sym, "i0": i, "entry": price,
                        "peak": price, "units": spend * (1 - COST) / price,
                    })

        # ---- park idle cash
        if park and cash > 1.0:
            parked_units += cash * (1 - COST) / park_idx[i]
            cash = 0.0

        equity = cash + parked_units * park_idx[i]
        for p in open_pos:
            price = px[p["sym"]][i]
            equity += p["units"] * (price if np.isfinite(price) else p["entry"])
        curve[i] = equity

    yrs = (cal[-1] - cal[0]).astype("timedelta64[D]").astype(int) / 365.25
    cagr = (curve[-1] / START_EQUITY) ** (1 / yrs) - 1
    dd = float(np.min(curve / np.maximum.accumulate(curve) - 1))
    daily = np.diff(curve) / curve[:-1]
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    return {"final": curve[-1], "cagr": cagr, "maxdd": dd, "sharpe": sharpe,
            "vol": daily.std() * np.sqrt(252)}


def main() -> None:
    ev, wide, cal = load()
    print(f"sessions {len(cal)}  signals {len(ev):,}  tickers {ev.sym.nunique()}")
    yrs = (cal[-1] - cal[0]).astype("timedelta64[D]").astype(int) / 365.25
    spy = wide["SPY"].to_numpy()
    print(f"window {cal[0]} -> {cal[-1]}  ({yrs:.2f} yrs)   "
          f"SPY {100*((spy[-1]/spy[0])**(1/yrs)-1):.2f}%/yr")
    pk = parking_index(wide, cal)
    print(f"parking basket (JNJ/PG/KO) {100*((pk[-1]/pk[0])**(1/yrs)-1):.2f}%/yr\n")

    for slots in (20, 40):
        print("=" * 98)
        print(f"{slots} SLOTS   (median of {len(list(SEEDS))} seeds; spread shows "
              f"sensitivity to same-day pick order)")
        print("=" * 98)
        print(f"{'policy':<22} {'park':<6} {'CAGR':>16} {'maxDD':>8} {'vol':>7} "
              f"{'Sharpe':>7} {'final $':>10}")
        rows = []
        for name, pol in POLICIES.items():
            for park in (True, False):
                res = [run_one(ev, wide, cal, pol, slots, s, park) for s in SEEDS]
                c = np.array([r["cagr"] for r in res])
                rows.append((name, park, np.median(c), c.min(), c.max(),
                             np.median([r["maxdd"] for r in res]),
                             np.median([r["vol"] for r in res]),
                             np.median([r["sharpe"] for r in res]),
                             np.median([r["final"] for r in res])))
        for name, park, med, lo, hi, dd, vol, sh, fin in sorted(
            rows, key=lambda r: -r[2]
        ):
            print(f"{name:<22} {'yes' if park else 'no':<6} "
                  f"{100*med:>6.2f}% [{100*lo:>5.1f},{100*hi:>5.1f}] "
                  f"{100*dd:>7.1f}% {100*vol:>6.1f}% {sh:>7.2f} {fin:>10,.0f}")
        print()


if __name__ == "__main__":
    main()
