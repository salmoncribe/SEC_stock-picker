"""Does the SIGNAL add anything, or is the exit rule doing all the work?

Runs the identical account simulation three ways:

  1. REAL     -- slots filled from insider-purchase signals
  2. PLACEBO  -- slots filled from random tickers drawn the same days, same
                 count, same exit rule
  3. SPY      -- buy and hold

If (1) and (2) land in the same place, the trading rules are the whole result
and the signal contributes nothing -- which is worth knowing before any money
moves. Also regresses the daily equity curve on SPY to split the return into
beta (market) and alpha (skill), because a 17% CAGR in a window where SPY did
15.5% is mostly not skill.

Also reports slot utilisation, which decides whether the "park idle cash in a
dividend stock" rule can matter at all.

Usage:
    uv run --with duckdb --with pandas --with numpy --with pyarrow \
        python scripts/research/placebo_and_alpha.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

PRICES = "/Volumes/Extreme/home-migrated/quant/data/parquet/daily_prices/*/*.parquet"
SAMPLES = "/Volumes/Extreme/home-migrated/quant/data/parquet/event_samples/**/*.parquet"
START, ENTRY_END, HARD_EXIT = "2016-07-25", "2019-11-29", "2020-02-19"
COST = 27.0 / 10_000.0
SLOTS = 40
SEEDS = range(12)
E0 = 2_000.0
POLICY = {"trail": 0.15}          # the best account-level policy from the sweep
POLICY_NAME = "trail 15%"


def load():
    con = duckdb.connect()
    con.execute("set memory_limit='8GB'; set threads=6;")
    ev = con.execute(f"""
        select distinct target_ticker as sym, t0
        from read_parquet('{SAMPLES}', union_by_name=true)
        where event_type='insider_transaction' and event_subtype='P'
          and horizon_days=60 and t0 >= date '{START}' and t0 <= date '{ENTRY_END}'
    """).df()
    # Placebo universe: everything liquid enough to have been tradeable.
    univ = con.execute(f"""
        select symbol from read_parquet('{PRICES}', hive_partitioning=true)
        where price_date between date '{START}' and date '{HARD_EXIT}'
          and adj_close > 5 and volume * close > 5000000
        group by symbol having count(*) > 600
    """).df().symbol.tolist()
    syms = tuple(sorted(set(ev.sym) | set(univ) | {"SPY"}))
    px = con.execute(f"""
        select symbol, price_date, adj_close
        from read_parquet('{PRICES}', hive_partitioning=true)
        where symbol in {syms} and adj_close > 0 and price_date <= date '{HARD_EXIT}'
    """).df()
    px["price_date"] = pd.to_datetime(px.price_date)
    cal = np.sort(px.price_date.unique()).astype("datetime64[ns]")
    wide = px.pivot_table(index="price_date", columns="symbol",
                          values="adj_close").reindex(pd.DatetimeIndex(cal)).ffill()
    ev["t0"] = pd.to_datetime(ev.t0)
    return ev, wide, cal, [u for u in univ if u in wide.columns]


def simulate(by_day, px_map, cal, seed, universe=None):
    """One account run. If ``universe`` is given, candidates are drawn from it
    at random instead of from the signal list (the placebo)."""
    rng = np.random.default_rng(seed)
    cash, open_pos = E0, []
    curve, filled = np.zeros(len(cal)), np.zeros(len(cal))
    trail = POLICY["trail"]

    for i in range(len(cal)):
        still = []
        for p in open_pos:
            price = px_map[p["sym"]][i]
            if not np.isfinite(price):
                still.append(p); continue
            p["peak"] = max(p["peak"], price)
            if price <= p["peak"] * (1 - trail) or i == len(cal) - 1:
                cash += p["units"] * price * (1 - COST)
            else:
                still.append(p)
        open_pos = still

        free = SLOTS - len(open_pos)
        n_today = len(by_day.get(i, []))
        if free > 0 and n_today:
            held = {p["sym"] for p in open_pos}
            if universe is None:
                cands = [c for c in by_day[i] if c in px_map and c not in held]
            else:
                cands = [c for c in rng.choice(universe, size=min(n_today * 3,
                         len(universe)), replace=False) if c not in held]
                cands = cands[:n_today]
            rng.shuffle(cands)
            equity = cash + sum(p["units"] * px_map[p["sym"]][i] for p in open_pos
                                if np.isfinite(px_map[p["sym"]][i]))
            target = equity / SLOTS
            for sym in cands[:free]:
                price = px_map[sym][i]
                if not np.isfinite(price) or price <= 0:
                    continue
                spend = min(target, cash)
                if spend < 1.0:
                    break
                cash -= spend
                open_pos.append({"sym": sym, "i0": i, "entry": price,
                                 "peak": price, "units": spend * (1 - COST) / price})

        eq = cash + sum(p["units"] * (px_map[p["sym"]][i]
                        if np.isfinite(px_map[p["sym"]][i]) else p["entry"])
                        for p in open_pos)
        curve[i], filled[i] = eq, len(open_pos)
    return curve, filled


def stats(curve, spy, cal):
    yrs = (cal[-1] - cal[0]).astype("timedelta64[D]").astype(int) / 365.25
    cagr = (curve[-1] / curve[0]) ** (1 / yrs) - 1
    dd = float(np.min(curve / np.maximum.accumulate(curve) - 1))
    r = np.diff(curve) / curve[:-1]
    b = np.diff(spy) / spy[:-1]
    beta = np.cov(r, b)[0, 1] / np.var(b)
    alpha = (r.mean() - beta * b.mean()) * 252
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() else 0.0
    return cagr, dd, beta, alpha, sharpe, r.std() * np.sqrt(252)


def main() -> None:
    ev, wide, cal, universe = load()
    px_map = {s: wide[s].to_numpy() for s in wide.columns}
    spy = px_map["SPY"]
    by_day: dict[int, list[str]] = {}
    for sym, t0 in zip(ev.sym, ev.t0):
        i = int(np.searchsorted(cal, np.datetime64(t0)))
        if i + 1 < len(cal):
            by_day.setdefault(i + 1, []).append(sym)

    print(f"policy: {POLICY_NAME}, {SLOTS} slots, {len(SEEDS)} seeds")
    print(f"placebo universe: {len(universe)} liquid tickers")
    yrs = (cal[-1] - cal[0]).astype("timedelta64[D]").astype(int) / 365.25
    print(f"window {str(cal[0])[:10]} -> {str(cal[-1])[:10]} ({yrs:.2f} yrs)\n")

    out = {}
    for label, univ in (("REAL signals", None), ("PLACEBO random", universe)):
        res = [simulate(by_day, px_map, cal, s, univ) for s in SEEDS]
        st = np.array([stats(c, spy, cal) for c, _ in res])
        util = np.mean([f.mean() for _, f in res]) / SLOTS
        out[label] = (st, util)

    print(f"{'run':<16} {'CAGR':>18} {'maxDD':>8} {'beta':>6} {'ALPHA':>8} "
          f"{'Sharpe':>7} {'vol':>7} {'slots full':>11}")
    print("-" * 92)
    for label, (st, util) in out.items():
        med = np.median(st, axis=0)
        lo, hi = np.percentile(st[:, 0], 10), np.percentile(st[:, 0], 90)
        print(f"{label:<16} {100*med[0]:>6.2f}% [{100*lo:>5.1f},{100*hi:>5.1f}] "
              f"{100*med[1]:>7.1f}% {med[2]:>6.2f} {100*med[3]:>7.2f}% "
              f"{med[4]:>7.2f} {100*med[5]:>6.1f}% {100*util:>10.1f}%")
    sc, sd = (spy[-1]/spy[0])**(1/yrs)-1, float(np.min(spy/np.maximum.accumulate(spy)-1))
    sr = np.diff(spy)/spy[:-1]
    print(f"{'SPY buy&hold':<16} {100*sc:>6.2f}%                "
          f"{100*sd:>7.1f}% {1.00:>6.2f} {0.0:>7.2f}% "
          f"{sr.mean()/sr.std()*np.sqrt(252):>7.2f} {100*sr.std()*np.sqrt(252):>6.1f}%")

    real, plac = np.median(out["REAL signals"][0], axis=0), np.median(out["PLACEBO random"][0], axis=0)
    print(f"\nSignal contribution over placebo: "
          f"{100*(real[0]-plac[0]):+.2f}pp CAGR, {100*(real[3]-plac[3]):+.2f}pp alpha")
    a = out["REAL signals"][0][:, 3]
    print(f"Real-run alpha across seeds: median {100*np.median(a):+.2f}%/yr, "
          f"range [{100*a.min():+.2f}, {100*a.max():+.2f}]")

    # ---- what leverage would be needed to reach the 20-30% target
    curves = [simulate(by_day, px_map, cal, s, None)[0] for s in SEEDS]
    print("\n" + "=" * 92)
    print("LEVERAGE REQUIRED FOR THE TARGET  (margin at 6.5%/yr on the borrowed part)")
    print("=" * 92)
    print(f"{'leverage':>9} {'CAGR':>8} {'maxDD':>9} {'vol':>7} {'Sharpe':>7}  note")
    for lev in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        cg, dds, vols, shs = [], [], [], []
        for c in curves:
            r = np.diff(c) / c[:-1]
            lr = lev * r - (lev - 1.0) * 0.065 / 252.0   # borrow cost, daily
            eq = np.cumprod(1.0 + lr)
            if eq[-1] <= 0:
                cg.append(-1.0); dds.append(-1.0); vols.append(np.nan); shs.append(np.nan)
                continue
            cg.append(eq[-1] ** (1 / yrs) - 1)
            dds.append(float(np.min(eq / np.maximum.accumulate(eq) - 1)))
            vols.append(lr.std() * np.sqrt(252))
            shs.append(lr.mean() / lr.std() * np.sqrt(252) if lr.std() else 0.0)
        note = ""
        m = np.median(cg)
        if m >= 0.30:
            note = "<-- hits the 30% goal"
        elif m >= 0.20:
            note = "<-- clears the 20% floor"
        print(f"{lev:>8.2f}x {100*m:>7.2f}% {100*np.median(dds):>8.1f}% "
              f"{100*np.median(vols):>6.1f}% {np.median(shs):>7.2f}  {note}")
    print("\nLeverage scales the drawdown as fast as the return. The window above")
    print("contains NO bear market -- the -17% maxDD is a bull-market number.")


if __name__ == "__main__":
    main()
