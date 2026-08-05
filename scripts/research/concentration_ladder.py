"""Concentration: does trading FEW, BIG positions work better?

Michael trades ~6 names at ~75% of account. The 14-config sweep found that
`slots 10` was the only configuration with POSITIVE out-of-sample alpha, so
concentration is the one lever that trended better in validation rather than
worse. This tests the full ladder properly.

The headline metric is NOT the median. With no ability to rank which signal is
best, a 6-slot book's outcome is dominated by WHICH six names you happened to
draw. So every config reports the full seed distribution: p10, median, p90,
worst seed, and the share of seeds that beat SPY. A strategy whose median beats
the index but whose 10th percentile is catastrophic is a lottery ticket, not an
edge.

24 seeds (up from 8) because dispersion is the whole point here.

Usage:
    uv run --with duckdb --with pandas --with numpy --with pyarrow \
        python scripts/research/concentration_ladder.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

PRICES = "/Volumes/Extreme/home-migrated/quant/data/parquet/daily_prices/*/*.parquet"
SAMPLES = "/Volumes/Extreme/home-migrated/quant/data/parquet/event_samples/**/*.parquet"
COST = 27.0 / 10_000.0
E0 = 10_000.0
SEEDS = range(24)
TRAIL = 0.15

FIT = ("2016-07-25", "2019-12-31")
VAL = ("2020-01-01", "2026-07-31")

LADDER = [3, 4, 5, 6, 8, 10, 15, 20, 40]
DEPLOY = [1.00, 0.75]      # fraction of equity allowed into positions


def load():
    con = duckdb.connect()
    con.execute("set memory_limit='8GB'; set threads=6;")
    ev = con.execute(f"""
        select distinct target_ticker as sym, t0
        from read_parquet('{SAMPLES}', union_by_name=true)
        where event_type='insider_transaction'
          and event_subtype in ('P','C') and horizon_days=60
    """).df()
    syms = tuple(sorted(set(ev.sym) | {"SPY"}))
    px = con.execute(f"""
        select symbol, price_date, adj_close
        from read_parquet('{PRICES}', hive_partitioning=true)
        where symbol in {syms} and adj_close > 0
    """).df()
    px["price_date"] = pd.to_datetime(px.price_date)
    cal = np.sort(px.price_date.unique()).astype("datetime64[ns]")
    wide = px.pivot_table(index="price_date", columns="symbol",
                          values="adj_close").reindex(pd.DatetimeIndex(cal)).ffill()
    ev["t0"] = pd.to_datetime(ev.t0)
    return ev, wide, cal


def simulate(by_day, px, cal, slots, deploy, seed, i_lo, i_hi):
    rng = np.random.default_rng(seed)
    cash, pos = E0, []
    curve = np.zeros(i_hi - i_lo + 1)
    for k, i in enumerate(range(i_lo, i_hi + 1)):
        keep = []
        for p in pos:
            price = px[p["sym"]][i]
            if not np.isfinite(price):
                keep.append(p); continue
            p["peak"] = max(p["peak"], price)
            if price <= p["peak"] * (1 - TRAIL) or i == i_hi:
                cash += p["units"] * price * (1 - COST)
            else:
                keep.append(p)
        pos = keep

        free = slots - len(pos)
        if free > 0:
            held = {q["sym"] for q in pos}
            cands = [c for c in by_day.get(i, []) if c in px and c not in held]
            rng.shuffle(cands)
            equity = cash + sum(q["units"] * px[q["sym"]][i] for q in pos
                                if np.isfinite(px[q["sym"]][i]))
            unit = equity * deploy / slots
            for sym in cands[:free]:
                price = px[sym][i]
                if not np.isfinite(price) or price <= 0:
                    continue
                spend = min(unit, cash)
                if spend < 1.0:
                    break
                cash -= spend
                pos.append({"sym": sym, "peak": price,
                            "units": spend * (1 - COST) / price})

        curve[k] = cash + sum(p["units"] * (px[p["sym"]][i]
                              if np.isfinite(px[p["sym"]][i]) else 0.0) for p in pos)
    return curve


def run(title, ev, wide, cal, window):
    lo, hi = window
    i_lo = int(np.searchsorted(cal, np.datetime64(lo)))
    i_hi = int(np.searchsorted(cal, np.datetime64(hi)))
    px = {s: wide[s].to_numpy() for s in wide.columns}
    spy = px["SPY"][i_lo:i_hi + 1]
    yrs = (cal[i_hi] - cal[i_lo]).astype("timedelta64[D]").astype(int) / 365.25
    spy_cagr = (spy[-1] / spy[0]) ** (1 / yrs) - 1
    spy_dd = float(np.min(spy / np.maximum.accumulate(spy) - 1))

    by_day: dict[int, list[str]] = {}
    for sym, t0 in zip(ev.sym, ev.t0):
        i = int(np.searchsorted(cal, np.datetime64(t0)))
        if i + 1 < len(cal):
            by_day.setdefault(i + 1, []).append(sym)

    print(f"\n{'=' * 104}")
    print(f"{title}   {lo} -> {hi} ({yrs:.2f} yrs)   "
          f"SPY {100*spy_cagr:.2f}%/yr, maxDD {100*spy_dd:.1f}%")
    print("=" * 104)
    print(f"{'slots':>6} {'deploy':>7} │ {'p10':>8} {'MEDIAN':>9} {'p90':>8} "
          f"{'WORST':>9} │ {'medDD':>8} {'worstDD':>9} │ {'beat SPY':>9}")
    print("-" * 104)
    rows = {}
    for slots in LADDER:
        for dep in DEPLOY:
            cs, dds = [], []
            for s in SEEDS:
                c = simulate(by_day, px, cal, slots, dep, s, i_lo, i_hi)
                if c[-1] <= 0:
                    cs.append(-1.0); dds.append(-1.0); continue
                cs.append((c[-1] / c[0]) ** (1 / yrs) - 1)
                dds.append(float(np.min(c / np.maximum.accumulate(c) - 1)))
            cs, dds = np.array(cs), np.array(dds)
            beat = 100.0 * np.mean(cs > spy_cagr)
            rows[(slots, dep)] = (np.median(cs), cs.min(), beat)
            print(f"{slots:>6} {dep:>6.0%} │ {100*np.percentile(cs,10):>7.2f}% "
                  f"{100*np.median(cs):>8.2f}% {100*np.percentile(cs,90):>7.2f}% "
                  f"{100*cs.min():>8.2f}% │ {100*np.median(dds):>7.1f}% "
                  f"{100*dds.min():>8.1f}% │ {beat:>8.0f}%")
    return rows, spy_cagr


def main() -> None:
    ev, wide, cal = load()
    print(f"signals: {len(ev):,} pairs across {ev.sym.nunique()} tickers (P + C)")
    print(f"trailing stop {TRAIL:.0%}, {len(list(SEEDS))} seeds per config")
    fit, spy_fit = run("FIT (no bear market)", ev, wide, cal, FIT)
    val, spy_val = run("VALIDATE (COVID + 2022 bear)", ev, wide, cal, VAL)

    print(f"\n{'=' * 104}\nDOES CONCENTRATION SURVIVE?  median CAGR, both windows"
          f"\n{'=' * 104}")
    print(f"{'slots':>6} {'deploy':>7} {'FIT':>9} {'VALIDATE':>10} "
          f"{'val WORST seed':>15} {'val beat SPY':>13}")
    for slots in LADDER:
        for dep in DEPLOY:
            f_med = fit[(slots, dep)][0]
            v_med, v_worst, v_beat = val[(slots, dep)]
            print(f"{slots:>6} {dep:>6.0%} {100*f_med:>8.2f}% {100*v_med:>9.2f}% "
                  f"{100*v_worst:>14.2f}% {v_beat:>12.0f}%")
    print(f"\nSPY: fit {100*spy_fit:.2f}%/yr, validate {100*spy_val:.2f}%/yr")
    print("\nRead the WORST-seed and p10 columns, not the median. With no way to")
    print("rank signals, the seed IS the strategy at low slot counts.")


if __name__ == "__main__":
    main()
