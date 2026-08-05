"""Can we reach 30%/yr without leverage? Test every untested lever.

Levers, each of which could plausibly add return and none of which has been
tested on this strategy:

  pyramid      add to a winner on each new high (Michael's actual style)
  reentry      re-buy a name after a trailing-stop exit if it makes a new high
  cells        trade C (derivative conversion) alongside P, for more candidates
  slots        concentration: fewer, larger positions
  adaptive     tight trail until the trade proves itself, loose trail after
  volscale     size inversely to trailing volatility
  regime       no new entries when SPY is below its 200-day moving average

FIT on 2016-07-25..2019-12-31, then VALIDATE on 2020-01-01..2026-07-31 --
which contains the COVID crash AND the 2022 bear market. A lever that only
works in the fit window is not a lever, it is a curve fit. Prior work measured
train->test rank correlation of -0.11 on a parameter sweep, so the validation
column is the only one that means anything.

Usage:
    uv run --with duckdb --with pandas --with numpy --with pyarrow \
        python scripts/research/reach_for_30.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

PRICES = "/Volumes/Extreme/home-migrated/quant/data/parquet/daily_prices/*/*.parquet"
SAMPLES = "/Volumes/Extreme/home-migrated/quant/data/parquet/event_samples/**/*.parquet"
COST = 27.0 / 10_000.0
E0 = 10_000.0
SEEDS = range(8)

FIT = ("2016-07-25", "2019-12-31")
VAL = ("2020-01-01", "2026-07-31")


def load(subtypes: tuple[str, ...]):
    con = duckdb.connect()
    con.execute("set memory_limit='8GB'; set threads=6;")
    subs = "','".join(subtypes)
    ev = con.execute(f"""
        select distinct target_ticker as sym, t0
        from read_parquet('{SAMPLES}', union_by_name=true)
        where event_type='insider_transaction'
          and event_subtype in ('{subs}') and horizon_days=60
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


def simulate(ev, wide, cal, cfg, seed, lo, hi):
    """One account run over [lo, hi]. Returns the daily equity curve."""
    rng = np.random.default_rng(seed)
    i_lo = int(np.searchsorted(cal, np.datetime64(lo)))
    i_hi = int(np.searchsorted(cal, np.datetime64(hi)))
    px = {s: wide[s].to_numpy() for s in wide.columns}
    spy = px["SPY"]
    sma200 = pd.Series(spy).rolling(200).mean().to_numpy()

    by_day: dict[int, list[str]] = {}
    for sym, t0 in zip(ev.sym, ev.t0):
        i = int(np.searchsorted(cal, np.datetime64(t0)))
        if i + 1 < len(cal):
            by_day.setdefault(i + 1, []).append(sym)

    slots = cfg["slots"]
    trail = cfg["trail"]
    adaptive = cfg.get("adaptive")        # (tight, arm_gain, loose)
    pyramid = cfg.get("pyramid", 0)       # max extra adds per position
    reentry = cfg.get("reentry", False)
    volscale = cfg.get("volscale", False)
    regime = cfg.get("regime", False)

    cash, pos = E0, []
    exited: dict[str, float] = {}         # sym -> peak price at exit (for reentry)
    curve = np.zeros(i_hi - i_lo + 1)

    # trailing 21d vol for sizing
    if volscale:
        rets = wide.pct_change()
        vol = rets.rolling(21).std().shift(1)

    for k, i in enumerate(range(i_lo, i_hi + 1)):
        # --- exits
        keep = []
        for p in pos:
            price = px[p["sym"]][i]
            if not np.isfinite(price):
                keep.append(p); continue
            p["peak"] = max(p["peak"], price)
            gain = p["peak"] / p["entry"] - 1.0
            t = trail
            if adaptive:
                tight, arm, loose = adaptive
                t = loose if gain >= arm else tight
            if price <= p["peak"] * (1 - t) or i == i_hi:
                cash += p["units"] * price * (1 - COST)
                exited[p["sym"]] = p["peak"]
            else:
                keep.append(p)
        pos = keep

        # --- pyramiding: add to positions making a new high
        if pyramid:
            equity = cash + sum(q["units"] * px[q["sym"]][i] for q in pos
                                if np.isfinite(px[q["sym"]][i]))
            unit = equity / slots
            for p in pos:
                price = px[p["sym"]][i]
                if (not np.isfinite(price)) or p["adds"] >= pyramid:
                    continue
                # add once per +10% above the level of the last add
                if price >= p["last_add"] * 1.10 and cash > unit * 0.5:
                    spend = min(unit * 0.5, cash)
                    cash -= spend
                    add_units = spend * (1 - COST) / price
                    p["entry"] = ((p["entry"] * p["units"] + price * add_units)
                                  / (p["units"] + add_units))
                    p["units"] += add_units
                    p["last_add"] = price
                    p["adds"] += 1

        # --- entries
        free = slots - len(pos)
        if free > 0 and not (regime and np.isfinite(sma200[i]) and spy[i] < sma200[i]):
            held = {q["sym"] for q in pos}
            cands = [c for c in by_day.get(i, []) if c in px and c not in held]
            if reentry:
                for s, pk in list(exited.items()):
                    if s in px and s not in held and np.isfinite(px[s][i]) \
                       and px[s][i] > pk:
                        cands.append(s)
                        del exited[s]
            rng.shuffle(cands)
            equity = cash + sum(q["units"] * px[q["sym"]][i] for q in pos
                                if np.isfinite(px[q["sym"]][i]))
            base = equity / slots
            for sym in cands[:free]:
                price = px[sym][i]
                if not np.isfinite(price) or price <= 0:
                    continue
                size = base
                if volscale:
                    v = vol[sym].iloc[i] if sym in vol.columns else np.nan
                    if np.isfinite(v) and v > 0:
                        size = base * min(2.0, max(0.4, 0.02 / v))
                spend = min(size, cash)
                if spend < 1.0:
                    break
                cash -= spend
                pos.append({"sym": sym, "entry": price, "peak": price,
                            "units": spend * (1 - COST) / price,
                            "last_add": price, "adds": 0})

        eq = cash + sum(p["units"] * (px[p["sym"]][i] if np.isfinite(px[p["sym"]][i])
                        else p["entry"]) for p in pos)
        curve[k] = eq
    return curve


def stats(curve, spy_slice, yrs):
    cagr = (curve[-1] / curve[0]) ** (1 / yrs) - 1
    dd = float(np.min(curve / np.maximum.accumulate(curve) - 1))
    r = np.diff(curve) / curve[:-1]
    b = np.diff(spy_slice) / spy_slice[:-1]
    n = min(len(r), len(b))
    beta = np.cov(r[:n], b[:n])[0, 1] / np.var(b[:n]) if np.var(b[:n]) else 0.0
    alpha = (r[:n].mean() - beta * b[:n].mean()) * 252
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() else 0.0
    return cagr, dd, beta, alpha, sharpe


CONFIGS = [
    ("baseline trail15 s40",      {"slots": 40, "trail": 0.15}),
    ("slots 10",                  {"slots": 10, "trail": 0.15}),
    ("slots 20",                  {"slots": 20, "trail": 0.15}),
    ("slots 60",                  {"slots": 60, "trail": 0.15}),
    ("pyramid x2",                {"slots": 40, "trail": 0.15, "pyramid": 2}),
    ("pyramid x4",                {"slots": 40, "trail": 0.15, "pyramid": 4}),
    ("reentry",                   {"slots": 40, "trail": 0.15, "reentry": True}),
    ("adaptive 8->25 @+15%",      {"slots": 40, "trail": 0.15,
                                   "adaptive": (0.08, 0.15, 0.25)}),
    ("adaptive 10->30 @+20%",     {"slots": 40, "trail": 0.15,
                                   "adaptive": (0.10, 0.20, 0.30)}),
    ("volscale",                  {"slots": 40, "trail": 0.15, "volscale": True}),
    ("regime 200dma",             {"slots": 40, "trail": 0.15, "regime": True}),
    ("pyramid+reentry",           {"slots": 40, "trail": 0.15, "pyramid": 2,
                                   "reentry": True}),
    ("pyr+reent+adaptive",        {"slots": 40, "trail": 0.15, "pyramid": 2,
                                   "reentry": True, "adaptive": (0.08, 0.15, 0.25)}),
    ("pyr+reent+adapt+s20",       {"slots": 20, "trail": 0.15, "pyramid": 2,
                                   "reentry": True, "adaptive": (0.08, 0.15, 0.25)}),
]


def report(title, ev, wide, cal, window):
    lo, hi = window
    i_lo = int(np.searchsorted(cal, np.datetime64(lo)))
    i_hi = int(np.searchsorted(cal, np.datetime64(hi)))
    spy = wide["SPY"].to_numpy()[i_lo:i_hi + 1]
    yrs = (cal[i_hi] - cal[i_lo]).astype("timedelta64[D]").astype(int) / 365.25
    print(f"\n{'=' * 96}\n{title}   {lo} -> {hi}  ({yrs:.2f} yrs)")
    sc = (spy[-1] / spy[0]) ** (1 / yrs) - 1
    sdd = float(np.min(spy / np.maximum.accumulate(spy) - 1))
    print(f"SPY: {100*sc:.2f}%/yr, maxDD {100*sdd:.1f}%\n{'=' * 96}")
    print(f"{'config':<24} {'CAGR':>8} {'maxDD':>8} {'beta':>6} {'alpha':>8} {'Sharpe':>7}")
    out = {}
    for name, cfg in CONFIGS:
        s = np.array([stats(simulate(ev, wide, cal, cfg, sd, lo, hi), spy, yrs)
                      for sd in SEEDS])
        m = np.median(s, axis=0)
        out[name] = m
        print(f"{name:<24} {100*m[0]:>7.2f}% {100*m[1]:>7.1f}% {m[2]:>6.2f} "
              f"{100*m[3]:>7.2f}% {m[4]:>7.2f}")
    return out


def main() -> None:
    print("### SINGLE CELL: insider purchases (P) only")
    ev, wide, cal = load(("P",))
    fit_p = report("FIT", ev, wide, cal, FIT)
    val_p = report("VALIDATE (includes COVID + 2022 bear)", ev, wide, cal, VAL)

    print("\n\n### MULTI CELL: P + C (derivative conversions) -- more candidates")
    ev2, wide2, cal2 = load(("P", "C"))
    print(f"signals: {len(ev2):,} pairs, {ev2.sym.nunique()} tickers "
          f"(vs {len(ev):,} / {ev.sym.nunique()} for P alone)")
    fit_pc = report("FIT", ev2, wide2, cal2, FIT)
    val_pc = report("VALIDATE", ev2, wide2, cal2, VAL)

    print(f"\n\n{'=' * 96}\nFIT -> VALIDATE STABILITY (did the lever survive?)\n{'=' * 96}")
    print(f"{'config':<24} {'P fit':>9} {'P val':>9} {'PC fit':>9} {'PC val':>9}")
    for name, _ in CONFIGS:
        print(f"{name:<24} {100*fit_p[name][0]:>8.2f}% {100*val_p[name][0]:>8.2f}% "
              f"{100*fit_pc[name][0]:>8.2f}% {100*val_pc[name][0]:>8.2f}%")
    rank_fit = [fit_p[n][0] for n, _ in CONFIGS]
    rank_val = [val_p[n][0] for n, _ in CONFIGS]
    rho = pd.Series(rank_fit).corr(pd.Series(rank_val), method="spearman")
    print(f"\nfit->validate RANK correlation across configs: {rho:+.2f}")
    print("Near zero or negative means the fit window cannot pick the winner --")
    print("the same result (-0.11) that made parameter sweeping useless before.")


if __name__ == "__main__":
    main()
