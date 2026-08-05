"""The "buyable dip" rule: deploy hard when insiders pile into a falling market.

From aggregate_insider_regime.py: aggregate insider buying does NOT predict the
market on its own (Spearman rho ~ -0.09, i.e. slightly backwards). But
CONDITIONAL on the market already being down, a spike in insider buying predicts
strongly:

    buying spike WHILE in drawdown   +27.55% over the next year (n=84)
    buying spike in a calm market     +5.25%                    (n=121)
    baseline                         +15.42%                    (n=2047)

So the rule is not a bull/bear forecaster. It is a "this dip is the buyable
kind" detector. This tests it as an actual sizing rule on top of the bull
config (10 slots / 120-day hold, 20.08% CAGR).

Four ways to express "deploy hard", because the baseline book is already ~90%
invested and cannot simply buy more without dry powder:

    A  control          always 10 slots / 120d
    B  dry powder       hold RESERVE in cash normally; deploy it on green light
    C  concentrate      normal 10 slots; on green light new entries use 5 slots
                        (bigger positions)
    D  ride longer      on green light, new entries hold 252d instead of 120d

POINT-IN-TIME DISCIPLINE, which is what makes or breaks this test:
  * the insider z-score uses a trailing 252-session mean/std and is lagged 2
    sessions;
  * the "top decile" threshold is an EXPANDING quantile of history up to that
    day, never the full-sample quantile. Using the full sample would let the
    rule know which spikes were the big ones.

Usage:
    uv run --with duckdb --with pandas --with numpy --with pyarrow \
        python scripts/research/green_light.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

import concentration_ladder as C

SAMPLES = "/Volumes/Extreme/home-migrated/quant/data/parquet/event_samples/**/*.parquet"
SEEDS = range(16)
E0 = 2_000.0
START, END = "2016-07-25", "2026-07-31"
WIN, Z, LAG = 21, 252, 2
DD_TRIGGER = -0.10
PCTL = 0.90


def insider_flow(cal, con) -> pd.Series:
    """Point-in-time z-score of the aggregate insider buy/sell ratio."""
    ev = con.execute(f"""
        select distinct target_ticker as sym, t0, event_subtype as code
        from read_parquet('{SAMPLES}', union_by_name=true)
        where event_type='insider_transaction' and horizon_days=60
          and event_subtype in ('P','S')
    """).df()
    ev["t0"] = pd.to_datetime(ev.t0)
    day = cal[np.searchsorted(cal.values, ev.t0.values).clip(0, len(cal) - 1)]
    ev["day"] = day
    piv = ev.pivot_table(index="day", columns="code", aggfunc="size",
                         fill_value=0).reindex(cal, fill_value=0)
    P = piv.get("P", pd.Series(0, index=cal)).rolling(WIN).sum()
    S = piv.get("S", pd.Series(0, index=cal)).rolling(WIN).sum()
    ratio = (P - S) / (P + S)
    z = (ratio - ratio.rolling(Z, min_periods=120).mean()) / \
        ratio.rolling(Z, min_periods=120).std()
    return z.shift(LAG)


def sim(by_day, px, cal, i_lo, i_hi, seed, green, mode, reserve=0.0):
    rng = np.random.default_rng(seed)
    cash, pos = E0, []
    curve = np.zeros(i_hi - i_lo + 1)
    g_tr, n_tr = [], []
    for k, i in enumerate(range(i_lo, i_hi + 1)):
        keep = []
        for p in pos:
            pr = px[p["sym"]][i]
            if not np.isfinite(pr):
                keep.append(p); continue
            if i - p["i0"] >= p["hold"] or i == i_hi:
                cash += p["units"] * pr * (1 - C.COST)
                (g_tr if p["green"] else n_tr).append(pr / p["entry"] - 1.0)
            else:
                keep.append(p)
        pos = keep

        on = bool(green[i])
        slots, hold, dep = 10, 120, 1.0 - reserve
        if on:
            dep = 1.0
            if mode == "concentrate":
                slots = 5
            elif mode == "ride":
                hold = 252
        free = slots - len(pos)
        if free > 0:
            held = {q["sym"] for q in pos}
            day = [c for c in by_day.get(i, []) if c in px and c not in held]
            rng.shuffle(day)
            eq = cash + sum(q["units"] * px[q["sym"]][i] for q in pos
                            if np.isfinite(px[q["sym"]][i]))
            unit = eq * dep / slots
            for s in day[:free]:
                pr = px[s][i]
                if not np.isfinite(pr) or pr <= 0:
                    continue
                # respect the cash reserve unless the green light is on
                spend = min(unit, max(0.0, cash - (0.0 if on else eq * reserve)))
                if spend < 1.0:
                    break
                cash -= spend
                pos.append({"sym": s, "i0": i, "entry": pr, "hold": hold,
                            "green": on, "units": spend * (1 - C.COST) / pr})
        curve[k] = cash + sum(p["units"] * (px[p["sym"]][i]
                              if np.isfinite(px[p["sym"]][i]) else 0.0) for p in pos)
    return curve, np.array(g_tr), np.array(n_tr)


def main() -> None:
    ev, wide, cal_np = C.load()
    px = {s: wide[s].to_numpy() for s in wide.columns}
    cal = pd.DatetimeIndex(cal_np)
    by_day: dict[int, list[str]] = {}
    for sym, t0 in zip(ev.sym, ev.t0):
        i = int(np.searchsorted(cal_np, np.datetime64(t0)))
        if i + 1 < len(cal_np):
            by_day.setdefault(i + 1, []).append(sym)

    con = duckdb.connect()
    con.execute("set memory_limit='8GB'; set threads=6;")
    z = insider_flow(cal, con)

    spy = px["SPY"]
    s = pd.Series(spy, index=cal)
    dd = (s / s.rolling(252, min_periods=60).max() - 1.0)
    # EXPANDING quantile — the threshold only ever uses history
    thr = z.expanding(min_periods=250).quantile(PCTL)
    green = ((dd <= DD_TRIGGER) & (z >= thr)).fillna(False).to_numpy()

    i_lo = int(np.searchsorted(cal_np, np.datetime64(START)))
    i_hi = min(int(np.searchsorted(cal_np, np.datetime64(END))), len(cal_np) - 1)
    yrs = (cal_np[i_hi] - cal_np[i_lo]).astype("timedelta64[D]").astype(int) / 365.25
    spy_w = spy[i_lo:i_hi + 1]
    spy_cagr = (spy_w[-1] / spy_w[0]) ** (1 / yrs) - 1
    spy_dd = float(np.min(spy_w / np.maximum.accumulate(spy_w) - 1))

    ng = int(green[i_lo:i_hi + 1].sum())
    print(f"{START} .. {END} ({yrs:.2f} yrs), {len(list(SEEDS))} seeds")
    print(f"GREEN LIGHT = drawdown <= {DD_TRIGGER:.0%} AND insider z >= expanding p{int(100*PCTL)}")
    print(f"  fires on {ng} sessions ({100*ng/(i_hi-i_lo+1):.1f}% of the period)")
    grp = (np.diff(np.concatenate([[0], green[i_lo:i_hi+1].astype(int)])) == 1).cumsum() * green[i_lo:i_hi+1]
    eps = [g for g in range(1, grp.max() + 1) if (grp == g).sum() >= 3]
    print(f"  {len(eps)} distinct episodes:")
    for g in eps:
        m = grp == g
        d = cal[i_lo:i_hi+1][m]
        print(f"     {d[0].date()} -> {d[-1].date()} ({m.sum()} sessions)")
    print(f"\nSPY {100*spy_cagr:.2f}%/yr, maxDD {100*spy_dd:.1f}%\n")

    print(f"{'variant':<26} {'CAGR':>8} {'maxDD':>8} {'Sharpe':>7} "
          f"{'green ret/tr':>13} {'normal ret/tr':>14} {'$2k':>10}")
    print("-" * 92)
    variants = [
        ("A control (10/120d)", "none", 0.0),
        ("B dry powder 15%", "powder", 0.15),
        ("B dry powder 25%", "powder", 0.25),
        ("C concentrate on green", "concentrate", 0.0),
        ("D ride 252d on green", "ride", 0.0),
        ("B+C powder 15% + conc", "concentrate", 0.15),
    ]
    for label, mode, res in variants:
        cg, dds, shp, gt, nt, fin = [], [], [], [], [], []
        for sd in SEEDS:
            c, g, n = sim(by_day, px, cal_np, i_lo, i_hi, sd, green, mode, res)
            if c[-1] <= 0:
                continue
            cg.append((c[-1] / c[0]) ** (1 / yrs) - 1)
            dds.append(float(np.min(c / np.maximum.accumulate(c) - 1)))
            r = np.diff(c) / c[:-1]
            shp.append(r.mean() / r.std() * np.sqrt(252) if r.std() else 0.0)
            if len(g): gt.append(g.mean())
            if len(n): nt.append(n.mean())
            fin.append(c[-1])
        print(f"{label:<26} {100*np.median(cg):>7.2f}% {100*np.median(dds):>7.1f}% "
              f"{np.median(shp):>7.2f} {100*np.median(gt) if gt else float('nan'):>12.2f}% "
              f"{100*np.median(nt):>13.2f}% ${np.median(fin):>9,.0f}")

    print("\nIf green ret/tr >> normal ret/tr, the detector is finding real")
    print("opportunities. If CAGR barely moves, the rule fires too rarely to matter.")


if __name__ == "__main__":
    main()
