"""Does the insider signal work better when the market is BEARISH?

Literature says insider purchases rise during downturns and insiders treat them
as buying opportunities (Insider trading patterns during the COVID period,
ScienceDirect 2025; bear markets show decreased sales and increased purchases).
If insiders have an informational edge, it should be worth most when the market
is mispricing their firm.

Only ENTRIES are gated by the regime. Open positions still exit on the 60-day
clock, so the book winds down naturally when the regime turns off rather than
being force-sold. Idle capital sits in cash.

Six regime definitions, from loosest to strictest:
    below_50dma     SPY under its 50-day moving average
    below_200dma    SPY under its 200-day moving average (the classic filter)
    dd_5            SPY 5%+ below its trailing 252-day high
    dd_10           10%+ below — a "correction"
    dd_20           20%+ below — a textbook "bear market"
    mom_neg_6m      SPY's trailing 126-day return is negative

Run continuously from 2016-07-25 so there is no cold-start bias
(see docs: a standalone window start is worth ~+13pp/yr).

The headline metric is NOT CAGR. A filter that is active 25% of the time will
lose to a filter that is always on, purely by being in cash. What matters is
RETURN PER DAY INVESTED — whether the signal is genuinely better in bearish
tape, or whether we are just sitting out.

Usage:
    uv run --with duckdb --with pandas --with numpy --with pyarrow \
        python scripts/research/bear_only.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import concentration_ladder as C

HOLD = 60
SEEDS = range(16)
E0 = 2_000.0
START, END = "2016-07-25", "2026-07-31"
SLOTS = (6, 10, 40)


def regimes(spy: np.ndarray) -> dict[str, np.ndarray]:
    s = pd.Series(spy)
    peak252 = s.rolling(252, min_periods=60).max()
    dd = s / peak252 - 1.0
    return {
        "always on": np.ones(len(spy), dtype=bool),
        "below_50dma": (s < s.rolling(50).mean()).to_numpy(),
        "below_200dma": (s < s.rolling(200).mean()).to_numpy(),
        "dd_5": (dd <= -0.05).to_numpy(),
        "dd_10": (dd <= -0.10).to_numpy(),
        "dd_20": (dd <= -0.20).to_numpy(),
        "mom_neg_6m": (s.pct_change(126) < 0).to_numpy(),
        # the inverse, as a control: does the edge live in BULL tape instead?
        "BULL only (200dma)": (s >= s.rolling(200).mean()).to_numpy(),
    }


def sim(by_day, px, cal, i_lo, i_hi, slots, seed, on: np.ndarray):
    rng = np.random.default_rng(seed)
    cash, pos = E0, []
    curve = np.zeros(i_hi - i_lo + 1)
    invested = np.zeros(i_hi - i_lo + 1)
    trades: list[float] = []
    for k, i in enumerate(range(i_lo, i_hi + 1)):
        keep = []
        for p in pos:
            pr = px[p["sym"]][i]
            if not np.isfinite(pr):
                keep.append(p); continue
            if i - p["i0"] >= HOLD or i == i_hi:
                cash += p["units"] * pr * (1 - C.COST)
                trades.append(pr / p["entry"] - 1.0)
            else:
                keep.append(p)
        pos = keep
        free = slots - len(pos)
        if free > 0 and bool(on[i]):
            held = {q["sym"] for q in pos}
            day = [c for c in by_day.get(i, []) if c in px and c not in held]
            rng.shuffle(day)
            eq = cash + sum(q["units"] * px[q["sym"]][i] for q in pos
                            if np.isfinite(px[q["sym"]][i]))
            unit = eq / slots
            for s in day[:free]:
                pr = px[s][i]
                if not np.isfinite(pr) or pr <= 0:
                    continue
                sp = min(unit, cash)
                if sp < 1.0:
                    break
                cash -= sp
                pos.append({"sym": s, "i0": i, "entry": pr,
                            "units": sp * (1 - C.COST) / pr})
        mv = sum(p["units"] * (px[p["sym"]][i] if np.isfinite(px[p["sym"]][i])
                 else 0.0) for p in pos)
        curve[k] = cash + mv
        invested[k] = mv / curve[k] if curve[k] > 0 else 0.0
    return curve, invested, np.array(trades)


def main() -> None:
    ev, wide, cal = C.load()
    px = {s: wide[s].to_numpy() for s in wide.columns}
    by_day: dict[int, list[str]] = {}
    for sym, t0 in zip(ev.sym, ev.t0):
        i = int(np.searchsorted(cal, np.datetime64(t0)))
        if i + 1 < len(cal):
            by_day.setdefault(i + 1, []).append(sym)

    i_lo = int(np.searchsorted(cal, np.datetime64(START)))
    i_hi = min(int(np.searchsorted(cal, np.datetime64(END))), len(cal) - 1)
    spy = px["SPY"]
    regs = regimes(spy)
    yrs = (cal[i_hi] - cal[i_lo]).astype("timedelta64[D]").astype(int) / 365.25
    spy_w = spy[i_lo:i_hi + 1]
    spy_cagr = (spy_w[-1] / spy_w[0]) ** (1 / yrs) - 1

    print(f"{START} .. {END} ({yrs:.2f} yrs), continuous — no cold start")
    print(f"SPY {100*spy_cagr:.2f}%/yr\n")
    print("regime activity (share of sessions the filter allows entries):")
    for name, on in regs.items():
        print(f"   {name:<22} {100*np.mean(on[i_lo:i_hi+1]):5.1f}%")

    for slots in SLOTS:
        print(f"\n{'=' * 104}\n{slots} SLOTS\n{'=' * 104}")
        print(f"{'regime':<22} {'CAGR':>8} {'maxDD':>8} {'Sharpe':>7} "
              f"{'invested':>9} {'trades':>7} {'ret/trade':>10} {'ret/yr INVESTED':>16}")
        for name, on in regs.items():
            cg, dds, shp, inv, rpt, ntr = [], [], [], [], [], []
            for sd in SEEDS:
                c, iv, tr = sim(by_day, px, cal, i_lo, i_hi, slots, sd, on)
                if c[-1] <= 0:
                    continue
                cg.append((c[-1] / c[0]) ** (1 / yrs) - 1)
                dds.append(float(np.min(c / np.maximum.accumulate(c) - 1)))
                r = np.diff(c) / c[:-1]
                shp.append(r.mean() / r.std() * np.sqrt(252) if r.std() else 0.0)
                inv.append(iv.mean())
                ntr.append(len(tr))
                rpt.append(tr.mean() if len(tr) else 0.0)
            if not cg:
                continue
            mi = np.median(inv)
            # return per year of full-capital exposure: de-scales the cash drag
            adj = (1 + np.median(cg)) ** (1 / mi) - 1 if mi > 0.02 else float("nan")
            print(f"{name:<22} {100*np.median(cg):>7.2f}% {100*np.median(dds):>7.1f}% "
                  f"{np.median(shp):>7.2f} {100*mi:>8.1f}% {np.median(ntr):>7.0f} "
                  f"{100*np.median(rpt):>9.2f}% {100*adj:>15.2f}%")

    print("\nret/trade and ret/yr-INVESTED are the columns that matter. If a bear")
    print("filter raises them, the signal really is better in falling markets and")
    print("the low CAGR is just cash drag. If it does not, the filter only removes")
    print("exposure and adds nothing.")


if __name__ == "__main__":
    main()
