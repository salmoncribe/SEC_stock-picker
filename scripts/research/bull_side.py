"""What configuration works best when the market is NOT in drawdown?

The bear side is settled (see bear_only.py): deep drawdown is where the insider
edge concentrates, and the switch must read drawdown depth rather than a moving
average. This finds the bull-side counterpart so the two can be combined without
the cash drag that makes either side useless alone.

Bull regime here = SPY within 10% of its trailing 252-day high. Bear = 10%+
below it. That threshold, not a moving average, because below-200dma was
measured to make per-trade returns WORSE (3.07% vs 5.56% baseline).

Grid: slot count x hold length x exit style, entries allowed only in the bull
regime. Positions still exit on their own rule regardless of regime, so the book
winds down naturally rather than being force-sold at a regime flip.

Continuous from 2016-07-25 — no cold start (worth ~+13pp/yr if you get it wrong).

Usage:
    uv run --with duckdb --with pandas --with numpy --with pyarrow \
        python scripts/research/bull_side.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import concentration_ladder as C

SEEDS = range(12)
E0 = 2_000.0
START, END = "2016-07-25", "2026-07-31"
BULL_DD = -0.10          # bull = shallower than a 10% drawdown


def sim(by_day, px, cal, i_lo, i_hi, seed, *, slots, hold, trail, on):
    rng = np.random.default_rng(seed)
    cash, pos = E0, []
    curve = np.zeros(i_hi - i_lo + 1)
    inv = np.zeros(i_hi - i_lo + 1)
    trades: list[float] = []
    for k, i in enumerate(range(i_lo, i_hi + 1)):
        keep = []
        for p in pos:
            pr = px[p["sym"]][i]
            if not np.isfinite(pr):
                keep.append(p); continue
            p["peak"] = max(p["peak"], pr)
            out = i == i_hi
            if hold is not None and i - p["i0"] >= hold:
                out = True
            if trail is not None and pr <= p["peak"] * (1 - trail):
                out = True
            if out:
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
                pos.append({"sym": s, "i0": i, "entry": pr, "peak": pr,
                            "units": sp * (1 - C.COST) / pr})
        mv = sum(p["units"] * (px[p["sym"]][i] if np.isfinite(px[p["sym"]][i])
                 else 0.0) for p in pos)
        curve[k] = cash + mv
        inv[k] = mv / curve[k] if curve[k] > 0 else 0.0
    return curve, inv, np.array(trades)


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
    s = pd.Series(spy)
    dd = (s / s.rolling(252, min_periods=60).max() - 1.0).to_numpy()
    bull = dd > BULL_DD
    yrs = (cal[i_hi] - cal[i_lo]).astype("timedelta64[D]").astype(int) / 365.25
    spy_w = spy[i_lo:i_hi + 1]

    print(f"{START} .. {END} ({yrs:.2f} yrs), continuous")
    print(f"SPY {100*((spy_w[-1]/spy_w[0])**(1/yrs)-1):.2f}%/yr")
    print(f"BULL regime (drawdown shallower than {abs(BULL_DD):.0%}): "
          f"{100*np.mean(bull[i_lo:i_hi+1]):.1f}% of sessions\n")

    print(f"{'slots':>6} {'hold':>6} {'exit':>10} {'CAGR':>8} {'maxDD':>8} "
          f"{'Sharpe':>7} {'invested':>9} {'trades':>7} {'ret/trade':>10}")
    print("-" * 84)
    rows = []
    grid = []
    for slots in (6, 10, 20, 40):
        for hold, trail, label in ((40, None, "fixed 40d"), (60, None, "fixed 60d"),
                                   (120, None, "fixed 120d"), (None, 0.15, "trail 15%"),
                                   (120, 0.15, "trail+120d")):
            grid.append((slots, hold, trail, label))
    for slots, hold, trail, label in grid:
        cg, dds, shp, iv, rt, nt = [], [], [], [], [], []
        for sd in SEEDS:
            c, i_, t = sim(by_day, px, cal, i_lo, i_hi, sd, slots=slots,
                           hold=hold, trail=trail, on=bull)
            if c[-1] <= 0 or not len(t):
                continue
            cg.append((c[-1] / c[0]) ** (1 / yrs) - 1)
            dds.append(float(np.min(c / np.maximum.accumulate(c) - 1)))
            r = np.diff(c) / c[:-1]
            shp.append(r.mean() / r.std() * np.sqrt(252) if r.std() else 0.0)
            iv.append(i_.mean()); rt.append(t.mean()); nt.append(len(t))
        if not cg:
            continue
        rows.append((slots, label, np.median(cg), np.median(dds), np.median(shp),
                     np.median(iv), np.median(nt), np.median(rt)))
    for slots, label, c, d, sh, i_, n, r in sorted(rows, key=lambda x: -x[2]):
        print(f"{slots:>6} {'':>6} {label:>10} {100*c:>7.2f}% {100*d:>7.1f}% "
              f"{sh:>7.2f} {100*i_:>8.1f}% {n:>7.0f} {100*r:>9.2f}%")

    print("\nBest bull-side configs by CAGR are at the top. Compare ret/trade to the")
    print("bear side's 29.89% (dd_20) and the always-on baseline's 5.56%.")


if __name__ == "__main__":
    main()
