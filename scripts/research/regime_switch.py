"""The combined regime-switching strategy: bull rules and bear rules, one book.

Established separately:
  * BULL (drawdown shallower than 10%): 10 slots, fixed 120-day hold ->
    21.61% CAGR, Sharpe 1.00, 12.77%/trade  (bull_side.py)
  * BEAR (deep drawdown): the insider edge runs ~5-6x baseline per trade, but
    the sample is two events, and bear-only holds cash 93% of the time
    (bear_only.py)

Switching removes the cash drag that makes either side useless alone. A position
keeps the exit rule it was ENTERED under, so a regime flip never force-sells the
book -- it just changes what new entries look like.

The bear side's slot count and hold length are swept here rather than assumed,
because the bull side's answer (120-day hold) was the opposite of the 60-day
hold the bear test happened to use.

OVERFITTING WARNING, stated in the file so it travels with the result: this
selects BOTH a bull config and a bear config on the same 10 years, and the bear
regime contains only two independent episodes (COVID 2020, the 2022 bear). The
combined number below is IN-SAMPLE. It is a hypothesis, not a validated edge.

Usage:
    uv run --with duckdb --with pandas --with numpy --with pyarrow \
        python scripts/research/regime_switch.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import concentration_ladder as C

SEEDS = range(16)
E0 = 2_000.0
START, END = "2016-07-25", "2026-07-31"
BEAR_DD = -0.10          # bear regime = 10%+ below the trailing 252-day high

BULL_CFG = {"slots": 10, "hold": 120}     # winner from bull_side.py


def sim(by_day, px, cal, i_lo, i_hi, seed, bull_cfg, bear_cfg, bear_on):
    """Regime-switching run. Each position keeps its entry-time exit rule."""
    rng = np.random.default_rng(seed)
    cash, pos = E0, []
    curve = np.zeros(i_hi - i_lo + 1)
    t_bull: list[float] = []
    t_bear: list[float] = []
    for k, i in enumerate(range(i_lo, i_hi + 1)):
        keep = []
        for p in pos:
            pr = px[p["sym"]][i]
            if not np.isfinite(pr):
                keep.append(p); continue
            if i - p["i0"] >= p["hold"] or i == i_hi:
                cash += p["units"] * pr * (1 - C.COST)
                (t_bear if p["bear"] else t_bull).append(pr / p["entry"] - 1.0)
            else:
                keep.append(p)
        pos = keep

        bear = bool(bear_on[i])
        cfg = bear_cfg if bear else bull_cfg
        free = cfg["slots"] - len(pos)
        if free > 0:
            held = {q["sym"] for q in pos}
            day = [c for c in by_day.get(i, []) if c in px and c not in held]
            rng.shuffle(day)
            eq = cash + sum(q["units"] * px[q["sym"]][i] for q in pos
                            if np.isfinite(px[q["sym"]][i]))
            unit = eq / cfg["slots"]
            for s in day[:free]:
                pr = px[s][i]
                if not np.isfinite(pr) or pr <= 0:
                    continue
                sp = min(unit, cash)
                if sp < 1.0:
                    break
                cash -= sp
                pos.append({"sym": s, "i0": i, "entry": pr, "bear": bear,
                            "hold": cfg["hold"],
                            "units": sp * (1 - C.COST) / pr})
        curve[k] = cash + sum(p["units"] * (px[p["sym"]][i]
                              if np.isfinite(px[p["sym"]][i]) else 0.0) for p in pos)
    return curve, np.array(t_bull), np.array(t_bear)


def stats(curve, yrs):
    cg = (curve[-1] / curve[0]) ** (1 / yrs) - 1 if curve[-1] > 0 else -1.0
    dd = float(np.min(curve / np.maximum.accumulate(curve) - 1))
    r = np.diff(curve) / curve[:-1]
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() else 0.0
    return cg, dd, sh


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
    bear_on = dd <= BEAR_DD
    yrs = (cal[i_hi] - cal[i_lo]).astype("timedelta64[D]").astype(int) / 365.25
    spy_w = spy[i_lo:i_hi + 1]
    spy_cagr = (spy_w[-1] / spy_w[0]) ** (1 / yrs) - 1
    spy_dd = float(np.min(spy_w / np.maximum.accumulate(spy_w) - 1))
    spy_r = np.diff(spy_w) / spy_w[:-1]

    print(f"{START} .. {END} ({yrs:.2f} yrs), continuous, {len(list(SEEDS))} seeds")
    print(f"BEAR regime = {abs(BEAR_DD):.0%}+ below trailing 252d high — "
          f"{100*np.mean(bear_on[i_lo:i_hi+1]):.1f}% of sessions")
    print(f"BULL config fixed at {BULL_CFG['slots']} slots / {BULL_CFG['hold']}d hold\n")
    print(f"SPY: {100*spy_cagr:>6.2f}%/yr  maxDD {100*spy_dd:>6.1f}%  "
          f"Sharpe {spy_r.mean()/spy_r.std()*np.sqrt(252):.2f}\n")

    print(f"{'bear config':<22} {'CAGR':>8} {'maxDD':>8} {'Sharpe':>7} "
          f"{'bull ret/tr':>12} {'bear ret/tr':>12} {'$2k becomes':>13}")
    print("-" * 88)
    rows = []
    bear_grid = [{"slots": sl, "hold": h}
                 for sl in (3, 6, 10, 20) for h in (60, 120)]
    # control: no switch at all — bull config runs everywhere
    bear_grid.append(dict(BULL_CFG))
    for bc in bear_grid:
        res, tb, tr = [], [], []
        for sd in SEEDS:
            c, b, e = sim(by_day, px, cal, i_lo, i_hi, sd, BULL_CFG, bc, bear_on)
            res.append(stats(c, yrs) + (c[-1],))
            if len(b): tb.append(b.mean())
            if len(e): tr.append(e.mean())
        a = np.array([r[:3] for r in res])
        fin = np.median([r[3] for r in res])
        m = np.median(a, axis=0)
        same = bc == BULL_CFG
        label = "NO SWITCH (control)" if same else f"{bc['slots']} slots / {bc['hold']}d"
        rows.append((label, m[0], m[1], m[2],
                     np.median(tb) if tb else np.nan,
                     np.median(tr) if tr else np.nan, fin, same))
    for label, c, d, sh, b, e, fin, same in sorted(rows, key=lambda x: -x[1]):
        mark = "  <-- control" if same else ""
        print(f"{label:<22} {100*c:>7.2f}% {100*d:>7.1f}% {sh:>7.2f} "
              f"{100*b:>11.2f}% {100*e:>11.2f}% ${fin:>12,.0f}{mark}")

    print("\nIf the best switch barely beats the NO SWITCH control, the bear side is")
    print("not adding anything and the bull config alone is the strategy.")
    print("Remember: bear regime = 2 independent episodes. This is IN-SAMPLE.")


if __name__ == "__main__":
    main()
