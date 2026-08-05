"""WHY does mean return per trade fall as slot count rises?

Measured: 3 slots -> 9.30%/trade, 40 slots -> 4.79%/trade, with capital
utilisation ~92% at every slot count (so it is not cash drag). Per-trade return
is purely price-based (exit/entry - 1), so it cannot be a sizing artifact
either. That leaves SELECTION: a 6-slot book and a 40-slot book are taking
different trades. This finds out which, and whether the difference is real.

Three tests:

  A  POPULATION. Bucket every signal by how many signals fired that day, and
     measure the raw 60-session forward return per bucket. No simulator
     involved. If busy days are worse, then any book with the capacity to take
     more of them is structurally worse -- and the effect is real.

  B  INSTRUMENTED. Record what each book actually took: entries per day, the
     candidate count on its entry days, holding period, realized return. Shows
     whether the books differ the way test A predicts.

  C  CONTROLLED. Re-run both slot counts with a hard cap of ONE entry per day,
     which forces both to sample days the same way. If the per-trade gap
     collapses, day/per-day selection was the mechanism. If it survives, there
     is a bug and the whole concentration result is void.

Force-closed trades at the end of the window are excluded from per-trade stats
throughout -- they have truncated holding periods and would bias short.

Usage:
    uv run --with duckdb --with pandas --with numpy --with pyarrow \
        python scripts/research/why_concentration.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import concentration_ladder as C

HOLD = 60
LO, HI = "2020-01-01", "2026-07-31"
SEEDS = list(C.SEEDS)[:12]


def build(ev, cal):
    by_day: dict[int, list[str]] = {}
    for sym, t0 in zip(ev.sym, ev.t0):
        i = int(np.searchsorted(cal, np.datetime64(t0)))
        if i + 1 < len(cal):
            by_day.setdefault(i + 1, []).append(sym)
    return by_day


def sim(by_day, px, slots, seed, i_lo, i_hi, max_per_day=None):
    """Returns a list of trade records; force-closed trades are flagged."""
    rng = np.random.default_rng(seed)
    cash, pos, trades = C.E0, [], []
    for i in range(i_lo, i_hi + 1):
        keep = []
        for p in pos:
            pr = px[p["sym"]][i]
            if not np.isfinite(pr):
                keep.append(p); continue
            forced = i == i_hi
            if i - p["i0"] >= HOLD or forced:
                cash += p["units"] * pr * (1 - C.COST)
                trades.append({"ret": pr / p["entry"] - 1.0, "held": i - p["i0"],
                               "cands": p["cands"], "forced": forced})
            else:
                keep.append(p)
        pos = keep
        free = slots - len(pos)
        if max_per_day is not None:
            free = min(free, max_per_day)
        if free > 0:
            held = {q["sym"] for q in pos}
            day = [c for c in by_day.get(i, []) if c in px and c not in held]
            n_cands = len(day)
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
                pos.append({"sym": s, "i0": i, "entry": pr, "cands": n_cands,
                            "units": sp * (1 - C.COST) / pr})
    return trades


def main() -> None:
    ev, wide, cal = C.load()
    px = {s: wide[s].to_numpy() for s in wide.columns}
    by_day = build(ev, cal)
    i_lo = int(np.searchsorted(cal, np.datetime64(LO)))
    i_hi = min(int(np.searchsorted(cal, np.datetime64(HI))), len(cal) - 1)

    # ---------------------------------------------------- A: population level
    print("=" * 88)
    print("TEST A — POPULATION: raw 60-session forward return by that day's signal count")
    print("(no simulator involved; SPY-relative and raw)")
    print("=" * 88)
    spy = px["SPY"]
    rows = []
    for i, syms in by_day.items():
        if i < i_lo or i + HOLD > i_hi:
            continue
        n = len(syms)
        for s in set(syms):
            if s not in px:
                continue
            a, b = px[s][i], px[s][i + HOLD]
            if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
                continue
            bench = spy[i + HOLD] / spy[i] - 1.0
            rows.append((n, b / a - 1.0, b / a - 1.0 - bench))
    df = pd.DataFrame(rows, columns=["n_cands", "raw", "hedged"])
    print(f"signals measured: {len(df):,}")
    df["bucket"] = pd.cut(df.n_cands, [0, 1, 2, 3, 5, 8, 100],
                          labels=["1", "2", "3", "4-5", "6-8", "9+"])
    g = df.groupby("bucket", observed=True).agg(
        n=("raw", "size"), mean_raw=("raw", "mean"),
        mean_hedged=("hedged", "mean"), win=("raw", lambda x: (x > 0).mean()))
    g["mean_raw"] = (100 * g.mean_raw).round(2)
    g["mean_hedged"] = (100 * g.mean_hedged).round(2)
    g["win"] = (100 * g.win).round(1)
    print(g.to_string())
    rho = df.n_cands.corr(df.raw, method="spearman")
    print(f"\nSpearman rho(signals that day, forward return) = {rho:+.4f}")
    print("Strongly negative => busy days really are worse, and a big book that")
    print("must take them is structurally disadvantaged. Near zero => not the cause.")

    # ------------------------------------------------- B: what each book took
    print("\n" + "=" * 88)
    print("TEST B — INSTRUMENTED: what each book actually traded (force-closed excluded)")
    print("=" * 88)
    print(f"{'slots':>6} {'trades/run':>11} {'mean ret':>10} {'median':>9} "
          f"{'mean held':>10} {'avg cands on entry day':>24}")
    for slots in (3, 6, 10, 20, 40):
        allt = []
        for sd in SEEDS:
            allt += [t for t in sim(by_day, px, slots, sd, i_lo, i_hi)
                     if not t["forced"]]
        r = np.array([t["ret"] for t in allt])
        c = np.array([t["cands"] for t in allt])
        h = np.array([t["held"] for t in allt])
        print(f"{slots:>6} {len(allt)/len(SEEDS):>11.0f} {100*r.mean():>9.2f}% "
              f"{100*np.median(r):>8.2f}% {h.mean():>10.1f} {c.mean():>24.2f}")

    # ------------------------------------------- C: controlled, 1 entry/day max
    print("\n" + "=" * 88)
    print("TEST C — CONTROLLED: max ONE entry per day, so both books sample days alike")
    print("=" * 88)
    print(f"{'slots':>6} {'trades/run':>11} {'mean ret':>10} {'median':>9} "
          f"{'avg cands':>11}")
    for slots in (3, 6, 10, 20, 40):
        allt = []
        for sd in SEEDS:
            allt += [t for t in sim(by_day, px, slots, sd, i_lo, i_hi, max_per_day=1)
                     if not t["forced"]]
        r = np.array([t["ret"] for t in allt])
        c = np.array([t["cands"] for t in allt])
        print(f"{slots:>6} {len(allt)/len(SEEDS):>11.0f} {100*r.mean():>9.2f}% "
              f"{100*np.median(r):>8.2f}% {c.mean():>11.2f}")
    print("\nIf the mean-return spread across slot counts COLLAPSES here, the effect")
    print("was per-day selection (real). If it PERSISTS, the simulator has a bug.")


if __name__ == "__main__":
    main()
