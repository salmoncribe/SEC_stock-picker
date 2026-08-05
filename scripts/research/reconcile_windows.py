"""Why does a 2020-2026 backtest disagree with the same years inside a 10-year run?

Measured discrepancy: running 2020-01-01..2026-07-31 standalone gave ~28.7%/yr
at 6 slots. The identical config inside a continuous 2016..2026 run produced
roughly 18%/yr over those same years. Same code, same seeds, same signals.

Until this is explained, no number produced tonight can be trusted. Four
candidate causes, tested one at a time:

  H1  COLD START — a standalone window begins 100% in cash and buys in over the
      first days. A continuous run enters 2020 already fully invested, so it
      eats the COVID drawdown on a full book. If this is it, every windowed
      backtest in this repo is flattered by its own start date.
  H2  ACCOUNT SIZE — the standalone runs used E0=10,000, the continuous run
      2,000. `spend = min(unit, cash)` plus the `if sp < 1.0: break` guard could
      behave differently at different scales.
  H3  SEED PATH — the RNG stream is consumed by every shuffled day, so seed N in
      a 2016-start run draws a different sequence than seed N in a 2020-start
      run. Not a bug, but it means "same seed" is not the same experiment.
  H4  ARITHMETIC — CAGR measured off different first/last equity points.

Usage:
    uv run --with duckdb --with pandas --with numpy --with pyarrow \
        python scripts/research/reconcile_windows.py
"""

from __future__ import annotations

import numpy as np

import concentration_ladder as C

SLOTS, HOLD = 6, 60
SEEDS = range(24)


def sim(by_day, px, cal, i_lo, i_hi, seed, e0):
    """Cold-start run over [i_lo, i_hi]. Returns the daily equity curve."""
    rng = np.random.default_rng(seed)
    cash, pos = e0, []
    curve = np.zeros(i_hi - i_lo + 1)
    for k, i in enumerate(range(i_lo, i_hi + 1)):
        keep = []
        for p in pos:
            pr = px[p["sym"]][i]
            if not np.isfinite(pr):
                keep.append(p); continue
            if i - p["i0"] >= HOLD or i == i_hi:
                cash += p["units"] * pr * (1 - C.COST)
            else:
                keep.append(p)
        pos = keep
        free = SLOTS - len(pos)
        if free > 0:
            held = {q["sym"] for q in pos}
            day = [c for c in by_day.get(i, []) if c in px and c not in held]
            rng.shuffle(day)
            eq = cash + sum(q["units"] * px[q["sym"]][i] for q in pos
                            if np.isfinite(px[q["sym"]][i]))
            unit = eq / SLOTS
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
        curve[k] = cash + sum(p["units"] * (px[p["sym"]][i]
                              if np.isfinite(px[p["sym"]][i]) else 0.0) for p in pos)
    return curve


def cagr(curve, cal, i_lo, i_hi):
    yrs = (cal[i_hi] - cal[i_lo]).astype("timedelta64[D]").astype(int) / 365.25
    return (curve[-1] / curve[0]) ** (1 / yrs) - 1 if curve[-1] > 0 else -1.0


def main() -> None:
    ev, wide, cal = C.load()
    px = {s: wide[s].to_numpy() for s in wide.columns}
    by_day: dict[int, list[str]] = {}
    for sym, t0 in zip(ev.sym, ev.t0):
        i = int(np.searchsorted(cal, np.datetime64(t0)))
        if i + 1 < len(cal):
            by_day.setdefault(i + 1, []).append(sym)

    i16 = int(np.searchsorted(cal, np.datetime64("2016-07-25")))
    i20 = int(np.searchsorted(cal, np.datetime64("2020-01-01")))
    i26 = min(int(np.searchsorted(cal, np.datetime64("2026-07-31"))), len(cal) - 1)
    yrs_2026 = (cal[i26] - cal[i20]).astype("timedelta64[D]").astype(int) / 365.25

    print(f"{SLOTS} slots / {HOLD}d hold, {len(list(SEEDS))} seeds")
    print(f"2020-01-01 .. 2026-07-31 = {yrs_2026:.2f} years\n")

    # --- H1/H2: cold-start standalone at two account sizes
    for e0 in (10_000.0, 2_000.0):
        cs = [cagr(sim(by_day, px, cal, i20, i26, s, e0), cal, i20, i26)
              for s in SEEDS]
        print(f"COLD start 2020, E0=${e0:>8,.0f}: median {100*np.median(cs):6.2f}%  "
              f"p10 {100*np.percentile(cs,10):6.2f}%")

    # --- H1: warm start — run from 2016, then measure only the 2020+ segment
    warm = []
    for s in SEEDS:
        full = sim(by_day, px, cal, i16, i26, s, 10_000.0)
        seg = full[i20 - i16:]
        warm.append((seg[-1] / seg[0]) ** (1 / yrs_2026) - 1 if seg[-1] > 0 else -1.0)
    print(f"WARM start (2016 run, 2020+ segment): median {100*np.median(warm):6.2f}%  "
          f"p10 {100*np.percentile(warm,10):6.2f}%")

    # --- H3: does the seed stream matter? cold-start 2020 with offset seeds
    off = [cagr(sim(by_day, px, cal, i20, i26, s + 1000, 10_000.0), cal, i20, i26)
           for s in SEEDS]
    print(f"COLD start 2020, different seed block:  median {100*np.median(off):6.2f}%")

    print("\n" + "=" * 72)
    cold = np.median([cagr(sim(by_day, px, cal, i20, i26, s, 10_000.0), cal, i20, i26)
                      for s in SEEDS])
    print(f"COLD 2020 start : {100*cold:6.2f}%/yr")
    print(f"WARM 2020 entry : {100*np.median(warm):6.2f}%/yr")
    print(f"difference      : {100*(cold-np.median(warm)):+6.2f}pp")
    print("=" * 72)
    print("\nIf COLD >> WARM, the standalone window was flattered by starting in")
    print("cash: it sat out the COVID crash it had no positions for, then bought")
    print("the bottom. Every windowed backtest in this repo inherits that bias,")
    print("and only the continuous run reflects an account that actually existed.")

    # --- how much of the gap is specifically the COVID quarter?
    iq = min(int(np.searchsorted(cal, np.datetime64("2020-04-01"))), len(cal) - 1)
    inv = []
    for s in SEEDS:
        full = sim(by_day, px, cal, i16, i26, s, 10_000.0)
        seg = full[i20 - i16: iq - i16 + 1]
        inv.append(seg[-1] / seg[0] - 1)
    cq = []
    for s in SEEDS:
        c = sim(by_day, px, cal, i20, iq, s, 10_000.0)
        cq.append(c[-1] / c[0] - 1)
    print(f"\nQ1 2020 (2020-01-01 .. 2020-04-01) total return:")
    print(f"  warm book (already invested): {100*np.median(inv):+6.2f}%")
    print(f"  cold book (starting in cash): {100*np.median(cq):+6.2f}%")


if __name__ == "__main__":
    main()
