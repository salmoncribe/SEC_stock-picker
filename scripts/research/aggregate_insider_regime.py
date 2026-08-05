"""Do insiders, in aggregate, tell us where the MARKET is going?

Michael's idea: stop asking "what does the market regime say about this trade"
and start asking "what does insider filing activity say about the coming
regime". There is strong academic support -- Seyhun (1992, QJE) found aggregate
net insider purchases predict up to 60% of the variation in one-year-ahead
aggregate stock returns, and later work (Lakonishok & Lee 2001; the
opportunistic-vs-routine literature) confirms the aggregate signal while
showing it is the OPPORTUNISTIC trades that carry it.

Builds market-wide daily indicators from Form 4 flow and tests whether they
predict forward SPY returns and regime transitions.

Two measurement traps this handles explicitly:

  * SEASONALITY. Insiders cannot trade during blackout windows around earnings,
    so raw filing counts have hard quarterly periodicity. Every indicator is
    z-scored against its own trailing 252-day history, which removes the level,
    the trend, and most of the seasonal shape.
  * POINT-IN-TIME. Indicators are lagged 2 sessions before being matched to
    forward returns, so nothing is measured against information that was not
    yet public.

Purchases are rare relative to sales (7,077 vs 61,489 in this panel) because
insiders receive stock as compensation and sell it routinely. That asymmetry is
why the ratio, not the raw count, is the signal.

Usage:
    uv run --with duckdb --with pandas --with numpy --with pyarrow \
        python scripts/research/aggregate_insider_regime.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

PRICES = "/Volumes/Extreme/home-migrated/quant/data/parquet/daily_prices/*/*.parquet"
SAMPLES = "/Volumes/Extreme/home-migrated/quant/data/parquet/event_samples/**/*.parquet"
LAG = 2          # sessions between an indicator and its first tradeable use
WIN = 21         # trailing window for the flow measures
Z = 252          # trailing window for the z-score


def load():
    con = duckdb.connect()
    con.execute("set memory_limit='8GB'; set threads=6;")
    ev = con.execute(f"""
        select distinct target_ticker as sym, t0, event_subtype as code
        from read_parquet('{SAMPLES}', union_by_name=true)
        where event_type='insider_transaction' and horizon_days=60
          and event_subtype in ('P','S','A','M','F')
    """).df()
    spy = con.execute(f"""
        select price_date, adj_close from read_parquet('{PRICES}', hive_partitioning=true)
        where symbol='SPY' and adj_close>0 order by price_date
    """).df()
    spy["price_date"] = pd.to_datetime(spy.price_date)
    ev["t0"] = pd.to_datetime(ev.t0)
    return ev, spy


def main() -> None:
    ev, spy = load()
    cal = pd.DatetimeIndex(spy.price_date)
    px = spy.adj_close.to_numpy()

    # daily counts per transaction code, on the trading calendar
    ev["day"] = cal[np.searchsorted(cal.values, ev.t0.values).clip(0, len(cal) - 1)]
    piv = ev.pivot_table(index="day", columns="code", aggfunc="size", fill_value=0)
    piv = piv.reindex(cal, fill_value=0)
    # distinct companies buying each day, the breadth measure
    brd = ev[ev["code"] == "P"].groupby("day").sym.nunique().reindex(cal, fill_value=0)

    P = piv.get("P", pd.Series(0, index=cal)).rolling(WIN).sum()
    S = piv.get("S", pd.Series(0, index=cal)).rolling(WIN).sum()
    B = brd.rolling(WIN).sum()

    ind = pd.DataFrame(index=cal)
    ind["net_ratio"] = (P - S) / (P + S)             # Seyhun's net measure
    ind["buy_share"] = P / (P + S)
    ind["breadth"] = B
    ind["buy_count"] = P
    ind["sell_count"] = S
    # z-score each against its own trailing year -- kills level and seasonality
    for c in list(ind.columns):
        m = ind[c].rolling(Z, min_periods=120).mean()
        sd = ind[c].rolling(Z, min_periods=120).std()
        ind[c + "_z"] = (ind[c] - m) / sd

    zc = [c for c in ind.columns if c.endswith("_z")]
    sig = ind[zc].shift(LAG)          # point-in-time lag

    # forward SPY returns
    fwd = {}
    for h in (21, 63, 126, 252):
        fwd[h] = pd.Series(np.concatenate([px[h:] / px[:-h] - 1.0,
                                           np.full(h, np.nan)]), index=cal)

    print(f"panel {cal[0].date()} .. {cal[-1].date()}, {len(cal)} sessions")
    print(f"indicators: {WIN}-session flow, z-scored on {Z} sessions, lagged {LAG}\n")
    print("SPEARMAN rho — indicator (t) vs forward SPY return")
    print(f"{'indicator':<16}" + "".join(f"{f'+{h}d':>10}" for h in fwd))
    print("-" * 56)
    for c in zc:
        row = f"{c.replace('_z',''):<16}"
        for h in fwd:
            d = pd.concat([sig[c], fwd[h]], axis=1).dropna()
            row += f"{d.iloc[:,0].corr(d.iloc[:,1], method='spearman'):>10.3f}"
        print(row)

    print("\nFORWARD SPY RETURN BY INDICATOR QUINTILE (net_ratio_z)")
    print(f"{'quintile':<12}" + "".join(f"{f'+{h}d':>10}" for h in fwd) + f"{'n':>8}")
    print("-" * 60)
    d = pd.concat([sig["net_ratio_z"].rename("s")] +
                  [fwd[h].rename(f"f{h}") for h in fwd], axis=1).dropna()
    d["q"] = pd.qcut(d.s, 5, labels=["Q1 most selling", "Q2", "Q3", "Q4",
                                     "Q5 most buying"])
    for q, g in d.groupby("q", observed=True):
        print(f"{str(q):<12}" + "".join(f"{100*g[f'f{h}'].mean():>9.2f}%" for h in fwd)
              + f"{len(g):>8}")

    # does it lead the drawdown regime?
    s = pd.Series(px, index=cal)
    dd = s / s.rolling(252, min_periods=60).max() - 1.0
    print("\nINDICATOR LEVEL AROUND DRAWDOWN STATES (contemporaneous)")
    st = pd.cut(dd, [-1, -0.20, -0.10, -0.05, 0.01],
                labels=["<-20%", "-20..-10%", "-10..-5%", "0..-5%"])
    t = pd.concat([sig["net_ratio_z"].rename("z"), st.rename("state")], axis=1).dropna()
    print(t.groupby("state", observed=True).z.agg(["mean", "median", "count"]).round(2).to_string())

    print("\nDOES A BUYING SPIKE PRECEDE A RECOVERY?")
    print("forward SPY return when the indicator is in its top decile, split by")
    print("whether the market was already in a drawdown at the time:")
    d2 = pd.concat([sig["net_ratio_z"].rename("z"), dd.rename("dd")] +
                   [fwd[h].rename(f"f{h}") for h in fwd], axis=1).dropna()
    hi = d2[d2.z >= d2.z.quantile(0.90)]
    for label, sub in (("in drawdown (<-10%)", hi[hi.dd <= -0.10]),
                       ("not in drawdown", hi[hi.dd > -0.10])):
        if len(sub) < 20:
            print(f"  {label:<22} n={len(sub)} — too few to report")
            continue
        print(f"  {label:<22}" + "".join(f"{100*sub[f'f{h}'].mean():>9.2f}%" for h in fwd)
              + f"  n={len(sub)}")
    print(f"  {'baseline (all days)':<22}"
          + "".join(f"{100*d2[f'f{h}'].mean():>9.2f}%" for h in fwd) + f"  n={len(d2)}")


if __name__ == "__main__":
    main()
