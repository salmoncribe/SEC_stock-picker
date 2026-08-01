# Backtest findings — the admitted cells as a portfolio

**Date:** 2026-07-29
**Status:** research complete; two structural findings change what the gate should admit
**Code:** `analytics/backtest.py`, `analytics/backtest_data.py`, `signals backtest` CLI
**Reproduce:** `market-intelligence signals backtest --holdout`

## What was measured

The portfolio replay in `analytics/backtest.py` had never been run against the
real database. This study wired it to the stored verdicts — 12 admitted cells in
`signal_status`, 1.52M daily bars, 633 symbols, 2016-10 → 2026-07 — and asked
what a portfolio trading those cells would actually have earned.

Discovery is 2016-10 → 2022-12; holdout is 2023-01 → 2026-07. Every parameter
and every cell-selection decision below was made on discovery. The holdout was
read once, at the end.

## Headline

| run | discovery | holdout |
|---|---|---|
| as-was | **−87.15%** | **−68.60%** |
| after the fixes below | **+19.39%** (DD 13.68%, Sharpe 0.28, PF 1.119) | **+11.02%** (DD 6.37%, Sharpe 0.30, PF 1.212) |

Holdout per-trade return (+0.982%) and profit factor (1.212) came in *above*
discovery, which is the shape you want: the choices were not fitted to the
discovery sample.

The honest reading of +11.02% over 3.6 years is ~2.9%/yr, market-hedged, at 6.4%
max drawdown and Sharpe 0.30. Real, out-of-sample, and thin.

## Finding 1 — most of the validated edge is not tradeable at all

`analytics/returns.py` defines

```
abnormal_return = total_return - (alpha + beta * market_return)
```

A trade can hedge `beta * market_return` by shorting an index. **Nothing hedges
`alpha`** — it is the name's own trailing drift over the window, and subtracting
it from the label removes exactly the drift a real position still eats.

Signed edge per admitted cell, weighted across all samples:

| definition | discovery | holdout |
|---|---|---|
| label (what the gate graded) | +0.763% | +0.962% |
| beta-hedged (what a trade can capture) | **−0.078%** | **+0.099%** |
| unhedged directional | −0.519% | −0.918% |

Per cell, the split is stark. The **sale** cells are artifacts of the alpha term:

| cell | label | beta-hedged | alpha term |
|---|---|---|---|
| `insider_transaction/C/20d` | +3.343% | **−1.504%** | −4.847% |
| `insider_transaction/S/20d` | +1.254% | **−0.293%** | −1.547% |
| `insider_transaction/M/20d` | +0.631% | **−0.076%** | −0.707% |
| `insider_transaction/G/20d` | +0.364% | **−0.219%** | −0.583% |

Insiders sell into strength, so a sale cell's measured edge *is* the subtracted
trailing alpha. The **purchase** cells survive, and improve, under hedging:

| cell | label | beta-hedged (disc) | beta-hedged (holdout) |
|---|---|---|---|
| `insider_transaction/P/20d` | +0.841% | **+1.604%** | **+1.662%** |
| `insider_cluster_buy/20d` | +1.847% | +1.113% | +0.119% |
| `insider_transaction/P/5d` | +0.433% | +0.624% | +0.986% |
| `insider_transaction/P/1d` | +0.375% | +0.413% | +0.671% |

`backtest_data.filter_tradeable` implements the consequence: a cell only trades
if its discovery beta-hedged edge clears `DEFAULT_MIN_HEDGED_EDGE` (15 bps, the
cost of trading it). That drops 5 of 12 cells — every one a sale cell.

**Implication for the gate.** `signals evaluate` admits on label CAR, so it
admits cells with no tradeable edge and no amount of execution tuning fixes
them. The gate should measure the beta-hedged edge alongside the label CAR, and
require both.

## Finding 2 — the label overstates what a compounding position earns

`forward_abnormal_return` is a *cumulative sum* of daily abnormal returns. A real
position compounds: `compounded ≈ sum − ½σ²H`. For the volatile small caps
insiders buy, 20 days at ~5% daily vol is ~2.5% of drag — the same order as the
entire claimed edge, and largest exactly where the signal looks strongest.

Trade-level attribution (3,000 discovery trades, all matched to their signal):

```
all 58,651 signals    : paper raw −0.8013%   paper hedged −0.0320%
the 3,000 traded      : paper raw +0.9706%   paper hedged +0.6912%
realised by the replay: +0.0570% per trade   (median +0.3583%)
```

The traded subset carries the edge, so this is not a selection problem. The
mean-vs-median gap (+0.057% vs +0.358%) is the fat left tail; the paper-vs-
realised gap is volatility drag.

## Finding 3 — the replay's own geometry was the largest single loss

The as-was configuration risked 2×ATR (typically 4–8% of price) to capture a
~1% mean move: reward:risk ≈ 0.15, which needs an ~87% win rate to break even.
It ran at 74%. Exit counts confirm the mechanism — 4,806 targets against 1,631
stops, and still −87%.

Stop width, swept on discovery, has a clean interior optimum:

| stop | discovery return | avg/trade | profit factor |
|---|---|---|---|
| 2×ATR | −65.85% | −0.581% | 0.782 |
| 3×ATR | −18.21% | −0.516% | 0.885 |
| 6×ATR | +20.20% | +0.453% | 1.098 |
| **8×ATR** | **+17.89%** | **+0.538%** | **1.103** |
| 25×ATR (≈none) | +4.80% | +0.057% | 1.082 |

8×ATR was chosen over the 6×ATR argmax because it sits at the centre of the
profitable plateau under *both* sizing modes — a less knife-edged pick.

## Changes made

In `analytics/backtest.py`:

1. **`exit_style="horizon"` (new default).** Hold the cell's calibrated window
   and exit at the close; the stop becomes a disaster brake, not a profit target.
2. **Horizons count trading days.** The old code used `as_of + timedelta(days=H)`,
   so a 20-bar cell was closed after ~14 bars.
3. **`hedge_symbol`.** Beta-weighted index offset, rebalanced daily, sized from
   `load_betas` — median market-model beta fitted strictly before the split date,
   so the hedge ratio is knowable at decision time.
4. **`sizing="equal_weight"`.** `shares = risk_budget / stop_distance` welds
   exposure to stop width, so widening the stop to stop noise-triggered exits
   silently shrank every position toward zero. Equal weight decouples them.
5. **`atr_lookback_bars=56`.** Matches the `atr_period * 4` window the live
   `signals.trade_alerts` path passes; the replay had been smoothing ATR over ten
   years of history and sizing trades production would never have taken.
6. **`max_gross_exposure_pct`.** Short proceeds previously financed an unbounded
   book — there was no cap covering the short side at all.
7. **Ranked capacity.** Scarce slots go to the largest expected move rather than
   to whichever ticker sorts first alphabetically.
8. **Bar date index and a bankruptcy stop.** O(1) day lookups instead of linear
   scans per position per day; a book replayed from non-positive equity stops.

## Caveats

- **No borrow cost, no dividends, no financing.** The surviving cells are almost
  all long, so borrow matters little now, but the hedge leg is short and unpriced.
- **Volume is not consulted.** No participation cap, so the replay assumes any
  size fills at the open.
- **A $10k account cannot harvest this.** At 1% weights integer share rounding
  dominates: the same config at $10k trades 427 times against 3,000 at $1M. The
  reported figures use $1M to measure the edge rather than the rounding.
- **`config/settings.yaml` was not changed.** `trading.atr_stop_multiple: 2.0`
  and the live trade-alert path still use the old geometry. Finding 3 says that
  geometry is the losing one — but changing what the live system texts is a
  separate decision, not a backtest result.

## Recommended next steps

1. Measure the beta-hedged edge inside `signals evaluate` and require it for
   admission. Cells admitted on label CAR alone are not tradeable.
2. Grade cells on compounded, not summed, forward returns.
3. Re-run the propagation cells (`build_propagation`) through the same
   tradeability screen before any of them reach the alert path.
4. Decide whether the live `trading:` block should adopt the horizon-hold, wide-
   stop geometry this study found, or whether the alert layer stays as-is.
