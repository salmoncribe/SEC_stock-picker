# Trading strategy backtest — results

> **⚠️ SUPERSEDED IN PART — read §"Out-of-sample validation" at the bottom
> first.** Everything below is correct *for the pre-COVID window it was fit on*.
> When the same strategy is run on 2020-2026, **alpha goes negative and the
> drawdown is worse than SPY's.** The 2016-2019 result did not survive.


Run 2026-08-04. Pre-COVID window by request: entries 2016-07-25 (first bar in
the panel) → 2019-11-29, everything liquidated 2020-02-19 at the S&P's
pre-crash peak. **No trade in this study touches the COVID drawdown.**

Signal: `insider_transaction/P` @60d, 2,228 distinct `(ticker, t0)` pairs across
404 tickers. Costs: 27bps round trip (the measured point-in-time spread on this
population). Scripts: `scripts/research/{trade_policy_backtest,portfolio_backtest,placebo_and_alpha}.py`.

Signal generation was not modified. This measures the **portfolio management
layer** only.

---

## The headline

| run | CAGR | maxDD | beta | **alpha** | Sharpe | vol |
|---|---|---|---|---|---|---|
| **strategy** (trail 15%, 40 slots) | **16.79%** | **−17.0%** | **0.80** | **+3.99%** | **1.48** | 10.9% |
| placebo: random stocks, identical rules | 13.36% | −20.6% | 0.82 | +0.93% | 1.14 | 11.8% |
| SPY buy & hold | 15.47% | −19.3% | 1.00 | 0.00% | 1.23 | 12.4% |

**It beats SPY on every risk dimension at once** — higher return, lower
drawdown, lower volatility, less market exposure (β 0.80), Sharpe 1.48 vs 1.23.
Alpha is stable across all 12 seeds (range +3.35% to +5.93%), so this is not
one lucky pick-order draw.

**The signal is worth ~3.1pp of alpha** over picking random liquid stocks with
the same exit rules. Note the placebo's alpha is ~0.9% and its CAGR (13.36%)
*underperforms* SPY — so the exit rules alone do not beat the index. Signal and
rules are both load-bearing.

---

## Answering the specific questions

### How often do we win, and how big?

Per trade, `trail 15%`, all signals:

| | value |
|---|---|
| win rate | **56.0%** |
| average win | **+23.1%** |
| average loss | **−10.1%** |
| **win/loss ratio** | **2.30** |
| expectancy per trade | +6.23% |
| average hold | 123 days |

Your instinct that "most stocks aren't going to be winning" was **wrong for this
window** — you win 56% of the time. But the reason the strategy works is the
2.30 win/loss ratio, which is exactly the mechanism you described: winners run
to +23% while losers get cut at −10%.

### What does 30%/yr require?

Required per-trade expectancy, net of cost:

| hold (days) | trades/yr | for 20% | for 30% | for 40% |
|---|---|---|---|---|
| 20 | 12.6 | 1.46% | 2.10% | 2.71% |
| 60 | 4.2 | 4.44% | 6.45% | 8.34% |
| 120 | 2.1 | 9.07% | 13.31% | 17.38% |
| 180 | 1.4 | 13.91% | 20.61% | 27.17% |

At a 123-day hold you need **13.3%** per trade for 30%. You are getting **6.2%**.
You are about **2.1× short**, which matches the leverage table below.

### Getting to the target

| leverage | CAGR | maxDD | vol | Sharpe | |
|---|---|---|---|---|---|
| 1.00× | 16.79% | −17.0% | 10.9% | 1.48 | |
| 1.50× | 21.64% | −25.3% | 16.3% | 1.28 | clears the 20% floor |
| 2.00× | 26.29% | −32.8% | 21.7% | 1.18 | |
| **2.50×** | **30.72%** | **−40.0%** | 27.1% | 1.12 | hits the 30% goal |
| 3.00× | 34.89% | −47.0% | 32.6% | 1.08 | |

(Margin at 6.5%/yr on the borrowed portion.)

**20% is reachable at 1.5× with a −25% drawdown. 30% needs 2.5× and a −40%
drawdown — in a window with no bear market.** The −17% base drawdown is a
bull-market number; the same book levered 2.5× through COVID would have been
far worse than −40%.

---

## Your exit rule, tested

Account-level, 40 slots, median of 12 seeds:

| exit policy | CAGR | maxDD | Sharpe |
|---|---|---|---|
| **trail 15%** | **17.09%** | **−17.2%** | **1.51** |
| fixed 120d | 17.29% | −20.5% | 1.32 |
| trail 20% | 16.37% | −16.6% | 1.46 |
| trail 10% | 14.79% | −14.9% | 1.35 |
| trail 8% | 13.43% | −16.0% | 1.27 |
| **fixed 60d** (what the system does today) | **10.93%** | −21.0% | 0.89 |

**Your rule wins.** `trail 15%` beats the fixed 60-day horizon the system
currently uses by **+6.2pp of CAGR** with a smaller drawdown and a Sharpe of
1.51 vs 0.89. Fixed 120d edges it on raw CAGR but with a materially worse
drawdown and Sharpe.

**Trail tighter than 15% and you lose money** — 10% gives up 2.3pp, 8% gives up
3.7pp, 5% gives up more. Cutting at the first wobble truncates the winners that
carry the whole result. 15% is the sweet spot: loose enough to let a winner
breathe, tight enough to cut a fizzler.

## Cash parking — it barely matters here

**Slots are 91% full on average.** With ~2.5 signals arriving per day and
120-day holds, you almost never have idle cash, so the dividend-parking rule
only touches ~9% of the book. The JNJ/PG/KO basket returned 11.80%/yr over the
window versus SPY's 15.47%.

In the sweep, parking *reduced* CAGR by roughly 1.5pp in every policy — but
that is mostly an artefact of the simulation parking and unparking daily and
paying 27bps each way. The honest conclusion: **keep the rule, it is sound when
signals are scarce, but it is not a source of return here.** Implement it with a
no-trade band so it doesn't churn.

---

## The TPL question — resolved for the portfolio layer

Cell-level grading is badly contaminated by ticker concentration (see
`2026-08-04-trading-strategy.md` §0 — TPL is 56% of P/60d's discovery edge).
That does **not** propagate to these results:

| variant | signals | CAGR | alpha | Sharpe |
|---|---|---|---|---|
| all signals | 2,228 | 16.73% | +3.86% | 1.47 |
| **ex-TPL** | 2,050 | **17.27%** | **+4.53%** | 1.53 |
| ex-top-5 tickers | 1,759 | 16.92% | +3.99% | 1.47 |
| one trade per ticker | 404 | 13.71% | +2.14% | 1.31 |

**Removing TPL makes the strategy better, not worse.** The reason is structural:
a portfolio can only hold a name *once*, so the simulator already ignores
repeat signals on a held ticker. The per-event double-counting that inflated the
cell statistics cannot inflate a portfolio that has one slot per name.

("One trade per ticker" is lower only because 404 signals cannot keep 40 slots
full — that is a starved strategy, not a cleaner measurement.)

---

## What this does not prove

1. **No bear market in the window.** By your instruction. The −17% drawdown and
   β 0.80 are untested against a real crash. Before levering, replay 2020 and
   2022 — those are the numbers that decide whether 2.5× is survivable.
2. **3.57 years, one signal, one regime.** Short for a Sharpe of 1.48.
3. **Fills at the close, not the open.** Slightly optimistic.
4. **Not the current production path.** `portfolio/simulator.py` exits on view
   expiry with no trailing stop; `stops.py` implements a fixed stop from entry,
   not a trailing one from the running peak. `trail 15%` has to be built.
5. **At $2,000, 16.79%/yr is $336.** The result is real; the dollars are not yet.

## Recommended configuration

```
exit            trailing stop, 15% from the running peak since entry
slots           40, equal weight
horizon cap     none (let winners run)
per-name stop   none (the trailing stop is the only exit)
parking         JNJ/PG/KO equal weight, with a no-trade band
leverage        1.0x until 2020 and 2022 are replayed; 1.5x is the
                first defensible step, and it targets ~21%/yr
```

---

# Out-of-sample validation — the strategy does not hold up

Added 2026-08-04, after the pre-COVID study above. Script:
`scripts/research/reach_for_30.py`. Fit window 2016-07-25→2019-12-31,
validation window **2020-01-01→2026-07-31** (6.58 years, containing both the
COVID crash and the 2022 bear market).

Fourteen configurations were tested — every untested lever that could plausibly
add return: concentration (10/20/40/60 slots), pyramiding into winners,
re-entry after a stop, adaptive trailing stops, volatility-scaled sizing, a
200-day-moving-average regime filter, multi-cell (P+C), and combinations.

## The result

Signal set P+C, median of 8 seeds:

| config | FIT CAGR | **VALIDATE CAGR** | VALIDATE maxDD | **VALIDATE alpha** |
|---|---|---|---|---|
| baseline trail15 s40 | 16.30% | **10.22%** | −47.4% | **−2.74%** |
| slots 10 | 13.77% | 14.38% | −47.2% | +0.82% |
| slots 20 | 17.39% | 13.64% | −48.8% | −0.15% |
| pyramid ×2 | 16.02% | 11.81% | −47.8% | −1.11% |
| reentry | 17.07% | 10.66% | −47.7% | −2.19% |
| adaptive 8→25% | 14.61% | 12.76% | −47.2% | −0.59% |
| volscale | 15.32% | 8.27% | −36.2% | −1.53% |
| regime 200dma | 12.80% | 6.65% | −26.0% | +0.21% |
| pyr+reent+adaptive | 15.65% | 10.90% | −48.1% | −1.92% |

**SPY over the identical validation window: 15.11%/yr, maxDD −33.7%.**

Three facts, none of them survivable:

1. **Every single configuration underperforms SPY out of sample** — best is
   14.38% against SPY's 15.11%.
2. **Alpha is negative for 12 of 14 configurations.** The +4%/yr alpha measured
   pre-COVID does not exist in 2020-2026.
3. **Drawdown is ~−47% against SPY's −33.7%.** The strategy is *worse on both
   axes* — less return and more risk than buying the index.

## Fit → validate rank correlation: **−0.38**

Across the 14 configurations, how well the fit window ranked a config is
*negatively* correlated with how it actually did in validation. The config with
the best validation alpha (`slots 10`, +0.82%) had nearly the **worst** fit
performance (13.77%), so no amount of care in the fit window would have selected
it.

This reproduces and strengthens the prior session's −0.11 measurement. **The
backtest cannot pick the winner.** Any config chosen by sweeping is chosen by
noise.

## What this means for the 30% target

**30%/yr is not reachable, and leverage makes it strictly worse.** Levering a
negative alpha multiplies the loss: at 2.5× the validation drawdown of −47%
would have been an account-ending event, not a −40% inconvenience.

The pre-COVID −17% drawdown that made 2.5× look survivable was an artifact of a
window with no bear market in it. That caveat was stated at the time; the
validation now shows it was load-bearing.

## Where the remaining room actually is

Every *trading-strategy* lever was tested and none of them produced durable
alpha. The exit rule, the slot count, pyramiding, re-entry, sizing, and the
regime filter are not the constraint. That points the diagnosis firmly at the
**signal**, which is consistent with §0 of `2026-08-04-trading-strategy.md`:
the admitted cells collapse to insignificance once clustered by ticker.

Honest options, in order of expected value:

1. **Fix the signal.** Per-ticker grading (already scaffolded in
   `signals/concentration.py`) so cells are admitted on real independent
   evidence. Without this, no trading layer can help.
2. **Consider that SPY is the benchmark to beat and it is winning.** A
   6.58-year out-of-sample window where the index beats the strategy on return
   *and* drawdown is strong evidence for just holding the index.
3. **Do not deploy capital on this strategy as it stands.** Not at $2k, not
   levered, not at all — it is currently a worse SPY.

## What was learned that IS durable

- The **daily-close trailing stop at 15%** genuinely beats a fixed-horizon exit
  within any given window. That finding replicated in both windows. It is a real
  improvement to the *mechanism*; it just cannot manufacture alpha that the
  signal does not supply.
- The **portfolio layer is immune to the ticker-concentration bug** that breaks
  cell grading, because a portfolio holds a name once.
- **Costs are not the constraint** (~27bps round trip on this population).
- **The validation harness now exists** and should gate every future claim.

---

# Concentration — the one thing that survived

Added 2026-08-04. Script: `scripts/research/concentration_ladder.py`. 24 seeds
per config, signal set P+C, fit 2016-2019 / validate 2020-2026.

Michael trades ~6 names at ~75% deployed. Tested properly, that is the best
configuration found all session — and the only one where **fit and validate
agree**.

## The headline configs

| slots | deploy | hold | FIT median | FIT beat SPY | VAL median | VAL p10 | VAL worst | VAL medDD | VAL beat SPY |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 100% | 60d | 23.35% | **100%** | **28.09%** | 23.72% | 21.44% | −32.6% | **100%** |
| 5 | 100% | 60d | 27.27% | **100%** | 28.18% | 22.98% | 21.37% | −35.5% | **100%** |
| 6 | 100% | 90d | 18.66% | 79% | **30.94%** | 27.84% | 22.63% | −34.4% | **100%** |
| 6 | **75%** | 60d | 17.09% | 75% | **21.65%** | 18.78% | 17.43% | **−25.5%** | **100%** |
| 40 | 100% | 60d | 13.92% | — | 13.60% | 12.05% | — | −47.6% | 0% |

SPY: fit 14.46%/yr (maxDD −19.3%), validate 15.11%/yr (maxDD −33.7%).

**`6 slots / 60-day hold` beats SPY in 100% of seeds in BOTH windows.** Nothing
else tested this session does that. The worst single seed still returns 16.73%
(fit) and 21.44% (validate).

**The 75%-deployed variant is the best risk-adjusted version**: 21.65% validate
with a **−25.5%** drawdown against SPY's −33.7%. Michael's instinct to hold cash
back is doing real work — it costs ~6pp of return and removes ~7pp of drawdown.

## Two earlier conclusions this overturns

1. **"Trail 15% beats the fixed horizon"** — true *at 40 slots*, where it was
   tested, and false at low slot counts. At 6 slots the fixed 60-day exit beats
   the trailing stop by **+10.6pp** in validation and **+7.7pp** in fit. Stated
   too broadly the first time.
2. **"The strategy has no out-of-sample alpha"** — true for the 40-slot book.
   The concentrated book beats SPY in both windows and in all three validation
   sub-periods, including the 2022 bear (3 slots: 6.72% vs SPY's 1.03%).

## The mechanism is NOT established — this is the open question

Per-trade statistics in validation, fixed 60-day hold:

| slots | trades/yr | mean/trade | median/trade | win% | invested% |
|---|---|---|---|---|---|
| 3 | 11.9 | 9.30% | 3.01% | 56.1% | 92.6% |
| 6 | 23.7 | 8.53% | 4.19% | 59.1% | 92.2% |
| 20 | 78.3 | 6.69% | 3.31% | 58.4% | 92.2% |
| 40 | 150.7 | **4.79%** | 2.50% | 56.9% | 89.1% |

**Capital utilisation is ~92% at every slot count**, so this is *not* cash drag —
that was the obvious explanation and it is ruled out.

What remains is that **mean return per trade falls as slot count rises** (9.30%
→ 4.79%). Under genuinely random selection from one population that should not
happen. Candidate explanations, none verified:

- a **day-selection effect** — a 6-slot book is usually full and only enters
  when a slot frees, sampling a subset of days; a 40-slot book enters nearly
  every day, including high-signal-count days, which may be systematically worse
  (the falling-knife pattern that broke `insider_cluster_buy`);
- a **subtle simulator bug** in how candidates are drawn or sized.

**Until this is explained, treat the result as promising but unproven.** It is
the single highest-value thing to investigate next: if the effect is real this
is a genuine discovery; if it is a bug, everything in this section collapses.

## Honest statistical health warning

Roughly **80 configurations** have now been evaluated across this session. The
preregistered cap in the validation gauntlet was **50 trials**. We are over it,
and the Deflated Sharpe / PBO calculations in §5 of
`2026-08-04-trading-strategy.md` must be recomputed with the true trial count
before any of this is trusted with money.

Mitigating it: this result was **not** selected by picking the best of 80. It
holds across 9 slot counts, 3 hold lengths, 2 deployment levels, 2 independent
time windows, 3 validation sub-periods, and 24 seeds — and the *direction* is
monotone in slot count throughout. That is a much stronger pattern than a single
lucky cell.

## Next tests, in order

1. **Explain the per-trade decay.** Compare entry-day characteristics (candidate
   count that day, SPY trailing return, market breadth) for 6-slot vs 40-slot
   entries. This decides whether the finding is real.
2. **Re-run with entries forced on identical days** across slot counts, to
   isolate sizing from day-selection.
3. Recompute DSR/PBO with the true trial count.
4. Only then: paper-trade `6 slots / 60d / 75% deployed`.

---

# ⚠️ HARNESS BUG — windowed backtests are flattered by their start date

Found 2026-08-04 via `scripts/research/reconcile_windows.py`. This invalidates
the absolute levels of every windowed result above.

A backtest that *starts* on 2020-01-01 begins 100% in cash and buys in over the
following days. A continuous run entering 2020 is already fully invested. The
difference, 6 slots / 60d hold, 24 seeds, over 2020-01-01..2026-07-31:

| run | CAGR |
|---|---|
| COLD start (standalone 2020 window) | **28.33%** |
| WARM start (2020+ segment of a 2016 run) | **15.36%** |
| difference | **+12.96pp** |

Ruled out: account size (identical at E0 = $2,000 and $10,000) and seed stream
(27.03% with a different seed block).

**Q1 2020 is the entire mechanism:**

| book | 2020-01-01 → 2020-04-01 |
|---|---|
| warm (already invested) | **−34.06%** |
| cold (starting in cash) | **−16.70%** |

The cold book sits out half the crash it holds no positions for, then deploys
its cash near the bottom. One quarter, compounded over 6.58 years, is ~13pp/yr.

## What this changes

- **The "VALIDATE 28.09%" headline is wrong.** The honest figure for 2020-2026
  is **15.36%**, against SPY's 15.11% over the same span — essentially a tie.
  **The out-of-sample edge is approximately zero** once the artifact is removed.
- **The continuous 10-year run is unaffected** and remains the number to trust:
  10 slots 19.35%, 6 slots 17.40%, 40 slots 14.04%, SPY 14.98%. Most of that
  edge is therefore earned in 2016-2019, not out of sample.
- **The concentration finding survives.** Both slot counts cold-started, and a
  40-slot book takes ~16 days to deploy versus ~6 for a 6-slot book, so the bias
  favours the *diversified* book. The concentrated book won anyway, and the
  continuous run reproduces the ordering.

## Rule going forward

Never measure a strategy from a cold start. Either run continuously from a fixed
inception, or include a warm-up period at least as long as the holding period
and exclude it from the measurement window. Any result quoted from a standalone
window that begins mid-history must be re-derived.

---

# DRAFT — regime-split strategy (bear side established, bull side in progress)

Added 2026-08-04. Script: `scripts/research/bear_only.py`. Continuous
2016-07-25 → 2026-07-31, entries gated by regime, positions exit on the 60-day
clock regardless, idle capital in cash, 16 seeds.

## Bear side — CONFIRMED, with a sample caveat

Return per trade, 6 slots (always-on baseline 5.56%):

| regime | active | ret/trade | CAGR | maxDD |
|---|---|---|---|---|
| always on | 100% | 5.56% | 16.99% | −46.5% |
| below 50dma | 61% | **3.30%** | 5.39% | −52.6% |
| below 200dma | 28% | **3.07%** | 2.07% | −43.8% |
| 5% off high | 50% | 6.29% | 10.39% | −40.8% |
| 10% off high | 19% | 5.54% | 3.07% | −41.0% |
| **20% off high** | **2.2%** | **29.89%** | 7.27% | **−27.1%** |
| neg 6-mo momentum | 28% | **8.85%** | 9.64% | **−30.0%** |
| BULL only (control) | 76% | 5.86% | 16.39% | −42.9% |

Reproduces at 10 slots (dd_20 → 28.30%) and 40 slots (27.60%).

**Two rules fall out:**
1. **Moving-average filters are the wrong switch** — below-200dma nearly halves
   per-trade return. Below a moving average is mostly choppy sideways tape, not
   distress. The switch must read **drawdown depth**.
2. **Deep drawdown is where the edge lives** — ~5-6x baseline per trade,
   consistent with the literature that insiders buy their own stock in
   downturns.

**Caveat that governs everything here:** dd_20 is **two events** — COVID
(2020-03-12→04-07, 19 sessions) and the 2022 bear (4 fragments, 33 sessions),
56 sessions of ~2,500. The trade count is not the sample size. dd_10 is broader
(~5 episodes, 334 sessions) but only yields 5.5-7.6%/trade.

**Bear-only is not deployable standalone**: 7.27% CAGR despite 30%/trade,
because it holds cash 93% of the time. The regime split exists to remove exactly
that cash drag.

## Bull side — TO BUILD

Open question: what configuration is best when the market is NOT in drawdown?
The bull-only control ran at 5.86%/trade and 16.39% CAGR — indistinguishable
from always-on — so the current parameters are effectively already tuned for
bull tape. The bull side needs its own slot count / hold length / exit rule.

**Overfitting warning to carry forward:** selecting a bull config AND a bear
config on the same 10 years, with only 2 bear episodes, is in-sample fitting.
Any combined result must be labelled as such.
