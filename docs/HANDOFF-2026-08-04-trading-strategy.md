# Handoff — trading strategy research, 2026-08-04 evening

Self-contained. Assumes no knowledge of the session that produced it. Written
for a fresh context starting on a new branch.

Companion docs (read in this order if you need depth):
1. `docs/HANDOFF-2026-08-04.md` — the *earlier* session, signal-side findings
2. `docs/plans/2026-08-04-trading-strategy.md` — the design + validation gauntlet
3. `docs/plans/2026-08-04-trading-strategy-RESULTS.md` — all backtest results
4. `docs/plans/2026-08-04-robinhood-mcp-integration.md` — broker integration (parked)

---

## 1. The one-paragraph state

Michael has **$2,000** in a Robinhood account and wants **30%/yr** (20% floor).
This session built and tested the **trading strategy layer** — entry, sizing,
holding period, exit — on top of the existing SEC insider signals. The honest
answer is **~21%/yr, Sharpe 0.95, max drawdown −41%**, versus SPY's 14.98% and
Sharpe 0.87 over the same 10 years. **30% is not reachable without leverage,
and leverage is not currently justified.** Six separate ideas were tested and
five converged on "the simple always-invested version already does it." Nothing
is deployed; no money is at risk; all research lives in `scripts/research/`.

---

## 2. THE STRATEGY (current best, and it is simple)

```
universe    insider_transaction subtypes P (open-market purchase)
            and C (derivative conversion), the 60d-horizon sample rows
slots       10, equal weight
entry       next session's close after the signal date; skip if already held
hold        120 trading sessions, FIXED. no trailing stop.
exit        time only
regime      do not OPEN new positions while SPY is 10%+ below its
            trailing 252-day high (defensive only)
cash        fully invested otherwise; no dry powder
leverage    none
```

**Measured, continuous 2016-07-25 → 2026-07-31, 16–24 seeds, 27bps round-trip
cost, starting $2,000:**

| | strategy | SPY |
|---|---|---|
| CAGR | **~21%** | 14.98% |
| Sharpe | 0.95 | 0.87 |
| max drawdown | −41% | −33.7% |
| $2,000 becomes | ~$14,100 | ~$8,100 |

The 10-slot / 120-day / always-invested control beat every more sophisticated
variant tried.

---

## 3. Established findings (each survived a real attempt to kill it)

### 3.1 Concentration beats diversification, ~+9 to +10pp/yr
Few big positions beat many small ones. Per-trade return **falls** as slot count
rises: 3 slots 9.30%, 6 slots 8.53%, 20 slots 6.69%, 40 slots 4.79%.

Seed-level paired test under a max-1-entry-per-day control:
**+2.05pp, t=8.85, p<0.0001, 22/24 seeds (validate)** and **+2.45pp, t=9.51,
23/24 (fit)**.

Ruled out as causes: cash drag (utilisation ~92% at every slot count), position
sizing (per-trade return is price-based), busy days being worse (rho=+0.04),
per-day concentration (explains half, +2.05pp survives), repeat signals
outperforming (rho=+0.007), TPL contamination (gap is a stable +9 to +10pp in
every ticker-removal variant).

**The mechanism is STILL UNIDENTIFIED.** Best remaining clue: the books take
structurally different trades — mean signal ordinal 141 (6 slots) vs 50 (40
slots), because a 40-name book usually already holds the high-frequency tickers
and the `not already held` rule excludes them. But ex-top-5 still shows the full
gap, so that is not sufficient. **This is the highest-value open question.**

### 3.2 Hold length depends on slot count — they interact
- At **40 slots**: a 15% trailing stop beats a fixed horizon (+3.3pp).
- At **6–10 slots**: the fixed horizon beats the trailing stop by **+10.6pp**
  (validate) and **+7.7pp** (fit).
- In the bull regime, **all four 120-day configs swept the top** (21.61, 20.45,
  19.95, 16.61) while 60-day holds sat at 11–15%.

Do not quote "trailing stop beats horizon" without the slot count attached.

### 3.3 Costs are not the constraint
Point-in-time Corwin–Schultz on the actual `/P` population: **median 16bps,
mean 27bps**, median dollar volume **$109M/day**. These are large caps (ED, IFF,
ABT, AMZN, T), not microcaps. The prior session's "118bps kills it" conclusion
was measured on the *cluster-buy* population and does not transfer.

### 3.4 Insiders buy the dip, and that dip is buyable
Aggregate insider buy/sell z-score by market state is monotone: calm −0.04,
−5..−10% +0.62, −10..−20% +0.90, below −20% **+1.74**.

Conditional forward SPY returns when aggregate insider buying hits its top
decile:

| condition | +21d | +63d | +126d | +252d | n |
|---|---|---|---|---|---|
| spike WHILE in drawdown | +5.38% | +9.38% | +14.13% | **+27.55%** | 84 |
| spike in calm market | −1.40% | −3.74% | −1.45% | +5.25% | 121 |
| baseline | +1.27% | +3.79% | +7.54% | +15.42% | 2047 |

**But aggregate insider flow does NOT predict the market on its own** —
Spearman rho ≈ −0.09, slightly backwards. Seyhun (1992, QJE)'s 60% R² does not
replicate on this panel. The signal is a "this dip is buyable" detector, not a
regime forecaster.

### 3.5 The green light is a LEVERAGE trigger, not a cash trigger
Green light = drawdown ≤ −10% AND insider z ≥ expanding p90. Fires 3.5% of
sessions across **7 episodes** (Jan 2019, Mar–Apr 2020, four in 2022, Apr 2025).
Green-light trades return **27.93%/trade vs 11.35% normal**, and **51.91%** if
held 252 days instead of 120.

**Yet no way of exploiting it beat the control**: dry powder 15% → 18.38%,
25% → 17.00%, concentrate → 18.82%, ride-252d → 20.24%, all versus the control's
21.55%. Reason: an always-invested book *already buys* those trades. Extracting
more requires capital you didn't have — i.e. leverage. Revisit when the account
supports margin.

---

## 4. Ideas that were tested and FAILED

| idea | result |
|---|---|
| bear-market-only trading | 7.27% CAGR — 93% in cash, drag kills it |
| moving-average regime filter | **actively harmful** — below-200dma cuts per-trade return to 3.07% from 5.56% |
| regime switching (bull cfg / bear cfg) | best switch +3.3pp over control, but bear-config choice swings results 6.5pp on 2 episodes = noise |
| trailing stop at low slot counts | −10.6pp vs fixed horizon |
| pyramiding into winners | no durable alpha |
| re-entry after stop | no durable alpha |
| vol-scaled sizing | no durable alpha |
| 75% deployment (cash buffer) | good in sub-windows, **loses over the full 10 years** (13.49%, beats SPY only 42%) |
| leverage to reach 30% | 2.5× → −40% modelled drawdown on a pre-COVID window; the honest out-of-sample drawdown is −45%, so 2.5× is account-ending |

---

## 5. METHODOLOGY TRAPS — read before running any backtest

### 5.1 Cold-start bias — worth +13pp/yr, found the hard way
A backtest that *starts* mid-history begins 100% in cash and buys in over the
following days. A continuous run reaching that date is already invested.
Measured on 2020-01-01: **cold start 28.33%/yr vs warm start 15.36%/yr.**
Q1 2020 alone: warm book −34.06%, cold book −16.70%.

**Rule: never measure from a cold start.** Run continuously from a fixed
inception, or warm up for at least the holding period and exclude it.

This invalidated the session's "2020–2026 validate = 28%" headline. The honest
figure is 15.36% vs SPY's 15.11% — **an out-of-sample tie.**

### 5.2 Per-event t-statistics are fiction on this corpus
Events repeat on the same tickers. Every admitted cell collapses when weighted
per-ticker:

| cell | split | per-event | per-TICKER | top ticker |
|---|---|---|---|---|
| P/60d | discovery | +3.04% t=8.01 | +2.35% **t=3.09** | TPL 56.4% |
| P/60d | holdout | +1.33% t=2.84 | +0.35% **t=0.53** | TPL **120.2%** |
| C/120d | discovery | +11.67% t=11.44 | +0.93% **t=0.26** | CRWD 37.4% |
| D/20d | discovery | −0.51% t=−2.50 | −0.07% **t=−0.17** | PENN −16.1% |

**TPL contributes >100% of P/60d's holdout edge** — ex-TPL it is negative.
`C` is only 43–44 distinct tickers.

`signals/impact.py::_placebo_samples` randomises the *control* ticker, so it
cannot detect concentration in the *treatment* group. A scaffold exists at
`src/market_intelligence/signals/concentration.py` — `per_ticker_stats()` is
implemented, `check_concentration()` raises `NotImplementedError` pending an
admission-policy decision.

**Note the nuance:** this breaks *cell grading* but NOT *portfolio results*,
because a portfolio holds a name once. Ex-TPL portfolio results are slightly
*better*, not worse.

### 5.3 The backtest cannot pick the winner
Fit(2016–19) → validate(2020–26) **rank correlation across 14 configs: −0.38.**
The best validation config had nearly the worst fit result. An earlier session
measured −0.11 on a different sweep. **Never select parameters by sweeping.**
Choose by mechanism, then verify pass/fail.

### 5.4 Trial count is blown
**~100 configurations were evaluated this session** against a preregistered
50-trial cap. Deflated Sharpe and PBO (see §5 of
`docs/plans/2026-08-04-trading-strategy.md`) **must be recomputed with the true
count** before anything is trusted with money.

Partial mitigation: the concentration effect is monotone across 9 slot counts,
3 holds, 2 deployment levels, 2 windows, 3 sub-periods and 24 seeds — a pattern,
not a lucky cell.

### 5.5 `event_samples` fans out duplicates
10,809 `/P` @120d rows collapse to **4,447 distinct `(ticker, t0)` pairs**.
Always `SELECT DISTINCT target_ticker, t0`. The sample builder needs fixing.

### 5.6 Good years are 3-name dependent, increasingly so
P&L concentration by year (top-3 share): 2017 35%, 2018 **81%**, 2019 59%,
2020 46%, 2021 41%, 2023 48%, 2024 **76%**, 2026 YTD **93%**. 2024 was
TPL 28.6% + COIN 23.7% + HOOD 23.2%. 2020 was the healthiest (99 distinct
names). This is a variance-plus-shots-on-goal strategy, not a picking-skill one.

---

## 6. Scripts (all in `scripts/research/`, all re-runnable)

| script | what it answers |
|---|---|
| `trade_policy_backtest.py` | per-trade exit-policy sweep; win rate, W/L, what 30% requires |
| `portfolio_backtest.py` | account-level sim with cash parking |
| `placebo_and_alpha.py` | random-stock placebo, alpha/beta decomposition, leverage table |
| `reach_for_30.py` | 14-config lever sweep, fit vs validate |
| `concentration_ladder.py` | slot-count ladder with full seed distribution — **the shared helper other scripts import** |
| `why_concentration.py` | mechanism tests (population, instrumented, controlled) |
| `reconcile_windows.py` | the cold-start bias proof |
| `bear_only.py` | six bear-regime definitions |
| `bull_side.py` | bull-regime config grid |
| `regime_switch.py` | combined bull/bear switching |
| `aggregate_insider_regime.py` | aggregate insider flow as market predictor |
| `green_light.py` | the buyable-dip sizing rule |

Run pattern:
```bash
cd scripts/research && uv run --with duckdb --with pandas --with numpy --with pyarrow python <script>.py
```

`concentration_ladder.py` exposes `load()`, `simulate()`, `SEEDS`, `COST`, `E0`
and is imported by several others — change it carefully.

---

## 7. Data and constraints

- **Read the parquet mirror**, not DuckDB: `/Volumes/Extreme/home-migrated/quant/data/parquet`.
  The `ai.quant.autopilot` launchd agent holds the DuckDB write lock.
- `daily_prices` has `open/high/low/close/adj_close/volume`, 4,005 symbols,
  2016-07-25 → 2026-08-03. Price history **starts 2016-07-25** — there is no
  earlier data.
- `event_samples` columns: `event_type, event_subtype, source_ticker,
  target_ticker, horizon_days, available_on, t0, split, forward_abnormal_return`.
- **`forward_abnormal_return` is the contaminated label** — never use it. Build
  returns from `adj_close` by trading-day offset, minus SPY over identical
  sessions.
- **The sealed holdout is SPENT** (read 2026-08-04 by `signals evaluate`).
  Out-of-sample evidence now only comes from forward confirmations.
- Only ~500 distinct tickers ever fire `/P` signals.

---

## 8. Open questions, ranked

1. **Why does concentration work?** (§3.1) Six causes ruled out. This decides
   whether the core finding is a durable edge or a decade-long coincidence.
2. **Is TPL a data artifact?** 178 signals from one ticker; Texas Pacific Land's
   trust structure may file its own share purchases as Form 4s. If so, exclude
   it from the universe and recompute everything.
3. **Recompute DSR/PBO** with the true ~100-trial count.
4. **Fix per-ticker grading** in `signals/impact.py::evaluate` — finish
   `signals/concentration.py::check_concentration`.
5. **Fix the `event_samples` duplicate fan-out.**
6. Paper-trade the §2 config before any money moves.

---

## 9. Things NOT to redo

- Don't re-sweep exit rules, slot counts, or hold lengths hoping for a better
  number. ~100 configs are done and the fit→validate rank correlation is −0.38.
- Don't build a bear-only or moving-average-regime strategy. Both measured worse
  than always-invested.
- Don't add leverage. Out-of-sample drawdown is −45%.
- Don't trust any windowed backtest that starts mid-history (§5.1).
- Don't quote the pre-COVID 16.79%/+4%-alpha numbers — they do not survive
  out-of-sample.

---

## 10. Relevant memory files

`~/.claude/projects/-Users-michaeltadlock-quant/memory/`

- `quant-concentration-works.md`
- `quant-cold-start-bias.md`
- `quant-strategy-fails-out-of-sample.md`
- `quant-tpl-dominates-insider-edge.md`
- `quant-p-events-are-liquid.md`
- `quant-bear-regime-edge.md`
- `quant-trailing-exit-beats-horizon.md`
- `quant-good-signals-2026-08-04.md` (signal side, prior session)

---

## 11. Broker integration (parked — do not build yet)

Robinhood shipped official agentic trading via MCP (May 2026),
`https://agent.robinhood.com/mcp/trading`, OAuth, Claude Code supported. 12
tools; separate dedicated Agentic account; equities only in beta. **Margin and
shorting in the agentic account are undocumented.** Full detail in
`docs/plans/2026-08-04-robinhood-mcp-integration.md`.

Key design note: **do not use a broker-native trailing stop.** The backtested
rule evaluates on the daily close; a native stop triggers intraday on wicks. A
prior session measured 279 stop exits, 100% realized worse than the stop level.

**Claude will not place trades.** The architecture is: this repo emits a
`daily_plan.json` with an `invariants` block; the user's own agent reads it,
calls `review_equity_order`, then places.
