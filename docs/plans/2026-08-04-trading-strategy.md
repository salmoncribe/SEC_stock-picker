# Trading strategy — signal to trade to exit

Written 2026-08-04, at commit `5ca6714` (master). Read after
`docs/HANDOFF-2026-08-04.md` and the `quant-good-signals-2026-08-04` memory.

The question this document set out to answer: **given a signal, how do we turn
it into a position, how long do we hold it, and what makes us sell?** Five
parallel research agents covered entry mechanics, holding/exit,
sizing/construction, codebase capability, and cost/validation. Their design is
in §3–§5 and it is sound.

But an empirical check run alongside them found that **the signals this design
was meant to trade do not survive a basic independence test.** That goes first,
because it changes what should be built next.

---

## 0. STOP — none of the admitted cells survive ticker clustering

On 2026-08-04 the fixed gate admitted `P/60d`, `C/40-120d` and `D/20d`. They sit
at `confirm_streak=1`, one autopilot cycle away from firing live to Telegram.

Their admission t-statistics are **per event**. But events are not independent:
the same ticker appears dozens or hundreds of times. Recomputing each cell's
edge weighting **each ticker once** instead of each event once:

| cell | split | n | tickers | per-event | **per-ticker** | top ticker's share |
|---|---|---|---|---|---|---|
| P/60d | discovery | 4,627 | 501 | +3.04% t=8.01 | +2.35% **t=3.09** | TPL **56.4%** |
| P/60d | holdout | 2,450 | 419 | +1.33% t=2.84 | +0.35% **t=0.53** | TPL **120.2%** |
| C/40d | discovery | 1,447 | **44** | +3.54% t=7.18 | −0.71% **t=−0.45** | TTD 31.3% |
| C/60d | discovery | 1,441 | **44** | +5.29% t=8.58 | −1.39% **t=−0.63** | CRWD 34.4% |
| C/90d | discovery | 1,425 | **44** | +8.31% t=10.40 | −0.81% **t=−0.29** | CRWD 37.4% |
| C/120d | discovery | 1,404 | **43** | +11.67% t=11.44 | +0.93% **t=0.26** | CRWD 37.4% |
| C/120d | holdout | 810 | **39** | +10.22% t=6.65 | +2.61% **t=0.47** | HOOD 40.9% |
| D/20d | discovery | 2,491 | 416 | −0.51% t=−2.50 | −0.07% **t=−0.17** | PENN −16.1% |

*(Label rebuilt independently: entry at first session on/after `t0`, exit at
+H trading sessions on `adj_close`, minus SPY over the identical sessions. It
reproduces production's per-event numbers closely — P/60d discovery +3.04% vs
production's +2.69%, holdout +1.33% vs +1.49% — so this clusters production's
actual numbers, not a different measurement.)*

**Read the two t-stat columns against each other.** Nothing about the data
changes between them. Only the assumption about what counts as an independent
observation. A per-event t-stat of 11.44 becoming a per-ticker t-stat of −0.29
is the same error as polling one person a thousand times.

Three specific failures:

1. **`P/60d`'s holdout confirmation is entirely TPL.** TPL contributes **120%**
   of the holdout edge — meaning *excluding TPL, the holdout edge is negative*,
   with a per-ticker median of **−0.79%**. Texas Pacific Land files insider
   purchases constantly (the trust's own repeated sub-share buying) and was a
   ~10x stock. The cell passed its holdout on one name.
2. **`C` is 44 tickers, not 1,447 events.** Derivative conversions cluster in a
   few high-growth names whose insiders convert constantly — TTD, CRWD, HOOD,
   CVNA. Every one is a major 2023–2026 winner. Per-ticker, `C`'s discovery edge
   is **negative at three of four horizons**. This is the cell the memory called
   "better-supported than it looked."
3. **`D/20d` is noise** — per-ticker t of −0.17 (discovery) and −0.79 (holdout).

**Why nothing caught this.** The placebo control in `signals/impact.py`
randomises the *control* ticker; it cannot detect concentration in the
*treatment* group. The K=2 promotion ladder counts confirmations, not
independent names. Neither mechanism looks at per-ticker contribution, so both
would have passed a cell that is one stock.

**Do not let these reach ACTIVE.** One more autopilot cycle promotes them.

---

## 1. The cost objection was aimed at the wrong population

Separately, and more encouragingly: the handoff's central negative result —
spreads of 118bps trade-weighted against a ~75bps break-even — **does not apply
to the insider-purchase population.**

Corwin-Schultz spreads computed point-in-time (trailing 22 sessions ending
strictly before each event's `t0`) on the `/P` population:

| | at event (point-in-time) | full universe |
|---|---|---|
| median spread | **16.0 bps** | 37.4 bps |
| mean spread | **26.8 bps** | — |
| median dollar volume | **$109,268,587/day** | $3,546,900/day |
| p10 dollar volume | $13,819,439/day | $42,375/day |

These are ED, IFF, TDG, ABT, KDP, T, SPG, AMZN, OXY — **large caps.** The 118bps
figure was a property of the *cluster-buy* population, the falling-knife small
caps the handoff independently proved was the broken detector. A liquidity
filter (ADV ≥ $5M, price ≥ $5, spread ≤ 75bps) retains ~92% of events, so it is
nearly free.

**Cost is not the binding constraint. Sample independence is.**

*Caveats: only 41% of events got a point-in-time estimate; Corwin-Schultz
truncates negatives to zero, biasing the median down. Trust the 26.8bps mean
over the 16bps median.*

---

## 2. What to do next

Ordered by what could invalidate what.

1. **Freeze promotion.** Prevent `P/60d`, `C/*`, `D/20d` from reaching ACTIVE
   until §5's test 2 exists. They are one cycle away.
2. **Add per-ticker clustering to the gate.** `signals/impact.py::evaluate`
   should compute its t-stat on per-ticker means, and reject any cell where one
   ticker exceeds ~10% of the summed edge or where fewer than ~100 distinct
   tickers contribute. This is a change to how *every* cell is graded, not a
   patch for these three.
3. **De-duplicate `event_samples`.** 10,809 `/P` @120d rows collapse to 4,447
   distinct `(ticker, t0)` pairs. Find the fan-out in the sample builder.
4. **Re-grade the whole corpus** under per-ticker clustering. Some cell may
   survive; none of the current three look likely to.
5. **Then, and only then**, build §3–§5.

Steps 1–3 are small and mechanical. Step 4 decides whether there is a strategy
at all.

---

## 3. The design (for whatever signal survives step 4)

Every parameter is chosen **by mechanism, never by sweep.** The prior grid
search measured train→test rank correlation of **−0.11** — backtest-optimal
parameters carry zero forward information.

### Entry

| parameter | value | rationale |
|---|---|---|
| decision→fill lag | 1 session | already correct: `simulator.py:335`, `available_on ≤ t−1`, fill at `t` open |
| `max_reporting_lag_days` | 10 | `filing_date − trans_date` |
| `min_adv_dollar` (20d median) | $5,000,000 | free — p10 of the population is $13.8M |
| `min_price` | $5.00 | avoids tick-dominated spreads |
| `max_spread_bps` | 75 | retains ~92% of events |
| duplicates | refresh horizon, never add | adding is concentration by the back door |

**Do not chase speed.** Jeng/Metrick/Zeckhauser (2003) find >6%/yr on a
six-month purchase portfolio; Lakonishok & Lee (2001) ~7.4% over 12 months.
Drift accrues over months. Marginal value of one day at a 60-day horizon is
~13bps against a ~13bps half-spread — hurrying is not profitable. *Finance
Research Letters* (2024) tested the fast version directly: returns vanish or go
negative once tradable size is capped.

### Sizing and construction

| parameter | value |
|---|---|
| weighting | **equal weight (1/N)** |
| `target_slots` | **40** at $2k; 60–80 at $100k |
| `max_position` | 0.05 |
| sector cap | 0.25 of gross |
| `max_cluster` | 0.15 |
| vintage cap | ≤10% of slots per 5 sessions |
| `max_gross` | 1.0 — no leverage until alpha has a t-stat |
| `max_beta` / `target_beta` | `None` at $2k |
| cash | an output, never a target; 3% floor |

**Bypass `optimizer.py`.** With a cell-average μ identical for every name, a
mean-variance optimizer has no return information and only fits covariance
noise. DeMiguel/Garlappi/Uppal (2009): across 14 optimal models and 7 datasets,
none beat 1/N consistently — sample MV on S&P sectors went from in-sample
Sharpe 0.3848 to **out-of-sample 0.0794** versus 0.1876 for 1/N. Michaud (1989)
calls optimizers "estimation-error maximizers." Jagannathan & Ma (2003): the
long-only constraint *is* the shrinkage.

Equal weight also **deletes the non-determinism bug** — no tiebreak means no
DuckDB row order to be lucky about. That bug swung ending equity from $25,559 to
$66,109 across five identical loads.

**Breadth is about volatility drag, not ranking.** At 45% idiosyncratic vol a
3-name book gives back ~5.6 points of a 120-day edge to `−σ²/2`; a 50-name book
gives back 0.9. That is the mechanism behind the 16.13% → 2.68% CAGR collapse
from 50 slots to 3 — arithmetic, not bad luck. Annual cost is **independent of
N**: `(2 × gross / hold) × spread` — the slot count cancels.

**Beta.** The edge is measured SPY-hedged but the book ran at β 0.92–1.05, so
realized P&L was mostly market. At $2,000 this cannot be fixed: one MES future
is ~$30k notional (15× the account), SPY is ~$600/share, SH carries 0.89%
expense plus daily-reset decay, ATM puts cost ~12–14%/yr — twice the alpha.
**Accept β≈1; build the switch so it is one config change at ~$40k.** Report
β-adjusted alpha and its t-stat on every replay regardless.

### Holding and exit

| parameter | value |
|---|---|
| `hold_days` | the admitted cell's horizon, in **trading** sessions |
| `min_hold_days` | 80% of horizon |
| exit rule | **time exit only** |
| `stop_loss_kind` | **`"none"`** |
| `rebalance_days` | 5 |
| `no_trade_band` | 0.03 |
| regime intake filter | no new entries when SPY < 200dma |

**Hold to the measured horizon.** Gross alpha is roughly invariant to holding
length while cost scales as 1/h. The last attempt paid 100% of the cost for
~18% of the drift.

**Every other exit is rejected on mechanism:**

- *Trailing stop* — needs positive autocorrelation (Kaminski & Lo). Our edge is
  event-conditional, not price-conditional; a trailing stop truncates the right
  tail that carries the entire mean.
- *Profit target* — already disproven in-repo (`analytics/backtest.py:13-16`):
  clips winners, turns a positive-mean cell into a losing strategy.
- *Rotation until replaced* — requires cross-sectional ranking we do not have.
- *Per-name stop* — **decisive evidence against.** 279 stop exits, 100% realized
  worse than the 8% level (mean −12.3%, worst −44.7%); turning it off gave a
  *smaller* 2020 drawdown (−38.6% vs −40.9%). Control per-name risk with
  position size, not with a stop that can be gapped through.

**Crash protection is an intake valve, not an exit.** No new tranches when SPY
closes below its 200dma: never forces a sale, never truncates a winner, costs
zero slippage, and on a rolling conveyor it de-grosses the book to ~0 over one
horizon automatically. This fixes the 2020 failure mode — cash at ~0% through
the whole crash, every freed dollar redeployed into a falling market. Behind it,
widen the governor to halve at −15% / flatten at −25% (−10%/−15% whipsaws
against our own base rate of a 10%+ drawdown every ~2 years), and only with
`account.decay_high_water_mark` wired in — `risk.py:495` documents that
flatten-without-re-entry is an absorbing state.

---

## 4. What the codebase can and cannot do

**Wired and working:** `max_beta`, `target_beta`, `target_vol`, `max_gross`,
`max_position`, `max_cluster`, `stop_loss_kind`, both governor thresholds,
`no_trade_band`, `rebalance_days`, `turnover_penalty_bps`, `slippage_bps`. The
t+1 execution lag and its three leak canaries are real and tested.

**Dead config:** `max_turnover_annual` (appears only at `config.py:458`),
`max_single_name_risk_share` (`risk.RiskLimits` never instantiated),
`graph_return_views_enabled`. `min_cvar` runs but has never held a position —
all 24 configs ended at exactly $10,000.

**Blocking gaps:**

1. **No liquidity data.** `MarketPanel` has no volume column;
   `backtest_data.load_bars` doesn't select it. `daily_prices.volume` exists —
   it just isn't plumbed. No ADV or spread filter is possible until it is.
2. **`CostModel` is a flat scalar** (5bps). It cannot express per-name cost.
3. **The traded cell is still `insider_cluster_buy`** — the one proven broken.
4. **`pit_guard` is imported by nothing.** The sealed clock is opt-in and unused
   in the replay path.
5. **`signals/corroboration.py` has zero callers** — nothing sets
   `EventAlert.corroborating_people`, so the confidence bonus never fires.
6. **`views._one_per_symbol` keeps the *earliest* expiry**, so a horizon refresh
   is silently discarded. Should keep the latest.
7. **Leverage is unfundable** — `_settle` clamps every buy to cash on hand.

---

## 5. Validation gauntlet — preregistered

Commit this list before the first run. A failure stops the pipeline.

| # | test | PASS | ABANDON |
|---|---|---|---|
| 0 | Determinism, 5 identical runs | max−min ending equity = **$0.00** | n/a — build bug |
| 1 | Label sanity: ρ(label, trailing 120d return) | \|ρ\| < **0.05** | ≥0.15 at 120d |
| 2 | **Ticker concentration** | no ticker > **10%** of summed edge; ≥**100** distinct tickers; per-ticker t ≥ **2.5** | one ticker > 25% |
| 3 | Placebo control, per-ticker weighted | edge ≥ **+1.5%**/horizon, bootstrap 95% CI lower bound > 0 | CI includes 0 |
| 4 | Shuffle canary, 500 shuffles | real edge > **99th pct** | p > 0.05 |
| 5 | Universe integrity (full remapped universe) | full ≥ **60%** of any filtered result | filtered > 1.5× full |
| 6 | Monte Carlo, 50 seeds | median CAGR ≥ SPY + **3%**, ≥70% of seeds beat SPY | <50% beat SPY |
| 7 | Cost stress ×1.0/1.5/2.0/3.0 | net > SPY at **2.0×** | fails at 1.0× |
| 8 | CPCV, purge = horizon, embargo 25d | ≥**8 of 11** paths positive | ≤5 of 11 |
| 9 | DSR / PBO | DSR ≥ **0.95**, PBO ≤ **0.20** | PBO ≥ 0.50 |
| 10 | Subperiod stability, 3 blocks | positive in **2 of 3** | negative in 2 of 3 |

**Test 2 is new and is the one that matters.** It would have caught all three
currently-admitted cells.

**There is no sealed holdout left.** It was read by `signals evaluate` on
2026-08-04 and is spent. Out-of-sample evidence now comes only from the K=2
ladder accumulating **forward** confirmations on genuinely new filings — which
makes test 2 more important, not less, since a contaminated cell will keep
confirming on the same contaminating ticker.

**Hard cap: 50 trials total**, counting every run including early failures.
Bailey–López de Prado's minimum backtest length at 50 trials is ~7.8 years
against a 10.0-year panel.

---

## 6. What this is honestly worth

**At $2,000, a 5% alpha is $100/year.** No sizing decision recovers that. The
purpose of this account is not income — it is to prove that modeled spreads
match realized fills, that the daily loop runs unattended, and that a measured
edge survives contact with a broker. Design for $100k, run at $2k, change
nothing but `N` when it grows. Meaningful income needs ~**$150k–$250k**.

**30%/yr should be retired as a target.** `PortfolioConfig`'s own sweep shows
2.5× gross → 33% CAGR at 66.6% vol and 73.8% drawdown, and 6× gross →
**−13.1% CAGR**. Past ~2.5× the `−σ²/2` term wins.

And as of today the honest expected alpha is **not yet a number** — every
candidate signal collapses under §0's clustering test. The design in §3 is
ready. It has nothing validated to trade.

---

## Sources

Lakonishok & Lee (2001) · Jeng, Metrick & Zeckhauser (2003), *REStat* 85(2) ·
Cohen, Malloy & Pomorski (2012), *JF* · "Insider filings as trading signals —
does it pay to be fast?", *Finance Research Letters* (2024) · Corwin & Schultz
(2012), *JF* 67(2) · Abdi & Ranaldo (2017), *RFS* 30(12) · Amihud (2002) ·
Almgren et al. (2005) · DeMiguel, Garlappi & Uppal (2009), *RFS* 22(5) ·
Michaud (1989), *FAJ* 45 · Jagannathan & Ma (2003), *JF* 58 · Tu & Zhou (2011),
*JFE* 99(1) · Kaminski & Lo, *When Do Stop-Loss Rules Stop Losses?* · Faber
(2007) · Bailey & López de Prado (2014) · Bailey, Borwein, López de Prado & Zhu
(2017) · Hansen (2005)
