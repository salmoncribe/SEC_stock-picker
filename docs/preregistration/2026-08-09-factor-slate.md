# Pre-registration — factor slate, round 1

**Committed:** 2026-08-09
**Design:** `docs/superpowers/specs/2026-08-09-sec-factor-signals-design.md`
**Status:** WRITTEN BEFORE ANY RESULT EXISTS

This document fixes the hypotheses, parameters, and decision rules *before* the
data is looked at. Its purpose is to make the Deflated Sharpe Ratio enforceable:
`N_trials` is read from this file, not recalled by the analyst.

Prior work on this dataset ran roughly 50 undocumented variants. At a 5%
threshold that manufactures ~2.5 spurious discoveries. This file exists so that
does not recur.

---

## 1. Trial count

**N_trials = 6.**

One per factor below. This is the `N` passed to the Deflated Sharpe Ratio.

Amendments are permitted and must be additive commits to this file. Adding a
seventh factor sets `N_trials = 7` and **requires recomputing DSR for all
previously reported factors at the higher bar.** Deleting a factor after seeing
its result is not permitted; a factor that disappoints is reported as a
disappointing result.

---

## 2. The six factors

Each is specified completely enough that two people would compute the same
number. `dir` is the hypothesised sign: `+` = high values predict high returns.

### 2.1 `net_issuance` — dir `−`

- **Construction:** year-over-year change in split-adjusted shares outstanding.
- **Source:** `dei:EntityCommonStockSharesOutstanding`, fallback
  `us-gaap:CommonStockSharesOutstanding`.
- **Split adjustment:** via `close / adj_close` ratio jumps in `daily_prices`.
- **Lookback:** most recent value as-of, versus the value as-of 12 months prior.
- **Hypothesis:** firms that shrink share count outperform; firms that dilute
  underperform.
- **Prior:** Pontiff & Woodgate (2008); Daniel & Titman (2006).

### 2.2 `pead` — dir `+`

- **Construction:** standardised unexpected earnings. `SUE = (EPS_q − EPS_{q−4}) /
  σ(EPS_q − EPS_{q−4})`, σ estimated over the trailing 8 available quarters.
- **Source (value):** `us-gaap:EarningsPerShareDiluted`, fallback
  `EarningsPerShareBasic`.
- **Source (announcement timing):** 8-K Item 2.02 event date where available
  (23,811 events, 626 CIKs); otherwise earliest `filed_date` of the EPS fact.
  Drift is measured from the announcement, not the 10-Q. Coverage split is
  reported with the result.
- **Hypothesis:** positive earnings surprises are underreacted to and drift.
- **Prior:** Ball & Brown (1968).

### 2.3 `accruals` — dir `−`

- **Construction:** `(NetIncomeLoss − NetCashProvidedByUsedInOperatingActivities)
  / average(Assets_t, Assets_{t−1})`.
- **Hypothesis:** earnings backed by cash persist; earnings backed by accruals
  reverse.
- **Prior:** Sloan (1996).

### 2.4 `profitability` — dir `+`

- **Construction:** `GrossProfit / Assets`. Where `GrossProfit` is untagged,
  `revenue − cost_of_revenue` using the §5.4 fallback chains.
- **Hypothesis:** gross profitability predicts returns and is roughly orthogonal
  to value.
- **Prior:** Novy-Marx (2013).

### 2.5 `asset_growth` — dir `−`

- **Construction:** year-over-year change in `us-gaap:Assets`.
- **Hypothesis:** firms that expand the balance sheet aggressively subsequently
  underperform.
- **Prior:** Cooper, Gulen & Schill (2008).

### 2.6 `insider_intensity` — dir `+`

- **Construction:** trailing-6-month net insider purchase dollars (subtype `P`
  buys minus `S` sales) divided by market capitalisation.
- **Source:** existing `events` table, 3.49M Form 4 rows.
- **Hypothesis:** insider net buying predicts returns cross-sectionally.
- **Secondary purpose:** harness sanity check. Prior work established the
  approximate magnitude of this effect; a result far outside that range indicates
  a harness defect rather than a discovery.

---

## 3. Test procedure, fixed in advance

- **Rebalance:** monthly, last trading day.
- **Sort:** deciles where ≥20 names per bucket are available; quintiles otherwise.
  Below 20 per bucket the month yields `NaN`.
- **Winsorisation:** factor values at 1% / 99%, cross-sectionally, before ranking.
- **Weighting:** equal-weighted **and** value-weighted, both reported. A factor
  surviving only equal-weighted is classified as a micro-cap artifact, not an edge.
- **Return:** long top bucket, short bottom bucket, one month forward.
  **Compounded** — `∏(1+r) − 1`, never summed. Computed from
  `daily_returns.total_return`; the existing `abnormal_return` column is not used.
- **Delisting return:** actual where priced; **−30%** for exchange-initiated
  (`25-NSE`) with no price (Shumway 1997); **0%** for voluntary (`25`, typically
  M&A). Names are never silently dropped.
- **Universe:** `universe_on(t)` = listed AND priced, with the `survivorship`
  label attached (`clean` / `bounded` / `biased`).
- **Splits:** DEV = `sha256(cik) % 100 < 70`; VAULT = remainder. Keyed on CIK so
  ticker changes cannot migrate a company across the boundary.
- **Statistics reported per factor, always together:**
  1. mean monthly spread and annualised Sharpe
  2. Fama-MacBeth t-statistic, Newey-West corrected
  3. CPCV Sharpe distribution across all 5 paths (N=6 groups, k=2, purged + embargoed)
  4. Deflated Sharpe Ratio at `N_trials` from §1
  5. `survivorship` label

No subset of these may be reported alone.

---

## 4. Decision rules, fixed in advance

A factor is **promoted** to a VAULT read only if, on DEV data:

- DSR > 0.95, **and**
- the CPCV Sharpe distribution is positive in ≥4 of 5 paths, **and**
- the value-weighted spread has the same sign as the equal-weighted spread, **and**
- the `survivorship` label is `clean` or `bounded`.

A factor failing any condition is reported as failed and is **not** re-specified
and retried. Re-specification after seeing a result is a new trial and increments
`N_trials`.

**The VAULT is read once, for the full promoted set at once, and the read is
logged to `docs/preregistration/vault-reads.log` with the git SHA.** There is no
second read.

---

## 5. What counts as a negative result

If zero factors promote, that is a publishable outcome of this round and the
correct next move is universe expansion (closing the survivorship hole, adding
text factors), **not** re-running the same six factors with different parameters.

Recorded here so the conclusion cannot be renegotiated later.
