# SEC factor signals + honest evaluation — design

**Date:** 2026-08-09
**Status:** design approved, not yet implemented
**Supersedes as the primary research direction:** the event-study insider work in
`docs/HANDOFF-2026-08-04-trading-strategy.md` (that work is retained and re-used,
but re-expressed as one cross-sectional factor among six — see §5.6)

---

## 1. Why this exists

Three months of research produced no deployable edge. The failure was not bad
luck, and it was not one bug. It was three structural problems, all of which this
design targets directly.

### 1.1 The wrong statistic

Every result to date comes from an *event-study* harness: an event fires, a
position opens, forward abnormal return is measured, results are averaged across
events. That statistic is dominated by whichever ticker emits the most events.
`docs/` records the consequence — 76% of the measured insider edge is one
ticker (TPL), and a `t=8.85, p<0.0001` concentration finding later collapsed
inside a ticker-clustered confidence interval containing zero.

Factor research uses a *cross-sectional portfolio sort*: each month, rank the
entire investable universe by the signal, form decile portfolios, and record the
long-short spread as a single monthly return. One ticker contributes at most one
name to one decile in one month. Ticker concentration cannot inflate the result,
and the output is a monthly return series that standard time-series statistics
apply to honestly.

**The data was mostly fine. The statistic computed on it was not.**

### 1.2 Survivorship

The price panel is a 2026 snapshot back-filled with history. Measured:

| Evidence | Observed | Expected for a real panel |
|---|---|---|
| Symbols that stop trading, 2016-07-25 → 2026-08-07 | **37 of 4,525** (0.8%) | ~5–8%/yr → ~40% cumulative |
| `companies.is_active` | `True` for all 8,018 | mixed |
| `companies.exchange` | `NULL` for all 8,018 | populated |
| `daily_prices.is_delisted_gap` | `False` for all 9,395,740 rows | column exists, never written |
| Form 25 / 25-NSE delisting notices on EDGAR, 2020 Q1 alone | 460 | — |
| S&P 500 constituents removed since 2016-07-25 | 203 | — |
| …of those, present in the price panel | **119 (59%)** | 100% |

The existing memory note estimates survivorship at ~1.4pp/yr. That estimate was
computed over tickers the signals referenced, but companies absent from the panel
were never available to be referenced. It measures the visible part of the hole
and is therefore a floor, not an estimate.

This is specifically fatal to a long-short design. The short leg of every
fundamental factor — high share issuance, high accruals, high asset growth, low
profitability — *is* the population that delists. Firms that dilute shareholders
and burn cash are the ones acquired at a discount or taken to zero. A long-short
test on this panel deletes the names the short leg exists to profit from, and
would report "the short leg does not work" when the truth is "the short leg's
winners are not in the file."

### 1.3 Multiple testing

`scripts/research/` holds 13 research scripts, and the handoff docs describe
dozens of tested variants (concentration ladders, regime switches, bear-only
windows, trailing-stop sweeps, seed sweeps). Under a 5% threshold, ~50 trials
produce ~2.5 spurious "discoveries" by construction. No correction was ever
applied, and the sealed time-based holdout (2023–2026) has been read and is spent.

---

## 2. Scope

**In scope.** Point-in-time universe reconstruction; XBRL fundamentals ingestion;
six pre-registered cross-sectional factors; a portfolio-sort / CPCV /
Fama-MacBeth / Deflated-Sharpe evaluation harness; a ticker-based vault with an
audit trail; two harness self-tests.

**Out of scope, deliberately.**

- **Trading, sizing, broker integration, the $2,000 book.** The agreed success bar
  for this round is "find a real edge, measured correctly." This work terminates
  at a spread and a t-statistic. What a small long-only account keeps of a
  surviving factor is the next round's question.
- **Text factors (Lazy Prices / 10-K similarity).** `filing_sections` has real
  text but covers only ~627 CIKs — 63 names per decile, all large cap. That
  cannot support the test. Phase 2, after the universe expands.
- **Paid market data.** Deferred until §4 quantifies the survivorship hole.

---

## 3. Architecture

Four new packages under `src/market_intelligence/`, each with one responsibility
and a narrow interface:

```
universe/       point-in-time listing membership      → fixes §1.2
fundamentals/   XBRL facts keyed on FILED date        → fixes lookahead
factors/        one small module per signal           → the evidence
evaluation/     sorts, CPCV, FM, DSR, vault guard     → fixes §1.1 and §1.3
```

Data flow:

```
EDGAR full-index (Form 25/25-NSE) ─┐
SEC submissions API                ├→ universe_membership ──┐
index_constituents (existing)     ─┘                        │
                                                            │
companyfacts.zip → fundamental_facts → factors ─────────────┼→ portfolio_sort
                                                            │       │
daily_prices / daily_returns (existing) ────────────────────┘       │
                                                                    ├→ CPCV        → Sharpe distribution
                                                                    ├→ FamaMacBeth → Newey-West t-stat
                                                                    └→ DeflatedSharpe (N from pre-registration)
                                                                              │
                                                                       vault guard (single audited read)
```

---

## 4. `universe/` — point-in-time membership

**Problem solved:** §1.2.

### 4.1 `delistings.py`

Parses the EDGAR quarterly full-index (`https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/form.idx`)
for forms `25`, `25-NSE`, `25/A`, `25-NSE/A`. Each row yields CIK, company name,
form, filing date, accession. Verified available: 2020 Q1 returns HTTP 200,
50.7 MB, containing 404 `25-NSE` + 56 `25` + 3 `25/A` + 2 `25-NSE/A`.

Form 25 is filed by the issuer, 25-NSE by the exchange. 25-NSE is the stronger
delisting signal (exchange-initiated); both are recorded with the form preserved
so downstream code can distinguish voluntary from involuntary.

### 4.2 `membership.py`

Builds table `universe_membership`:

| column | meaning |
|---|---|
| `cik`, `ticker`, `company_name` | identity |
| `first_listed_date` | earliest evidence of listing (first periodic filing, or index add) |
| `last_listed_date` | Form 25/25-NSE date, index removal, or `NULL` if still listed |
| `delist_form` | `25`, `25-NSE`, `25/A`, `25-NSE/A`, or `NULL` |
| `delist_reason` | `exchange_initiated`, `voluntary`, `index_removal`, `unknown` |
| `source`, `source_url`, `content_hash` | provenance, matching existing table conventions |

Exposes exactly one query function:

```python
def universe_on(as_of: date) -> set[str]:
    """Tickers listed on as_of. Excludes anything with no membership record."""
```

**Absence is exclusion, never assumption.** A ticker with no membership row on
date `D` is not in the universe on `D`. This is the opposite of the current
behaviour, where absence from the delisting record silently implies "still alive."

### 4.3 Backfill of existing columns

`companies.is_active`, `companies.exchange`, and `daily_prices.is_delisted_gap`
are populated from `universe_membership`. These columns already exist and have
never been written; leaving them uniformly `True`/`NULL`/`False` is an active
hazard because they read as answered questions.

### 4.4 Output: the hole, quantified

`membership.py` emits a coverage report — for each year, how many companies were
listed per EDGAR versus how many appear in `daily_prices`. This is the deliverable
that decides whether the paid survivorship-free price feed is worth $49/mo. That
decision is explicitly deferred until this number exists.

### 4.5 Interim universe

Until the hole is closed, the trustworthy universe is S&P 500 point-in-time
membership from the existing `index_constituents` table (900 rows, 380 with
`removed_date`, verified real: Alcoa removed 2016-11-01, AAL removed 2024-09-23).
~500 names supports **quintile** sorts (100/bucket), not deciles.

Every evaluation result carries a `survivorship` label:

- `clean` — point-in-time universe, all constituents priced
- `bounded` — point-in-time universe, some constituents unpriced, hole size reported
- `biased` — full 4,525-name panel, no survivorship correction

**No result may be quoted without its label.**

---

## 5. `fundamentals/` — XBRL facts

**Problem solved:** silent lookahead from restatements.

### 5.1 The motivating case

Apple, `us-gaap:Assets`, three facts for one period end:

```
end=2008-09-27   val=$39,572,000,000   filed=2009-07-22   form=10-Q
end=2008-09-27   val=$39,572,000,000   filed=2009-10-27   form=10-K
end=2008-09-27   val=$36,171,000,000   filed=2010-01-25   form=10-K/A   ← −8.6%
```

A store keyed on `end` holds one number and reports `36,171` for a period the
market valued at `39,572` for sixteen months. Every derived ratio inherits
information that did not exist. Apple is the most scrutinised issuer on the
exchange; small caps restate considerably more.

### 5.2 Ingestion

Source: `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip` —
verified HTTP 200, `content-length: 1,400,391,677` (1.4 GB), one file covering
every filer. This replaces a 4,500-CIK crawl. SSD has 1.6 TiB free.

### 5.3 Storage

Table `fundamental_facts`, one row per `(cik, concept, unit, period, accession)`:

| column | note |
|---|---|
| `cik`, `taxonomy`, `concept`, `unit` | `taxonomy` ∈ {`dei`, `us-gaap`} |
| `period_start`, `period_end` | `period_start` NULL for instantaneous facts |
| `value` | as filed |
| `filed_date` | **the point-in-time key** |
| `accession`, `form`, `fy`, `fp` | provenance |

**Every vintage is retained.** A restatement is a new row with a later
`filed_date`, not an update. The as-of accessor is:

```python
def facts_as_of(cik, concept, as_of: date) -> DataFrame:
    """Latest filed value per period, using only facts filed on or before as_of."""
```

Selection is by `filed_date <= as_of` in the query itself. There is no
post-filter step that can be forgotten — the same structural principle as the
existing sealed-clock guard.

### 5.4 Concept whitelist

All confirmed present in Apple's companyfacts (n = datapoints observed):

| purpose | concepts, in fallback order |
|---|---|
| shares outstanding | `dei:EntityCommonStockSharesOutstanding` (70), `us-gaap:CommonStockSharesOutstanding` (144) |
| total assets | `us-gaap:Assets` (146) |
| net income | `us-gaap:NetIncomeLoss` (338) |
| operating cash flow | `us-gaap:NetCashProvidedByUsedInOperatingActivities` (134) |
| revenue | `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` (117), `us-gaap:Revenues` (11), `us-gaap:SalesRevenueNet` |
| cost of revenue | `us-gaap:CostOfGoodsAndServicesSold` (234), `us-gaap:CostOfRevenue`, `us-gaap:CostOfGoodsSold` |
| gross profit | `us-gaap:GrossProfit` (338), else revenue − cost |
| EPS | `us-gaap:EarningsPerShareDiluted` (338), `us-gaap:EarningsPerShareBasic` |
| equity | `us-gaap:StockholdersEquity` (264) |

Fallback chains are ordered and explicit. Tagging varies by filer and by year;
a single hard-coded tag silently drops issuers.

**A missing concept resolves to `NaN`, never `0`.** A zero would rank a company
as worst-in-market on several factors rather than excluding it.

### 5.5 Split adjustment

Share counts are as-reported and not split-adjusted. The split factor is derived
from the existing panel as the ratio `close / adj_close`, whose discrete jumps
identify split dates. Share-count factors compare split-adjusted counts only.

### 5.6 Factor slate (pre-registered)

Committed to `docs/preregistration/2026-08-09-factor-slate.md` **before any
result is computed**. `N_trials` for the Deflated Sharpe is read from that file.

| factor | construction | dir | literature |
|---|---|---|---|
| `net_issuance` | YoY Δ split-adjusted shares outstanding | − | Pontiff & Woodgate (2008); Daniel & Titman (2006) |
| `pead` | standardised unexpected earnings vs seasonal random walk | + | Ball & Brown (1968) |
| `accruals` | (NetIncome − CFO) / average assets | − | Sloan (1996) |
| `profitability` | gross profit / assets | + | Novy-Marx (2013) |
| `asset_growth` | YoY Δ total assets | − | Cooper, Gulen & Schill (2008) |
| `insider_intensity` | trailing-6mo net insider buy $ / market cap | + | existing Form 4 corpus, re-expressed |

`insider_intensity` reuses the 3.49M-row Form 4 corpus in the correct statistic.
It doubles as a harness sanity check: its approximate expected behaviour is known
from prior work, so a wildly different result indicates a harness bug rather than
a discovery.

Interface, one small module per factor:

```python
class Factor(Protocol):
    name: str
    direction: int            # +1 high-is-good, -1 low-is-good
    def compute(self, as_of: date, universe: set[str]) -> pd.Series: ...
```

---

## 6. `evaluation/` — the judge

### 6.1 `portfolio_sort.py`

Monthly rebalance. Rank `universe_on(t)` by factor value, form deciles (quintiles
where breadth requires it), record the long-short spread as one monthly return.

- Factor values winsorised at 1%/99% cross-sectionally before ranking.
- **Minimum 20 names per bucket.** A month that cannot fill buckets yields `NaN`,
  never a silently-averaged partial sort.
- Both equal-weighted and value-weighted spreads are reported. A factor that
  works only equal-weighted is a micro-cap illiquidity artifact.

### 6.2 `cpcv.py`

Combinatorial Purged Cross-Validation (López de Prado, *Advances in Financial
Machine Learning*, ch. 12). Partition the sample into N=6 contiguous groups,
withhold k=2 as test → C(6,2)=15 train/test splits → 15·2/6 = **5 complete
backtest paths**.

- **Purge:** training observations whose label window overlaps the test window are
  dropped.
- **Embargo:** a further fixed span after each test block is dropped, removing
  serial-correlation leakage across the boundary.

Output is a *distribution* of Sharpe ratios, not a single path. With monthly
rebalancing and one-month forward returns, label overlap is minimal and purging
is close to a no-op — but it becomes essential the moment a longer holding period
is tested, which prior work did extensively at 120 days.

### 6.3 `fama_macbeth.py`

Cross-sectional regression of forward return on factor rank, each month; the
t-statistic is computed on the *time series of monthly coefficients* with
Newey-West standard errors.

This is the statistically correct replacement for the per-event t-statistics used
previously. Memory records the rule already learned the hard way: never quote a
per-event t-statistic on this corpus.

### 6.4 `deflated_sharpe.py`

Bailey & López de Prado (2014). Given observed Sharpe `SR`, sample length `T`,
skew `γ₃`, kurtosis `γ₄`, and trial count `N`:

```
SR₀ = σ(SR) · [ (1−γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]      γ = 0.5772…

DSR = Z[ (SR − SR₀)·√(T−1) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²) ]
```

`N` comes from the pre-registration file, not from the analyst's memory.

### 6.5 `vault.py`

```python
def split_of(ticker: str) -> Literal["dev", "vault"]:
    return "dev" if int(sha256(ticker.encode()).hexdigest(), 16) % 100 < 70 else "vault"
```

Deterministic, permanent, no state to lose or corrupt.

**Guard.** Every evaluation run records whether it touched VAULT tickers, and
VAULT reads append to a git-tracked log at `docs/preregistration/vault-reads.log`
(timestamp, factor, git SHA, result). Reading the vault twice without a visible
commit is not possible. This mirrors the existing sealed-clock guard: make the
error structurally impossible rather than remembered.

---

## 7. Error handling

| failure | handling |
|---|---|
| XBRL concept absent for a filer | ordered fallback chain; then `NaN`. Never `0`. |
| Fact filed after `as_of` | excluded by query construction, not post-filter |
| Ticker with no membership record on `D` | excluded from universe on `D` |
| Bucket below 20 names | month yields `NaN`, not a partial sort |
| Ticker/CIK remapping over time | join on CIK, carry ticker as a display attribute |
| `companyfacts.zip` download interrupted | checksum + resume; parse is idempotent on `(cik, concept, period, accession)` |
| Restated fact | new row, never an update |

---

## 8. Testing

Per-factor unit tests against hand-computed fixtures, plus two tests of the
harness itself. These are the highest-value tests in the design, because the
failure mode being defended against is *a harness that reports edges that are not
there* — which is what the last three months produced.

### 8.1 Lookahead canary

A synthetic factor constructed from **future** returns. It must:

- post a very large spread when run through a deliberately naive harness, and
- collapse to approximately zero once the point-in-time guard is active.

If the canary does not scream in the naive configuration, the guard is untested
and every downstream result is unverified.

### 8.2 Null canary

A random factor, 100 seeds. Required:

- spread ≈ 0,
- the distribution of Fama-MacBeth t-statistics ≈ N(0,1),
- DSR ≈ 0.5.

If random noise scores well, the harness has a bug. This is found *before*
trusting a real result rather than after publishing one.

### 8.3 Regression fixture

A frozen small-universe fixture with known-correct expected outputs, so harness
refactors cannot silently change results.

---

## 9. Build order

1. `universe/` — delisting parse, membership table, coverage report, column backfill
2. `evaluation/` — sorts, CPCV, Fama-MacBeth, DSR, vault + guard, **both canaries**
3. `fundamentals/` — companyfacts ingest, as-of accessor, split adjustment
4. `factors/` — six modules against the pre-registered slate
5. Run: S&P 500 point-in-time (`survivorship: clean`) → full panel (`biased`), reported side by side

Order is deliberate: the judge is built and self-tested **before** the evidence,
so no factor result is ever produced by an unvalidated harness. Step 2 completing
with both canaries passing is the gate for steps 3–5.

---

## 10. What would falsify this design

Stated in advance, so the answer is not negotiated after the fact:

- If the null canary does not produce ~N(0,1) t-statistics, the harness is wrong
  and no factor result from it counts.
- If all six factors post DSR < 0.5 on `survivorship: clean` data, the honest
  conclusion is that this data cannot support a fundamental factor strategy at
  this universe size, and the next move is universe expansion — not another
  parameter sweep.
- If a factor works only equal-weighted, only in one CPCV path, or only before
  costs, it is not an edge.
