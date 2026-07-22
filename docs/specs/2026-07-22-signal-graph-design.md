# Event-Propagation Signal Graph — Design

**Date:** 2026-07-22
**Status:** Approved
**Scope:** Turn the baseline data platform into an event-triggered alerting system: a filing
event at company A produces a ranked, sourced prediction for related companies B.
**Non-goal:** No order execution, no brokerage integration, no position sizing, no investment
advice. The system produces model scores and measured track records; a human decides what,
if anything, to do with them.

---

## 1. Problem

The baseline collects filing metadata, filing documents, Item sections, and macro series. It
has **no price data**, **no notion of a relationship between two companies**, and **no way to
tell whether anything it stores predicts anything at all**.

The hypothesis to be tested and, if it holds, operationalized:

> A disclosed event at company A (spending commitment, guidance shift, material agreement,
> relationship change) predicts abnormal returns at economically linked company B over a
> horizon of days to weeks, because information diffuses across supply chains more slowly
> than it diffuses within a single company's analyst coverage.

This is a documented effect (Cohen & Frazzini 2008, *Economic Links and Predictable Returns*).
It is not assumed here — it is measured, per event type and per edge type, on held-out data,
and only the combinations that survive are allowed to fire alerts.

## 2. Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| A | Output shape | **Event-triggered alerts** | Every alert is a discrete, falsifiable prediction, so the system grades itself continuously. A daily ranking makes every name a prediction every day and is far harder to attribute. |
| B | Universe | **S&P 500, point-in-time** | Most large supply-chain counterparties are in-universe; price data is clean. Constituents are stored with `added_date`/`removed_date` so backtests see the index as it was, not as it is. |
| C | Edge source | **Extracted from filings** | ASC 280 compels disclosure of any customer >10% of revenue; Item 1/1A name suppliers, partners, competitors in prose. Every edge cites a filing and a date, so an alert can state *why* it fired. |
| D | Edge discovery via return correlation | **Excluded** | Circular (using returns to find links used to predict returns) and unexplainable. 500 names is 124,750 pairs; ~6,000 clear p<0.05 by chance alone. |
| E | Event taxonomy | **Open registry, not an enum** | New event types are config rows, not schema migrations — mirroring `config/fred_series.yaml`. Broad ingestion is safe *because* alerting is gated separately (decision F). |
| F | Alerting gate | **Admission on discovery split, track record on holdout** | Selecting winners and grading them on the same sample inflates reported hit rates by construction. The two splits must be disjoint. |
| G | Model | **Conditional impact table first; GBM later** | The rule table produces the target alert format directly, has few parameters, and makes thin cells visible. It is also the honest baseline any ML model must beat out-of-sample. |
| H | Graph mutation | **Bitemporal, append-only; no deletes** | "Cleanup" corrects state and dedupes identity. Deleting a dead edge destroys the ability to backtest the years in which it was alive. |
| I | Obsidian vault | **Derived projection, regenerable** | The vault is rendered *from* DuckDB. Hand-edits are never authoritative, so `render-vault` can always rebuild from scratch. |
| J | Price provider | **`yfinance` behind the existing `MarketDataProvider` ABC** | Free, matches the "cheap to run" constraint. It is an unofficial endpoint, so it is isolated behind the existing interface and its data is quality-gated (§7) rather than trusted. |

## 3. Architecture

Six stages. Each has a hard input/output contract and is independently testable.

```
 0  PRICE SPINE          PIT constituents · adjusted OHLCV · abnormal returns
        │                 └─ the measuring instrument; nothing validates without it
        ▼
 1  EXTRACTION           events (XBRL + LLM + 8-K item codes) · edge observations
        │                 └─ candidate tables. NOT the graph yet.
        ▼
 2  DATASET BUILDER      (event, edge, target) rows joined point-in-time
        │                 └─ one contract; impact table and any future model both consume it
        ▼
 3  VALIDATION GATE      per (event_type, edge_type, horizon): n · hit rate · mean CAR · t
        │                 admission decided on DISCOVERY; track record reported on HOLDOUT
        ▼
 4  GRAPH + LIFECYCLE    dedup to one edge per (source, target, type)
        │                 candidate → active → dormant → retired   (bitemporal, no deletes)
        ▼
 5  PROJECTION           Obsidian vault render · alerts
                          only ACTIVE edges × ADMITTED event types fire
```

New modules, following the established `clients → collectors → validators → storage` pattern:

- `clients/market_yfinance.py` — `YFinanceMarketDataProvider(MarketDataProvider)`
- `clients/constituents.py` — point-in-time index membership
- `collectors/prices.py`, `collectors/constituents.py`
- `analytics/returns.py` — abnormal-return computation (pure)
- `extraction/events.py` — event extraction registry
- `extraction/edges.py` — relationship extraction from Item 1 / 1A / segment notes
- `graph/lifecycle.py` — dedup, state machine, bitemporal writes
- `signals/dataset.py` — the (event, edge, target) row builder
- `signals/impact.py` — the conditional impact table + gate
- `projection/vault.py` — Obsidian renderer
- `alerts/engine.py` — alert generation

## 4. Schema

All tables carry the standard provenance block (`source`, `source_url`, `content_hash`,
`schema_version`, `validation_status`, `validation_errors`, `collected_time`) and follow the
existing natural-key upsert contract.

**`index_constituents`** — natural key `(index_id, company_id, added_date)`
`index_id`, `company_id`, `cik`, `ticker`, `added_date`, `removed_date` (NULL = current).
Answers "who was in the index on date D?" — the survivorship-bias defence.

**`daily_prices`** — natural key `(symbol, price_date)`
`symbol`, `price_date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`,
`is_delisted_gap` (BOOLEAN), `provider`.

**`daily_returns`** — natural key `(symbol, price_date)`
`symbol`, `price_date`, `total_return`, `market_return`, `sector_return`,
`abnormal_return`, `beta`, `estimation_window_start`. Abnormal return is the label.

**`events`** — natural key `(filing_id, event_type, event_key)`
`event_id` (PK), `company_id`, `event_type`, `event_subtype`, `filing_id`,
`acceptance_time` (**the point-in-time clock — never `report_date`**), `magnitude`,
`direction`, `payload` (JSON), `extraction_method`, `extraction_confidence`.

**`edge_observations`** — natural key `(filing_id, source_company_id, target_company_id, edge_type)`
One row per *sighting* of a relationship in one filing. The raw evidence, never deduped:
`observed_at`, `item_code`, `quote`, `extraction_confidence`.

**`graph_edges`** — natural key `(source_company_id, target_company_id, edge_type)`
The deduped graph. **This is the fix for "added 4 times":** four filings naming the same
relationship produce four `edge_observations` and **one** `graph_edges` row with
`confirmation_count = 4`.
`edge_id` (PK), `strength`, `first_seen`, `last_confirmed`, `confirmation_count`,
`valid_from`, `valid_to`, `state` (`candidate|active|dormant|retired`), `state_reason`.

**`event_pair_samples`** — natural key `(event_id, edge_id, horizon_days)`
Stage 2 output: `features` (JSON), `forward_abnormal_return`, `split` (`discovery|holdout`).

**`impact_stats`** — natural key `(event_type, edge_type, horizon_days, split)`
Stage 3 output: `n`, `hit_rate`, `mean_car`, `t_stat`, `admitted` (BOOLEAN), `computed_at`.

**`alerts`** — natural key `(event_id, target_company_id, horizon_days)`
`predicted_direction`, `predicted_magnitude`, `confidence`, `impact_stat_id`,
`fired_at`, `realized_abnormal_return` (backfilled after the horizon closes).

## 5. Graph lifecycle

| State | Entry condition | Fires alerts | In vault |
|---|---|---|---|
| `candidate` | extracted from a filing, not yet validated | no | no |
| `active` | confirmed within staleness window **and** its `(event_type, edge_type)` cell was admitted | **yes** | yes |
| `dormant` | real relationship, no measurable predictive value | no | yes, greyed |
| `retired` | contradicted, or unconfirmed for N consecutive filings | no | history only |

Retirement sets `valid_to`; it never deletes. Every graph query is `as_of`-parameterized, so
a 2019 backtest sees the 2019 graph including edges that later retired.

**Deduplication** happens at write time on the `graph_edges` natural key, reusing the existing
DuckDB insert/update split. Re-running extraction over the same filings reports
`0 inserted, N updated` and the edge count stays flat.

## 6. The validation gate

1. `signals/dataset.py` emits one row per `(event, edge, target, horizon)` with the event's
   `acceptance_time` as t=0 and forward abnormal returns at 1/5/20 days.
2. Rows are split **chronologically** — discovery = earlier period, holdout = later period.
   Not randomly: random splits leak, because adjacent days share information.
3. `signals/impact.py` computes per-cell statistics on discovery and admits a cell only if it
   clears configured thresholds (minimum `n`, minimum hit rate, `t_stat`).
4. The same cells are then recomputed on holdout. **Holdout numbers are what the alert
   displays.** A cell admitted on discovery that collapses on holdout is reported as such and
   demoted.

Purged/embargoed splitting: samples whose forward-return window overlaps the split boundary
are dropped, so no holdout label is partly determined by discovery-period prices.

## 7. Data quality and error handling

- **Delisted symbols.** A free price API returns an empty series for a dead ticker, which
  reads as "no data" rather than "-100%". Any constituent with `removed_date` set and no
  price rows near that date is flagged `is_delisted_gap` and excluded from return
  calculations rather than silently treated as flat.
- **Provider unreliability.** `yfinance` is unofficial and can change without notice. It is
  isolated behind `MarketDataProvider`; the collector validates every batch (monotonic dates,
  positive prices, `high >= max(open, close)`, non-negative volume) and rejects rather than
  stores anomalies, matching the existing `validation_status` contract.
- **Split/dividend adjustment.** Returns are computed from `adj_close` only. A raw-close
  return series would show a 2-for-1 split as a -50% day.
- **LLM extraction drift.** Extraction confidence is stored per record, the prompt version is
  part of `extraction_method`, and re-extraction is idempotent on the natural key. Low
  confidence edges enter as `candidate` and cannot reach `active` without corroboration.
- **Point-in-time discipline.** Every join uses `acceptance_time`, never `report_date` or
  `filing_date`. A single test asserts no sample's feature window extends past its t=0.

## 8. Testing

Mirrors the existing offline discipline — no test touches the network or real `data/`.

- **Pure functions** (`analytics/returns.py`, `extraction/events.py`, `graph/lifecycle.py`)
  get direct unit tests with hand-built fixtures.
- **Leakage tests** are first-class: an assertion that no `event_pair_samples` row uses price
  data dated before its event's `acceptance_time`, and that no holdout label overlaps the
  discovery window.
- **Idempotency tests** re-run each collector twice and assert row counts are unchanged and
  `confirmation_count` increments without creating new edges.
- **Bitemporal tests** assert `graph_as_of(D)` returns edges retired after D and excludes
  edges first seen after D.
- **Two controls, not one.** A deliberately null signal (random events joined to random
  edges) must produce an unadmitted cell — if noise passes the gate, the gate is broken.
  And a known-real signal (Form 4 insider purchases) must produce an admitted one — if a
  documented effect fails the gate, the gate is equally broken, just in the direction that
  looks like rigour instead of like a bug. A gate that only has the null control cannot
  distinguish "correctly sceptical" from "detects nothing at all".

## 9. Build order

Stages are sequential; each ships working and testable before the next begins.

| Stage | Deliverable | Gate to proceed |
|---|---|---|
| 0 | Price spine + PIT constituents + abnormal returns | **Done 2026-07-22.** 1.55M price rows, 1.55M returns, 900 PIT membership windows, zero lookahead violations |
| 1a | **Form 4 insider events** (structured XML, no LLM) | Extraction idempotent; event count reconciles against `filings` |
| 1b | 8-K item-code events (structured metadata, no LLM) | Item codes parsed for every 8-K in the universe |
| 1c | 10-K/10-Q documents + LLM edge extraction | Spot-check edge accuracy on 20 filings |
| 2 | Dataset builder | Leakage tests pass |
| 3 | Validation gate | **Both** controls behave: null signal rejected, Form 4 detected |
| 4 | Graph lifecycle | Dedup + bitemporal tests pass |
| 5 | Obsidian projection + alerts | Vault regenerates from empty; alerts cite sources |

### Why Form 4 comes before LLM extraction

The universe-wide filing collection (2026-07-22) returned 684,301 filings, and
the mix argues for reordering what Stage 1 builds first:

| Form | Count | CIKs | Structured? | Storage |
|---|---|---|---|---|
| 4 (insider) | 344,392 | 627 | **XML, no LLM** | ~3 GB |
| 8-K | 73,249 | 626 | **item codes, no LLM** | ~10 GB |
| 10-Q | 17,112 | 623 | prose, needs LLM | ~35 GB |
| 10-K | 5,780 | 623 | prose, needs LLM | ~12 GB |

Form 4 yields roughly 60x the events of 10-K at a fifteenth of the storage and
no inference cost. That alone would justify doing it first under the decision
rule, but the decisive reason is different:

**Form 4 is the positive control this design was missing.** Section 8 already
requires a null-signal control — random events joined to random edges must fail
the gate. That proves the gate can reject noise. It does not prove the gate can
*detect* anything, and a gate that rejects everything passes it just as well.

Insider trading is a well-documented return predictor with exact transaction
and filing dates. If the harness cannot recover it, the harness is broken — a
bad point-in-time join, a mis-signed return, a horizon off by one — and the
finding is about the code, not the market. Without that check, a weak
propagation result at Stage 3 is unattributable: "no effect exists" and "my
join is wrong" look identical.

So Stage 3's gate must satisfy **both** controls before any propagation result
from it is believed.

## 10. Risks

| Risk | Mitigation |
|---|---|
| No real edge exists in this data | Gate reports it honestly rather than manufacturing one; stage 3 can return "nothing admitted" and that is a valid outcome |
| Multiple-testing false positives | Chronological holdout + minimum `n` + null-signal control |
| Survivorship bias | Point-in-time constituents with `removed_date` |
| Lookahead bias | `acceptance_time` as the only clock; explicit leakage tests |
| Sector shocks masquerading as propagation | Abnormal returns are sector-adjusted, not raw |
| `yfinance` breaking | Isolated behind the ABC; swap in a licensed provider without touching stages 1-5 |
| Vault/DB drift | Vault is generated only; `render-vault` rebuilds from scratch |

## 11. Explicit non-goals

No order execution, no broker API, no position sizing, no portfolio construction, no
investment advice. The system's output is a scored, sourced, track-recorded prediction. Any
decision made from it is the user's own.
