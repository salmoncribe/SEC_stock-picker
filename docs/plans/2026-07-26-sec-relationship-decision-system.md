# SEC Relationship Decision System -- Build Plan

**Status:** proposed -- research and design only; no live trading alert is authorized by this plan.

**Goal:** turn public SEC disclosures into a reproducible research pipeline that calls an
LLM only for a small set of independently strong, still-tradeable relationship patterns.
The LLM can approve, hold, or reject a human-executed long/short *candidate*. It never
places, modifies, or closes an order.

**Primary policy:** no candidate can be described as a ``2% net`` opportunity unless
the system has shown that it can capture at least 2% after the estimated full cost of
execution, at the planned quantity, before the stop or time exit. A gross chart target
is not enough.

## 1. Decisions Locked By This Plan

1. DuckDB is the source of truth. Obsidian is a regenerable research projection and
   Telegram is a delivery channel; neither is a trigger.
2. The LLM is never the first selector. Deterministic evidence, timing, tradeability,
   and statistical gates choose the candidates it may inspect.
3. Evidence quality and tradeability are separate scores. A compelling filing with no
   remaining executable move does not get an LLM call.
4. The live system defaults closed. Missing, late, ambiguous, corrected, stale, or
   untradeable data suppresses an alert and records the reason.
5. Until a suitable real-time market-data source, a first-public-disclosure source, and
   a broker/borrow data source are selected, the system remains research/paper-only.
6. Existing no-auto-execution policy remains in force. A Telegram message is a research
   candidate with a human decision, not an order instruction.

## 2. Current Foundation

The existing relationship-to-alert path is worth preserving:

```text
company_edges -> propagation samples -> active signal_status
  -> EventAlert -> TradeAlertRecord -> trade_alerts -> Telegram
```

- SEC relationship extraction already uses schema-constrained Ollama output in
  `src/market_intelligence/collectors/relationships.py`.
- Events, edges, impact studies, promotion status, trade plans, Telegram delivery, and
  an immutable alert ledger already exist in `src/market_intelligence/database.py`,
  `src/market_intelligence/signals/`, and `src/market_intelligence/autopilot/`.
- The current price source is daily YFinance OHLCV. That is sufficient for historical
  daily research, but not for deciding whether a filing is already priced at a specific
  intraday time.
- `analytics/eventstudy.py` correctly starts after public availability and uses
  chronological/purged splits. Preserve that discipline.

## 3. Target Architecture

```text
SEC dissemination + issuer-release sources                  real-time market / borrow sources
                 |                                                        |
                 v                                                        v
         immutable filing package --------------------------------> market snapshot
                 |                                                        |
                 v                                                        |
    deterministic facts / graph edges                                    |
                 |                                                        |
                 +----------> opportunity score and hard gates <---------+
                                      |
                                      | eligible only
                                      v
                         frozen evidence packet (no DB lock)
                                      |
                                      v
                         LLM decision: approve / hold / reject
                                      |
                                      v
                    deterministic verifier + portfolio risk gate
                                      |
                           notify only if every gate passes
                                      |
                    DuckDB decision ledger -> Obsidian -> Telegram
                                      |
                                      v
                   paper/live outcome grader -> calibration dashboard
```

The decision worker must read a frozen packet, release the DuckDB connection before the
network/model call, then reopen a short-lived connection to persist a result. An LLM must
never hold the single-writer DuckDB lock.

## 4. The Three-Gate Strategy

### Gate A -- Evidence: should an LLM read this?

The score is transparent and versioned. Initial components:

| Component | Initial points |
| --- | ---: |
| New material agreement, acquisition, customer/supplier, or ownership fact | +4 |
| Named public-company counterparty | +3 |
| Form 4 open-market purchase by CEO, CFO, or director | +3 |
| Two or more distinct insiders buying in the configured cluster window | +2 |
| Existing validated supply/customer/partner relationship corroborates it | +2 |
| Contract/value/magnitude is material to the affected company | +2 |
| Facts became public within seven days of each other | +2 |
| Shared board member is a corroborating channel | +1 |
| Vague or unnamed counterparty, weak extraction, or ambiguous entity | -3 |
| Grant, exercise, gift, withholding, or other non-open-market transaction | -3 |
| Amendment, duplicate, or already-public disclosure | reject |

Initial rule: evidence score `>= 8` is necessary but not sufficient for an LLM call.
Every component, source quote, version, and rejection reason is retained. Score changes
produce a new policy version; historic scores are never overwritten.

### Gate B -- Tradeability: can 2% net actually be captured?

Compute the economics at the intended quantity rather than from a chart screenshot:

```text
net_reward = directional(target_exit_vwap - entry_vwap) * quantity
             - fees - borrow - P95 spread - P95 slippage - P95 impact

R = directional(entry_vwap - stop_exit_vwap) * quantity
    + fees + borrow + P95 exit slippage
```

Use a separate stress value for halt/reopen and gap-through-stop risk. The candidate is
eligible only when all are true:

```text
conservative remaining net move >= 2.0% of entry notional
net_reward / max(R, gap_R_p95) >= 2.0
fresh, firm, uncrossed quote and valid market status
liquidity, spread, and participation limits pass
the price has not already absorbed the expected move
no earnings, halt, LULD, correction, duplicate, or conflict block exists
short candidates also have a live broker-approved locate/borrow
```

The estimated remaining move must use a lower confidence bound from comparable, fully
matured, costed historical opportunities -- never a mean return or LLM confidence. For
a live strategy family, begin with at least 30 independent root-event clusters; below
that, candidates are research-only.

### Gate C -- LLM: does the original evidence support the thesis?

The LLM receives only a frozen, cited evidence packet: filing/exhibit spans, structured
facts, relationship edges, Form 4 transaction details, prior related filings, timing
state, market snapshot, and deterministic gate results. Its JSON verdict is one of:

```yaml
verdict: approve_long | approve_short | hold | reject
causal_chain: string
why_now: string
evidence_ids: [source_span_id]
disconfirming_evidence_ids: [source_span_id]
risk_flags: [string]
missing_information: [string]
```

The post-LLM validator rejects unsupported quotes, invalid entities, stale provenance,
unsupported relation types, missing disconfirming analysis, or non-schema output. LLM
self-confidence is not an alpha signal. A rejected or failed model call produces no edge
and no alert.

## 5. Required Data Contracts

### Filing and disclosure timing

Store all times independently in UTC and preserve raw source values:

```text
event_occurred_at
issuer_claimed_release_at
edgar_accepted_at
sec_dissemination_observed_at
our_fetch_at
parse_complete_at
candidate_scored_at
decision_at
market_snapshot_at
alert_sent_at
paper_fill_at / actual_fill_at
```

`edgar_accepted_at` is an authoritative regulatory timestamp, not proof of first public
distribution. The live path needs monitored issuer IR/releases/webcasts or another
reliable broad-distribution source to set `first_public_at = verified`. With EDGAR-only
coverage, `first_public_at = unknown` and the candidate stays research-only.

Treat the complete submission, primary document, and relevant exhibits as one disclosure
package. Persist raw bytes and SHA-256 hashes; parse the header first. Form 4 uses SEC
public-dissemination time as its signal clock, never its transaction date.

### Market, borrow, and operational snapshot

For every decision record, persist:

- exchange event and local receipt timestamps; feed sequence; NBBO bid/ask/size/venue;
  midpoint; trade conditions; depth used; session/auction indicator; LULD/SSR/halt state;
  one-minute volume; historical ADV; volatility; market and sector reference returns;
- expected P50/P95 spread, slippage, impact, commissions, borrow fee, and gap risk;
- borrow/locate identifier, exact quantity, expiration, recall state, hard-to-borrow
  status, and broker restriction status for every short;
- policy/config hash, source/data snapshot identifiers, parser/model/prompt versions,
  score components, all gate decisions, and all suppression reasons.

### New persisted objects

Add, do not overload, the following tables:

1. `filing_packages` and `source_observations` -- filing/document hashes, timing,
   first-public state, parser version, and retrieval audit trail.
2. `market_snapshots` -- immutable quote/status/borrow data used by one evaluation.
3. `relationship_opportunities` -- natural key `(event_id, edge_id, target_ticker,
   horizon_days, strategy_version)`; evidence score, tradeability result, evidence
   snapshot hash, status, and timestamps.
4. `opportunity_decisions` -- append-only LLM attempt/result keyed by opportunity,
   input hash, prompt version, and model version.
5. `strategy_versions` and `experiment_registry` -- configuration/code/data/prompt/cost
   versions and every attempted threshold/model/feature family.
6. `paper_orders` and `paper_fills` -- simulated executable orders and outcomes, separate
   from the existing `trade_alerts` ledger.
7. `kill_switch_events` and `system_health` -- when, why, and who paused alerts.

Use `trade_alerts` only after a candidate passes every decision and portfolio gate. It
remains the Telegram-deduplication and eventual grading ledger.

## 6. Implementation Phases

### Phase 0 -- Freeze the baseline and remove known blockers

**Purpose:** make current research reproducible before layering a new system on top.

- Inventory and test the existing dirty people/relationship/graph changes without
  reverting them. The relationship extractor currently owns a database write lock and has
  recorded model timeouts; resolve this before adding a second model workflow.
- Verify the event-sample uniqueness migration and propagation fan-out against a clean
  database copy.
- Capture baseline test results, database schema version, current configuration hash,
  model build, and launchd state.
- Confirm that the active graph watcher cannot collide with planned workers.

**Exit gate:** full existing suite passes; timeout/retry behavior is understood; no unknown
schema migration remains; one reproducible graph/propagation run completes.

### Phase 1 -- Provenance and first-public timing

**Purpose:** establish whether a filing is new and when it was actually usable.

- Add complete-submission header retrieval and parse `ACCEPTANCE-DATETIME` using
  `America/New_York` conversion rules.
- Persist immutable filing packages, document hashes, exhibits, observations, corrections,
  amendments, and dedup/supersession links.
- Add issuer-release/webcast/IR source adapter interface. Do not claim live eligibility
  without a configured source that can verify public-distribution time.
- Implement a bounded SEC discovery queue for `4`, `4/A`, `8-K`, `8-K/A`, `6-K`, `10-Q`,
  `10-K`, `13D`, `13G`, and amendments; use global rate limiting below SEC limits.
- Add daily/quarterly archive reconciliation. A missing live observation becomes a data
  incident, not a silently backfilled signal.

**Tests:** timestamp conversion around daylight saving; duplicate exhibit; prior release;
amendment/correction cancellation; stale/unknown first-public state; reconciliation gap.

**Exit gate:** 100 sampled filing packages have exact raw hashes, correct acceptance times,
and correct live/research-only eligibility classification.

### Phase 2 -- Real-time market and borrow adapters

**Purpose:** replace daily-only assumptions at decision time.

- Select and integrate a real-time source with NBBO, trades, market status, LULD, and
  one-minute bars for the defined universe. Daily YFinance remains historical-only.
- Add a broker read-only adapter for account state, positions, broker restrictions, and
  short borrow/locate status. No order routing is in scope.
- Build immutable `market_snapshots` and market-data health checks.
- Implement configurable session controls: regular hours only by default; opening auction,
  first five minutes, last fifteen minutes, extended hours, halts, LULD, and SSR all fail
  closed unless a separately validated policy permits them.

**Tests:** stale/locked/crossed quote; feed sequence gap; no quote; halt/LULD/SSR; borrow
recall; quantity exceeds locate; delayed broker acknowledgment; clock drift.

**Exit gate:** replayed snapshots and paper broker data prove the system never evaluates a
candidate with stale, invalid, or unborrowable conditions.

### Phase 3 -- Deterministic opportunity score and policy registry

**Purpose:** choose candidates without an LLM and make score tuning honest.

- Add `schemas/opportunities.py` and `signals/relationship_opportunities.py` as pure
  modules. Inputs are active event/edge evidence, first-public state, timing, historical
  cell metrics, and a market snapshot.
- Add `relationship_decision` configuration: score weights, score threshold, minimum
  independent samples, net target, R threshold, liquidity/spread/participation limits,
  timing budget, and kill-switch thresholds.
- Persist the complete component breakdown, policy version, and every suppression reason.
- Define `p_net_2`: probability of reaching a fully costed `+2%` before stop/time exit;
  and `mu_net_lcb`: the lower confidence bound of expected executable net return.
- Do not use the present display `confidence` value as either forecast. It is a heuristic
  blend and remains display-only until recalibrated.

**Tests:** component additivity; hard rejections override score; score-version immutability;
same input is idempotent; no future timestamp can affect a result; no duplicate filing can
multiply a root event.

**Exit gate:** every historical candidate can be replayed exactly from its input hashes and
policy version, with a human-readable explanation of why it did or did not reach the LLM.

### Phase 4 -- Cost-aware statistical calibration

**Purpose:** prove that “2% net” refers to executable economics rather than raw returns.

- Split data chronologically into immutable discovery, later calibration, and sealed final
  test windows. Purge/embargo boundaries by maximum holding period.
- Define independent root-event clusters: same issuer event, same target/day, repeated
  filings, and fan-out edges count once for uncertainty and promotion.
- Build quote-aware fill simulation: entry at ask for long/bid for short after measured
  latency; exit at opposite side; fees, borrow, spread, impact, stop gaps, halts, and
  adverse ordering when intrabar ordering is unknown.
- Use moving/block bootstrap over root-event clusters; track multiple testing through an
  append-only experiment registry. Tune on calibration only; never tune sealed results.
- Calibrate monotone score-to-`p_net_2` mapping and report reliability, Brier score,
  score-band monotonicity, net-return lower bounds, and regime slices.

**Tests:** 100 duplicated rows cannot strengthen a strategy; common sector shock widens
uncertainty; future data cannot change a historical decision; split/adjustment/stop-gap
cases; stressed-cost scenario; calibration leakage.

**Exit gate:** a strategy family may use `2% net` only when the sealed test clears
pre-registered net-return, probability, calibration, and regime-stability thresholds. If
not, it remains research-only and may not send a trade candidate.

### Phase 5 -- LLM decision worker and verifier

**Purpose:** add qualitative reasoning after deterministic eligibility, not before it.

- Create `llm/opportunity_decider.py` and a versioned schema-constrained prompt.
- Build frozen evidence packets with exact source spans and both supporting and potentially
  disconfirming facts. Include no live database connection in the model call.
- Persist every attempt, timeout, schema failure, abstention, and verdict in
  `opportunity_decisions`.
- Add deterministic quote-span verifier, entity resolver, relation allow-list, stale-input
  validator, and model/prompt version checks.
- Build a frozen, double-reviewed LLM evaluation corpus with hard negatives, boilerplate,
  aliases, stale relationships, conflicting language, and unsupported claims.

**Tests:** malformed JSON; hallucinated quote; mismatched entity; missing counterevidence;
model timeout; retry dedup; prompt/model version drift; same evidence with different
temperature is not silently merged.

**Exit gate:** quote support, entity resolution, abstention, and schema-validity thresholds
are met on the held-out corpus. An LLM may not create an edge or alert when verification
fails.

### Phase 6 -- Portfolio gate, Obsidian decision notes, and Telegram

**Purpose:** deliver only approved, non-duplicative, portfolio-safe candidates.

- Add portfolio state/risk reservation: pending orders, open candidates, issuer/sector/
  catalyst overlap, gross/net exposure, daily loss, and correlated factor exposure.
- Add kill switches for data freshness, unknown provenance, parser/model/prompt drift,
  invalid corporate actions, LULD/halt, missing borrow, failed fill assumptions, and
  calibration degradation.
- Project one Obsidian decision note per opportunity with linked company/person/filing/
  evidence notes and score/gate history. Do not expose raw private broker state in the
  vault.
- Extend Telegram only for `approve_long`/`approve_short` candidates after all hard gates;
  `hold` and `reject` stay in DuckDB/Obsidian. The message includes evidence score,
  net-2% estimate, R, maximum acceptable entry condition, invalidation, time limit, and
  why it could still be wrong.
- Keep existing `trade_alerts` first-firing-wins semantics. A correction, halt, or later
  re-evaluation cancels eligibility; it never duplicates a notification.

**Tests:** duplicate alert; competing correlated candidates; daily risk breach; stale quote
between decision and send; kill switch; Telegram failure; Obsidian projection deletion;
decision-to-alert audit reconstruction.

**Exit gate:** shadow notifications can be replayed end-to-end without duplicate alerts,
missing audit fields, or portfolio-limit bypasses.

### Phase 7 -- Staged rollout and score-tuning cadence

**Purpose:** learn without disguising research as proven alpha.

1. **Historical replay:** no notifications; reconstruct opportunities and costs at original
   timestamps.
2. **Paper shadow:** run live sources and LLM, log every candidate/rejection, send only a
   daily operations summary. Continue for the longer of six months or 100 independent
   matured root events.
3. **Live notification shadow:** Telegram labels candidates as paper/research; compare live
   observed quotes, latency, and fill assumptions against the simulator.
4. **Human-executed candidate mode:** only after a written go/no-go review against the
   sealed-test and paper metrics. No broker order routing is introduced.

Score changes are experiments, not edits in place. Each change creates a new
`strategy_version`, declares its hypothesis, is tuned only on calibration data, and earns a
new sealed test. The dashboard reports score-band outcomes, `p_net_2` calibration, net P&L
after modeled costs, rejected candidates, latency percentiles, LLM quote support, and every
kill-switch activation.

## 7. Subagent Operating Plan

### Rules for every worker

- This repository has important uncommitted user work. No worker may revert, reformat, or
  overwrite changes it did not create.
- Use isolated worktrees only after the coordinator captures the intended working-tree
  baseline. Workers receive explicit file ownership and may edit only their assigned files.
- No two workers edit `database.py`, `storage/duckdb.py`, `config.py`, `cli.py`,
  `autopilot/orchestrator.py`, `autopilot/notify.py`, or launchd files. Those are integration
  files and have one coordinator owner.
- Each worker supplies unit tests, fixtures, migration notes, and a short handoff describing
  inputs, outputs, idempotency key, version behavior, and unresolved assumptions.
- A reviewer agent checks the final assembled branch for leakage, duplicate notifications,
  DB-lock duration, stale data, and test omissions before any stage advances.

### Workstreams and ownership

| Workstream | Subagent ownership | Files it may own | Dependencies |
| --- | --- | --- | --- |
| A. Provenance | Filing/package timing worker | new `schemas/provenance.py`, `collectors/disclosure_packages.py`, `extraction/submission_header.py`, tests/fixtures | Phase 0 |
| B. Market snapshot | Market-data worker | new `clients/realtime_market.py`, `collectors/market_snapshots.py`, `schemas/market_snapshot.py`, tests | vendor choice |
| C. Opportunity score | Quant-score worker | new `schemas/opportunities.py`, `signals/relationship_opportunities.py`, `signals/net_return.py`, tests | A + B contracts |
| D. LLM decision | LLM worker | new `llm/opportunity_decider.py`, prompts, schemas, fixtures/tests | C packet contract |
| E. Validation | Research-validation worker | new `analytics/executable_validation.py`, `analytics/experiment_registry.py`, tests | B + C |
| F. Integration | Coordinator only | database, storage, config, CLI, orchestrator, notifier, vault projection, launchd | A-E completed |
| G. Independent audit | Review-only agent | no edits unless explicitly assigned | assembled integration |

### Execution order

1. Coordinator completes Phase 0 and freezes the shared baseline.
2. Workstreams A and B proceed in parallel because they create new contracts/files.
3. Once the data contracts are approved, C and E proceed in parallel; C uses interfaces,
   not a concrete vendor implementation.
4. D starts when C can produce a frozen evidence packet.
5. Coordinator integrates schema/config/CLI/orchestration serially, then runs all tests.
6. Review-only agent performs the adversarial audit, then Phase 7 begins in shadow mode.

### Example worker prompts

- **Quant-score worker:** “Own only the new opportunity schema and pure deterministic
  scoring modules. Implement evidence/tradeability gates, policy-version immutability, and
  tests. Do not edit database, config, CLI, autopilot, or existing dirty files. Preserve
  existing user work.”
- **LLM worker:** “Own only the new decision module and offline fixtures. The model receives
  a frozen packet and returns strict JSON with cited span IDs and disconfirming evidence.
  No database I/O and no Telegram code. Add failure/timeout/schema tests.”
- **Validation worker:** “Own only new executable-validation modules. Enforce chronological
  discovery/calibration/sealed splits, root-event clustering, cost-aware fill scenarios,
  and multiple-testing accounting. Do not alter current signal gates without coordinator
  review.”
- **Audit worker:** “Read assembled code and tests only. Find point-in-time leakage,
  first-public ambiguity, stale quote use, lock-held model calls, duplicate notification
  risk, wrong short eligibility, and invalid 2%-net claims. Report findings ordered by
  severity with exact paths.”

## 8. Required User Decisions Before Live-Candidate Alerts

These choices materially change the system and cannot be safely guessed:

1. Real-time market-data provider and budget: must supply at least NBBO, trades, market
   status, LULD, and historical intraday replay for the intended universe.
2. First-public-disclosure coverage: issuer IR source, news/wire source, or a decision to
   remain EDGAR-only research mode.
3. Broker integration: read-only account/position/borrow data provider; whether short
   candidates are enabled at all in the first rollout.
4. Initial eligible universe and liquidity tiers.
5. Risk policy values: account basis, per-candidate risk, gross/net/sector limits, daily
   loss limit, holding periods, and whether extended-hours candidates remain disabled.
6. Definition of a paper fill and how human-entered actual trades, if any, are recorded.

## 9. Research Basis

- SEC EDGAR APIs update submission data throughout the day and document processing delays:
  https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- SEC developer resources set fair-access expectations and request limits:
  https://www.sec.gov/about/developer-resources
- SEC documents acceptance timestamps and filing/dissemination behavior:
  https://www.sec.gov/about/webmaster-frequently-asked-questions
- SEC Form 8-K instructions describe current-report disclosure and exhibits:
  https://www.sec.gov/files/form8-k.pdf
- SEC Regulation SHO overview covers locate, close-out, price-test, and short-sale risks:
  https://www.sec.gov/investor/pubs/regsho.htm
- SEC Rule 605 guidance explains execution quality and NBBO-based measurement:
  https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions/frequently-asked-questions-rule-605-regulation-nms
- FINRA explains extended-hours risks and execution/liquidity differences:
  https://www.finra.org/investors/insights/extended-hours-trading
- Bailey and Lopez de Prado describe selection bias and backtest-overfitting controls:
  https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

## 10. Definition of Done

The system is ready for the first human-executed candidate only when every statement below
is true:

- It can reproduce a decision from source bytes, times, quote snapshot, policy/model/prompt
  versions, and gate results.
- It has verified first-public timing or marks the candidate research-only.
- It has live-quality market and short-borrow data where required.
- It has passed sealed cost-aware validation for the claimed `2% net`/`2R` strategy family.
- It has completed the paper-shadow acceptance period and shown acceptable latency, fill,
  calibration, and LLM-evidence metrics.
- All kill switches default closed, have tests, send an operator notification, and require a
  documented review to resume.
- Telegram sends only a deduplicated, fully-auditable candidate; the system never sends an
  order and never places a trade.
