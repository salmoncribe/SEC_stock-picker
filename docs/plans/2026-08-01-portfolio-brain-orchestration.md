# Portfolio Brain — Build Orchestration

**Status:** orchestration plan, derived from the weekend build plan
**Authored:** 2026-07-31 (Fri evening)
**Orchestrator artifact.** The build plan says *what*; this says *who, when, and against which contract*.

---

## 1. Pre-flight findings (verified against the repo, 2026-07-31)

### Confirmed
| Plan claim | Verified |
|---|---|
| No `portfolio/` package | ✅ greenfield, no collision |
| 770 existing tests | ✅ 770 test functions |
| `paper_orders` / `paper_fills`, zero code | ✅ `database.py:664-690` |
| `experiment_registry` / `strategy_versions` | ✅ `database.py:592-621` |
| `build_propagation` disabled over 2 defects | ✅ `orchestrator.py:124`; the inline comment names both defects verbatim |
| `backtest_data.py` loaders (5) | ✅ `load_bars`, `load_signals`, `load_hedged_edges`, `filter_tradeable`, `load_betas` |
| `executable_validation.py` primitives | ✅ `moving_block_bootstrap`, `CostScenario`, `TwoPercentProof` |
| `experiment_registry.py` chain | ✅ `ExperimentRecord`, `ExperimentRegistry` |
| briefing untrusted-ticker join | ✅ `briefing.py:216` — `ON ce.source_ticker = e.ticker` |
| `company_edges.report_date` for PIT filter | ✅ column exists, typed `DATE` |
| graph-watchdog contends every 10 min | ✅ `StartInterval 600` |
| `portfolio_gate.evaluate`, `trade_plan.wilder_atr` | ✅ present |

### Corrections to the build plan
1. **Workstream P item 5 is CUT.** `scripts/extract_relationships.sh` is 107 lines; there is no line 109, and `bash -n` parses clean. Already fixed or misdiagnosed. Workstream P is now **4 items, not 5**.
2. **ECOS is the only wheel landmine.** `cvxpy 1.9.2` / `scipy 1.18.0` / `pyportfolioopt 1.6.0` / Clarabel (`cp39-abi3`) / SCS (`cp314` native) all resolve on cp314-arm64. **ECOS 2.0.14 has no macOS arm64 wheel** — pin the solver set explicitly and assert no source build.
3. **The pypfopt timebox is a runtime test, not an install test.** pypfopt is `py3-none-any` so it always installs; pandas is already **3.0.3**. The 30-minute verdict must execute `BlackLittermanModel` + Idzorek against real pandas 3 and compare to the hand-rolled numpy path.

### Environment facts that constrain scheduling
- DuckDB: **7.2 GB** at `/Volumes/Extreme/home-migrated/quant/data/database/market_intelligence.duckdb`
- `MARKET_INTELLIGENCE_HOME=/Volumes/Extreme/home-migrated/quant` → all Parquet artifacts land there
- Python **3.14.6**, pandas **3.0.3**, numpy **2.5.1** already installed
- Five `ai.quant.*` launchd agents loaded: `autopilot` (06:00), `autopilot-watchdog`, `morning-scan`, `graph-watchdog` (600 s), `ramguard`
- `pyproject.toml` floor stays `>=3.12`; resolved wheels are cp314

---

## 1b. Wave 0 — COMPLETE (Fri 2026-07-31 evening)

Executed rather than scheduled, on Michael's call ("no reason to delay it, it's simulation data not real data").

**Dependencies.** `uv sync` in **8.8 s**, all wheels, **zero source builds**. Installed: `cvxpy 1.9.2`, `scipy 1.18.0`, `pyportfolioopt 1.6.0`, `clarabel 0.11.1`, `scs 3.2.11`, `osqp 1.1.3`, `highspy 1.15.1`. **ECOS never entered the dependency graph** — the one identified landmine did not fire. Note `scikit-learn 1.9.0` arrives transitively via pypfopt; Ledoit-Wolf is still hand-rolled so domain code carries no sklearn import and the dep leaves with pypfopt if `bl_engine` ever flips.

**Solver proof.** `cvxpy.installed_solvers()` → `['CLARABEL', 'HIGHS', 'OSQP', 'SCIPY', 'SCS']`. Both Clarabel and SCS solved the exact `max μ'w − γw'Σw − κ‖w−w₀‖₁` objective, with caps, on:
- a well-conditioned 7-name covariance, and
- **a rank-2 covariance over 7 names, min eigenvalue −3.6e-18** (numerically singular)

returning identical weights in both cases. **The plan's headline optimizer risk is retired.** `nearest_psd` + ridge + the three-stage fallback are still built — the replay cannot stop to debug — but as a guarantee, not a crutch.

**bl_engine verdict → `pypfopt`.** On pandas **3.0.3** / numpy 2.5.1 / Python 3.14.6, PyPortfolioOpt's `BlackLittermanModel` posterior matched the hand-rolled closed form to **1.4e-17** on a fixed 3-asset case — machine epsilon, i.e. the same computation. Idzorek ran and produced a correctly-ordered omega (confidence 0.25 → ω 7.6e-3; confidence 0.90 → ω 5.6e-5). The numpy path remains a tested substitute, not an expected fallback. Verdict script retained in the session scratchpad.

**Contract shipped.** `PortfolioConfig` (30 fields, all defaulted, two cross-field validators: flatten > halve, cluster ≥ position) + `SettingsFile.portfolio` + a documented `portfolio:` block in `settings.yaml`. Typed stubs for all 10 modules — every dataclass fully defined, every function signature final, bodies `raise NotImplementedError`.

**Verification:** 770/770 existing tests green · `ruff check src/ tests/` clean · `mypy` clean on all 11 new files · all 10 modules import.

**Michael's decision recorded:** the drawdown governor **bypasses** the no-trade band. Encoded as `PortfolioConfig.governor_bypasses_band = True` and as the `bypass` parameter on `optimizer.apply_no_trade_band`.

---

## 2. Orchestration principle

Agents may run in parallel **only when no two write the same file.** The build plan's module split gives that naturally — one module + one test file per agent. The two shared surfaces (`pyproject.toml`, `config.py`/`settings.yaml`) are written **serially by the orchestrator in Wave 0** and never touched by an agent.

The unlock is the **typed stub contract**: Wave 0 materializes every dataclass and function signature from the build plan into real files with `raise NotImplementedError` bodies. Agents then import each other's *types* without waiting for each other's *implementations*. Without this, Wave 2 blocks on Wave 1 and the weekend serializes.

---

## 3. Dependency DAG

```
Wave 0 (orchestrator, serial) ── deps · solver proof · pypfopt verdict · PortfolioConfig · TYPED STUBS
   │
   ├── Wave 1 (5 parallel, zero cross-deps) ──────────────────────────────┐
   │      A1 account.py     A2 costs.py    A3 risk.py    A4 metrics.py    │
   │      A5 Workstream P fixes (existing files — disjoint from portfolio/)│
   │                                                                       │
   ├── Wave 2 (2 parallel) ── B1 feed.py (DB) · B2 views.py (needs risk)  │
   │                                                                       │
   ├── Wave 3 (1) ────────── C1 optimizer.py (needs risk + views)         │
   │                                                                       │
   ├── Wave 4 (1, hardest) ─ D1 simulator.py + leak canaries              │
   │                                                                       │
   └── Wave 5 (3 parallel) ─ E1 store.py · E2 report.py · E3 CLI          │
                                                                           │
   Long-pole side track (starts end of Wave 1) ────────────────────────────┘
        build-propagation run → evaluate → GATE DECISION (Sun PM)
```

**Sequencing change vs. the build plan:** Workstream P moves from Saturday *evening* to Wave 1. Its fixes unblock `build-propagation`, which is the longest-running job of the weekend (prior attempt: 3 h 40 m hang). Starting it Saturday afternoon rather than Saturday night gives the gate evaluation a full extra shift of runway, and lets it fail early if it's going to.

---

## 4. Agent roster and contracts

Each agent receives: its module's signatures from the build plan, its exact test list, the style reference (`tests/test_backtest.py`), and the hard rules below.

**Rules binding on every agent:**
- TDD — test first, watch it fail, then implement.
- Frozen dataclasses, numpy in domain math, **no pandas in domain code**, structlog, ruff line-length 100.
- Offline tests only. Use `memory_db` / `tmp_config` fixtures. **No agent opens the 7.2 GB DuckDB.**
- Do not edit `pyproject.toml`, `config.py`, `settings.yaml`, or any file outside your assigned module + test file.
- Report back: tests passing/failing, deviations from the assigned signature, anything the contract got wrong.

| Agent | Writes | Tests | Notes |
|---|---|---|---|
| **A1 account** | `portfolio/account.py` | `test_portfolio_account.py` (10) | Highest-value invariants. Equity identity ≡ cash+Σshares·px to 1e-9; overdraft/oversell/backdated raise. Golden P&L is the noon checkpoint. |
| **A2 costs** | `portfolio/costs.py` | `test_portfolio_costs.py` (5) | Small. Slippage signs must mirror `backtest._fill` exactly — that's the parity contract. |
| **A3 risk** | `portfolio/risk.py` | `test_portfolio_risk.py` (11) | Largest Wave-1 module. Hand-rolled Ledoit-Wolf (~40 lines), union-find (~30). Clusters exclude interlocks *and* future edges. |
| **A4 metrics** | `portfolio/metrics.py` | `test_portfolio_metrics.py` (8) | Fully independent (scipy.stats.norm only). PSR against the published Bailey–López de Prado example. |
| **A5 workstream-P** | `signals/dataset.py`, `autopilot/briefing.py` | 3 new P tests | Items 1–4 only (item 5 is cut). Dedup before upsert; batch-scoped `_existing_keys`; PIT edge filter; CIK join replacing `source_ticker = e.ticker`. **Must not run the build** — that's supervised. |
| **B1 feed** | `portfolio/feed.py` | `test_portfolio_feed.py` (7) | `trailing_returns(end_exclusive)` is the no-lookahead choke point — test it hardest. Seeded fixture DB only. |
| **B2 views** | `portfolio/views.py` | `test_portfolio_views.py` (8) | Drag adjustment `μ = hedged_edge − ½σ²H`. Closed-form BL vs. textbook 3-asset golden. `bl_engine` default set by Wave 0's verdict. |
| **C1 optimizer** | `portfolio/optimizer.py` | `test_portfolio_optimizer.py` (9) | Must **never raise mid-replay** — Clarabel → SCS → deterministic inverse-vol fallback. Degenerate-Σ test is the one that matters. |
| **D1 simulator** | `portfolio/simulator.py` | `test_portfolio_simulator.py` (11) | Hardest module. The three leak canaries live here. Recommend orchestrator-driven rather than fully delegated. |
| **E1 store** | `portfolio/store.py` + 4 helpers in `storage/duckdb.py` | `test_portfolio_store.py` (8) | Only Wave-5 agent touching a shared file (`storage/duckdb.py`) — runs alone or with a narrow diff. |
| **E2 report** | `portfolio/report.py` | `test_portfolio_report.py` (5) | Pure renderers, never take a connection. First on the cut list. |
| **E3 cli** | `cli.py` portfolio sub-app | `test_portfolio_cli.py` (4) | Touches shared `cli.py` — additive sub-app only. |

---

## 5. Checkpoints (unchanged from the build plan, plus one)

| When | Gate | Failure action |
|---|---|---|
| Wave 0 end | cvxpy solves a trivial QP; pypfopt verdict recorded | pypfopt fails → `bl_engine="numpy"`, proceed |
| Sat noon | A1 golden P&L green | slip → account is never cut, cut Wave 5 instead |
| Sat ~19:00 | panel→Σ→views→posterior on synthetic 5-asset data | — |
| Sat evening | experiment **preregistered** before any sweep | non-negotiable |
| **NEW: Sat evening** | propagation build launched, watchdog paused, RSS logged | hangs → kill, gate reports "not yet measured" |
| **Sun 13:00** | full discovery replay < ~2 min, RSS < 1 GB, Parquet written | slip past 14:30 → start cutting |
| Sun PM | DSR ≥ 0.95, PBO ≤ 0.20, stress passes | fail → no holdout read, record stands |
| Sun last action | **exactly one** sealed holdout read | — |

**Cut order:** Telegram/Obsidian rendering → min-CVaR → sweep automation → step polish → propagation evaluation slips to Monday.
**Never cut:** account invariants · the three leak canaries · PIT edge filter · DSR/PBO before unsealing · exactly one sealed read · bootstrap stress scenario.

---

## 5b. Contract amendments discovered during the build

Recorded as they surface, so later agents inherit the correction rather than rediscovering it.

**Amendment 1 — `DEPOSIT` lifts the high-water mark (A1, verified).** The original stub said the HWM never moves in `apply_fill`. That was wrong for deposits: `replay([genesis_fill(10_000, d)])` produced `high_water_mark == 0.0` while `genesis(10_000, d)` produced `10_000.0`, breaking the module's own invariant #5 (replay reproduces state exactly) — which is load-bearing, because `portfolio status` rebuilds the account of record from `paper_fills` by folding it. It is also correct capital-flow accounting: an unadjusted mark reports later deposits as a permanent phantom drawdown, and adding capital must not shrink a measured drawdown. Verified directly: `genesis` and `replay` now agree on both cash and HWM.

**Amendment 2 — id format pinned, and `write_genesis` must not construct its own row (A1 flagged, orchestrator closed).** `simulation_version` is a **suffix**: `pf:CASH:2026-01-05:DEPOSIT:portfolio-sim-v1`. `store.write_genesis` must call `account.genesis_fill()` and persist its return value rather than building the row independently — two constructions of the opening deposit will drift, and since `paper_fills` is keyed for idempotent re-runs, each would write its own "opening" row and silently double the account's starting capital on a second run. Closed structurally in the `store.py` contract before E1 starts.

**Amendment 3 — `vol_target_scale` gains `periods_per_year: int = 252` (A3 flagged, orchestrator ordered).** The original signature mixed an annual `target_vol` (0.12) with a daily covariance and had no periodicity parameter. Handed a daily sigma it returns `1.0` on essentially every day — **volatility targeting silently disabled**, which would quietly void one of Michael's four explicit risk decisions. Documenting the caller's responsibility was rejected as the fix: a knob that is off while appearing on is worse than one that is absent. The function already mixed periodicities, so the period count is a missing parameter, not an invented assumption.

**Amendment 4 — `blend_cov(sample_cov, ewma, *, lw_weight)`.** Renamed from `weight`. Semantics: `lw_weight * sample_cov + (1 - lw_weight) * ewma`, matching `PortfolioConfig.lw_blend`. Both arguments are the same shape and dtype, so a flipped blend would produce plausible numbers with no error — the name is the only guard.

**Amendment 5 — the board-interlock edge type is `"shared_board_member"`, not `"interlock"` (A3).** The build plan and the original stub both used the wrong string. `risk.py` uses a strict allow-list of the four commercial types (`competitor`, `customer`, `partner`, `supplier`), so both spellings are correctly excluded, with a drift guard in the tests asserting the literal set matches `schemas.edges.EdgeType`.

**Amendment 6 — `scipy.*` added to pyproject's mypy `ignore_missing_imports` override (orchestrator).** scipy ships no `py.typed`. Declared centrally alongside duckdb/pyarrow/yfinance rather than silenced inline. If `scipy-stubs` is ever added to the dev extra, remove `scipy.*` in the same change — `warn_unused_ignores = true` would otherwise start failing.

**Amendment 7 — `load_feed`'s `as_of` fallback is split, because betas and edges are dated differently (B1).** Edges carry `report_date` and get re-filtered per day by `risk.commercial_clusters`, so they fall back to unbounded; bounding them at the first signal date would starve later days of graph structure they legitimately knew. Betas have no row-level date, so they fall back to the **earliest signal date** — an unbounded beta fetch would fit the hedge ratio on the very era it hedges, a leak that is invisible in any output and would flatter every hedged return in the replay. A single shared fallback would have been silently wrong for one of the two.

**Amendment 8 — `BLPosterior.sigma` is the input covariance, NOT Black-Litterman's `Σ + M` (B2, modeling decision).** `M` is uncertainty about the *mean*, and with τ=0.05 letting it through would scale every risk budget by roughly `(1 + τ)`. The optimizer, the position caps and `vol_target_scale` would then all be reading a covariance that no test in `risk.py` ever measured. The estimate that was measured is the one that ships. Documented in the module docstring; test 4 asserts `posterior.sigma` is elementwise the input.

**Amendment 9 — `BacktestSignal` carries no cell key, so signal→cell matching is by `(horizon_days, direction)` only (B2).** Where several admitted cells share that pair, the **smallest** hedged edge wins: the signal does not say which cell fired it, and taking the largest would let an unrelated strong cell price the trade — upward-biased selection wearing the costume of a match. A cell absent from `hedged_edges` is skipped entirely rather than falling back to its raw `mean_car`, which is the unhedged label this module exists to keep out. If `BacktestSignal` ever gains a cell key, `views._matchable_cells` is the single function to change.

Related limitation: `AdmittedCell` has no `edge_type`, so `views_from_propagation` cannot tell which relationship type a cell was measured on and applies each admitted cell to every eligible edge (`customer`/`supplier`/`partner`; `competitor` excluded for unknown sign). Currently moot — the propagation gate admits zero cells — but it must be revisited before `graph_return_views_enabled` is ever flipped true.

**Convention note for downstream modules:** `AccountState.positions` and `MarkedAccount.prices` are `MappingProxyType` — immutability is real, not conventional. They compare equal to plain dicts but reject item assignment.

---

## 5c. ⛔ Blocking discovery — `event_samples` constraint mismatch (Workstream P)

**Verified against the live database 2026-07-31.** The 7.2 GB table carries `UNIQUE(event_id, edge_id, horizon_days)`; the code expects `UNIQUE(event_id, edge_id, horizon_days, target_ticker)`. Commit `5672094` created the table with the narrow form and `b89a156` widened it **later the same day**, but `CREATE TABLE IF NOT EXISTS` plus a `migrate_db()` that only issues `ALTER TABLE ... ADD COLUMN` can never rewrite a constraint on an existing table.

Self-control samples write one row per event and never collide. **Propagation fans out** — one event, many targets, sharing `event_id`/`edge_id`/`horizon_days` and differing only in `target_ticker` — so every row after the first is a duplicate-key abort. This is almost certainly what killed the 6am run on 2026-07-29, and **no code change reaches it**: Workstream P's in-Python dedup is defensive only, since these rows are legitimately distinct.

Remediation is a supervised, backed-up table rebuild — see [propagation-build-runbook.md](propagation-build-runbook.md) Step 0. Until it happens, propagation evaluation cannot run, which per the cut list means the gate reports "not yet measured" and nothing else is blocked.

---

## 5d. ⛔ FIRST REAL REPLAY — the drawdown governor is an absorbing state

**Run 2026-07-31, discovery split, default config, against the live database. 2,518 sessions in 10.5 s.**

The machinery works end to end. The *risk rule* has a gap, and the first honest run found it.

| | |
|---|---|
| Day 914 | equity peaks at **$15,942**, then draws down **18.97%** to $12,917.69 |
| Governor | fires `flatten` (drawdown > 15%) → account goes fully to cash |
| Days 915–2518 | equity moves on **0 of 1,603** days. Governor returns to full gross on **0 of 1,604** |
| Reported result | "+29.18% total, +2.60%/yr" — earned entirely before day 914, then frozen for **6.4 years** |

### The mechanism

Drawdown is measured from the high-water mark. Once the account is flat, equity is constant, so the high-water mark is constant, so the drawdown is *permanently* pinned at whatever level triggered the flatten. Recovery requires being invested; the governor forbids being invested. **It is mathematically absorbing.**

**This is not a code defect.** `risk.drawdown_governor` implements exactly the decided rule — halve at −10%, flatten at −15% — and its tests pin those boundaries correctly. The *specification* names an exit and no re-entry, and a one-way rule applied to a mean-reverting quantity is a trapdoor.

### Consequences for every other number in that run

- **Turnover 831.70%/yr fails the 400%/yr preregistered cap** — and is itself distorted, since all trading happened in the first 36% of the calendar while the denominator spans the whole of it.
- **MaxDD 18.97% breaches the 15% flatten threshold** (within the stress suite's halt+5pts margin, but the governor demonstrably did not hold the line it exists to hold — it fires on `t-1` information, so it always concedes a day).
- **Sharpe 0.42, IR −0.771** are computed over a series that is flat for two-thirds of its length. They describe a mothballed account, not a strategy.

### The decision this needs (Michael's, not the system's)

A re-entry rule. The plausible shapes, and what each costs:

1. **Reset the high-water mark on flatten** — the account restarts from its flattened equity. Simple, and the drawdown clock genuinely resets. Cost: a slow bleed can never trigger a second halt, because the mark keeps following the losses down.
2. **Decay the high-water mark toward current equity** over a stated horizon (standard fund practice). Preserves protection against a slow bleed; adds one parameter and a decay shape to choose.
3. **Time-boxed flatten** — flat for N sessions, then resume at `halve_scale`. Easiest to reason about and to test; the re-entry is arbitrary rather than evidence-driven.

Until one is chosen the account cannot be run forward for real: `portfolio step` would flatten once and then no-op indefinitely, reporting a stable equity that is stable only because nothing is happening.

**Do not tune anything else against this run.** Its metrics are an artifact of the trapdoor, not a measurement of the strategy.

---

## 5e. Findings from the first real replays (2026-07-31 evening)

### The three risk controls that were silently disabled

Each passed its unit tests. Each was found only by running ten years of real prices.

| Control | Failure | Found by |
|---|---|---|
| Drawdown governor | Fired once, never un-fired — absorbing state | Replay: 1,604 dead sessions |
| Volatility targeting | Bound on 1.5% of days; **1.000 through all of COVID** | Replay: `vol_scale` inspection |
| Live-path governor | Reconstructed HWM always == opening deposit | Agent reasoning about `replay()` vs `mark()` |

All three fixed. Replay after: $12,917 → **$22,072**, Sharpe 0.42 → 0.78, dead sessions 1,604 → 64.

### Raising the risk budget does NOT raise returns (measured)

| Risk budget | CAGR | Realized vol | Sharpe |
|---|---|---|---|
| 12% vol (shipped) | **8.25%** | 10.92% | **0.78** |
| 20% vol | 6.50% | 10.83% | 0.64 |
| 30% vol | 5.36% | 10.91% | 0.53 |
| 45% vol + 35% pos + γ=1 | 7.43% | 13.58% | 0.60 |

`vol_target_scale` is capped at 1.0 — it can only *cut* exposure, never lever up — so raising the target merely makes it bind less. **The binding constraint is signal supply, not risk appetite**: 38% of sessions have zero live views, average gross 73.9%. This is a breadth problem (Grinold: IR ≈ IC·√breadth) and a risk dial cannot fix it.

### The edge does not decay at 20 days — it compounds

Raw forward returns on 6,229 discovery signals:

| Hold | Mean | Hit rate |
|---|---|---|
| 5d | +0.35% | 55.0% |
| **20d** *(what ships)* | +1.60% | 57.6% |
| 60d | +6.07% | 61.2% |
| 120d | **+10.86%** | **62.6%** |

Raw, unhedged, not drag-adjusted — roughly half of the 120d figure is market drift. But the hit rate climbing monotonically from 55% → 62.6% is not drift.

### The signal library has only ever been asked one question

**471 cells judged, 24 admitted — all at horizons 1, 5, 20 and nothing else, ever.**

| Family | Cells | Admitted | Samples |
|---|---|---|---|
| `filing_item` | 285 | **0** | 419,255 |
| `insider_transaction` | 174 | 18 | 2,076,463 |
| `insider_cluster_buy` | 12 | 6 | 3,125 |
| `placebo` (control) | 6 | **0** | — |

419k samples of 8-K events (material agreements, earnings, officer changes) produced nothing at 20 days. Whether they produce anything at 60–120 days **has never been tested**. That is the rebuild launched 2026-07-31 21:00 (`--horizons 40,60,90,120`).

**Read the placebo result first.** Market drift grows with holding period, so a long-horizon test can admit noise purely because the numbers got bigger. Placebo currently admits 0/6. If it starts admitting at 90 days, the long-horizon admissions are drift and the correct conclusion is "we measured the wrong thing", not "we found breadth".

Thresholds must stay at their defaults (`min_clusters 200`, `min_t 3.0`, `min_car 0.002`, hedged ≥15bps). Loosening them to admit more cells is the overfitting the whole gate exists to prevent.

---

## 5f. The horizon rebuild — result (2026-07-31, 20:56–22:30)

`signals build-dataset --horizons 40,60,90,120`, 1h33m, **3,159,530 rows inserted, 0 rejected**. 3.89M samples were unmeasurable (a 120-day window needs 120 days of forward prices, so recent events are unscoreable by construction) and 77,920 purged at the 2023-01-01 split. DB 7.2 GB → 8.25 GB. Backup at `.bak-20260731-horizons`.

**`rejected: 0` confirms the constraint analysis**: self-control samples key on `(event, 'self', horizon)`, so horizon 60 cannot collide with horizon 20. The `event_samples` UNIQUE defect bites propagation only.

### The placebo control held — read this before anything else

| horizon | placebo cells | admitted |
|---|---|---|
| 1 / 5 / 20 | 2 each | **0** |
| 40 / 60 / 90 / 120 | 2 each | **0** |

Market drift grows with holding period, so the risk was that long horizons would admit noise purely because the numbers got bigger. They did not. The long-horizon admissions are signal, not drift.

### Admissions: 24 → 94 cells; tradeable 7 → 18

After `filter_tradeable`'s beta-hedged ≥15bps screen:

| cell | hedged edge | hit | n |
|---|---|---|---|
| `insider P / 120d` | **+5.16%** | 58.3% | 4,447 |
| `insider P / 90d` | +4.74% | 56.9% | 4,543 |
| `cluster_buy / 60d` | +4.52% | **64.8%** | 656 |
| `insider P / 20d` *(shipped)* | +1.60% | 54.6% | 4,752 |

`filing_item/5.07` (shareholder votes) clears at 40d and 60d — **the first non-insider family this platform has ever admitted.** The P cells' hedged edge exceeds their raw CAR, i.e. beta-hedging *improved* them.

### But the replay got worse, and the reason is instructive

| | 20d cells + 3 fixes | 18 cells + all fixes |
|---|---|---|
| ending equity | $22,072 | $17,568 |
| CAGR | 8.25% | 5.80% |
| Sharpe | 0.78 | 0.66 |
| **turnover** | 1401% (fails cap) | **361% (passes)** |
| avg gross | 73.9% | 70.1% |
| max drawdown | 17.92% | 17.07% |

**Turnover compliance was the prediction that landed** — long holds don't churn, exactly as the arithmetic `turnover ≈ 2·gross·252/holding_days` said. **Deployment was the prediction that did not**: 73.9% → 70.1%, essentially flat.

The mechanism is the covariance units fix. With Σ correctly scaled by holding period, a 120-day view carries 120× the variance penalty of a one-day view, so the optimizer prices a long hold as the longer risk exposure it actually is and sizes it down. A +5.16% edge earned over 6× the holding time, against 6× the accumulated variance, is not 3.2× better than +1.60% — it is roughly a wash. The old 130× return/risk imbalance was concealing that.

### ⚠️ The comparison above is confounded — do not select on it

Three things changed between those runs: the cell set, the covariance units fix, and the band exit fix. Part of the $22,072 figure was the optimizer over-betting against a risk term 130× too small. **Neither number is a clean read, and neither should be chosen by eye.** Horizon selection is now a legitimate declared dimension of the preregistered sweep; that is where it gets decided, with DSR and PBO, not here.

---

## 6. Resolved contract questions

**Governor × no-trade band — RESOLVED 2026-07-31: the governor bypasses the band.**
When the drawdown governor halves gross at −10%, every target weight moves ~50%. Had the 1% band applied afterward, positions falling inside it would refuse to shrink, leaving the account more exposed than the governor intended precisely during a drawdown. Michael's call: de-risking always executes in full; the band continues to damp ordinary rebalance churn only. Encoded as `PortfolioConfig.governor_bypasses_band` and `optimizer.apply_no_trade_band(..., bypass=...)`. D1's crash-panel test must assert the governor's target gross is actually reached, not merely requested.

**bl_engine — RESOLVED 2026-07-31 by measurement: `pypfopt`.** See §1b.

---

## 7. Supervised (not delegated) actions

These touch the live system or the 7.2 GB DB and stay with Michael + orchestrator:

1. **Pausing `ai.quant.graph-watchdog`** during the propagation build and the Sunday checkpoint.
2. **Running `signals build-propagation`** — 7.2 GB DB, prior 3 h 40 m hang, bounded write windows.
3. **Any `--write` run** against the account of record.
4. **The sealed holdout read** — once, ever.
