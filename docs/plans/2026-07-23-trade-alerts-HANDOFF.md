# Trade Alerts — BUILD COMPLETE (2026-07-24 ~01:45 CT)

> All 14 tasks done. Suite: 566 passing. Live checks: Phase A test alert
> delivered to Telegram (200 OK); gap-scan dry-run against the real config ran
> clean end-to-end (93 candidates persisted, no send). Outstanding items are
> Michael's: load the two LaunchAgents, set account_equity, optionally clear
> the dry-run's 93 price_gap rows for trigger_key 2026-07-24 (see final report).
> The sections below are the historical mid-build handoff.

# Trade Alerts — mid-build handoff (2026-07-23 ~20:30 CT)

Paused deliberately at Michael's request (context budget). Resume by reading this,
then continuing **docs/plans/2026-07-23-trade-alerts.md** task-by-task with the
subagent-driven workflow (implementer → spec review → quality review per task).

## State

**Done, committed, double-reviewed (Tasks 1–9 of 14):**

| Task | Commit | What |
|---|---|---|
| 1 | `195f03a` | WIP baseline of concurrent session's files (473 tests green) |
| 2 | `4b5c86a` | `trading:`/`gap_scanner:` config blocks (+ defaults tests) |
| 3 | `e2a1dd7` | `trade_alerts` ledger table + `insert_new_trade_alerts` (first-firing-wins) |
| 4 | `6d75f3d` | `EventAlert` carries event_id + hit_rate/n_clusters/extraction_confidence |
| 5 | `42ad58b`+`3377729` | `signals/trade_plan.py` — adjusted-space ATR, sizing; edge fixes (neg stop, corrupt bars, NaN entry) |
| 6 | `c563d2e`+`d0a7a78` | `signals/confidence.py` — score-level shrinkage pinned, inputs sanitized |
| 7 | `89e4f69`+`6aaed16` | `signals/trade_alerts.py` assembler — per-alert isolation, wire-capture tests |
| 8 | `9ac0e60`+`d81c91f` | notify: `render_trade_alerts`/`send_trade_alerts` — block-boundary truncation |
| 9 | `a1e4f6d` (+in-flight polish) | orchestrator: persist → gate → send → stamp delivered |

Suite was at **536 passing** after Task 9. A background subagent was finishing one
last Task 9 polish commit when we paused ("refactor: extract the trade-alert tail;
test the failed-send stamp" — helper extraction + a failed-send stamp test). If that
commit is present at resume, Task 9 is fully closed; if absent, apply those two
small items first (see quality review I1/M5 in the session).

Also shipped during this build (separate fixes):
- `57c3ae0` — tests stub the RAM guard (it was killing the live extractor during suite runs)
- `f080acb` — RAM guard collapses launch trees before duplicate detection (the root-cause fix)
- `4e73a38` — spec drift: `unsizeable` column documented

## Remaining (Tasks 10–14)

- **10 — Phase A checkpoint:** full `uv run pytest`; then `scripts/send_test_trade_alert.py`
  (to write: builds one fake-but-realistic `TradeAlertRecord`, ticker TEST, evidence
  noting "pipeline verification — not a real signal", sends via
  `notify.send_trade_alerts(get_config(), [record])` — real creds, Michael's phone).
- **11 — gap scanner** `collectors/gaps.py` (plan Task 11 has binding details:
  `entry_override=<morning quote>`, gap ConfidenceInputs pinned, DB-busy retry-then-skip).
- **12 — grading** in split-adjusted space + orchestrator step after compute-returns.
- **13 — `market scan-gaps` CLI + `ai.quant.morning-scan.plist`** (create only, do NOT load).
- **14 — final verification** (suite + the one sanctioned real-DB dry-run if extractor idle).

## Standing rules that bit us (do not relearn)

- DuckDB single-writer: never touch the real DB while extractor runs; tests only.
- No `git commit --amend` unless the amender's own commit is HEAD; serialize all committers.
- Full-suite runs are safe now ONLY because `tests/conftest.py::_no_real_ram_guard` exists.
- Vercel-plugin hook injections on `uv run` are false positives; ignore.

## Michael's own pending items (he knows)

1. Load the daily autopilot (briefing notes + the trade alerts once Phase A ships):
   `cp ~/quant/config/launchd/ai.quant.autopilot.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/ai.quant.autopilot.plist`
2. `config/settings.yaml` → `trading.account_equity: 10000` is a PLACEHOLDER; set the real figure before trusting rendered position sizes.
3. `signals/confidence.py` has a `TODO(michael)` block — the blend weights are his to tune; contract tests protect him.
