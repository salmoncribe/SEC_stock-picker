# Autopilot: a self-checking daily loop, then a connection engine

Status: approved 2026-07-22. Phase 1 is the current build.

The manual work of this session — pull new data, validate it, build samples,
run the validation gate — becomes a scheduled loop that runs itself and hands a
briefing to a human. The design turns on one fact about the domain and two
hazards that fact creates.

## The fact that shapes everything: maturation lag

An event that happens today cannot be scored today. Labelling it at the 20-day
horizon needs 20 trading days of *future* returns that do not exist yet. So the
data has a maturation lag, and the loop runs at **two speeds**:

- **Fast loop (daily):** pull new prices and filings, validate, compute
  returns, emit events, and fire alerts on today's events using the
  *already-proven* gate.
- **Slow loop (as windows close):** events whose forward window just matured
  become measurable; rebuild their samples, re-run the gate, and update which
  cells are trusted.

Acting uses yesterday's proven model (instant, no waiting); re-proving happens
in the background as reality fills in the windows. An alert never blocks on
validation, and an unproven cell never fires an alert.

## Hazard A — repeated testing, and the promotion ladder

Re-running the gate on a schedule is sequential hypothesis testing: run it
enough times and noise clears any fixed threshold by luck. Two mechanisms
contain it.

1. **Frozen discovery.** The discovery/holdout boundary is fixed (2023-01-01).
   Admission is decided *once* per cell against frozen discovery data;
   re-running never mints new admissions from the growing pool. New data only
   ever grows the **holdout**, and surviving holdout repeatedly is the
   out-of-sample confirmation we want.

2. **The promotion ladder.** A cell must confirm on holdout for **K consecutive
   evaluations** (default K=2) before it is trusted to fire alerts. A cell that
   flickers admitted-on-noise never earns a streak, so it never earns trust.

   | Status | Meaning | Fires alerts? |
   |--------|---------|---------------|
   | `candidate` | cleared frozen discovery, unconfirmed | no |
   | `active`    | confirmed on holdout K times running | **yes** |
   | `dormant`   | was active; holdout weakened below the floor | no, watched |
   | `retired`   | holdout reversed sign, or dead N runs | no, archived |

   The ladder is a pure function of `(current status, streaks, today's gate
   verdict)`. `judge()` already returns ADMITTED / DEMOTED / REJECTED /
   INSUFFICIENT; the ladder maps that verdict onto a status transition:
   ADMITTED confirms (→ active at streak K), a sign reversal retires, a
   collapse below the economic floor makes an active cell dormant, INSUFFICIENT
   holds.

## Hazard B — a small local LLM will get some links wrong (phase 2)

The connection engine's LLM only *proposes* edges; the event-study harness is
the *referee*. A hallucinated edge becomes a propagation hypothesis that fails
validation and never fires an alert. So an imperfect model is safe: it need
only generate testable hypotheses, each carrying the quoted source sentence for
provenance. "Training" is a later optimisation — once the pipeline has produced
validated edges, those become labels to distill a smaller, faster model from.

## Phase 1 — the self-checking loop (this build)

One launchd agent (`ai.quant.autopilot`, daily ~06:00) runs one Python entry
point, `market-intelligence autopilot run`, that sequences existing collectors
and adds the promotion ladder and the briefing. Not a shell script: a
production loop needs one run-record for the whole loop, structured logging,
and partial-failure handling (a later step failing must not discard an earlier
step's work, and must still produce a briefing that says what failed).

Daily DAG:

1. `market sync-prices` (incremental) → `market compute-returns` (incremental).
2. Refresh SEC submissions for the universe → detect new filings →
   `sec ingest-documents` (new only) → `sec sync-filing-events` (new only) →
   insider refresh.
3. **Mature:** build samples for events whose forward window has now closed.
4. **Re-gate + ladder:** run the gate; advance each cell's status.
5. **Diff:** compare today's statuses against the last run's snapshot.
6. **Brief:** render an Obsidian dated note (permanent record + graph home) and
   push a short Telegram nudge. A failed run still notifies — silence must
   never look like success.

New components:

- `signals/promotion.py` — the ladder (pure logic) + a `signal_status` table.
- `autopilot/types.py` — the `Briefing` contract both projections consume.
- `autopilot/briefing.py` — the day-over-day diff that produces a `Briefing`.
- `autopilot/orchestrator.py` — the DAG runner with partial-failure handling.
- `autopilot/obsidian.py` — `Briefing` → dated markdown note in the vault.
- `autopilot/notify.py` — `Briefing` → Telegram message via the Bot API.
- `config/launchd/ai.quant.autopilot.plist` — the schedule.

Secrets (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) live in the git-ignored
`.env`, never the repo. The Obsidian vault path is configuration.

Honest early state: the only active self-edge signal today is insider buying,
so early briefings are mostly "insider purchase at X" alerts plus
validation-status changes. That is real output, not a placeholder; it enriches
as the connection engine lands.

## Phase 2 — the connection engine (later)

A local LLM (llama.cpp; Qwen2.5-14B-Instruct quantized as the starting model)
reads new 10-K/10-Q sections and proposes typed edges — `buys_from`,
`sells_to`, `competitor`, `partner` — each with a quoted source sentence,
confidence, and filing provenance. Each edge is a propagation hypothesis the
harness measures. Validated edges fire alerts and grow the graph; unvalidated
edges die. Edges are deduplicated bitemporally: an edge asserted in four
consecutive 10-Ks is one edge with a `valid_from`, not four rows.

## Phase 3 — Obsidian projection (later)

The vault is a regenerable projection of the database: one note per company
(its validated edges link to other company notes, lighting up the graph view),
one per active signal, plus the daily briefing. The database is truth; Obsidian
is the view, rebuildable from scratch whenever extraction is re-run.

## What the automation never does

It stops at generating a note or an alert. It never places a trade, moves
money, or sizes a position. Money moves only by a human hand.
