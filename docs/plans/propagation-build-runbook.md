# Propagation Build — Supervised Runbook

**This procedure is not delegated to an agent.** It writes to a 7.2 GB single-writer DuckDB that a launchd agent contends for every 600 seconds, and the previous attempt hung for 3 h 40 m. Michael runs it, or the orchestrator runs it with Michael present.

**Prerequisite:** Workstream P fixes 1–3 landed and their tests green. Without the dedup fix this run dies on a duplicate-key collision; without the batch-scoped upsert it degrades into a full-table scan per call; without the point-in-time edge filter its results are contaminated by edge-dimension lookahead and would have to be thrown away anyway.

---

## ⛔ Step 0 — REBUILD THE `event_samples` CONSTRAINT FIRST

**Verified on the live database 2026-07-31. The build will abort again without this, and no code change reaches it.**

```
live table:  UNIQUE(event_id, edge_id, horizon_days)                   ← 2,498,843 rows
code expects: UNIQUE(event_id, edge_id, horizon_days, target_ticker)
```

### Why this happens

`event_samples` was created by commit `5672094` (2026-07-22) with the narrow constraint. Commit `b89a156`, **later the same day**, widened it to include `target_ticker`. But the schema is applied with `CREATE TABLE IF NOT EXISTS`, and `database.migrate_db()` only ever issues `ALTER TABLE ... ADD COLUMN` — neither can rewrite a constraint on an existing table. The live table was created in that same-day window and has carried the narrow constraint ever since.

### Why it is fatal to propagation specifically

Self-control samples write one row per event, so `(event_id, edge_id, horizon_days)` is naturally unique for them and the narrow constraint never bites. **Propagation fans out**: one event at a source company produces one row per *target* it propagates to, all sharing `event_id`, `edge_id` (which holds the relationship type, not a per-edge id) and `horizon_days`, differing only in `target_ticker`. Against the narrow constraint every row after the first is a duplicate-key abort.

This is almost certainly what killed the 6am run on 2026-07-29. Note the consequence for Workstream P: **fix 1's in-Python dedup is defensive only** — it cannot help here, because these rows are legitimately distinct and de-duplicating them would silently discard real samples.

### The rebuild

This is a supervised, backed-up operation on a 7.2 GB file. **Back up first** — the DB is the accumulated output of every collector run to date:

```bash
cp /Volumes/Extreme/home-migrated/quant/data/database/market_intelligence.duckdb \
   /Volumes/Extreme/home-migrated/quant/data/database/market_intelligence.duckdb.bak
```

Then, with every `ai.quant.*` agent unloaded (Step 1 below) and no other process holding the DB, rebuild the table with the correct constraint and copy the rows across. Confirm afterwards:

```sql
SELECT constraint_text FROM duckdb_constraints() WHERE table_name = 'event_samples';
SELECT count(*) FROM event_samples;   -- expect 2,498,843 preserved
```

Do not skip the row-count check. A rebuild that silently drops rows on the new constraint would destroy self-control samples the promotion ladder depends on — if the count comes back lower, **stop and restore the backup**, because that means real rows collided and the correct constraint is not yet understood.

**Consider instead:** if the rebuild looks risky against a 7.2 GB file late at night, propagation evaluation is on the cut list and may slip to Monday without blocking anything else (see the end of this document). Rebuilding a 2.5M-row table that every other pipeline reads is not a thing to rush.

---

## Why this is the long pole

`build_propagation` pairs every commercial edge's source events against the target's forward returns, writing samples into `event_samples`. Two properties make it slow and lock-hostile at once: it is O(edges × events × horizons), and it writes into the same table the daily pipeline reads. Everything else this weekend can proceed while it grinds — but nothing downstream of the *gate decision* can.

---

## Step 1 — Stop the lock contenders

`ai.quant.graph-watchdog` fires every 600 s and holds the write lock roughly 8 minutes in 10. Running the build against that guarantees retry storms.

```bash
launchctl unload ~/Library/LaunchAgents/ai.quant.graph-watchdog.plist
```

Confirm nothing else is mid-run before starting:

```bash
launchctl list | grep ai.quant
```

A `-` in the PID column means not currently running. If `ai.quant.autopilot` shows a live PID, **wait** — the 06:00 daily run must finish first. Do not race it.

---

## Step 2 — Bound the scope before launching

Full-universe propagation is not the goal and will not finish. Restrict to:

- **Commercial edges only** — customer / supplier / competitor / partner (~3–6k of the 24,485 total). Board interlocks are excluded: a shared director is a channel, not a measured relationship.
- **Event families of the 7 tradeable cells** — all long insider-purchase. Sale cells are alpha artifacts that invert under hedging; propagating them would spend hours measuring a known artifact.
- **Horizons (1, 5, 20)** only.

---

## Step 3 — Launch, logged, in the background

```bash
uv run market-intelligence signals build-propagation 2>&1 | tee -a logs/propagation_build.log
```

Check `--help` first for the flags that bound event type, subtype, and horizon — pass them explicitly rather than relying on defaults.

### How to tell "working" from "hung" — the log will be silent either way

Learned the hard way on the 2026-07-31 horizon rebuild: these builders emit **nothing** to the log for their entire run, so log silence carries no information at all. Three signals do:

```bash
P=$(pgrep -f "build-dataset" | head -1); C=$(pgrep -P "$P" | head -1)

# 1. CPU duty cycle -- the decisive one. Sample twice, 3s apart.
ps -o pid,stat,%cpu,time -p "$C"
```
`STAT=RN` with CPU time advancing ~1s per wall second is **healthy batch work**. The 3h40m hang was the opposite: a process *waiting*, burning no CPU. Compute-bound and hung look nothing alike once you measure it.

```bash
# 2. Database file size -- the only real progress metric while the lock is held.
ls -lh <db>.duckdb <db>.duckdb.wal
```
A read-only connection **will fail with `IOException`** while the writer holds the lock; that is expected, not a fault. File growth is the substitute: the horizon rebuild added ~400 MB in its first 30 minutes.

```bash
# 3. RSS should SAWTOOTH, not climb monotonically.
```
Memory rising then dropping (e.g. 950 MB → 1.95 GB → 1.06 GB) is batch accumulation followed by a flush — healthy. A monotonic climb is a leak; kill it.

**Kill on:** CPU near zero for minutes (hang), RSS climbing without ever dropping, or any `traceback`/`constraint`/`duplicate` in the log.

**`ram_guard` will not reap this run**, and it is worth knowing why rather than worrying: it only considers marker groups with `len(procs) > 1`, and only acts below `DEFAULT_ACT_BELOW_FREE_PCT = 20.0` free memory. A single legitimate build is never a candidate no matter how large it grows, and `market-intelligence` is in `_DB_TOUCHING_MARKERS`, so even a genuine duplicate is checked against `lsof` before being killed.

Killing this is cheap and safe. The build flushes per target, so an interrupted run keeps what it finished.

---

## Step 4 — Evaluate, then read the gate

```bash
uv run market-intelligence signals evaluate
```

A propagation cell may feed return views **only** if it clears both bars:

1. **ADMITTED** via `impact.judge` — ≥ 200 clusters, |t| ≥ 3, |mean_car| ≥ 20 bps, hit-rate edge ≥ 2%
2. **Tradeable** — passes `filter_tradeable`'s beta-hedged ≥ 15 bps screen on discovery

Competitor edges are risk-structure only regardless of the outcome: no signed return semantics have ever been measured for them, so their direction is unknown.

---

## Step 5 — Record the verdict honestly

**If zero cells pass:** `graph_return_views_enabled` stays `false`, and the report reads *"propagation gate: 0 admitted."* This is a legitimate, publishable result — the machinery shipped and the data declined. No coefficient is ever hand-waived in to make the graph look useful.

**If cells pass:** flip `portfolio.graph_return_views_enabled` to `true` in `config/settings.yaml` and record which cells qualified, with their statistics, in the plan document.

Either way the graph still earns its place in the **covariance** independently, via `risk.linked_pair_correlation_study` — that measurement needs no propagation coefficient at all, which is exactly why the build plan put risk-structure and return-views on separate tracks.

---

## Step 6 — Restore the watchdog (do not skip)

```bash
launchctl load ~/Library/LaunchAgents/ai.quant.graph-watchdog.plist
launchctl list | grep ai.quant
```

The daily loop is degraded until this is back. If the build is still running when you stop for the night, kill the build and restore the watchdog rather than leaving it unloaded overnight.

---

## If this slips to Monday

It is allowed to. Per the cut list, propagation evaluation slipping does **not** block the core weekend deliverable: Workstream P's code fixes land regardless, and the gate simply reports "not yet measured." The portfolio brain ships with `graph_return_views_enabled=false`, which is its default and its honest state until a measurement says otherwise.
