# Trade Alerts — Gap Detection, Trade Plans, Telegram — Design

**Date:** 2026-07-23
**Status:** Approved
**Scope:** Turn validated signals and overnight price gaps into actionable Telegram trade
alerts: each alert carries a full trade plan (entry reference, stop loss, position size,
exit), the evidence that produced it, and a confidence score. Every alert is persisted and
later graded so the alerting layer builds its own track record.
**Non-goal:** No order execution, no brokerage integration, no auto-trading. The alert ends
with the plan; the human decides. All risk parameters (account equity, risk per trade,
multipliers) are user-owned configuration, never inferred.

---

## 1. Problem

The signal graph produces validated predictions and the autopilot texts a daily briefing,
but the alert line stops at `TICKER ↑ +CAR / Nd — basis`. Missing:

1. **No price-gap detection.** A stock opening far from prior close — often the visible
   footprint of the very events the platform ingests — is never noticed, so the "why did
   it gap" question the data could answer is never asked.
2. **No trade plan.** An alert names a direction and horizon but not a stop, a size, or an
   exit, so acting on one requires redoing the risk math by hand every time.
3. **No proof or confidence in the message.** The evidence (filing quotes in
   `edge_observations`, measured hit rates in `impact_stats`) exists in the DB but never
   reaches the phone.
4. **No alert ledger.** Fired alerts are not persisted as discrete predictions, so the
   alerting layer — unlike the signal layer with its discovery/holdout gate — has no
   track record of its own.

## 2. Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| A | Gap definition | **Both reaction-lag and overnight price gaps** | Reaction-lag gaps (event at A, linked B not yet repriced) are what the signal graph already finds. Price gaps (open vs prior close) are a new scanner whose candidates are explained by joining the event tables. Both feed one downstream pipeline. |
| B | Risk model | **Volatility-based: k×ATR stop, fixed-fractional sizing** | Stop distance adapts to each stock's volatility; sizing off stop distance makes every stop-out cost the same configured fraction of equity. Parameters (`atr_stop_multiple`, `risk_pct_per_trade`) are config, not code. |
| C | Cadence | **Morning scan (short-lived process) + existing daily loop** | Price gaps are an open-of-day phenomenon; a ~15-min-after-open run catches them while actionable. No daemon: DuckDB is single-writer, so alerting runs as sequenced short-lived steps, never a long-lived watcher holding a lock. |
| D | Build shape | **Extend the pipeline; no new dependencies** | ATR and gap math are a few lines over `daily_prices` already in DuckDB. External scanners cannot see the proprietary event/edge data, which is the entire edge. |
| E | Confidence score | **Derived from measured stats; blend authored by the user; components always shown** | Inputs: holdout hit rate + sample size (`impact_stats`), edge `confirmation_count`/`strength` (`graph_edges`), `extraction_confidence`. The weighting is a trader's judgment call, so Michael writes the blend function against a scaffolded interface. The message renders the components alongside the score so it stays auditable. A `min_confidence` gate keeps low-quality alerts off the phone; gated alerts are still persisted and graded so the ledger can verify the score discriminates. |
| F | Alert ledger | **Every alert persisted to `trade_alerts`, graded on later runs** | Each alert is a discrete falsifiable prediction (hit stop / hit target / expired at horizon). Grading closes the loop the same way the promotion ladder does for signals: the alerting layer accumulates its own honest track record. |
| G | Execution | **None, ever** | The system renders a plan; it never places, modifies, or closes a position. This preserves the signal-graph spec's non-goal and keeps the human as the only actor. |

## 3. Architecture

Two trigger paths converge on one alert pipeline:

```
 DAILY LOOP (existing autopilot)          MORNING SCAN (new, ~15 min after open)
   validated signal fires                   gap candidates: |open − prior_close| ≥ threshold
   (event at A → prediction for B)          liquidity floor · catalyst join vs events/edges
        │                                        │
        └────────────────┬───────────────────────┘
                         ▼
              signals/trade_plan.py        entry ref · stop (k×ATR) · target · shares · exit date
                         ▼
              signals/confidence.py        measured stats + edge strength + extraction conf → 0–100
                         ▼
              trade_alerts table           the ledger: plan + evidence persisted, graded later
                         ▼
              autopilot/notify.py          plain-text Telegram render (never raises)
```

New modules, following the established pattern:

- `signals/trade_plan.py` — pure: (signal, prices, trading config) → `TradePlan`
- `signals/confidence.py` — pure: (`ConfidenceInputs`) → 0–100 score; blend function user-authored
- `collectors/gaps.py` — morning gap scan via the existing `MarketDataProvider` ABC
- `autopilot/notify.py` — extended with a trade-alert renderer; same never-raise contract
- CLI: `scan-gaps` (morning entry point), grading folded into the existing daily run
- LaunchAgent: `config/launchd/ai.quant.morning-scan.plist`

## 4. Trade plan math

- **Entry reference:** last close (daily path) or the morning quote (gap path). The entry
  is a reference for the plan's math, not an order.
- **ATR(14):** Wilder's smoothing over **split-adjusted** OHLC — each bar's open/high/low
  scaled by that bar's `adj_close / close` factor, so a recent split cannot inflate the
  true range. Insufficient history (< period + 1 bars) → the ticker is skipped with a
  logged reason, never a fabricated stop.
- **Stop:** `entry − k × ATR` for longs, `entry + k × ATR` for shorts
  (`k = atr_stop_multiple`).
- **Target:** daily path `entry × (1 + predicted_car)`; gap path has no model prediction,
  so target = `entry ± gap_target_r_multiple × stop_distance` (R-multiple, config).
- **Size:** `shares = floor((account_equity × risk_pct_per_trade) / stop_distance)`,
  then capped so `shares × entry ≤ max_position_pct × account_equity`. If the cap or floor
  produces 0 shares the alert is still sent, flagged `unsizeable at current risk settings`
  — the information is still useful even when the position isn't.
- **Exit:** earliest of stop hit, target hit, or the time exit (`horizon_days` from the
  signal on the daily path; `gap_max_hold_days` config on the gap path).

## 5. Confidence score

`ConfidenceInputs` per alert:

- `hit_rate`, `n` — holdout track record for the (event_type, edge_type, horizon) cell
  from `impact_stats` (daily path only)
- `confirmation_count`, `strength` — from `graph_edges`
- `extraction_confidence` — from the LLM extraction rows
- gap path substitutes: catalyst present (bool), gap size, liquidity — flagged
  `no track record yet` in the message until graded ledger history accumulates

Small-`n` cells must not masquerade as certainty: the scaffold shrinks `hit_rate` toward
50 as `n` falls, and the blend operates on the shrunk value. The blend function itself
(weighting of the three inputs) is deliberately left to the user; tests pin its contract
(monotonic in each input, bounded 0–100), not its weights.

## 6. Schema

**`trade_alerts`** — natural key `(kind, ticker, trigger_key)`; standard provenance block
and the house natural-key upsert contract (re-firing is a no-op, so nothing alerts twice).
`trigger_key` is defined per path:

- daily path: `"{event_id}:{edge_id}:{horizon_days}"` — one alert per prediction, even
  though the briefing's 4-day event lookback re-surfaces the same event on consecutive
  runs. A null `edge_id` (self-edge cells) renders canonically as `self`, never empty.
- gap path: the gap's trading date (`"2026-07-24"`) — a re-run of `scan-gaps` cannot
  double-text the morning's gaps

Columns: `alert_id` (PK), `fired_at`, `kind` (`reaction_lag|price_gap`), `ticker`,
`direction`, `entry_ref`, `stop`, `target`, `shares`, `notional`, `risk_amount`,
`time_exit_date`, `confidence`, `evidence` (JSON: filing quotes, event ids, cell stats,
gap metrics), `event_id`/`edge_id` (nullable, daily path), `delivered` (bool),
`delivery_note` (nullable: `gated_below_min_confidence|telegram_failed`),
`outcome` (`open|hit_stop|hit_target|expired`), `outcome_return`, `graded_at`,
`unsizeable` (bool — the account couldn't fund even one share at the configured
risk; the alert still fires with the geometry, flagged).

This table **supersedes** the `alerts` table designed (but never built) in the
signal-graph spec §4: same purpose — persist fired predictions and backfill outcomes —
now carrying the trade plan as well. The signal-graph spec's `alerts` table will not be
built separately.

Grading runs inside the existing daily loop: for each `open` alert, walk subsequent
`daily_prices` bars **in entry-day adjusted price space** — the stored raw `stop`/`target`
and each later bar are converted through the `adj_close / close` factor chain, so a split
or large dividend mid-hold never fakes a stop-out or corrupts `outcome_return` (the
raw-price plan stays in the message; adjusted math stays in the grader). Conservative
tie-break — a bar that touches both stop and target grades as `hit_stop` (assume the
worst path through the bar).

## 7. Message format

```
🎯 TRADE SIGNAL — DAL ↑ (confidence 72)
Why: BA 10-K names DAL as >10% customer (filed 2024-02-05);
     BA guidance-cut event 2026-07-21; cell hit 67% (n=41, holdout)
Plan: entry ~$44.20 · stop $41.80 (2×ATR) · target $46.90 (+6.1%)
Size: 18 sh (~$795 → risks 1.0% of account) · exit by 2026-08-20
```

Plain text, no `parse_mode`, rendered by a pure function, truncated under the 4096-char
ceiling — all inherited from the existing notifier contract. Multiple alerts in one run
are batched into one message, highest confidence first. Trade alerts are **their own
message**, separate from the daily briefing nudge: the briefing stays a status glance,
the trade alert is an actionable ping.

## 8. Configuration

```yaml
trading:
  account_equity: 10000        # PLACEHOLDER — set to your real figure
  risk_pct_per_trade: 1.0      # % of equity one stop-out may cost
  atr_period: 14
  atr_stop_multiple: 2.0
  max_position_pct: 20.0       # no single position above this % of equity
  min_confidence: 60           # below this: persisted + graded, but not texted

gap_scanner:
  min_gap_pct: 3.0             # |open − prior close| / prior close threshold
  min_avg_dollar_volume: 5000000   # 20-day average; skip illiquid names
  gap_target_r_multiple: 2.0
  gap_max_hold_days: 5
  gap_catalyst_lookback_days: 7    # events within this window count as the gap's catalyst
  require_catalyst: false      # true → only gaps explained by an event are texted
  min_confidence: 40           # gap alerts cap at 50 (no track record yet); this
                               # separate floor lets them text anyway — raise to
                               # 60+ to silence gaps until the ledger matures
```

Defaults are placeholders to make the pipeline runnable; the user owns every value.

## 9. Scheduling and failure behaviour

- **LaunchAgent** `ai.quant.morning-scan`: weekdays, ~15 minutes after market open
  (08:45 America/Chicago). Short-lived process; exits when done.
- **DB busy** (single-writer lock held by a collector): bounded retry with backoff, then
  skip the scan and log. A missed scan is a missed message, not a broken pipeline.
- **Provider failure** (yfinance error/empty): log and skip; never fabricate prices.
- **Notifier:** existing never-raise contract unchanged. Ledger write happens before the
  send, so a Telegram outage never loses the record (`delivered = false`).
- **Weekends/holidays:** scan exits early when the market calendar says closed (prior
  trading day's bar absent → no-op).

## 10. Testing

- Pure-function unit tests with fixtures: ATR, stop/size/target math (long, short,
  unsizeable, missing history), gap detection (threshold, liquidity, catalyst join),
  confidence contract (bounds, monotonicity, small-n shrinkage), renderer output.
- Telegram through `httpx.MockTransport`, as the suite already does.
- Ledger round-trip and grading tests, including the both-sides-touched conservative case.

## 11. Build order

- **Phase A (today):** `trading:` config block, `trade_plan.py`, `confidence.py` scaffold
  + user-authored blend, `trade_alerts` table, renderer extension, wiring into the daily
  loop, manual end-to-end run → first trade-alert text delivered.
- **Phase B (next):** `collectors/gaps.py`, `scan-gaps` CLI, morning LaunchAgent, alert
  grading in the daily loop, `require_catalyst` refinement after observing live gaps.
