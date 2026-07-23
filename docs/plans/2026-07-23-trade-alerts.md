# Trade Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `docs/specs/2026-07-23-trade-alerts-design.md` — turn validated signals and overnight price gaps into Telegram trade alerts carrying a full trade plan (entry ref, ATR stop, position size, exit), evidence, and a confidence score, persisted to a self-grading `trade_alerts` ledger.

**Architecture:** Extend the existing pipeline (`market_intelligence` package). Two trigger paths — reaction-lag alerts from the daily autopilot loop, and a new morning gap scan — converge on one chain: `signals/trade_plan.py` → `signals/confidence.py` → `trade_alerts` table → `autopilot/notify.py`. Pure functions everywhere possible; DuckDB single-writer respected (short-lived sequenced processes, never a daemon).

**Tech Stack:** Python 3.11 (`uv run`), DuckDB, pydantic config, typer CLI, httpx (+MockTransport in tests), pytest. NO new dependencies.

---

## Ground rules for every task

- Run commands from repo root `/Users/michaeltadlock/quant`. Test with `uv run pytest tests/<file> -q`.
- **Never touch the real database** (`MARKET_INTELLIGENCE_HOME` on the external SSD). A relationship-extraction job is running and DuckDB is single-writer. All tests use `memory_db` / `tmp_config` fixtures (see `tests/conftest.py`). Do not run `market-intelligence autopilot run` or any collector against the real home during this build.
- Commit ONLY the files named in your task (`git add <explicit paths>`). The working tree may contain other sessions' work.
- Match house style: module docstrings explaining *why*, `from __future__ import annotations`, frozen dataclasses for values, `get_logger(__name__)`, no type-annotation churn on existing code.
- Read the spec `docs/specs/2026-07-23-trade-alerts-design.md` before starting any task.

### Codebase facts (verified 2026-07-23 — trust these, don't rediscover)

- `EventAlert` / `Briefing` in `src/market_intelligence/autopilot/types.py` (frozen dataclasses).
- `event_alerts()` in `src/market_intelligence/autopilot/briefing.py` fires only through **self-edge** cells (`SELF_EDGE = "self"`), 4-day lookback; SELECT joins `events e` × `signal_status s`.
- `events` table has: `event_id, ticker, event_type, event_subtype, available_time, payload, extraction_confidence, filing_id, accession_number`.
- `signal_status` has: `hit_rate, n_clusters, mean_car, direction, status`.
- `daily_prices` has: `symbol, price_date, open, high, low, close, adj_close, volume` — natural key `(symbol, price_date)`.
- Company graph table is **`company_edges`** (has `evidence`, `times_asserted`, `extraction_confidence`); the spec's `graph_edges`/`edge_observations` names are the design-level names for this.
- Schema DDL lives in `src/market_intelligence/database.py` (`SCHEMA_STATEMENTS` dict + `TABLES` tuple; `init_db` is CREATE TABLE IF NOT EXISTS).
- Per-table write helpers live in `src/market_intelligence/storage/duckdb.py` (generic `upsert(con, table, rows, key_cols, ...)`).
- Config: pydantic models in `src/market_intelligence/config.py`; `SettingsFile` holds sub-models with full defaults (see `LLMConfig` for the pattern — YAML entry optional).
- Notifier `src/market_intelligence/autopilot/notify.py`: plain text, no parse_mode, `_SAFE_CHARS = 3900` truncation, never raises, `httpx.Client(transport=...)` injectable; tests use `httpx.MockTransport` (see `tests/test_notify.py`).
- Orchestrator `src/market_intelligence/autopilot/orchestrator.py`: `run()` builds briefing inside `database.connection(...)` block then `_deliver(config, briefing)` (best-effort).
- Collectors use `pipeline_run(config, name)` contextmanager from `src/market_intelligence/collectors/__init__.py` yielding `(con, RunSummary)`.
- CLI is typer: sub-apps `market_app`, `signals_app`, `autopilot_app` in `src/market_intelligence/cli.py`.
- Market provider ABC `src/market_intelligence/clients/market.py`: `get_daily_prices`, `get_latest_price`, `get_corporate_actions`; real impl `clients/market_yfinance.py`; `MockMarketDataProvider` for tests.
- `utcnow()` helper: `from market_intelligence.schemas.common import utcnow`.
- LaunchAgent template: `config/launchd/ai.quant.autopilot.plist`.

---

## Task 1: Clean baseline — commit the concurrent session's in-flight work

The tree has uncommitted phase-2b/2c work from another session, including two files this plan also edits (`database.py`, `config/settings.yaml`). Commit it as-is first so later commits are cleanly scoped.

**Files:** the currently dirty/untracked set only.

- [ ] **Step 1:** `git status --porcelain` and confirm the dirty set is (5 modified): `config/settings.yaml`, `scripts/extract_relationships.sh`, `src/market_intelligence/collectors/relationships.py`, `src/market_intelligence/database.py`, `tests/test_relationships.py`; (untracked): `config/launchd/ai.quant.graph-watchdog.plist`, `config/launchd/ai.quant.ramguard.plist`, `scripts/watch_relationship_graph.py`, `tests/test_ram_guard.py`, `docs/specs/.obsidian/`.
- [ ] **Step 2:** Add `docs/specs/.obsidian/` to `.gitignore` (it's the Obsidian app's per-vault config, not source).
- [ ] **Step 3:** `uv run pytest -q` — confirm the suite passes BEFORE committing (record the count). If tests fail, STOP and report; do not commit a broken baseline.
- [ ] **Step 4:** Commit: `git add config/settings.yaml scripts/extract_relationships.sh src/market_intelligence/collectors/relationships.py src/market_intelligence/database.py tests/test_relationships.py config/launchd/ai.quant.graph-watchdog.plist config/launchd/ai.quant.ramguard.plist scripts/watch_relationship_graph.py tests/test_ram_guard.py .gitignore` then `git commit -m "wip: phase-2b in-flight work from concurrent session (pre-trade-alerts baseline)"`.

## Task 2: Config — `trading:` and `gap_scanner:` blocks

**Files:**
- Modify: `src/market_intelligence/config.py` (add models after `LLMConfig`; add fields to `SettingsFile`)
- Modify: `config/settings.yaml` (append blocks)
- Test: `tests/test_config.py` (append tests)

- [ ] **Step 1: Failing tests** — append to `tests/test_config.py`:

```python
class TestTradingConfig:
    def test_defaults_load_without_yaml_entry(self, tmp_config):
        t = tmp_config.settings.trading
        assert t.account_equity > 0
        assert 0 < t.risk_pct_per_trade <= 100
        assert t.atr_period >= 2
        assert t.atr_stop_multiple > 0
        assert 0 < t.max_position_pct <= 100
        assert 0 <= t.min_confidence <= 100

    def test_gap_scanner_defaults(self, tmp_config):
        g = tmp_config.settings.gap_scanner
        assert g.min_gap_pct > 0
        assert g.min_avg_dollar_volume > 0
        assert g.gap_target_r_multiple > 0
        assert g.gap_max_hold_days >= 1
        assert g.gap_catalyst_lookback_days >= 1
        assert g.require_catalyst in (True, False)
```

- [ ] **Step 2:** `uv run pytest tests/test_config.py -q` → FAIL (`AttributeError: trading`).
- [ ] **Step 3: Implement** — in `config.py` after `LLMConfig`:

```python
class TradingConfig(BaseModel):
    """User-owned risk parameters for rendered trade plans.

    Every number here belongs to Michael, not the system: the alert layer only
    does arithmetic with them. ``account_equity`` ships as an obvious
    placeholder so a rendered size is never mistaken for advice about a real
    account until he sets it.
    """

    account_equity: float = Field(default=10_000.0, gt=0.0)
    risk_pct_per_trade: float = Field(default=1.0, gt=0.0, le=100.0)
    atr_period: int = Field(default=14, ge=2)
    atr_stop_multiple: float = Field(default=2.0, gt=0.0)
    max_position_pct: float = Field(default=20.0, gt=0.0, le=100.0)
    min_confidence: int = Field(default=60, ge=0, le=100)


class GapScannerConfig(BaseModel):
    """Morning price-gap scan thresholds (see trade-alerts design §8)."""

    min_gap_pct: float = Field(default=3.0, gt=0.0)
    min_avg_dollar_volume: float = Field(default=5_000_000.0, gt=0.0)
    gap_target_r_multiple: float = Field(default=2.0, gt=0.0)
    gap_max_hold_days: int = Field(default=5, ge=1)
    gap_catalyst_lookback_days: int = Field(default=7, ge=1)
    require_catalyst: bool = False
```

Add to `SettingsFile`: `trading: TradingConfig = TradingConfig()` and `gap_scanner: GapScannerConfig = GapScannerConfig()`.

Append to `config/settings.yaml`:

```yaml
# Trade-alert layer (docs/specs/2026-07-23-trade-alerts-design.md).
# These are Michael's numbers, not the system's. account_equity is a
# PLACEHOLDER — set your real figure before trusting rendered sizes.
trading:
  account_equity: 10000
  risk_pct_per_trade: 1.0
  atr_period: 14
  atr_stop_multiple: 2.0
  max_position_pct: 20.0
  min_confidence: 60

gap_scanner:
  min_gap_pct: 3.0
  min_avg_dollar_volume: 5000000
  gap_target_r_multiple: 2.0
  gap_max_hold_days: 5
  gap_catalyst_lookback_days: 7
  require_catalyst: false
```

- [ ] **Step 4:** `uv run pytest tests/test_config.py -q` → PASS.
- [ ] **Step 5:** Commit: `git add src/market_intelligence/config.py config/settings.yaml tests/test_config.py && git commit -m "feat: trading and gap-scanner config blocks"`

## Task 3: Schema — `trade_alerts` table

**Files:**
- Modify: `src/market_intelligence/database.py` (`TABLES` + `SCHEMA_STATEMENTS`)
- Modify: `src/market_intelligence/storage/duckdb.py` (add `insert_new_trade_alerts`)
- Test: `tests/test_storage.py` (append)

- [ ] **Step 1: Failing tests** — append to `tests/test_storage.py`:

```python
class TestTradeAlerts:
    @staticmethod
    def _row(**overrides):
        row = {
            "alert_id": "a1", "kind": "reaction_lag", "ticker": "NKE",
            "trigger_key": "ev1:self:20", "fired_at": None, "direction": 1,
            "entry_ref": 74.2, "stop": 71.1, "target": 76.4, "shares": 13,
            "notional": 964.6, "risk_amount": 100.0, "time_exit_date": None,
            "confidence": 72, "evidence": "{}", "event_id": "ev1",
            "edge_id": "self", "delivered": False, "delivery_note": None,
            "outcome": "open", "outcome_return": None, "graded_at": None,
            "unsizeable": False, "schema_version": "1.0.0",
        }
        row.update(overrides)
        return row

    def test_insert_new_inserts_and_skips_existing(self, memory_db):
        from market_intelligence.storage import duckdb as duckdb_store
        first = duckdb_store.insert_new_trade_alerts(memory_db, [self._row()])
        assert len(first) == 1
        again = duckdb_store.insert_new_trade_alerts(
            memory_db, [self._row(alert_id="a2", confidence=99)]
        )
        assert again == []  # same (kind, ticker, trigger_key): re-fire is a no-op
        n = memory_db.execute("SELECT count(*) FROM trade_alerts").fetchone()[0]
        assert n == 1
        kept = memory_db.execute("SELECT confidence FROM trade_alerts").fetchone()[0]
        assert kept == 72  # first firing wins; nothing overwritten
```

- [ ] **Step 2:** Run → FAIL (table/function missing).
- [ ] **Step 3: Implement.** In `database.py`, add `"trade_alerts"` to `TABLES` and this statement to `SCHEMA_STATEMENTS` (with a comment block explaining the ledger, dedup key, and that it supersedes the signal-graph spec's unbuilt `alerts` table):

```sql
CREATE TABLE IF NOT EXISTS trade_alerts (
    alert_id          TEXT PRIMARY KEY,
    kind              TEXT NOT NULL,
    ticker            TEXT NOT NULL,
    trigger_key       TEXT NOT NULL,
    fired_at          TIMESTAMPTZ,
    direction         INTEGER,
    entry_ref         DOUBLE,
    stop              DOUBLE,
    target            DOUBLE,
    shares            INTEGER,
    notional          DOUBLE,
    risk_amount       DOUBLE,
    time_exit_date    DATE,
    confidence        INTEGER,
    evidence          TEXT,
    event_id          TEXT,
    edge_id           TEXT,
    delivered         BOOLEAN,
    delivery_note     TEXT,
    outcome           TEXT,
    outcome_return    DOUBLE,
    graded_at         TIMESTAMPTZ,
    unsizeable        BOOLEAN,
    source            TEXT,
    source_url        TEXT,
    content_hash      TEXT,
    schema_version    TEXT,
    validation_status TEXT,
    validation_errors TEXT,
    collected_time    TIMESTAMPTZ,
    UNIQUE (kind, ticker, trigger_key)
)
```

In `storage/duckdb.py` add (docstring: first-firing-wins insert-only semantics; the generic `upsert` would overwrite the original plan on re-fire, which is exactly what the ledger must never do):

```python
TRADE_ALERT_KEY = ("kind", "ticker", "trigger_key")

def insert_new_trade_alerts(
    con: duckdb.DuckDBPyConnection, rows: Iterable[Row]
) -> list[Row]:
    """Insert rows whose (kind, ticker, trigger_key) is unseen; return them."""
    materialized = [dict(r) for r in rows]
    if not materialized:
        return []
    materialized = _dedupe_last_wins(materialized, TRADE_ALERT_KEY)
    existing = _existing_keys(con, "trade_alerts", TRADE_ALERT_KEY)
    fresh = [
        r for r in materialized
        if tuple(r.get(c) for c in TRADE_ALERT_KEY) not in existing
    ]
    if fresh:
        columns = [r[1] for r in con.execute('PRAGMA table_info("trade_alerts")').fetchall()]
        _insert(con, "trade_alerts", fresh, [c for c in columns])
    return fresh
```

(Insert must tolerate missing keys in the row dict — `_insert` already uses `row.get(col)`.)

- [ ] **Step 4:** `uv run pytest tests/test_storage.py -q` → PASS.
- [ ] **Step 5:** Commit: `git add src/market_intelligence/database.py src/market_intelligence/storage/duckdb.py tests/test_storage.py && git commit -m "feat: trade_alerts ledger table with first-firing-wins inserts"`

## Task 4: Carry `event_id` + cell stats on `EventAlert`

**Files:**
- Modify: `src/market_intelligence/autopilot/types.py` (extend `EventAlert`)
- Modify: `src/market_intelligence/autopilot/briefing.py` (extend the SELECT)
- Test: `tests/test_autopilot.py` (extend the `event_alerts` test)

- [ ] **Step 1: Failing test** — in `tests/test_autopilot.py` find the existing test that seeds `events` + `signal_status` and calls `briefing.event_alerts` (or add one modeled on it, using `memory_db`); assert the returned alert now has `event_id == <seeded id>`, `hit_rate` matching the seeded cell, `n_clusters` matching, `extraction_confidence` matching the seeded event.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** In `types.py` append defaulted fields to `EventAlert` (defaults keep every existing constructor call working):

```python
    event_id: str = ""
    hit_rate: float | None = None
    n_clusters: int = 0
    extraction_confidence: float | None = None
```

In `briefing.py::event_alerts` add `e.event_id, s.n_clusters, e.extraction_confidence` to the SELECT list and populate the new fields in the `EventAlert(...)` construction.

- [ ] **Step 4:** `uv run pytest tests/test_autopilot.py tests/test_notify.py tests/test_obsidian.py -q` → PASS (renderers must be unaffected).
- [ ] **Step 5:** Commit: `git add src/market_intelligence/autopilot/types.py src/market_intelligence/autopilot/briefing.py tests/test_autopilot.py && git commit -m "feat: event alerts carry event_id and cell stats for the trade-alert layer"`

## Task 5: `signals/trade_plan.py` — ATR, stop, size, target, exit (pure)

**Files:**
- Create: `src/market_intelligence/signals/trade_plan.py`
- Test: `tests/test_trade_plan.py`

- [ ] **Step 1: Failing tests** — create `tests/test_trade_plan.py`:

```python
"""Trade-plan math: pure functions over bars. No I/O, no network."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from market_intelligence.signals.trade_plan import Bar, TradePlan, build_trade_plan, wilder_atr


def _bars(n=30, close=100.0, spread=2.0, start=date(2026, 6, 1)):
    """n flat-ish bars with a known true range of `spread` each day."""
    out = []
    for i in range(n):
        d = start + timedelta(days=i)
        out.append(Bar(date=d, open=close, high=close + spread / 2,
                       low=close - spread / 2, close=close, adj_close=close))
    return out


class TestWilderAtr:
    def test_constant_range_converges_to_that_range(self):
        atr = wilder_atr(_bars(40, spread=2.0), period=14)
        assert atr == pytest.approx(2.0, rel=0.05)

    def test_insufficient_history_returns_none(self):
        assert wilder_atr(_bars(10), period=14) is None

    def test_split_adjusted_bars_ignore_raw_split_cliff(self):
        # 20 bars at raw 100 (adj 50 — pre 2:1 split), then 20 at raw 50 (adj 50).
        pre = [Bar(date=date(2026, 5, 1) + timedelta(days=i), open=100, high=101,
                   low=99, close=100, adj_close=50.0) for i in range(20)]
        post = [Bar(date=date(2026, 5, 21) + timedelta(days=i), open=50, high=50.5,
                    low=49.5, close=50, adj_close=50.0) for i in range(20)]
        atr = wilder_atr(pre + post, period=14)
        # In adjusted space the split cliff does not exist: ATR stays ~ the
        # true daily range (~1 pre-split-adjusted / ~1 post), never ~50.
        assert atr < 3.0


class TestBuildTradePlan:
    def _plan(self, direction=1, predicted_move=0.03, equity=10_000.0, **kw):
        return build_trade_plan(
            bars=_bars(40, close=100.0, spread=2.0),
            direction=direction,
            predicted_move=predicted_move,
            horizon_days=20,
            as_of=date(2026, 7, 23),
            account_equity=equity,
            risk_pct_per_trade=1.0,
            atr_period=14,
            atr_stop_multiple=2.0,
            max_position_pct=20.0,
            **kw,
        )

    def test_long_plan_geometry(self):
        plan = self._plan()
        assert plan.entry_ref == pytest.approx(100.0)
        assert plan.stop == pytest.approx(100.0 - 2.0 * 2.0, rel=0.05)   # k*ATR below
        assert plan.target == pytest.approx(103.0)                        # 1 + predicted
        assert plan.time_exit_date == date(2026, 7, 23) + timedelta(days=20)
        # risk-based size would be floor((10_000 * 1%) / 4) = 25 shares, but the
        # 20% position cap (2_000 notional / 100) binds first → 20 shares.
        assert plan.shares == 20
        assert plan.risk_amount == pytest.approx(plan.shares * (plan.entry_ref - plan.stop))
        assert not plan.unsizeable

    def test_short_plan_flips_stop_and_target(self):
        plan = self._plan(direction=-1, predicted_move=-0.03)
        assert plan.stop > plan.entry_ref
        assert plan.target < plan.entry_ref

    def test_position_cap_binds(self):
        # equity 1M: risk budget 10k / 4 = 2500 shares uncapped (250k notional);
        # the 20% cap (200k) binds → exactly floor(200_000 / 100) = 2000 shares.
        plan = self._plan(equity=1_000_000.0)
        assert plan.shares == 2000

    def test_entry_override_anchors_the_plan_at_the_quote(self):
        # The gap path plans off the live morning quote, not the last bar's
        # close — the whole point is that those differ by the gap.
        plan = self._plan(entry_override=105.0)
        assert plan.entry_ref == pytest.approx(105.0)
        assert plan.stop == pytest.approx(105.0 - 4.0, rel=0.05)

    def test_tiny_account_is_unsizeable_but_still_planned(self):
        plan = self._plan(equity=100.0)
        assert plan.shares == 0
        assert plan.unsizeable
        assert plan.stop < plan.entry_ref  # the plan geometry still renders

    def test_insufficient_history_returns_none(self):
        plan = build_trade_plan(
            bars=_bars(5), direction=1, predicted_move=0.02, horizon_days=5,
            as_of=date(2026, 7, 23), account_equity=10_000.0,
            risk_pct_per_trade=1.0, atr_period=14, atr_stop_multiple=2.0,
            max_position_pct=20.0,
        )
        assert plan is None
```

- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3: Implement** `src/market_intelligence/signals/trade_plan.py`:

```python
"""Trade-plan math: from a signal and price history to stop, size, and exits.

Pure functions over in-memory bars so the whole module tests without a
database. Two deliberate choices from the design doc:

* **All range math runs in adjusted price space.** Each bar's open/high/low is
  scaled by its own ``adj_close / close`` factor before the true range is
  computed, so a split or large dividend inside the ATR window cannot inflate
  the stop distance. The most recent bar's factor is ~1, so the result is in
  current-price terms and composes directly with a raw entry reference.
* **A plan that cannot be sized is still a plan.** ``shares == 0`` (account too
  small for the risk budget at this stop distance) sets ``unsizeable`` rather
  than suppressing the alert — the geometry is still information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Bar:
    """One daily OHLC bar; ``adj_close`` carries the split/dividend factor."""

    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float


@dataclass(frozen=True)
class TradePlan:
    entry_ref: float
    stop: float
    target: float
    shares: int
    notional: float
    risk_amount: float
    time_exit_date: date
    atr: float
    unsizeable: bool


def _factor(bar: Bar) -> float:
    if bar.close <= 0:
        return 1.0
    return bar.adj_close / bar.close


def wilder_atr(bars: list[Bar], period: int) -> float | None:
    """ATR over adjusted OHLC with Wilder's smoothing; None if history is thin."""
    if len(bars) < period + 1:
        return None
    ordered = sorted(bars, key=lambda b: b.date)
    highs, lows, closes = [], [], []
    for bar in ordered:
        f = _factor(bar)
        highs.append(bar.high * f)
        lows.append(bar.low * f)
        closes.append(bar.close * f)
    ranges = [
        max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        for i in range(1, len(ordered))
    ]
    atr = sum(ranges[:period]) / period
    for tr in ranges[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def build_trade_plan(
    *,
    bars: list[Bar],
    direction: int,
    predicted_move: float,
    horizon_days: int,
    as_of: date,
    account_equity: float,
    risk_pct_per_trade: float,
    atr_period: int,
    atr_stop_multiple: float,
    max_position_pct: float,
    entry_override: float | None = None,
) -> TradePlan | None:
    """Turn one directional signal into a full plan, or None without history.

    ``entry_override`` anchors the plan at a live quote instead of the last
    bar's close — the gap path passes the morning quote here, because prior
    close and the quote differ by exactly the gap being traded.
    """
    atr = wilder_atr(bars, atr_period)
    if atr is None or atr <= 0 or direction == 0:
        return None
    entry = entry_override if entry_override is not None else (
        sorted(bars, key=lambda b: b.date)[-1].close
    )
    if entry <= 0:
        return None

    stop_distance = atr_stop_multiple * atr
    if direction > 0:
        stop = entry - stop_distance
        target = entry * (1.0 + abs(predicted_move))
    else:
        stop = entry + stop_distance
        target = entry * (1.0 - abs(predicted_move))

    risk_budget = account_equity * (risk_pct_per_trade / 100.0)
    shares = math.floor(risk_budget / stop_distance)
    max_notional = account_equity * (max_position_pct / 100.0)
    if shares * entry > max_notional:
        shares = math.floor(max_notional / entry)
    shares = max(shares, 0)

    return TradePlan(
        entry_ref=entry,
        stop=stop,
        target=target,
        shares=shares,
        notional=shares * entry,
        risk_amount=shares * stop_distance,
        time_exit_date=as_of + timedelta(days=horizon_days),
        atr=atr,
        unsizeable=shares == 0,
    )


__all__ = ["Bar", "TradePlan", "build_trade_plan", "wilder_atr"]
```

- [ ] **Step 4:** `uv run pytest tests/test_trade_plan.py -q` → PASS. If the geometry assertions fail on rounding, fix the TEST tolerance, not the math.
- [ ] **Step 5:** Commit: `git add src/market_intelligence/signals/trade_plan.py tests/test_trade_plan.py && git commit -m "feat: trade-plan math — adjusted-space ATR stops and fixed-fractional sizing"`

## Task 6: `signals/confidence.py` — auditable 0–100 score

**Files:**
- Create: `src/market_intelligence/signals/confidence.py`
- Test: `tests/test_confidence.py`

- [ ] **Step 1: Failing tests** — create `tests/test_confidence.py` pinning the CONTRACT (bounds, monotonicity, shrinkage), not the weights:

```python
"""Confidence contract tests: bounds, monotonicity, small-n humility.

Deliberately weight-agnostic — the blend weights are Michael's to tune; these
tests only pin properties any sane blend must satisfy.
"""

from __future__ import annotations

from market_intelligence.signals.confidence import ConfidenceInputs, score, shrunk_hit_rate


def _inputs(**kw):
    base = dict(hit_rate=0.65, n_clusters=40, times_asserted=0,
                extraction_confidence=0.9, has_track_record=True)
    base.update(kw)
    return ConfidenceInputs(**base)


class TestShrinkage:
    def test_small_n_pulls_toward_half(self):
        assert abs(shrunk_hit_rate(0.9, 2) - 0.5) < abs(shrunk_hit_rate(0.9, 200) - 0.5)

    def test_zero_n_is_half(self):
        assert shrunk_hit_rate(0.9, 0) == 0.5


class TestScore:
    def test_bounded(self):
        for hr in (0.0, 0.5, 1.0):
            for n in (0, 5, 500):
                s = score(_inputs(hit_rate=hr, n_clusters=n))
                assert 0 <= s <= 100

    def test_monotonic_in_hit_rate(self):
        assert score(_inputs(hit_rate=0.8)) >= score(_inputs(hit_rate=0.55))

    def test_more_evidence_never_hurts(self):
        assert score(_inputs(times_asserted=5)) >= score(_inputs(times_asserted=0))
        assert score(_inputs(extraction_confidence=0.95)) >= score(
            _inputs(extraction_confidence=0.30)
        )

    def test_no_track_record_is_capped(self):
        s = score(_inputs(has_track_record=False, hit_rate=None, n_clusters=0))
        assert s <= 50

    def test_none_hit_rate_treated_as_no_signal(self):
        s = score(_inputs(hit_rate=None))
        assert 0 <= s <= 60
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement** `src/market_intelligence/signals/confidence.py`:

```python
"""Confidence: compress an alert's evidence into one auditable 0–100 number.

The score is derived from *measured* quantities — the cell's holdout hit rate
(shrunk toward 50 when the sample is thin), how often the underlying
relationship was independently asserted, and the extractor's own confidence.
The rendered message always shows the components next to the score, so the
number can be audited rather than believed.

The blend weights are deliberately a judgment call, not a discovery:
# TODO(michael): these weights are yours to tune. The contract tests in
# tests/test_confidence.py pin bounds/monotonicity/shrinkage only — any
# weighting that keeps those properties is fair game.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Pseudo-observations pulling a thin cell's hit rate toward a coin flip.
SHRINKAGE_N = 20


@dataclass(frozen=True)
class ConfidenceInputs:
    hit_rate: float | None          # holdout hit rate of the cell, 0..1
    n_clusters: int                 # independent holdout observations behind it
    times_asserted: int             # independent filings asserting the edge (0 for self)
    extraction_confidence: float | None  # extractor's own 0..1 confidence
    has_track_record: bool          # False for gap alerts until the ledger matures


def shrunk_hit_rate(hit_rate: float | None, n: int) -> float:
    """Hit rate pulled toward 0.5 as evidence thins; exactly 0.5 with none."""
    if hit_rate is None or n <= 0:
        return 0.5
    return 0.5 + (hit_rate - 0.5) * (n / (n + SHRINKAGE_N))


def score(inputs: ConfidenceInputs) -> int:
    """Blend the components into 0–100. Weights: see module TODO."""
    base = shrunk_hit_rate(inputs.hit_rate, inputs.n_clusters) * 100.0

    # Corroboration bonuses are small on purpose: they refine a measured base,
    # they must never manufacture confidence a track record didn't earn.
    edge_bonus = min(inputs.times_asserted, 5) * 1.0
    extract_bonus = (inputs.extraction_confidence or 0.0) * 5.0

    value = base + edge_bonus + extract_bonus
    if not inputs.has_track_record:
        value = min(value, 50.0)
    return int(max(0.0, min(100.0, round(value))))


__all__ = ["SHRINKAGE_N", "ConfidenceInputs", "score", "shrunk_hit_rate"]
```

- [ ] **Step 4:** `uv run pytest tests/test_confidence.py -q` → PASS.
- [ ] **Step 5:** Commit: `git add src/market_intelligence/signals/confidence.py tests/test_confidence.py && git commit -m "feat: auditable confidence score with small-n shrinkage"`

## Task 7: `signals/trade_alerts.py` — assemble, persist, mark delivered

**Files:**
- Create: `src/market_intelligence/signals/trade_alerts.py`
- Test: `tests/test_trade_alerts.py`

This is the glue: `Briefing.alerts` + DB → `TradeAlertRecord` rows → first-firing-wins persistence → the sendable subset.

- [ ] **Step 1: Failing tests** — create `tests/test_trade_alerts.py` using `memory_db` + `tmp_config`. Seed `daily_prices` (40 bars for NKE around $100 with real spread), then exercise:
  - `build_records(con, alerts, config, as_of)` with one self-edge `EventAlert(ticker="NKE", event_id="ev1", horizon_days=20, direction=1, predicted_car=0.03, hit_rate=0.65, n_clusters=40, extraction_confidence=0.9, ...)` → one record with `kind == "reaction_lag"`, `trigger_key == "ev1:self:20"`, `edge_id == "self"`, plan fields populated, `confidence` in 0..100, `evidence` JSON parseable and naming the event type + cell stats.
  - An alert whose ticker has no price history → no record, and the returned `skipped` note names the ticker (return shape: `(records, notes)`).
  - `persist_new(con, records)` → inserts, second call returns `[]`.
  - `sendable(records, min_confidence)` → filters gated ones; gated rows keep `delivery_note == "gated_below_min_confidence"`.
  - `mark_delivered(con, records, delivered=True)` flips the flag; `delivered=False` sets `delivery_note == "telegram_failed"`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** Shape:

```python
@dataclass(frozen=True)
class TradeAlertRecord:
    alert_id: str
    kind: str            # "reaction_lag" | "price_gap"
    ticker: str
    trigger_key: str
    fired_at: datetime
    direction: int
    plan: TradePlan
    confidence: int
    evidence: dict       # rendered to JSON at persist time
    event_id: str | None
    edge_id: str | None
    delivered: bool = False
    delivery_note: str | None = None
```

- `build_records(con, alerts, config, as_of)`: for each `EventAlert`, load the last `atr_period * 4` bars from `daily_prices` (`SELECT price_date, open, high, low, close, adj_close FROM daily_prices WHERE symbol = ? AND price_date <= ? ORDER BY price_date DESC LIMIT ?`, reversed into `Bar`s), call `build_trade_plan` (skip + note when None), `confidence.score(...)` with `times_asserted=0`, `has_track_record=True` for reaction-lag. Evidence dict: `{"event_type", "event_subtype", "available_on", "basis", "hit_rate", "n_clusters", "predicted_car", "extraction_confidence"}`. `trigger_key = f"{event_id}:self:{horizon_days}"` (null/empty edge renders as `self` per spec).
- `persist_new(con, records, schema_version="1.0.0")` (default so both call shapes in this plan work): rows via `dataclasses` fields + `json.dumps(evidence)`, `outcome="open"`, delegate to `duckdb_store.insert_new_trade_alerts`, map returned rows back to records (match on `alert_id`).
- `sendable(records, min_confidence)` and `gated(...)` split; `mark_gated(con, records)` stamps the note.
- `mark_delivered(con, records, *, delivered)` → `UPDATE trade_alerts SET delivered = ?, delivery_note = ? WHERE alert_id = ?` per record.
- [ ] **Step 4:** `uv run pytest tests/test_trade_alerts.py -q` → PASS.
- [ ] **Step 5:** Commit: `git add src/market_intelligence/signals/trade_alerts.py tests/test_trade_alerts.py && git commit -m "feat: trade-alert assembly and first-firing-wins ledger persistence"`

## Task 8: Notifier — separate trade-alert message

**Files:**
- Modify: `src/market_intelligence/autopilot/notify.py`
- Test: `tests/test_notify.py` (append)

- [ ] **Step 1: Failing tests** — append to `tests/test_notify.py`:
  - `render_trade_alerts([record])` output contains: `TRADE SIGNAL`, ticker + arrow, `confidence NN`, a `Why:` line naming event type and `hit`/`n=`, a `Plan:` line with entry/stop/target, a `Size:` line with share count and risk %, an `exit by` date. Multiple records sorted highest-confidence first. `unsizeable` record renders `unsizeable at current risk settings` instead of a Size line. Output length ≤ `TELEGRAM_MAX_CHARS` even with 30 records (truncation marker).
  - `send_trade_alerts(config, records, transport=MockTransport)` posts to the Telegram URL and returns True on 200 / False on 500 / False and no request without creds — mirror the existing `send` tests.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** Extract the credential-check + POST + logging body of `send()` into `_post_text(config, text, *, transport) -> bool`; re-express `send()` with it (behavior identical — existing tests must stay green). Add pure `render_trade_alerts(records: Sequence[TradeAlertRecord]) -> str` (import under `TYPE_CHECKING` from `market_intelligence.signals.trade_alerts`) and `send_trade_alerts(config, records, *, transport=None) -> bool` = render + `_post_text`, empty-records → return False without posting. Reuse `_SAFE_CHARS` truncation. Format per spec §7 (see the spec's example; risk % line = `risk_amount / account_equity` is NOT available here — render `~$<notional> → risks $<risk_amount>` instead, amounts from the record's plan).
- [ ] **Step 4:** `uv run pytest tests/test_notify.py -q` → PASS, including all pre-existing tests untouched.
- [ ] **Step 5:** Commit: `git add src/market_intelligence/autopilot/notify.py tests/test_notify.py && git commit -m "feat: trade-alert telegram rendering as its own message"`

## Task 9: Orchestrator wiring — daily loop fires trade alerts

**Files:**
- Modify: `src/market_intelligence/autopilot/orchestrator.py`
- Test: `tests/test_autopilot.py` (append)

- [ ] **Step 1: Failing tests** — in `tests/test_autopilot.py`, following the file's existing orchestrator-test pattern (injected steps, tmp_config):
  - A run whose briefing contains one alert (seed `events` + `signal_status` + `daily_prices` in the tmp DB) leaves one row in `trade_alerts` and the row survives a second `run()` unchanged (dedup: still exactly one row).
  - A run where trade-alert building raises (e.g. monkeypatch `trade_alerts.build_records` to raise) still completes and the briefing notes mention `trade-alerts`.
  - Delivery: monkeypatch `notify.send_trade_alerts` to record its argument and return True → the persisted row has `delivered = True`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** In `orchestrator.run()`, inside the second `database.connection(...)` block after `briefing = briefing_builder.build(...)`, add a best-effort block (never raises; failure appends a note):

```python
        sendable_records: list = []
        try:
            records, ta_notes = trade_alerts_builder.build_records(
                con, briefing.alerts, config, as_of=briefing_date
            )
            notes.extend(ta_notes)
            new_records = trade_alerts_builder.persist_new(
                con, records, schema_version=config.settings.app.schema_version
            )
            min_conf = config.settings.trading.min_confidence
            sendable_records = trade_alerts_builder.sendable(new_records, min_conf)
            trade_alerts_builder.mark_gated(
                con, trade_alerts_builder.gated(new_records, min_conf)
            )
        except Exception as exc:  # the ledger must not sink the briefing
            logger.error("autopilot_trade_alerts_failed", error=str(exc))
            briefing.notes.append(f"step trade-alerts FAILED: {exc}")
```

After `_deliver(config, briefing)`, send and stamp:

```python
    if sendable_records:
        sent = notify.send_trade_alerts(config, sendable_records)
        try:
            with database.connection(config.paths.database_path) as con:
                trade_alerts_builder.mark_delivered(con, sendable_records, delivered=sent)
        except Exception as exc:
            logger.error("autopilot_trade_alerts_stamp_failed", error=str(exc))
```

(Note: `briefing.notes` is a list on a frozen dataclass — appending to the list is fine; do NOT reassign the field. Build the trade-alert block BEFORE `briefing_builder.build` if notes must land inside the briefing — check the existing construction order and keep notes accurate; acceptable alternative: collect `notes` list before building the briefing, exactly as the step loop does.)

- [ ] **Step 4:** `uv run pytest tests/test_autopilot.py -q` → PASS.
- [ ] **Step 5:** Commit: `git add src/market_intelligence/autopilot/orchestrator.py tests/test_autopilot.py && git commit -m "feat: daily loop persists and sends trade alerts"`

## Task 10: Phase A checkpoint — full suite + live Telegram proof

- [ ] **Step 1:** `uv run pytest -q` → full suite green. Fix anything broken before proceeding.
- [ ] **Step 2:** Live message proof (network + real Telegram creds, NO database):
  write `scripts/send_test_trade_alert.py` — builds one `TradeAlertRecord` from hardcoded REALISTIC-BUT-FAKE numbers, ticker `TEST`, evidence noting `pipeline verification — not a real signal`, and calls `notify.send_trade_alerts(get_config(), [record])`. Run `uv run python scripts/send_test_trade_alert.py`; expect `sent=True` and the message on Michael's phone.
- [ ] **Step 3:** Commit the script: `git add scripts/send_test_trade_alert.py && git commit -m "chore: manual trade-alert send verification script"`

## Task 11: Gap scan — `collectors/gaps.py`

**Files:**
- Create: `src/market_intelligence/collectors/gaps.py`
- Modify: `src/market_intelligence/clients/market_yfinance.py` (only if it lacks a usable latest-quote path — read it first)
- Test: `tests/test_gaps.py`

- [ ] **Step 1:** Read `clients/market_yfinance.py`. If `get_latest_price` is implemented, use it; otherwise add a minimal batch quote helper to the provider following its existing style (yfinance `download`/`fast_info`, isolated behind the ABC).
- [ ] **Step 2: Failing tests** — `tests/test_gaps.py`, all offline via `MockMarketDataProvider` or a purpose-built stub provider + `memory_db`:
  - `detect_gaps(prior_bars, quotes, cfg)` (pure): finds `|quote − prior_close| / prior_close ≥ min_gap_pct`; drops tickers under `min_avg_dollar_volume` (20-day avg of `close × volume`); direction = sign of the gap (continuation).
  - Catalyst join: seeded event within `gap_catalyst_lookback_days` → candidate carries the event in evidence and `has_catalyst=True`; `require_catalyst=True` drops explanation-less gaps.
  - `scan(config, provider, as_of)` end-to-end against `tmp_config`-backed DB: seeds `index_constituents` (current members), `daily_prices`; stub provider quotes one ticker gapping +5%; asserts one `trade_alerts` row `kind="price_gap"`, `trigger_key == as_of.isoformat()`, target = entry ± `gap_target_r_multiple × stop_distance`, confidence ≤ 50 (`has_track_record=False`), evidence includes `"no track record yet"`. Re-run → no second row.
- [ ] **Step 3:** Run → FAIL. **Step 4: Implement** `collectors/gaps.py` with `scan(config, provider=None, as_of=None, notify=True)` using the `pipeline_run(config, "gaps.scan")` pattern; universe = `SELECT ticker FROM index_constituents WHERE removed_date IS NULL`; prior bars + liquidity from `daily_prices`; plan via `build_trade_plan(..., entry_override=<morning quote>)` — the plan MUST anchor at the live quote, never the prior close (they differ by exactly the gap). `predicted_move` = `gap_target_r_multiple × stop_distance / entry` so the target math reuses the one code path (document this in the module docstring). A ticker with no prior bar in `daily_prices` (fresh listing, market holiday, stale DB) is skipped silently — this is also the weekend/holiday no-op from spec §9, worth one test (quote equal to prior close → no candidate). Gap confidence: `ConfidenceInputs(hit_rate=None, n_clusters=0, times_asserted=<1 if catalyst else 0>, extraction_confidence=None, has_track_record=False)` — the spec §5 gap substitutes (catalyst/size/liquidity) are deliberately collapsed into the ≤50 no-track-record cap plus the catalyst-as-corroboration bonus until graded ledger history exists; note this in the docstring. Persistence + gating + `notify.send_trade_alerts` exactly as the orchestrator does. DB-busy: wrap the `pipeline_run` entry in a bounded retry (3 attempts, 30s apart) catching `duckdb.IOException`; on final failure log `gap_scan_skipped_db_busy` and return a `RunSummary(status="skipped")` — a missed scan must not crash.
- [ ] **Step 5:** `uv run pytest tests/test_gaps.py -q` → PASS. Commit: `git add src/market_intelligence/collectors/gaps.py tests/test_gaps.py <provider file if touched> && git commit -m "feat: morning gap scanner feeding the trade-alert ledger"`

## Task 12: Grading — close the loop

**Files:**
- Create: `src/market_intelligence/signals/grading.py`
- Modify: `src/market_intelligence/autopilot/orchestrator.py` (add non-critical step `grade-trade-alerts` before `sync-prices`? NO — after `compute-returns`, so today's bars exist)
- Test: `tests/test_grading.py`

- [ ] **Step 1: Failing tests** — `tests/test_grading.py` with `memory_db`: seed one open `trade_alerts` row (long, entry 100, stop 96, target 103, exit date D+20) plus subsequent `daily_prices` bars, assert:
  - bar low ≤ 96 → `outcome="hit_stop"`, `outcome_return ≈ (96−100)/100`, `graded_at` set;
  - bar high ≥ 103 (no stop touch) → `hit_target`;
  - a bar touching BOTH → `hit_stop` (conservative);
  - no touch through exit date → `expired`, return from the exit-date close;
  - **split case:** bars after a 2:1 split (raw halves, `adj_close` continuous) do NOT grade as `hit_stop` — grading compares in adjusted space via each bar's `adj_close/close` factor and the entry-day factor applied to stop/target;
  - short alert: stop/target logic mirrored;
  - already-graded rows untouched on re-run.
- [ ] **Step 2:** Run → FAIL. **Step 3: Implement** `grade_open_alerts(con, as_of) -> dict[str, int]` (counts per outcome) + a `grade(config)` wrapper using `pipeline_run(config, "signals.grade-trade-alerts")`. Add `Step("grade-trade-alerts", lambda c: grading.grade(c))` to `default_steps()` after `compute-returns`.
- [ ] **Step 4:** `uv run pytest tests/test_grading.py tests/test_autopilot.py -q` → PASS.
- [ ] **Step 5:** Commit: `git add src/market_intelligence/signals/grading.py src/market_intelligence/autopilot/orchestrator.py tests/test_grading.py && git commit -m "feat: grade open trade alerts in split-adjusted space"`

## Task 13: CLI + LaunchAgent

**Files:**
- Modify: `src/market_intelligence/cli.py` (add `market scan-gaps`, update the docstring command list)
- Create: `config/launchd/ai.quant.morning-scan.plist`
- Test: `tests/test_cli.py` (append, following its existing typer-runner pattern)

- [ ] **Step 1:** Failing test: `market scan-gaps` invokes `gaps.scan` (monkeypatched) and exits 0; `--dry-run` flag passes `notify=False` (the kwarg already exists on `scan` from Task 11 — verify the pass-through: build+persist but skip the send). Run → FAIL. Implement following the existing `market sync-prices` command's shape. Update the module docstring command list.
- [ ] **Step 2:** Create `config/launchd/ai.quant.morning-scan.plist` modeled byte-for-byte on `config/launchd/ai.quant.autopilot.plist` (read it; adapt Label to `ai.quant.morning-scan`, ProgramArguments to run `market-intelligence market scan-gaps` via the same uv/interpreter invocation, StartCalendarInterval to Weekday entries 1–5 at Hour 8 / Minute 45 local time, its own log paths). **Do NOT `launchctl load` it** — installation is Michael's call; the final report gives him the one-liner.
- [ ] **Step 3:** `uv run pytest tests/test_cli.py -q` → PASS.
- [ ] **Step 4:** Commit: `git add src/market_intelligence/cli.py config/launchd/ai.quant.morning-scan.plist tests/test_cli.py && git commit -m "feat: scan-gaps CLI and morning launchd agent"`

## Task 14: Final verification

- [ ] **Step 1:** `uv run pytest -q` — entire suite green. Record the final count vs the Task 1 baseline.
- [ ] **Step 2:** `uv run market-intelligence market scan-gaps --dry-run` against the REAL config — this is the ONE sanctioned exception to the "never touch the real database" ground rule, and only if `pgrep -f extract_relationships` shows the extractor idle; otherwise skip (DB is locked — say so in the report; exercising the scan's own busy-retry-then-skip path is also an acceptable outcome). `--dry-run` skips the Telegram send but does persist ledger + pipeline_runs rows; that write is authorized here. Expect: clean run or clean skip, no traceback.
- [ ] **Step 3:** Update `docs/specs/2026-07-23-trade-alerts-design.md` status line if anything shipped differs from the spec (it shouldn't).
- [ ] **Step 4:** Final commit of any stragglers; `git log --oneline` to confirm one clean commit per task.
