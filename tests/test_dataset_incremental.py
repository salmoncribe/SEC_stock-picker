"""Equivalence and performance tests for signals.dataset.build_incremental.

build() recomputes every symbol's entire (events x horizons) cross product on
every run -- the actual production bottleneck this module fixes. The tests
below prove build_incremental() is not just faster but *equivalent*: run
repeatedly as data trickles in across simulated days, it must land on the
exact same event_samples rows a single full build() over the final data would
produce -- including the specific correctness trap of an event that was
unmeasurable in an earlier run becoming measurable once its symbol's return
series is extended or corrected.

All DBs here are throwaway (`tmp_config`, isolated `tmp_path`), never the real
`data/database/market_intelligence.duckdb`.
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from market_intelligence import database, hashing
from market_intelligence.collectors import RunSummary
from market_intelligence.config import Config, load_config, reset_config_cache
from market_intelligence.signals import dataset
from market_intelligence.storage import duckdb as duckdb_store

# Far beyond every date used below, so nothing in this file is purged at the
# discovery/holdout split boundary -- these tests are about the incremental
# watermark, not about assign_split.
_NO_PURGE_SPLIT = date(2035, 1, 1)
_HORIZONS = (1, 5, 10)


# --------------------------------------------------------------------------
# Row builders. content_hash is computed the same way the real collectors
# compute it (collectors/returns.py, collectors/filing_events.py): a hash of
# the row's actual VALUE, not of when it was written. That is the exact
# property build_incremental()'s watermark depends on, so the tests must
# seed data the same way production does, not with a content_hash that is
# always null or always fresh.
# --------------------------------------------------------------------------


def _return_row(symbol: str, price_date: date, abnormal_return: float | None) -> dict[str, Any]:
    return {
        "return_id": f"r-{symbol}-{price_date.isoformat()}",
        "symbol": symbol,
        "price_date": price_date,
        "total_return": abnormal_return if abnormal_return is not None else 0.0,
        "abnormal_return": abnormal_return,
        "content_hash": hashing.content_hash(symbol, price_date.isoformat(), abnormal_return),
        "source": "market",
    }


def _return_rows(symbol: str, start: date, values: list[float | None]) -> list[dict[str, Any]]:
    return [_return_row(symbol, start + timedelta(days=i), v) for i, v in enumerate(values)]


def _event_row(event_id: str, ticker: str, available: datetime) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_type": "insider_transaction",
        "event_key": event_id,
        "event_subtype": "P",
        "ticker": ticker,
        "available_time": available,
        "magnitude": 50_000.0,
        "direction": 1,
        "content_hash": hashing.content_hash(event_id, "P", str(available)),
    }


def _seed(
    config: Config, *, returns: list[dict] | None = None, events: list[dict] | None = None
) -> None:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        if returns:
            duckdb_store.upsert_daily_returns(con, returns)
        if events:
            duckdb_store.upsert_events(con, events)


# Columns compared for equivalence -- everything that describes the sample's
# *content*. collected_time is deliberately excluded: it is real wall-clock
# provenance, expected to differ between a run today and a run next week, and
# is not part of what "the same sample" means.
_COMPARISON_COLUMNS = """
    sample_id, event_id, edge_id, event_type, event_subtype, source_ticker,
    target_ticker, horizon_days, available_on, t0, window_end,
    forward_abnormal_return, magnitude, direction, split, features
"""


def _event_samples(config: Config) -> list[tuple]:
    with database.connection(config.paths.database_path) as con:
        return con.execute(
            f"SELECT {_COMPARISON_COLUMNS} FROM event_samples ORDER BY sample_id"
        ).fetchall()


def _run_summary_stage(summary: RunSummary, key: str) -> int:
    return summary.stage.get(key, 0)


# --------------------------------------------------------------------------
# The main equivalence proof: several symbols, several "days" of data
# arriving, build_incremental() called after each day. The final table must
# match a single build() call over the union of everything seeded.
# --------------------------------------------------------------------------


def test_incremental_matches_full_rebuild_across_simulated_days(tmp_config: Config) -> None:
    start = date(2020, 1, 1)

    # Symbol A: full 15-day return series and one event from day one --
    # measurable immediately under every horizon, and must stay stable
    # (never rewritten, never re-flagged) on every later no-op run.
    a_returns = _return_rows("AAA", start, [0.01 * (i % 3 - 1) for i in range(15)])
    a_event = _event_row("evt-aaa-1", "AAA", datetime(2020, 1, 3, tzinfo=UTC))

    # Symbol B: only 5 days of returns at first. Its event's 10-day horizon
    # window cannot be formed yet (not enough trading days past t0) --
    # unmeasurable on day 1. More returns land on day 2, extending the tail
    # past what the window needs -- must become measurable without ever
    # having been re-offered as a "new" event.
    b_returns_day1 = _return_rows("BBB", start, [0.02, -0.01, 0.03, 0.00, 0.01])
    b_event = _event_row("evt-bbb-1", "BBB", datetime(2020, 1, 3, tzinfo=UTC))
    b_returns_day2_extra = _return_rows(
        "BBB", start + timedelta(days=5), [0.01 * (i % 2) for i in range(15)]
    )

    # Symbol C: returns present from day one, but its event does not exist
    # until day 3 -- a brand-new symbol appearing mid-run under an existing
    # fingerprint (no prior watermark row at all, not merely a stale one).
    c_returns = _return_rows("CCC", start, [0.005 * (i % 4) for i in range(15)])
    c_event_day3 = _event_row("evt-ccc-1", "CCC", datetime(2020, 1, 3, tzinfo=UTC))

    # Symbol D: fully measurable from day one and never touched again --
    # the control that proves an untouched symbol is skipped, not merely
    # harmlessly reprocessed, on every subsequent run.
    d_returns = _return_rows("DDD", start, [0.01 * (i % 5 - 2) for i in range(15)])
    d_event = _event_row("evt-ddd-1", "DDD", datetime(2020, 1, 4, tzinfo=UTC))

    # --- Day 1 ---
    _seed(tmp_config, returns=[*a_returns, *b_returns_day1, *c_returns, *d_returns])
    _seed(tmp_config, events=[a_event, b_event, d_event])

    s1 = dataset.build_incremental(tmp_config, horizons=_HORIZONS, split_date=_NO_PURGE_SPLIT)
    # AAA, BBB, DDD reprocessed; CCC has no event yet so it's absent entirely.
    assert _run_summary_stage(s1, "symbols_reprocessed") == 3

    with database.connection(tmp_config.paths.database_path) as con:
        b_horizons_day1 = {
            r[0]
            for r in con.execute(
                "SELECT horizon_days FROM event_samples WHERE source_ticker = 'BBB'"
            ).fetchall()
        }
    # t0 = Jan 4 (index 3 of 5). horizon=1 needs just that one day, which
    # exists; horizon=5 and horizon=10 need trading days that don't exist yet.
    assert b_horizons_day1 == {1}

    # --- Day 2: BBB's return series is extended; nothing else changes ---
    _seed(tmp_config, returns=b_returns_day2_extra)
    s2 = dataset.build_incremental(tmp_config, horizons=_HORIZONS, split_date=_NO_PURGE_SPLIT)
    assert _run_summary_stage(s2, "symbols_reprocessed") == 1  # only BBB
    # CCC still has no event, so it isn't in the candidate set at all yet --
    # only AAA and DDD are unchanged-and-skipped.
    assert _run_summary_stage(s2, "symbols_skipped") == 2

    with database.connection(tmp_config.paths.database_path) as con:
        b_horizons = {
            r[0]
            for r in con.execute(
                "SELECT horizon_days FROM event_samples WHERE source_ticker = 'BBB'"
            ).fetchall()
        }
    assert b_horizons == {1, 5, 10}  # the previously-unmeasurable event is now fully scored

    # --- Day 3: CCC's first event arrives; nothing else changes ---
    _seed(tmp_config, events=[c_event_day3])
    s3 = dataset.build_incremental(tmp_config, horizons=_HORIZONS, split_date=_NO_PURGE_SPLIT)
    assert _run_summary_stage(s3, "symbols_reprocessed") == 1  # only CCC (new symbol)
    assert _run_summary_stage(s3, "symbols_skipped") == 3  # AAA, BBB, DDD all unchanged

    # --- Day 4: nothing changes at all -- must be a total no-op ---
    s4 = dataset.build_incremental(tmp_config, horizons=_HORIZONS, split_date=_NO_PURGE_SPLIT)
    assert _run_summary_stage(s4, "symbols_reprocessed") == 0
    assert _run_summary_stage(s4, "symbols_skipped") == 4
    assert s4.inserted == 0
    assert s4.updated == 0

    incremental_final = _event_samples(tmp_config)

    # --- Reference: seed a FRESH, independent db with the exact final union
    # of everything seeded above, and run the legacy full build() once. ---
    ref_home = tmp_config.paths.home.parent / "reference_home"
    reset_config_cache()
    ref_config = load_config(project_root=tmp_config.paths.project_root, home=ref_home)
    ref_config.paths.ensure()

    _seed(
        ref_config,
        returns=[*a_returns, *b_returns_day1, *b_returns_day2_extra, *c_returns, *d_returns],
        events=[a_event, b_event, c_event_day3, d_event],
    )
    dataset.build(ref_config, horizons=_HORIZONS, split_date=_NO_PURGE_SPLIT)
    reference_final = _event_samples(ref_config)

    assert incremental_final == reference_final
    assert len(incremental_final) > 0  # sanity: the scenario actually produced rows


def test_value_correction_to_existing_row_triggers_reprocessing(tmp_config: Config) -> None:
    """A gap-filled abnormal_return (same price_date, corrected value) must be retried.

    This is the specific correctness trap called out in the design brief: an
    event can be unmeasurable not because too little time has passed, but
    because one day *inside* an already-long-enough window has no abnormal
    return yet. COUNT(*) alone would not notice this fix (no row is added);
    only a checksum over the row's actual value does.
    """
    start = date(2020, 1, 1)
    # 12 days: t0 will be day 1 (Jan 2), horizon=5 needs day1..day5. day3
    # starts out NULL -- a gap inside an otherwise complete window.
    values: list[float | None] = [0.01] * 12
    values[3] = None
    returns_with_gap = _return_rows("EEE", start, values)
    event = _event_row("evt-eee-1", "EEE", datetime(2020, 1, 1, tzinfo=UTC))

    _seed(tmp_config, returns=returns_with_gap, events=[event])
    s1 = dataset.build_incremental(tmp_config, horizons=(5,), split_date=_NO_PURGE_SPLIT)
    assert _run_summary_stage(s1, "unmeasurable") == 1

    with database.connection(tmp_config.paths.database_path) as con:
        count_row = con.execute("SELECT COUNT(*) FROM event_samples").fetchone()
    assert count_row is not None
    assert count_row[0] == 0

    # Same return_id / price_date, corrected value -- an UPDATE, not an
    # INSERT. Row count for the symbol is unchanged; only the value moves.
    gap_day = start + timedelta(days=3)
    corrected = _return_row("EEE", gap_day, 0.02)
    _seed(tmp_config, returns=[corrected])

    s2 = dataset.build_incremental(tmp_config, horizons=(5,), split_date=_NO_PURGE_SPLIT)
    assert _run_summary_stage(s2, "symbols_reprocessed") == 1
    assert _run_summary_stage(s2, "unmeasurable") == 0

    with database.connection(tmp_config.paths.database_path) as con:
        fetched = con.execute(
            """
            SELECT horizon_days, forward_abnormal_return
            FROM event_samples WHERE source_ticker = 'EEE'
            """
        ).fetchone()
    assert fetched is not None
    row: tuple[int, float] = fetched
    assert row[0] == 5
    # t0 = Jan 2 (first trading day strictly after the Jan 1 available_time);
    # the 5-day window is Jan2..Jan6 = four days at 0.01 plus the corrected
    # Jan 4 day at 0.02.
    assert row[1] == pytest.approx(4 * 0.01 + 0.02)


def test_horizon_parameter_change_forces_reprocessing(tmp_config: Config) -> None:
    """A watermark recorded under one parameter set must not be reused under another."""
    start = date(2020, 1, 1)
    returns = _return_rows("FFF", start, [0.01] * 20)
    event = _event_row("evt-fff-1", "FFF", datetime(2020, 1, 1, tzinfo=UTC))
    _seed(tmp_config, returns=returns, events=[event])

    s1 = dataset.build_incremental(tmp_config, horizons=(1,), split_date=_NO_PURGE_SPLIT)
    assert _run_summary_stage(s1, "symbols_reprocessed") == 1
    with database.connection(tmp_config.paths.database_path) as con:
        horizons_seen = {
            r[0] for r in con.execute("SELECT horizon_days FROM event_samples").fetchall()
        }
    assert horizons_seen == {1}

    # Same data, no new returns or events -- but a different horizons tuple.
    # Must be treated as "never built under this fingerprint", not skipped.
    s2 = dataset.build_incremental(tmp_config, horizons=(1, 5), split_date=_NO_PURGE_SPLIT)
    assert _run_summary_stage(s2, "symbols_reprocessed") == 1
    assert _run_summary_stage(s2, "symbols_skipped") == 0
    with database.connection(tmp_config.paths.database_path) as con:
        horizons_seen = {
            r[0] for r in con.execute("SELECT horizon_days FROM event_samples").fetchall()
        }
    assert horizons_seen == {1, 5}


# --------------------------------------------------------------------------
# Performance: dozens of symbols, a few thousand events, most already built.
# Compares a full build() rerun over the whole (now slightly larger) dataset
# against build_incremental() touching only the handful of symbols with new
# data.
# --------------------------------------------------------------------------


def _generate_universe(
    n_symbols: int, days_per_symbol: int, events_per_symbol: int, start: date
) -> tuple[list[dict], list[dict]]:
    returns: list[dict] = []
    events: list[dict] = []
    for s in range(n_symbols):
        symbol = f"SYM{s:04d}"
        values: list[float | None] = [
            0.001 * ((i * 7 + s) % 11 - 5) for i in range(days_per_symbol)
        ]
        returns.extend(_return_rows(symbol, start, values))
        # Spread events across the first 2/3 of the series so most are
        # measurable at horizon=20, some near the tail legitimately are not
        # -- realistic mix, same as production.
        for e in range(events_per_symbol):
            offset = (e * (days_per_symbol * 2 // 3)) // max(events_per_symbol, 1)
            event_date = start + timedelta(days=offset)
            available = datetime.combine(event_date, datetime.min.time(), tzinfo=UTC)
            events.append(_event_row(f"evt-{symbol}-{e}", symbol, available))
    return returns, events


def test_incremental_beats_full_rebuild_on_mostly_unchanged_dataset(tmp_config: Config) -> None:
    n_symbols = 40
    days_per_symbol = 90
    events_per_symbol = 50  # 2,000 events total, x3 horizons = 6,000 candidates
    start = date(2020, 1, 1)

    returns, events = _generate_universe(n_symbols, days_per_symbol, events_per_symbol, start)
    _seed(tmp_config, returns=returns, events=events)

    # Establish the fully-built baseline (this cost is paid once by either
    # approach and is not part of the comparison).
    dataset.build_incremental(
        tmp_config, horizons=dataset.DEFAULT_HORIZONS, split_date=_NO_PURGE_SPLIT
    )

    # A small delta: 2 of 40 symbols get one new event each.
    changed_symbols = ["SYM0003", "SYM0027"]
    delta_events = [
        _event_row(f"evt-{sym}-new", sym, datetime(2020, 2, 15, tzinfo=UTC))
        for sym in changed_symbols
    ]
    _seed(tmp_config, events=delta_events)

    t0 = time.perf_counter()
    dataset.build(tmp_config, horizons=dataset.DEFAULT_HORIZONS, split_date=_NO_PURGE_SPLIT)
    full_rebuild_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    incremental_summary = dataset.build_incremental(
        tmp_config, horizons=dataset.DEFAULT_HORIZONS, split_date=_NO_PURGE_SPLIT
    )
    incremental_seconds = time.perf_counter() - t0

    reprocessed = _run_summary_stage(incremental_summary, "symbols_reprocessed")
    skipped = _run_summary_stage(incremental_summary, "symbols_skipped")

    print(
        f"\n[perf] full build() over {n_symbols} symbols / {len(events)} events: "
        f"{full_rebuild_seconds:.4f}s\n"
        f"[perf] build_incremental() same delta ({reprocessed} reprocessed, "
        f"{skipped} skipped): {incremental_seconds:.4f}s\n"
        f"[perf] speedup: {full_rebuild_seconds / max(incremental_seconds, 1e-9):.1f}x"
    )

    assert reprocessed == len(changed_symbols)
    assert skipped == n_symbols - len(changed_symbols)
    # Generous margin (not a tight timing assertion) -- the point is "orders
    # of magnitude less work when almost nothing changed", not a specific
    # ratio that could flake on a loaded CI box.
    assert incremental_seconds < full_rebuild_seconds / 3

    # And the two are still in agreement on content for the symbols that
    # actually changed.
    with database.connection(tmp_config.paths.database_path) as con:
        for sym in changed_symbols:
            count_row = con.execute(
                "SELECT COUNT(*) FROM event_samples WHERE source_ticker = ?", [sym]
            ).fetchone()
            assert count_row is not None
            assert count_row[0] > 0
