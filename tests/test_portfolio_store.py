"""Persistence tests: the fill ledger and the experiment chain.

Every test runs against an in-memory DuckDB built from ``database.py``'s own
schema dict (the ``memory_db`` fixture). The 7.2 GB production file is never
opened -- it is single-writer and contended every 600 seconds, so a test that
touched it would both stall and be able to corrupt the account of record.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import duckdb
import pytest

from market_intelligence.analytics.experiment_registry import (
    ExperimentRecord,
    ExperimentRegistry,
    ExperimentStage,
)
from market_intelligence.portfolio import account, store
from market_intelligence.portfolio.account import Fill, Order, Side

VERSION = "portfolio-sim-v1"
SCRATCH = "portfolio-sim-scratch"
START = 10_000.0
D0 = date(2026, 1, 5)
D1 = date(2026, 1, 6)
D2 = date(2026, 1, 7)


def _order(symbol: str, side: Side, shares: float, as_of: date, version: str = VERSION) -> Order:
    return Order(
        order_id=store.order_id(symbol, as_of, side, version),
        symbol=symbol,
        side=side,
        shares=shares,
        as_of=as_of,
        reason="test",
    )


def _fill(
    symbol: str,
    side: Side,
    shares: float,
    price: float,
    as_of: date,
    *,
    commission: float = 0.0,
    version: str = VERSION,
) -> Fill:
    return Fill(
        fill_id=store.fill_id(symbol, as_of, side, version),
        order_id=store.order_id(symbol, as_of, side, version),
        symbol=symbol,
        side=side,
        shares=shares,
        price=price,
        commission=commission,
        as_of=as_of,
    )


def _trading_day(version: str = VERSION) -> tuple[tuple[Order, ...], tuple[Fill, ...]]:
    """A genesis-free order/fill pair set: two buys, then a partial sell."""
    specs = [
        ("AAPL", Side.BUY, 12.3456, 100.0, D1, 1.0),
        ("MSFT", Side.BUY, 5.0, 50.0, D1, 0.5),
        ("AAPL", Side.SELL, 2.3456, 110.0, D2, 0.75),
    ]
    orders = tuple(_order(s, side, sh, day, version) for s, side, sh, _, day, _ in specs)
    fills = tuple(
        _fill(s, side, sh, px, day, commission=fee, version=version)
        for s, side, sh, px, day, fee in specs
    )
    return orders, fills


def _experiment_chain(family: str) -> tuple[ExperimentRecord, ...]:
    """registration -> calibration -> sealed evaluation, hash-linked."""
    sealed_start = datetime(2026, 6, 1, tzinfo=UTC)
    common = {
        "experiment_id": f"exp-{family}",
        "strategy_family": family,
        "hypothesis": "graph propagation adds return",
        "code_hash": "code-1",
        "configuration_hash": "cfg-1",
        "data_snapshot_hash": "data-1",
        "cost_model_version": "cost-1",
        "feature_version": "feat-1",
    }
    registered = ExperimentRecord(
        record_id=f"{family}-r0",
        stage=ExperimentStage.REGISTERED,
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
        sealed_window_start=sealed_start,
        **common,
    )
    calibrated = ExperimentRecord(
        record_id=f"{family}-r1",
        stage=ExperimentStage.CALIBRATED,
        recorded_at=datetime(2026, 2, 1, tzinfo=UTC),
        parameters={"tau": 0.05, "symbols": ["AAPL", "MSFT"]},
        metrics={"discovery": {"sharpe": 0.8}},
        parent_record_hash=registered.record_hash,
        **common,
    )
    sealed = ExperimentRecord(
        record_id=f"{family}-r2",
        stage=ExperimentStage.SEALED_EVALUATED,
        recorded_at=datetime(2026, 7, 1, tzinfo=UTC),
        metrics={"sealed": {"sharpe": 0.4, "dsr": 0.91}},
        parent_record_hash=calibrated.record_hash,
        **common,
    )
    return (registered, calibrated, sealed)


# DuckDB's declared type -> the Python type a round-trip must produce. Written
# values are checked against this so a column whose type changed in
# database.py surfaces here rather than as a silent cast in production.
_TYPE_FAMILIES: dict[str, type | tuple[type, ...]] = {
    "VARCHAR": str,
    "BIGINT": int,
    "INTEGER": int,
    "DOUBLE": float,
    "BOOLEAN": bool,
    "TIMESTAMP WITH TIME ZONE": datetime,
    "DATE": date,
}


def _assert_rows_fit_ddl(con: duckdb.DuckDBPyConnection, table: str) -> int:
    """Every stored row honours ``table``'s declared columns, types and NOT NULLs.

    Driven off ``PRAGMA table_info`` against a database created by
    ``database.init_db`` from ``SCHEMA_STATEMENTS``, so this reads the real DDL
    rather than a hand-written copy of it.
    """
    info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
    names = [row[1] for row in info]
    declared = {row[1]: (row[2], bool(row[3])) for row in info}
    rows = con.execute(f'SELECT * FROM "{table}"').fetchall()
    assert rows, f"{table} has no rows to check"
    for row in rows:
        for name, value in zip(names, row, strict=True):
            type_name, not_null = declared[name]
            if not_null:
                assert value is not None, f"{table}.{name} is NOT NULL but was written NULL"
            if value is None:
                continue
            expected = _TYPE_FAMILIES.get(type_name)
            assert expected is not None, f"{table}.{name} has untested type {type_name}"
            assert isinstance(value, expected), (
                f"{table}.{name} declared {type_name} but round-tripped {type(value).__name__}"
            )
    return len(rows)


# --------------------------------------------------------------------------- #
# 1. schema fit                                                                #
# --------------------------------------------------------------------------- #
def test_every_written_column_fits_the_real_ddl(memory_db: duckdb.DuckDBPyConnection) -> None:
    """A write against the shipped schema succeeds and fills every NOT NULL column.

    DuckDB rejects an unknown column outright, so a clean write already proves
    the column names exist; the type and NOT NULL sweep proves the values are
    storable rather than silently coerced or left null.
    """
    orders, fills = _trading_day()
    store.write_genesis(memory_db, cash=START, as_of=D0, simulation_version=VERSION)
    store.write_orders(memory_db, orders, simulation_version=VERSION)
    store.write_fills(memory_db, fills, simulation_version=VERSION)
    for record in _experiment_chain("portfolio"):
        store.write_experiment_record(memory_db, record)

    assert _assert_rows_fit_ddl(memory_db, "paper_orders") == len(orders) + 1
    assert _assert_rows_fit_ddl(memory_db, "paper_fills") == len(fills) + 1
    assert _assert_rows_fit_ddl(memory_db, "experiment_registry") == 3

    # The columns added for the ledger must be present and typed, not just
    # writable: symbol names the instrument, quantity_exact holds a quantity
    # BIGINT cannot, and reason records why an order existed.
    for table in ("paper_orders", "paper_fills"):
        info = memory_db.execute(f'PRAGMA table_info("{table}")').fetchall()
        declared = {row[1]: row[2] for row in info}
        assert declared["symbol"] == "VARCHAR"
        assert declared["quantity_exact"] == "DOUBLE"
        assert declared["quantity"] == "BIGINT"
    order_info = memory_db.execute('PRAGMA table_info("paper_orders")').fetchall()
    assert {row[1]: row[2] for row in order_info}["reason"] == "VARCHAR"

    symbol, quantity, quantity_exact = memory_db.execute(
        "SELECT symbol, quantity, quantity_exact FROM paper_fills WHERE paper_fill_id = ?",
        [store.fill_id("AAPL", D1, Side.BUY, VERSION)],
    ).fetchone()
    assert symbol == "AAPL"
    assert quantity_exact == pytest.approx(12.3456)
    assert quantity == 12  # the rounded index, which is exactly why it is not authoritative

    order_symbol, order_exact, reason = memory_db.execute(
        "SELECT symbol, quantity_exact, reason FROM paper_orders WHERE paper_order_id = ?",
        [store.order_id("AAPL", D1, Side.BUY, VERSION)],
    ).fetchone()
    assert order_symbol == "AAPL"
    assert order_exact == pytest.approx(12.3456)
    assert reason == "test"  # Order.reason now persists rather than being dropped

    # Nothing load-bearing hides in the JSON blob any more.
    payload = json.loads(
        memory_db.execute("SELECT fill_assumptions FROM paper_fills LIMIT 1").fetchone()[0]
    )
    assert "shares" not in payload and "symbol" not in payload


# --------------------------------------------------------------------------- #
# 2. idempotent double-write                                                   #
# --------------------------------------------------------------------------- #
def test_double_write_changes_neither_row_count_nor_values(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    """Re-running a step rewrites the same rows. This is what makes a re-run safe."""
    orders, fills = _trading_day()

    def write() -> None:
        store.write_genesis(memory_db, cash=START, as_of=D0, simulation_version=VERSION)
        store.write_orders(memory_db, orders, simulation_version=VERSION)
        store.write_fills(memory_db, fills, simulation_version=VERSION)

    write()
    before_orders = memory_db.execute("SELECT * FROM paper_orders ORDER BY 1").fetchall()
    before_fills = memory_db.execute("SELECT * FROM paper_fills ORDER BY 1").fetchall()

    write()
    assert memory_db.execute("SELECT * FROM paper_orders ORDER BY 1").fetchall() == before_orders
    assert memory_db.execute("SELECT * FROM paper_fills ORDER BY 1").fetchall() == before_fills
    assert len(before_orders) == len(orders) + 1
    assert len(before_fills) == len(fills) + 1

    # And the account it rebuilds to is the same account, not a doubled one.
    rebuilt = store.load_account(
        memory_db, simulation_version=VERSION, expected_starting_cash=START
    )
    assert rebuilt.high_water_mark == pytest.approx(START, abs=1e-9)


# --------------------------------------------------------------------------- #
# 3. exact account rebuild                                                     #
# --------------------------------------------------------------------------- #
def test_load_account_reproduces_the_in_memory_state_exactly(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    orders, fills = _trading_day()
    genesis = account.genesis_fill(START, D0, VERSION)
    expected = account.replay((genesis, *fills))

    store.write_genesis(memory_db, cash=START, as_of=D0, simulation_version=VERSION)
    store.write_orders(memory_db, orders, simulation_version=VERSION)
    store.write_fills(memory_db, fills, simulation_version=VERSION)

    # The fold must read ``quantity_exact``, not the JSON blob. Blanking the
    # blob first means a rebuild that still lands exactly can only have come
    # from the typed column.
    memory_db.execute("UPDATE paper_fills SET fill_assumptions = '{}'")

    rebuilt = store.load_account(
        memory_db, simulation_version=VERSION, expected_starting_cash=START
    )

    assert rebuilt.as_of == expected.as_of
    assert rebuilt.cash == pytest.approx(expected.cash, abs=1e-9)
    assert rebuilt.high_water_mark == pytest.approx(expected.high_water_mark, abs=1e-9)
    assert set(rebuilt.positions) == set(expected.positions) == {"AAPL", "MSFT"}
    for symbol, position in expected.positions.items():
        stored = rebuilt.positions[symbol]
        assert stored.shares == pytest.approx(position.shares, abs=1e-9)
        assert stored.cost_basis == pytest.approx(position.cost_basis, abs=1e-9)
    # The fractional leg is the one a BIGINT quantity column would have eaten:
    # 12.3456 bought less 2.3456 sold, which rounds to 12 - 2 = 10 only by luck.
    assert rebuilt.positions["AAPL"].shares == pytest.approx(10.0, abs=1e-9)
    assert rebuilt.cash == pytest.approx(8771.206, abs=1e-9)

    # A NULL in the authoritative column refuses to fold rather than silently
    # substituting the rounded BIGINT.
    memory_db.execute("UPDATE paper_fills SET quantity_exact = NULL WHERE symbol = 'AAPL'")
    with pytest.raises(ValueError, match="quantity_exact"):
        store.load_account(memory_db, simulation_version=VERSION, expected_starting_cash=START)


# --------------------------------------------------------------------------- #
# 4. genesis mismatch fails closed -- stored deposit disagrees                  #
# --------------------------------------------------------------------------- #
def test_genesis_mismatch_raises_and_names_simulation_version(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    store.write_genesis(memory_db, cash=START, as_of=D0, simulation_version=VERSION)

    with pytest.raises(store.GenesisMismatchError) as excinfo:
        store.load_account(
            memory_db, simulation_version=VERSION, expected_starting_cash=25_000.0
        )

    message = str(excinfo.value)
    assert "simulation_version" in message
    assert "10000" in message.replace(",", "") or "10,000" in message


def test_a_second_genesis_cannot_silently_double_the_account(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    """Two deposits under one version fund the account twice, invisibly.

    The fill id embeds its date, so a second ``write_genesis`` on a different
    day lands under a new id, the fold sums both, and $10,000 becomes $20,000
    with every individual row valid and nothing anomalous to notice. Both the
    writer and the loader refuse -- the loader independently, so a deposit that
    arrives by any other route (a direct INSERT, a restored backup, a future
    writer) is caught too.
    """
    store.write_genesis(memory_db, cash=START, as_of=D0, simulation_version=VERSION)
    opened = store.load_account(
        memory_db, simulation_version=VERSION, expected_starting_cash=START
    )
    assert opened.cash == pytest.approx(START)

    # Idempotence is about the SAME genesis: an identical re-run changes nothing.
    store.write_genesis(memory_db, cash=START, as_of=D0, simulation_version=VERSION)
    assert store.load_account(
        memory_db, simulation_version=VERSION, expected_starting_cash=START
    ).cash == pytest.approx(START)

    # A genesis on a different date is a *different* deposit, and is refused.
    with pytest.raises(store.GenesisMismatchError, match="already funded"):
        store.write_genesis(
            memory_db, cash=START, as_of=date(2026, 2, 2), simulation_version=VERSION
        )

    # And the loader refuses one that got in behind the writer's back.
    sneaked = account.genesis_fill(5_000.0, date(2026, 3, 2), VERSION)
    store.write_orders(
        memory_db,
        (
            Order(
                order_id=sneaked.order_id,
                symbol=sneaked.symbol,
                side=sneaked.side,
                shares=sneaked.shares,
                as_of=sneaked.as_of,
                reason="injected",
            ),
        ),
        simulation_version=VERSION,
    )
    store.write_fills(memory_db, (sneaked,), simulation_version=VERSION)
    with pytest.raises(store.GenesisMismatchError, match="DEPOSIT fills"):
        store.load_account(memory_db, simulation_version=VERSION, expected_starting_cash=START)


# --------------------------------------------------------------------------- #
# 5. genesis mismatch fails closed -- no deposit at all                        #
# --------------------------------------------------------------------------- #
def test_ledger_without_a_genesis_deposit_raises_rather_than_returning_unfunded(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    orders, fills = _trading_day()
    store.write_orders(memory_db, orders, simulation_version=VERSION)
    store.write_fills(memory_db, fills, simulation_version=VERSION)

    with pytest.raises(store.GenesisMismatchError) as excinfo:
        store.load_account(
            memory_db, simulation_version=VERSION, expected_starting_cash=START
        )
    assert "simulation_version" in str(excinfo.value)

    # An entirely empty ledger is the same picture and must fail the same way.
    with pytest.raises(store.GenesisMismatchError):
        store.load_account(
            memory_db, simulation_version="never-written", expected_starting_cash=START
        )


# --------------------------------------------------------------------------- #
# 6. version isolation                                                         #
# --------------------------------------------------------------------------- #
def test_simulation_versions_share_tables_without_seeing_each_other(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    """A scratch run must not contaminate the account of record."""
    orders, fills = _trading_day(SCRATCH)
    store.write_genesis(memory_db, cash=START, as_of=D0, simulation_version=VERSION)
    store.write_genesis(memory_db, cash=25_000.0, as_of=D0, simulation_version=SCRATCH)
    store.write_orders(memory_db, orders, simulation_version=SCRATCH)
    store.write_fills(memory_db, fills, simulation_version=SCRATCH)

    of_record = store.load_account(
        memory_db, simulation_version=VERSION, expected_starting_cash=START
    )
    scratch = store.load_account(
        memory_db, simulation_version=SCRATCH, expected_starting_cash=25_000.0
    )

    assert of_record.cash == pytest.approx(START, abs=1e-9)
    assert dict(of_record.positions) == {}
    assert set(scratch.positions) == {"AAPL", "MSFT"}
    assert scratch.cash < 25_000.0
    # Both live in the same tables.
    assert memory_db.execute("SELECT count(*) FROM paper_fills").fetchone()[0] == len(fills) + 2

    # And the account of record fails closed against the scratch run's capital.
    with pytest.raises(store.GenesisMismatchError):
        store.load_account(
            memory_db, simulation_version=VERSION, expected_starting_cash=25_000.0
        )


# --------------------------------------------------------------------------- #
# 7. registry chain round-trip                                                 #
# --------------------------------------------------------------------------- #
def test_experiment_chain_survives_a_round_trip_with_hash_links_intact(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    chain = _experiment_chain("portfolio")
    other = _experiment_chain("gap-scanner")
    for record in (*chain, *other):
        store.write_experiment_record(memory_db, record)

    loaded = store.load_experiment_chain(memory_db, family="portfolio")

    assert [r.record_id for r in loaded] == [r.record_id for r in chain]
    assert [r.record_hash for r in loaded] == [r.record_hash for r in chain]
    assert [r.stage for r in loaded] == [r.stage for r in chain]
    assert loaded[1].parent_record_hash == loaded[0].record_hash
    assert loaded[2].parent_record_hash == loaded[1].record_hash
    # The record freezes lists into tuples on construction, so the round-trip
    # is compared against the original record rather than the literal input.
    assert dict(loaded[1].parameters) == dict(chain[1].parameters)
    assert dict(loaded[1].parameters)["symbols"] == ("AAPL", "MSFT")
    assert dict(loaded[2].metrics) == dict(chain[2].metrics)
    # The registry's own rules are the authority on validity: a chain that
    # replays cleanly through it is a chain whose links verify.
    registry = ExperimentRegistry(loaded)
    assert registry.is_preregistered_for_sealed_test("exp-portfolio") is True

    # Re-writing the identical record is a no-op, not a duplicate.
    store.write_experiment_record(memory_db, chain[0])
    assert len(store.load_experiment_chain(memory_db, family="portfolio")) == 3


# --------------------------------------------------------------------------- #
# 8. kill switch fails closed                                                  #
# --------------------------------------------------------------------------- #
def test_kill_switch_is_active_when_it_cannot_be_read(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    """A safety switch that defaults to "off" when unreadable is not a safety switch."""
    bare = duckdb.connect(":memory:")
    try:
        assert store.kill_switch_active(bare) is True  # no table at all
    finally:
        bare.close()

    # Present and empty: nothing has ever tripped, so trading is permitted.
    assert store.kill_switch_active(memory_db) is False

    memory_db.execute(
        """
        INSERT INTO kill_switch_events (
            kill_switch_event_id, switch_name, state, reason, actor,
            occurred_at, review_required, resolved_at
        ) VALUES ('k1', 'portfolio', 'engaged', 'drawdown halt', 'governor', ?, TRUE, NULL)
        """,
        [datetime(2026, 7, 1, tzinfo=UTC)],
    )
    assert store.kill_switch_active(memory_db) is True

    memory_db.execute(
        """
        INSERT INTO kill_switch_events (
            kill_switch_event_id, switch_name, state, reason, actor,
            occurred_at, review_required, resolved_at
        ) VALUES ('k2', 'portfolio', 'cleared', 'reviewed', 'michael', ?, FALSE, ?)
        """,
        [datetime(2026, 7, 2, tzinfo=UTC), datetime(2026, 7, 2, tzinfo=UTC)],
    )
    assert store.kill_switch_active(memory_db) is False

    # An unrecognised state is treated as engaged, not as permission to trade.
    memory_db.execute(
        """
        INSERT INTO kill_switch_events (
            kill_switch_event_id, switch_name, state, reason, actor,
            occurred_at, review_required, resolved_at
        ) VALUES ('k3', 'portfolio', 'wat', 'typo', NULL, ?, TRUE, NULL)
        """,
        [datetime(2026, 7, 3, tzinfo=UTC)],
    )
    assert store.kill_switch_active(memory_db) is True


# --------------------------------------------------------------------------- #
# 9. daily account marks                                                       #
# --------------------------------------------------------------------------- #
def test_account_marks_round_trip_and_answer_as_of_a_given_session(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    """Latest mark at or before a date, and ``None`` before any exist."""
    assert store.load_latest_mark(memory_db, simulation_version=VERSION) is None

    for day, equity, cash, high_water in (
        (D0, START, START, START),
        (D1, 11_000.0, 500.0, 11_000.0),
        (D2, 10_450.0, 250.0, 11_000.0),
    ):
        store.write_account_mark(
            memory_db,
            simulation_version=VERSION,
            as_of=day,
            equity=equity,
            cash=cash,
            high_water_mark=high_water,
            drawdown=(high_water - equity) / high_water,
        )

    latest = store.load_latest_mark(memory_db, simulation_version=VERSION)
    assert latest is not None
    as_of, equity, high_water = latest
    assert as_of == D2
    assert equity == pytest.approx(10_450.0)
    assert high_water == pytest.approx(11_000.0)

    # "At or before" is the contract: asking as of D1 must not see D2's mark.
    assert store.load_latest_mark(memory_db, simulation_version=VERSION, as_of=D1) == (
        D1,
        pytest.approx(11_000.0),
        pytest.approx(11_000.0),
    )
    assert store.load_latest_mark(memory_db, simulation_version=VERSION, as_of=D0) == (
        D0,
        pytest.approx(START),
        pytest.approx(START),
    )
    # Before the first mark there is nothing to return, not a zero mark.
    assert (
        store.load_latest_mark(
            memory_db, simulation_version=VERSION, as_of=date(2025, 12, 31)
        )
        is None
    )

    # ``marked_at`` is derived from ``as_of``, never the wall clock, so a re-run
    # is byte-identical -- see the note on ``_at_midnight``.
    marked_at = memory_db.execute(
        "SELECT marked_at FROM paper_account_marks WHERE as_of = ?", [D1]
    ).fetchone()[0]
    assert marked_at == datetime(2026, 1, 6, tzinfo=UTC)
    assert _assert_rows_fit_ddl(memory_db, "paper_account_marks") == 3


def test_rewriting_a_session_mark_leaves_one_row_with_identical_values(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    """A re-run of a session rewrites its mark rather than minting a second one.

    ``portfolio step`` is re-runnable by design, and the mark id is derived from
    ``(simulation_version, as_of)`` precisely so that a second run of the same
    session cannot leave the account with two contradictory peaks for one day.
    """

    def write() -> None:
        store.write_account_mark(
            memory_db,
            simulation_version=VERSION,
            as_of=D1,
            equity=11_000.0,
            cash=500.0,
            high_water_mark=11_000.0,
            drawdown=0.0,
        )

    write()
    before = memory_db.execute("SELECT * FROM paper_account_marks ORDER BY 1").fetchall()

    write()
    after = memory_db.execute("SELECT * FROM paper_account_marks ORDER BY 1").fetchall()

    assert len(before) == 1
    assert after == before
    assert before[0][0] == store.mark_id(VERSION, D1)


def test_load_account_restores_the_peak_the_fold_can_never_see(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    """The pinned bug: a rebuilt account's high-water mark was the opening deposit.

    ``replay`` folds through ``apply_fill``, and the ratchet lives in
    ``account.mark`` -- which the fold never calls. Only ``DEPOSIT`` lifts the
    mark, so every peak since the opening $10,000 was lost, and the drawdown
    governor on the ``portfolio step`` path measured its drawdown from $10,000
    forever and could never fire.

    Both sides are asserted deliberately. Showing only the restored value would
    be consistent with the fold having been right all along; showing the broken
    $10,000 beside it proves the marks table is what fixed it.
    """
    peak = 15_942.0
    equity = 12_917.0
    orders, fills = _trading_day()
    store.write_genesis(memory_db, cash=START, as_of=D0, simulation_version=VERSION)
    store.write_orders(memory_db, orders, simulation_version=VERSION)
    store.write_fills(memory_db, fills, simulation_version=VERSION)
    store.write_account_mark(
        memory_db,
        simulation_version=VERSION,
        as_of=D2,
        equity=equity,
        cash=8_771.206,
        high_water_mark=peak,
        drawdown=(peak - equity) / peak,
    )

    restored = store.load_account(
        memory_db, simulation_version=VERSION, expected_starting_cash=START
    )
    folded = store.load_account(
        memory_db,
        simulation_version=VERSION,
        expected_starting_cash=START,
        restore_high_water_mark=False,
    )

    # Fixed: the peak the account actually struck.
    assert restored.high_water_mark == pytest.approx(peak, abs=1e-9)
    # Broken: what the fold alone produces, and what the governor used to see.
    assert folded.high_water_mark == pytest.approx(START, abs=1e-9)
    assert restored.high_water_mark > folded.high_water_mark

    # A 19% drawdown that the old code reported as a *gain* over its mark.
    assert (peak - equity) / peak == pytest.approx(0.1898, abs=5e-5)
    assert equity > folded.high_water_mark

    # Only the mark differs: the restore must not disturb the fold's account.
    assert restored.as_of == folded.as_of
    assert restored.cash == pytest.approx(folded.cash, abs=1e-9)
    assert dict(restored.positions) == dict(folded.positions)


def test_account_marks_do_not_leak_across_simulation_versions(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    """A scratch experiment's peak must not become the ledger of record's."""
    store.write_genesis(memory_db, cash=START, as_of=D0, simulation_version=VERSION)
    store.write_genesis(memory_db, cash=START, as_of=D0, simulation_version=SCRATCH)
    store.write_account_mark(
        memory_db,
        simulation_version=VERSION,
        as_of=D1,
        equity=11_000.0,
        cash=11_000.0,
        high_water_mark=11_000.0,
        drawdown=0.0,
    )
    store.write_account_mark(
        memory_db,
        simulation_version=SCRATCH,
        as_of=D2,
        equity=40_000.0,
        cash=40_000.0,
        high_water_mark=99_000.0,
        drawdown=0.5959,
    )

    assert store.load_latest_mark(memory_db, simulation_version=VERSION) == (
        D1,
        pytest.approx(11_000.0),
        pytest.approx(11_000.0),
    )
    of_record = store.load_account(
        memory_db, simulation_version=VERSION, expected_starting_cash=START
    )
    scratch = store.load_account(
        memory_db, simulation_version=SCRATCH, expected_starting_cash=START
    )
    assert of_record.high_water_mark == pytest.approx(11_000.0, abs=1e-9)
    assert scratch.high_water_mark == pytest.approx(99_000.0, abs=1e-9)
    # Both live in one table, isolated only by the version column.
    assert memory_db.execute("SELECT count(*) FROM paper_account_marks").fetchone()[0] == 2


def test_an_unmarked_account_falls_back_to_the_fold_rather_than_raising(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    """A brand-new account has no marks yet, and neither does a pre-table one.

    Both must load. An empty table is a successful read saying "never marked";
    a missing one is an account written before ``paper_account_marks`` shipped.
    Refusing either would strand the ledger of record behind a schema change.
    """
    store.write_genesis(memory_db, cash=START, as_of=D0, simulation_version=VERSION)

    # Present and empty.
    assert store.load_latest_mark(memory_db, simulation_version=VERSION) is None
    unmarked = store.load_account(
        memory_db, simulation_version=VERSION, expected_starting_cash=START
    )
    assert unmarked.high_water_mark == pytest.approx(START, abs=1e-9)

    # Absent entirely.
    memory_db.execute("DROP TABLE paper_account_marks")
    assert store.load_latest_mark(memory_db, simulation_version=VERSION) is None
    pre_table = store.load_account(
        memory_db, simulation_version=VERSION, expected_starting_cash=START
    )
    assert pre_table.high_water_mark == pytest.approx(START, abs=1e-9)
    assert pre_table.cash == pytest.approx(START, abs=1e-9)
