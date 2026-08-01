"""Persistence: the fill ledger, the daily marks, and the experiment chain.

Three write surfaces, all bounded and all idempotent:

  * ``paper_orders`` / ``paper_fills`` -- the account of record. Ids are
    deterministic (``pf:{symbol}:{date}:{side}``) and rows are keyed
    ``UNIQUE(opportunity_id, simulation_version)``, so re-running a step writes
    the same rows rather than duplicating the account. ``simulation_version``
    isolates experiments: a scratch run cannot contaminate the real ledger.

  * ``paper_account_marks`` -- one row per marked session, and the only place
    the high-water mark is durable. The fold cannot rebuild it: the ratchet
    lives in ``account.mark`` and a fill log carries no prices, so
    ``account.replay`` returns an account whose mark is forever the opening
    deposit. ``load_account`` therefore restores the mark from this table.

  * ``experiment_registry`` / ``strategy_versions`` -- a ~50-line adapter over
    the existing hash-chained ``ExperimentRecord``, which already forbids
    calibration after sealing. This module persists the chain; it does not
    reimplement its rules.

Write protocol, non-negotiable given a 7.2 GB single-writer database that the
graph-watchdog contends for every 600 seconds: one bounded write phase,
chunked at <=10k rows, via ``with_db_retry``. No network inside the lock. Bulky
artifacts (per-day weights, trial return matrices) go to Parquet on the SSD,
never into the database.

``load_account`` fails closed on a genesis mismatch. If the stored opening
deposit disagrees with the configured ``starting_cash``, that is two different
accounts wearing one name, and the remedy is a new ``simulation_version`` --
never a silent reconciliation.

**Two shipped-schema facts that shape every row here.**

  1. ``quantity`` is ``BIGINT`` on both tables and predates the decision to
     trade fractional shares. It could not be dropped -- ``migrate_db`` only
     adds columns -- so it stays as a rounded convenience index and
     ``quantity_exact DOUBLE`` carries the authoritative quantity that
     ``load_account`` folds. Nothing in this module reads the ``BIGINT`` back.
     ``symbol`` and ``reason`` are likewise real columns now, so no load-bearing
     value hides in ``fill_assumptions``: that blob is an audit note about what
     the simulated fill assumed, and it deliberately duplicates nothing.
  2. ``paper_fills`` has no ``simulation_version`` column. Isolation therefore
     runs through the parent order: a fill is joined to ``paper_orders`` on
     ``paper_order_id`` and filtered by that table's version. ``write_fills``
     refuses a fill whose parent order is absent, because such a fill would be
     invisible to that join -- an account that silently under-reports is worse
     than a write that fails.

``symbol`` and ``quantity_exact`` arrived by ``ALTER TABLE ADD COLUMN``, which
cannot add ``NOT NULL``, so a row could carry them as NULL. ``_load_fills``
raises on that rather than substituting the rounded ``quantity``: folding a
silently wrong quantity into the account of record is the failure this whole
module exists to prevent.

Every function takes an open connection and does its own chunking. The bounded
retry lives at the caller, as in ``feed.load_feed``: ``with_db_retry`` expects
``fn`` to own a whole connect-use-close cycle so a retry is a clean do-over,
which a half-applied write on an already-open connection is not.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, time
from typing import Any

import duckdb

from market_intelligence.analytics.experiment_registry import ExperimentRecord, ExperimentStage
from market_intelligence.hashing import canonical_json
from market_intelligence.logging_config import get_logger
from market_intelligence.portfolio import account
from market_intelligence.portfolio.account import EQUITY_TOLERANCE, AccountState, Fill, Order, Side
from market_intelligence.storage import duckdb as duckdb_store

_log = get_logger("portfolio.store")

# Matches ``storage.duckdb``'s own batching bound -- see the note above its
# ``_BATCH_CHUNK_ROWS``: one giant VALUES relation drove a live process to
# ~16 GB RSS and had to be killed.
_CHUNK_ROWS = 10_000

# An order is an intent that may go unfilled (see ``account.Order``), so
# "submitted" is the only status truthful for every row at write time.
_ORDER_STATUS = "submitted"

# ``account.genesis_fill`` mints the opening row; this is only its audit label.
_GENESIS_REASON = "genesis deposit"

# ``experiment_registry`` demands three window strings that ``ExperimentRecord``
# does not model. Writing this rather than an empty string keeps "we never
# recorded a discovery window" distinguishable from "the window was blank".
_UNSPECIFIED_WINDOW = "unspecified"

# Kill-switch states that mean *not* engaged. Everything else -- including a
# typo, a renamed state, or a value from a future writer -- counts as engaged.
# The asymmetry is deliberate: an unrecognised state must not read as
# permission to trade.
_KILL_SWITCH_CLEAR_STATES = frozenset({"cleared", "inactive", "released", "resolved"})

# Replay order within one day. Sells settle before buys, so a sale can fund a
# purchase decided on the same date -- the ordering ``simulator.step_day``
# applies in memory, reproduced here so a rebuild folds the same sequence.
_SIDE_RANK = "CASE o.side WHEN 'DEPOSIT' THEN 0 WHEN 'SELL' THEN 1 ELSE 2 END"

# ``paper_account_marks``' natural key, matching the table's own UNIQUE
# constraint: one mark per session per experiment.
_ACCOUNT_MARK_KEY: tuple[str, ...] = ("simulation_version", "as_of")

# Written explicitly rather than inferred from the row, so a column added to the
# table without a value here fails the write instead of being filled with NULL.
_ACCOUNT_MARK_COLUMNS: tuple[str, ...] = (
    "mark_id",
    "simulation_version",
    "as_of",
    "equity",
    "cash",
    "high_water_mark",
    "drawdown",
    "marked_at",
)


class GenesisMismatchError(RuntimeError):
    """Stored opening deposit disagrees with config. Bump ``simulation_version``."""


def _natural_key(symbol: str, as_of: date, side: str) -> str:
    """``{symbol}:{iso date}:{side}`` -- one order/fill pair's identity.

    Kept in one place because ``account.genesis_fill`` builds the same string
    for the opening row, and the two must not drift.
    """
    return f"{symbol}:{as_of.isoformat()}:{side}"


def order_id(symbol: str, as_of: date, side: str, simulation_version: str) -> str:
    """``po:{symbol}:{iso date}:{side}:{simulation_version}``. Version is a SUFFIX.

    Deterministic so a re-run rewrites rather than duplicates.
    """
    return f"po:{_natural_key(symbol, as_of, side)}:{simulation_version}"


def fill_id(symbol: str, as_of: date, side: str, simulation_version: str) -> str:
    """``pf:{symbol}:{iso date}:{side}:{simulation_version}``. Version is a SUFFIX.

    The suffix position is fixed by ``account.genesis_fill``, which mints the
    opening row's id before this module ever sees it. Prefixing instead would
    make the genesis row's id disagree with every other row's, and
    ``paper_fills`` is keyed for idempotent re-runs -- so the two would each
    write their own "opening" deposit and the account would silently double its
    starting capital on the second run.
    """
    return f"pf:{_natural_key(symbol, as_of, side)}:{simulation_version}"


def _at_midnight(day: date) -> datetime:
    """The day as a UTC timestamp.

    Deterministic on purpose: a wall-clock stamp would make every re-run write
    a different ``submitted_at``/``filled_at``, and ``filled_at`` is half of
    ``paper_fills``' natural key -- a clock in there is a duplicated ledger.
    """
    return datetime.combine(day, time.min, tzinfo=UTC)


def _chunks[T](items: Sequence[T], size: int = _CHUNK_ROWS) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def write_genesis(
    con: duckdb.DuckDBPyConnection, *, cash: float, as_of: date, simulation_version: str
) -> None:
    """Write the opening DEPOSIT fill. Idempotent.

    **Must call ``account.genesis_fill(cash, as_of, simulation_version)`` and
    persist what it returns** rather than constructing the row here. The
    genesis fill's shape is a real convention -- ``symbol="CASH"``,
    ``shares=cash``, ``price=1.0``, so ``shares * price`` is the amount with no
    special case in the fold -- and two independent constructions of it will
    drift. ``load_account`` compares the stored deposit against
    ``expected_starting_cash``, so a drift here surfaces as
    ``GenesisMismatchError`` rather than as a wrong account.

    The parent ``paper_orders`` row is derived from that same returned fill,
    not built alongside it, because ``load_account`` reaches every fill through
    its order and a genesis fill without one would be invisible.

    Idempotent means *the same* genesis, not *another* one. Re-running with an
    identical ``cash``/``as_of`` rewrites one row and changes nothing. Calling
    it a second time with a different date or amount raises: the fill id embeds
    the date, so a second deposit lands under a new id, the fold sums both, and
    the account is silently funded twice -- $10,000 becomes $20,000 with every
    row individually valid and nothing to notice. The remedy is a new
    ``simulation_version``, never a second deposit.
    """
    existing = _deposit_ids(con, simulation_version)
    opening = account.genesis_fill(cash, as_of, simulation_version)
    stray = existing - {opening.fill_id}
    if stray:
        raise GenesisMismatchError(
            f"simulation_version {simulation_version!r} is already funded by "
            f"{sorted(stray)!r}; writing {opening.fill_id!r} on top of it would fold two "
            f"deposits into one account and double its capital. Use a new "
            f"simulation_version rather than re-opening this one"
        )
    opening_order = Order(
        order_id=opening.order_id,
        symbol=opening.symbol,
        side=opening.side,
        shares=opening.shares,
        as_of=opening.as_of,
        reason=_GENESIS_REASON,
    )
    write_orders(con, (opening_order,), simulation_version=simulation_version)
    write_fills(con, (opening,), simulation_version=simulation_version)


def _order_row(order: Order, simulation_version: str) -> dict[str, Any]:
    return {
        "paper_order_id": order.order_id,
        "opportunity_id": _natural_key(order.symbol, order.as_of, order.side),
        "symbol": order.symbol,
        "side": str(order.side),
        # A rounded index kept only because the BIGINT column cannot be dropped.
        "quantity": round(order.shares),
        "quantity_exact": order.shares,
        "reason": order.reason,
        # Every simulated order executes at the open; there is no limit to record.
        "limit_price": None,
        "submitted_at": _at_midnight(order.as_of),
        "status": _ORDER_STATUS,
        "simulation_version": simulation_version,
    }


def _fill_row(fill: Fill, simulation_version: str) -> dict[str, Any]:
    # Every value the fold needs now has a typed column, so this blob records
    # only what a reader cannot see in one: which run produced the row, and why
    # the borrow/slippage columns are zero. Duplicating symbol or quantity here
    # would create a second authority that can drift from the first.
    assumptions = {
        "simulation_version": simulation_version,
        "costs": "slippage and borrow already folded into fill_price and fees",
    }
    return {
        "paper_fill_id": fill.fill_id,
        "paper_order_id": fill.order_id,
        "symbol": fill.symbol,
        "fill_price": fill.price,
        # A rounded index kept only because the BIGINT column cannot be dropped.
        "quantity": round(fill.shares),
        # Authoritative. This is the column ``load_account`` folds.
        "quantity_exact": fill.shares,
        "filled_at": _at_midnight(fill.as_of),
        "fees": fill.commission,
        # The simulator folds slippage and borrow into the executed price and
        # the commission; recording them again here would double-count.
        "borrow_cost": 0.0,
        "slippage": 0.0,
        # Sorted keys, so a re-run produces a byte-identical blob and the
        # idempotent rewrite really is a no-op.
        "fill_assumptions": canonical_json(assumptions),
    }


def write_orders(
    con: duckdb.DuckDBPyConnection, orders: tuple[Order, ...], *, simulation_version: str
) -> int:
    """Chunked idempotent upsert into ``paper_orders``. Returns rows written.

    "Rows written" is inserted + updated: a second run reports the same count
    as the first, so a caller cannot read idempotency as "nothing persisted".
    """
    if not orders:
        return 0
    rows = [_order_row(order, simulation_version) for order in orders]
    written = 0
    for chunk in _chunks(rows):
        written += duckdb_store.upsert_paper_orders(con, chunk).total
    return written


def _missing_parent_orders(
    con: duckdb.DuckDBPyConnection, order_ids: Sequence[str], simulation_version: str
) -> list[str]:
    """Order ids in this batch with no ``paper_orders`` row in this version."""
    wanted = sorted(set(order_ids))
    missing: list[str] = []
    for chunk in _chunks(wanted):
        placeholders = ", ".join("?" for _ in chunk)
        found = {
            row[0]
            for row in con.execute(
                "SELECT paper_order_id FROM paper_orders "
                f"WHERE simulation_version = ? AND paper_order_id IN ({placeholders})",
                [simulation_version, *chunk],
            ).fetchall()
        }
        missing.extend(order_id_ for order_id_ in chunk if order_id_ not in found)
    return missing


def write_fills(
    con: duckdb.DuckDBPyConnection, fills: tuple[Fill, ...], *, simulation_version: str
) -> int:
    """Chunked idempotent upsert into ``paper_fills``. Returns rows written.

    Raises when a fill's parent order is not already in ``paper_orders`` for
    this ``simulation_version``. ``paper_fills`` carries no version column, so
    ``load_account`` reaches fills only through that join: an orphaned fill
    would be written successfully and then silently vanish from the account.
    Write the orders first.
    """
    if not fills:
        return 0
    missing = _missing_parent_orders(con, [fill.order_id for fill in fills], simulation_version)
    if missing:
        raise ValueError(
            f"{len(missing)} fill(s) have no paper_orders row under simulation_version "
            f"{simulation_version!r} and would be invisible to load_account; "
            f"write the orders first. First missing: {missing[:3]}"
        )
    rows = [_fill_row(fill, simulation_version) for fill in fills]
    written = 0
    for chunk in _chunks(rows):
        written += duckdb_store.upsert_paper_fills(con, chunk).total
    return written


def _deposit_ids(con: duckdb.DuckDBPyConnection, simulation_version: str) -> set[str]:
    """Fill ids of every DEPOSIT already recorded for one simulation version.

    Its own query rather than a filter over ``_load_fills`` so that funding a
    fresh account costs one narrow lookup instead of rebuilding a ledger that
    may hold thousands of fills. Missing tables read as "nothing funded yet" --
    the caller is about to create them.
    """
    try:
        rows = con.execute(
            """
            SELECT f.paper_fill_id
            FROM paper_fills f
            JOIN paper_orders o ON o.paper_order_id = f.paper_order_id
            WHERE o.simulation_version = ? AND o.side = ?
            """,
            [simulation_version, str(Side.DEPOSIT)],
        ).fetchall()
    except duckdb.Error:
        return set()
    return {str(row[0]) for row in rows}


def _load_fills(con: duckdb.DuckDBPyConnection, simulation_version: str) -> tuple[Fill, ...]:
    """Stored fills for one version, in the order the fold must see them."""
    rows = con.execute(
        f"""
        SELECT f.paper_fill_id, f.paper_order_id, f.symbol, f.quantity_exact,
               f.fill_price, f.fees, f.filled_at, o.side
        FROM paper_fills f
        JOIN paper_orders o ON o.paper_order_id = f.paper_order_id
        WHERE o.simulation_version = ?
        ORDER BY f.filled_at, {_SIDE_RANK}, f.paper_fill_id
        """,
        [simulation_version],
    ).fetchall()
    loaded: list[Fill] = []
    for pf_id, po_id, symbol, quantity_exact, price, fees, filled_at, side in rows:
        if symbol is None or quantity_exact is None:
            # Added by ALTER TABLE, so NULL is representable. Falling back to
            # the rounded ``quantity`` would fold a wrong number into the
            # account of record silently, which is the one outcome worse than
            # refusing to rebuild at all.
            raise ValueError(
                f"paper_fills row {pf_id!r} has a NULL symbol or quantity_exact and cannot "
                f"be folded; the account of record will not be rebuilt from a guess"
            )
        loaded.append(
            Fill(
                fill_id=str(pf_id),
                order_id=str(po_id),
                symbol=str(symbol),
                side=Side(str(side)),
                # The authoritative column, never the rounded BIGINT.
                shares=float(quantity_exact),
                price=float(price),
                commission=float(fees),
                as_of=filled_at.date(),
            )
        )
    return tuple(loaded)


def mark_id(simulation_version: str, as_of: date) -> str:
    """``pm:{simulation_version}:{iso date}``. Version is a PREFIX here.

    Unlike ``order_id``/``fill_id``, which carry the version as a suffix because
    ``account.genesis_fill`` fixed that shape before this module existed, a mark
    has no second author. Its id is derived from exactly the two columns of the
    table's UNIQUE constraint, so a re-run of a session rewrites its own row and
    cannot mint a second mark for the same day.
    """
    return f"pm:{simulation_version}:{as_of.isoformat()}"


def write_account_mark(
    con: duckdb.DuckDBPyConnection,
    *,
    simulation_version: str,
    as_of: date,
    equity: float,
    cash: float,
    high_water_mark: float,
    drawdown: float,
) -> None:
    """Persist one session's mark. Idempotent on ``(simulation_version, as_of)``.

    Must be called on **every** session the account is stepped, including
    sessions where nothing traded: the high-water mark is a function of marked
    equity, so it moves whether or not a fill happens, and a marks table that
    only records fill days loses every peak struck between rebalances.

    ``marked_at`` is derived from ``as_of`` rather than the wall clock, matching
    ``submitted_at``/``filled_at``. This module is deliberately clock-free so a
    re-run is byte-identical to the run it repeats; a timestamp that moved would
    turn each idempotent rewrite into a visible change.
    """
    row = {
        "mark_id": mark_id(simulation_version, as_of),
        "simulation_version": simulation_version,
        "as_of": as_of,
        "equity": float(equity),
        "cash": float(cash),
        "high_water_mark": float(high_water_mark),
        "drawdown": float(drawdown),
        "marked_at": _at_midnight(as_of),
    }
    duckdb_store.upsert(
        con,
        "paper_account_marks",
        [row],
        key_cols=_ACCOUNT_MARK_KEY,
        columns=_ACCOUNT_MARK_COLUMNS,
    )


def load_latest_mark(
    con: duckdb.DuckDBPyConnection, *, simulation_version: str, as_of: date | None = None
) -> tuple[date, float, float] | None:
    """Most recent ``(as_of, equity, high_water_mark)`` at or before ``as_of``.

    ``None`` when this version has never been marked -- which is the honest
    answer for a brand-new account, and the reason ``load_account`` treats it as
    "fall back to the fold" rather than as an error.

    A missing table reads as ``None`` for the same reason ``_deposit_ids``
    tolerates one: an account written before this table shipped has no marks,
    and refusing to load it would strand the ledger of record. This is safe in a
    way ``kill_switch_active``'s fail-closed rule is not, because the fallback
    is a *lower* mark and therefore a smaller reported drawdown only in the
    direction of trading less, never more.
    """
    clause = "" if as_of is None else " AND as_of <= ?"
    params: list[Any] = [simulation_version] if as_of is None else [simulation_version, as_of]
    try:
        row = con.execute(
            f"""
            SELECT as_of, equity, high_water_mark FROM paper_account_marks
            WHERE simulation_version = ?{clause}
            ORDER BY as_of DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
    except duckdb.Error as exc:
        _log.warning(
            "paper_account_marks_unreadable",
            simulation_version=simulation_version,
            error=str(exc),
        )
        return None
    if row is None:
        return None
    return (row[0], float(row[1]), float(row[2]))


def load_account(
    con: duckdb.DuckDBPyConnection,
    *,
    simulation_version: str,
    expected_starting_cash: float,
    fractional: bool = True,
    restore_high_water_mark: bool = True,
) -> AccountState:
    """Rebuild the account by folding ``account.replay`` over stored fills.

    Raises ``GenesisMismatchError`` when the stored deposit disagrees with
    ``expected_starting_cash``.

    The fold alone cannot produce the high-water mark. ``replay`` folds through
    ``apply_fill``, and the ratchet lives in ``account.mark`` -- a function of
    *marked* equity, which a fill log has no prices to compute. ``apply_fill``
    lifts the mark only on a ``DEPOSIT``, so a reconstructed account's mark was
    always the opening deposit and every peak since was lost. That is not
    cosmetic: ``portfolio step`` trades from precisely this reconstructed
    account, so its drawdown governor measured every drawdown from $10,000 and
    could never fire. ``restore_high_water_mark`` (default on) replaces the
    folded mark with the last one ``portfolio step`` persisted.

    Set it to ``False`` only to observe the fold's own answer -- the tests use
    it to prove the restore is doing the work.
    """
    fills = _load_fills(con, simulation_version)
    remedy = (
        f"the remedy is a new simulation_version, not a reconciliation "
        f"(current simulation_version={simulation_version!r})"
    )
    if not fills or fills[0].side is not Side.DEPOSIT:
        opened_with = "an empty ledger" if not fills else f"a {fills[0].side} fill"
        raise GenesisMismatchError(
            f"no opening DEPOSIT for simulation_version {simulation_version!r}: the ledger "
            f"begins with {opened_with}. An unfunded account and an unwritten one are the "
            f"same picture, and neither is safe to trade from; {remedy}"
        )
    extra_deposits = [fill.fill_id for fill in fills[1:] if fill.side is Side.DEPOSIT]
    if extra_deposits:
        # Defence in depth behind write_genesis's own guard: this catches a
        # second deposit however it arrived -- a direct INSERT, a restored
        # backup, a future writer -- because the fold would otherwise sum them
        # into an account holding capital that was never deposited.
        raise GenesisMismatchError(
            f"simulation_version {simulation_version!r} holds {len(extra_deposits) + 1} "
            f"DEPOSIT fills ({extra_deposits!r} beyond the opening one). Folding them sums "
            f"into capital this account was never funded with; {remedy}"
        )
    genesis = fills[0]
    stored_cash = genesis.shares * genesis.price - genesis.commission
    if abs(stored_cash - expected_starting_cash) > EQUITY_TOLERANCE:
        raise GenesisMismatchError(
            f"stored opening deposit {stored_cash!r} disagrees with the configured "
            f"starting_cash {expected_starting_cash!r}: these are two different accounts "
            f"wearing one name, so {remedy}"
        )
    state = account.replay(fills, fractional=fractional)
    if not restore_high_water_mark:
        return state
    latest = load_latest_mark(con, simulation_version=simulation_version)
    if latest is None:
        # Legitimate for an account that has never been stepped: there is no
        # mark yet because no session has been marked. Logged rather than
        # silent, because the same shape appears when the marks table is
        # unreadable, and a governor quietly measuring from the opening deposit
        # is the failure this whole restore exists to prevent.
        _log.info(
            "account_mark_absent_folding_high_water_mark",
            simulation_version=simulation_version,
            high_water_mark=state.high_water_mark,
        )
        return state
    _, _, high_water = latest
    return replace(state, high_water_mark=high_water)


def _jsonable(value: Any) -> Any:
    """Undo ``_freeze_snapshot`` so the value is a plain JSON type.

    ``ExperimentRecord`` stores ``parameters``/``metrics`` as mappingproxy and
    tuples, which ``json.dumps`` cannot encode -- and ``canonical_json``'s
    ``default=str`` would silently stringify the whole mapping instead of
    raising. This mirrors the registry's own ``_plain_snapshot`` exactly, so a
    round-tripped record re-hashes to the value it was written with.
    """
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(item) for item in value]
    return value


def _record_payload(record: ExperimentRecord) -> dict[str, Any]:
    """The whole record as JSON. ``result_json`` is what the chain round-trips."""
    return {
        "record_id": record.record_id,
        "experiment_id": record.experiment_id,
        "strategy_family": record.strategy_family,
        "stage": str(record.stage),
        "recorded_at": record.recorded_at.isoformat(),
        "hypothesis": record.hypothesis,
        "code_hash": record.code_hash,
        "configuration_hash": record.configuration_hash,
        "data_snapshot_hash": record.data_snapshot_hash,
        "cost_model_version": record.cost_model_version,
        "feature_version": record.feature_version,
        "parameters": _jsonable(record.parameters),
        "metrics": _jsonable(record.metrics),
        "parent_record_hash": record.parent_record_hash,
        "sealed_window_start": (
            None if record.sealed_window_start is None else record.sealed_window_start.isoformat()
        ),
        "record_hash": record.record_hash,
    }


def _record_from_payload(payload: Mapping[str, Any]) -> ExperimentRecord:
    """Rebuild a record, passing ``record_hash`` so its own check re-verifies it."""
    sealed = payload["sealed_window_start"]
    return ExperimentRecord(
        record_id=payload["record_id"],
        experiment_id=payload["experiment_id"],
        strategy_family=payload["strategy_family"],
        stage=ExperimentStage(payload["stage"]),
        recorded_at=datetime.fromisoformat(payload["recorded_at"]),
        hypothesis=payload["hypothesis"],
        code_hash=payload["code_hash"],
        configuration_hash=payload["configuration_hash"],
        data_snapshot_hash=payload["data_snapshot_hash"],
        cost_model_version=payload["cost_model_version"],
        feature_version=payload["feature_version"],
        parameters=payload["parameters"],
        metrics=payload["metrics"],
        parent_record_hash=payload["parent_record_hash"],
        sealed_window_start=None if sealed is None else datetime.fromisoformat(sealed),
        record_hash=payload["record_hash"],
    )


def write_experiment_record(con: duckdb.DuckDBPyConnection, record: ExperimentRecord) -> None:
    """Persist one link of the hash chain into ``experiment_registry``.

    The declared table keys on ``experiment_id`` alone, but a chain holds many
    records per experiment, so the *record* id occupies that primary key and
    the experiment id travels inside ``result_json`` with the rest of the
    record. ``result_json`` is the round-trip surface; the typed columns are an
    index over it.

    Append-only, and loudly so: re-writing an identical record is a no-op, but
    a *different* record claiming an existing ``record_id`` raises rather than
    being dropped -- silently discarding it would erase the audit that the
    registry exists to provide.
    """
    payload = canonical_json(_record_payload(record))
    existing = con.execute(
        "SELECT result_json FROM experiment_registry WHERE experiment_id = ?",
        [record.record_id],
    ).fetchone()
    if existing is not None:
        if existing[0] != payload:
            raise ValueError(
                f"experiment_registry already holds a different record under record_id "
                f"{record.record_id!r}; the chain is append-only and immutable"
            )
        return
    row = {
        "experiment_id": record.record_id,
        "strategy_version_id": record.strategy_family,
        "hypothesis": record.hypothesis,
        "discovery_window": _UNSPECIFIED_WINDOW,
        "calibration_window": _UNSPECIFIED_WINDOW,
        "sealed_test_window": (
            _UNSPECIFIED_WINDOW
            if record.sealed_window_start is None
            else record.sealed_window_start.isoformat()
        ),
        # UNIQUE (strategy_version_id, feature_hash): the record hash is unique
        # per link, so the constraint separates chain links instead of
        # collapsing them.
        "feature_hash": record.record_hash,
        "result_json": payload,
        "registered_at": record.recorded_at,
        "completed_at": None,
    }
    duckdb_store.insert_append_only(
        con, "experiment_registry", [row], key_cols=("experiment_id",)
    )


def load_experiment_chain(
    con: duckdb.DuckDBPyConnection, *, family: str
) -> tuple[ExperimentRecord, ...]:
    """Load a family's chain in order. Chain validity is the record's own concern.

    Ordered by ``registered_at`` (the record's ``recorded_at``), which is
    monotone along a lineage by the registry's own rules, with the record id as
    a stable tiebreak. Rebuilding each record re-verifies its ``record_hash``;
    handing the result to ``ExperimentRegistry`` re-verifies the links.
    """
    rows = con.execute(
        """
        SELECT result_json FROM experiment_registry
        WHERE strategy_version_id = ?
        ORDER BY registered_at, experiment_id
        """,
        [family],
    ).fetchall()
    return tuple(_record_from_payload(json.loads(row[0])) for row in rows)


def kill_switch_active(con: duckdb.DuckDBPyConnection) -> bool:
    """Fail-closed: an unreadable kill-switch table means active, not inactive.

    A missing table, a busy lock, a renamed column -- every one of them is a
    read that did not happen, and a safety switch whose unknown state reads as
    "off" is not a safety switch. An empty table is a different thing: it is a
    successful read saying no switch has ever tripped, so trading is permitted.
    """
    try:
        rows = con.execute(
            """
            SELECT e.state, e.resolved_at
            FROM kill_switch_events e
            WHERE e.occurred_at = (
                SELECT max(x.occurred_at) FROM kill_switch_events x
                WHERE x.switch_name = e.switch_name
            )
            """
        ).fetchall()
    # Deliberately broad: every failed read, whatever its class, must fail closed.
    except Exception as exc:
        _log.warning("kill_switch_unreadable_assuming_active", error=str(exc))
        return True
    for state, resolved_at in rows:
        if resolved_at is None and str(state).strip().lower() not in _KILL_SWITCH_CLEAR_STATES:
            return True
    return False


__all__ = [
    "GenesisMismatchError",
    "fill_id",
    "kill_switch_active",
    "load_account",
    "load_experiment_chain",
    "load_latest_mark",
    "mark_id",
    "order_id",
    "write_account_mark",
    "write_experiment_record",
    "write_fills",
    "write_genesis",
    "write_orders",
]
