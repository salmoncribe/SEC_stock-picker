"""Portfolio CLI tests (offline, throwaway home, never the real database).

The production DuckDB is 7.2 GB, single-writer, and contended every 600
seconds; opening it from a test would block a live collector and be blocked by
one. Every test here points ``MARKET_INTELLIGENCE_HOME`` at ``tmp_path``, so
the CLI builds and reads its own empty database under that directory and the
real one is never touched.

Six things are pinned, each because getting it wrong is silent:

1. ``replay`` completes on a seeded feed, prints its metrics, and -- because
   ``--no-write`` is the default -- leaves ``paper_orders``/``paper_fills``
   empty even though the replay itself executed fills.
2. ``status`` on an unwritten ledger says so instead of printing a zero account
   that looks tradeable.
3. ``status`` reports the high-water mark restored from ``paper_account_marks``
   and the drawdown measured from it, and falls back honestly when a funded
   account has not been marked yet. The fold cannot rebuild that mark, so
   without the restore the governor measures every drawdown from $10,000.
4. ``step`` writes a mark on every session it runs, including one that traded
   nothing -- the case a column on ``paper_fills`` would have missed.
5. ``holdout-read`` refuses without a preregistered chain, and refuses again
   when a chain exists but its recorded gate verdict failed.
6. Every command in the sub-app is actually registered and actually documented
   -- the check that catches a command silently not wired up.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from market_intelligence import database
from market_intelligence.analytics.experiment_registry import ExperimentRecord, ExperimentStage
from market_intelligence.cli import app
from market_intelligence.config import reset_config_cache
from market_intelligence.portfolio import store as portfolio_store
from market_intelligence.portfolio.account import Fill, Order, Side

runner = CliRunner()

#: Every command this sub-app is contracted to expose. A command dropped from
#: the registration list fails ``test_every_portfolio_command_is_registered``
#: rather than quietly disappearing from ``--help``.
EXPECTED_COMMANDS = frozenset(
    {"replay", "stress", "step", "status", "sweep", "study-graph", "holdout-read"}
)

#: Fourteen sessions in May 2020 (weekends excluded, no holiday in the window).
DAYS = [
    date(2020, 5, 1),
    date(2020, 5, 4),
    date(2020, 5, 5),
    date(2020, 5, 6),
    date(2020, 5, 7),
    date(2020, 5, 8),
    date(2020, 5, 11),
    date(2020, 5, 12),
    date(2020, 5, 13),
    date(2020, 5, 14),
    date(2020, 5, 15),
    date(2020, 5, 18),
    date(2020, 5, 19),
    date(2020, 5, 20),
]

#: The signal fires on DAYS[3]. Two sessions of history precede it, so the
#: volatility estimate that ``views_from_signals`` needs exists by the first
#: scheduled rebalance (panel row 5, ``rebalance_days`` = 5).
SIGNAL_DAY = DAYS[3]

#: Deterministic day-over-day moves. Non-zero and non-constant so the sample
#: covariance is estimable and the realized-vol figures are not degenerate.
WIGGLE = (
    0.0,
    0.012,
    -0.008,
    0.015,
    -0.011,
    0.009,
    0.014,
    -0.013,
    0.010,
    -0.006,
    0.011,
    -0.009,
    0.013,
    0.007,
)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the app at a throwaway home with valid test credentials."""
    monkeypatch.setenv("MARKET_INTELLIGENCE_HOME", str(tmp_path))
    monkeypatch.setenv("SEC_USER_AGENT", "TestSuite/0.1 (tests@market-intel.test)")
    monkeypatch.setenv("FRED_API_KEY", "test-fred-key")
    reset_config_cache()
    yield
    reset_config_cache()


def _db_path(home: Path) -> Path:
    return home / "data" / "database" / "market_intelligence.duckdb"


def _prices(base: float, offset: int) -> list[float]:
    """A deterministic price path; ``offset`` de-syncs one symbol from another."""
    price = base
    out: list[float] = []
    for index in range(len(DAYS)):
        price *= 1.0 + WIGGLE[(index + offset) % len(WIGGLE)]
        out.append(round(price, 4))
    return out


def _seed_feed(home: Path) -> None:
    """One admitted, hedge-surviving cell firing on AAA and BBB, plus SPY.

    Mirrors ``tests/test_portfolio_feed.py::_seed_feed``, widened to fourteen
    sessions so the replay reaches a scheduled rebalance and actually executes
    fills -- which is what makes the "``--no-write`` wrote nothing" assertion
    mean something.
    """
    with database.connection(_db_path(home)) as con:
        database.init_db(con)
        con.execute(
            """
            INSERT INTO signal_status (
                signal_id, event_type, event_subtype, edge_type, horizon_days, status,
                confirm_streak, fail_streak, last_verdict, mean_car, hit_rate, n_clusters,
                direction, first_seen_time, schema_version
            ) VALUES ('sig-p', 'insider_transaction', 'P', 'self', 20, 'active',
                      2, 0, 'admitted', 0.02, 0.58, 600, 1, ?, '1.0.0')
            """,
            [datetime(2026, 1, 1, tzinfo=UTC)],
        )
        for index, ticker in enumerate(("AAA", "BBB")):
            con.execute(
                """
                INSERT INTO event_samples (
                    sample_id, event_id, edge_id, event_type, event_subtype, source_ticker,
                    target_ticker, horizon_days, available_on, t0, window_end,
                    forward_abnormal_return, split, schema_version
                ) VALUES (?, ?, 'self', 'insider_transaction', 'P', ?, ?, 20,
                          ?, ?, ?, 0.02, 'discovery', '1.0.0')
                """,
                [f"s{index}", f"e{index}", ticker, ticker, SIGNAL_DAY, SIGNAL_DAY, DAYS[5]],
            )
        for ticker in ("AAA", "BBB"):
            for day in DAYS:
                con.execute(
                    """
                    INSERT INTO daily_returns (
                        return_id, symbol, price_date, total_return, market_return,
                        abnormal_return, beta, method, schema_version
                    ) VALUES (?, ?, ?, 0.01, 0.002, 0.0076, 1.2, 'market_model', '1.0.0')
                    """,
                    [f"r-{ticker}-{day}", ticker, day],
                )
        for offset, (symbol, base) in enumerate((("AAA", 40.0), ("BBB", 25.0), ("SPY", 300.0))):
            for day, close in zip(DAYS, _prices(base, offset * 3), strict=True):
                con.execute(
                    """
                    INSERT INTO daily_prices (
                        price_id, symbol, price_date, open, high, low, close, adj_close,
                        volume, provider, schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1000, 'test', '1.0.0')
                    """,
                    [
                        f"p-{symbol}-{day}",
                        symbol,
                        day,
                        close,
                        close * 1.01,
                        close * 0.99,
                        close,
                        close,
                    ],
                )
        con.execute(
            """
            INSERT INTO company_edges (
                edge_id, edge_key, source_cik, source_ticker, target, target_name,
                target_ticker, edge_type, resolution_status, report_date, schema_version
            ) VALUES ('edge-1', 'key-1', '0000001', 'AAA', 'BBB', 'BBB', 'BBB',
                      'customer', 'resolved', ?, '1.0.0')
            """,
            [date(2020, 1, 1)],
        )


def _row_count(home: Path, table: str) -> int:
    with database.connection(_db_path(home)) as con:
        database.init_db(con)
        row = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()
    return int(row[0]) if row else 0


def _write_failing_chain(home: Path, *, family: str) -> None:
    """A real preregistration whose calibrated child records a FAILED gate."""
    recorded_at = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    common = {
        "experiment_id": f"{family}:test",
        "strategy_family": family,
        "hypothesis": "the declared grid contains a configuration that survives deflation",
        "code_hash": "code-hash",
        "configuration_hash": "config-hash",
        "data_snapshot_hash": "data-hash",
        "cost_model_version": "slippage_bps=5.0",
        "feature_version": family,
    }
    registration = ExperimentRecord(
        record_id=f"{family}:registration:test",
        stage=ExperimentStage.REGISTERED,
        recorded_at=recorded_at,
        sealed_window_start=datetime(2026, 7, 2, 12, 0, tzinfo=UTC),
        parameters={"grid": ["baseline"]},
        **common,
    )
    calibration = ExperimentRecord(
        record_id=f"{family}:calibration:test",
        stage=ExperimentStage.CALIBRATED,
        recorded_at=recorded_at,
        parameters={"grid": ["baseline"]},
        metrics={
            "dsr": 0.41,
            "pbo": 0.63,
            "n_trials": 1,
            "stress_passed": False,
            "gate_passed": False,
        },
        parent_record_hash=registration.record_hash,
        **common,
    )
    with database.connection(_db_path(home)) as con:
        database.init_db(con)
        portfolio_store.write_experiment_record(con, registration)
        portfolio_store.write_experiment_record(con, calibration)


# --------------------------------------------------------------------------- #
# 1. replay: it runs, it reports, and by default it writes nothing              #
# --------------------------------------------------------------------------- #
def test_replay_reports_metrics_and_no_write_is_the_default(tmp_path: Path) -> None:
    """The workhorse completes, prints a metrics table, and touches no ledger.

    The seeded feed is deliberately rich enough to trade: if the replay never
    executed a fill, "nothing was written" would be true for the wrong reason
    and the default would go untested. So the fills produced in memory are
    asserted alongside the zero rows on disk.
    """
    _seed_feed(tmp_path)

    result = runner.invoke(app, ["portfolio", "replay"])

    assert result.exit_code == 0, result.output
    assert "portfolio replay" in result.output
    assert "ending equity" in result.output
    assert "max drawdown" in result.output
    assert "Sharpe" in result.output
    # The gate has judged nothing yet, and the report says so rather than
    # implying a pass. Zero admitted propagation cells is a published result.
    assert "propagation" in result.output.lower()

    # --no-write is the default: the account of record is untouched.
    assert _row_count(tmp_path, "paper_orders") == 0
    assert _row_count(tmp_path, "paper_fills") == 0

    # ... and the evidence went to Parquet on the SSD, not into the database.
    artifacts = tmp_path / "data" / "parquet" / "portfolio"
    summaries = list(artifacts.rglob("summary.json"))
    assert summaries, f"no summary.json under {artifacts}"
    assert list(artifacts.rglob("equity.parquet"))
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["counts"]["fills"] > 0, "the seeded replay never traded: the default is untested"


def test_replay_write_persists_a_mark_so_the_live_governor_can_see_a_peak(
    tmp_path: Path,
) -> None:
    """A replay-created ledger must carry its high-water mark, not just its fills.

    The ratchet lives in ``account.mark`` and a fill log carries no prices, so
    ``account.replay`` alone reconstructs an account whose mark is forever the
    opening deposit. An account materialised by ``replay --write`` and then
    traded by ``portfolio step`` would measure every future decline from
    $10,000 and the drawdown governor would never fire -- the same silent
    disarming this ledger already suffered once, arriving through a second door.
    """
    _seed_feed(tmp_path)

    result = runner.invoke(app, ["portfolio", "replay", "--write"])
    assert result.exit_code == 0, result.output

    assert _row_count(tmp_path, "paper_fills") > 0, "the seeded replay never traded"
    assert _row_count(tmp_path, "paper_account_marks") == 1, (
        "replay --write must persist the closing mark, or the live governor is blind"
    )

    with database.connection(_db_path(tmp_path)) as con:
        database.init_db(con)
        stored_mark = con.execute(
            "SELECT high_water_mark, equity FROM paper_account_marks "
            "WHERE simulation_version = ?",
            ["portfolio-sim-v1"],
        ).fetchone()
        restored = portfolio_store.load_account(
            con, simulation_version="portfolio-sim-v1", expected_starting_cash=10_000.0
        )
        folded = portfolio_store.load_account(
            con,
            simulation_version="portfolio-sim-v1",
            expected_starting_cash=10_000.0,
            restore_high_water_mark=False,
        )

    # Both sides asserted: the restore must be doing the work, not the fold
    # having quietly been fine all along.
    assert folded.high_water_mark == pytest.approx(10_000.0)
    assert restored.high_water_mark == pytest.approx(float(stored_mark[0])), (
        "the restored mark must come from the stored row"
    )
    # The persisted mark is a real peak the replay reached, not the deposit
    # echoed back -- otherwise this test would pass on a ledger that stored
    # nothing useful.
    assert float(stored_mark[0]) >= float(stored_mark[1]), (
        "a mark below its own equity is not a peak"
    )


# --------------------------------------------------------------------------- #
# 2. status: an unwritten ledger is not a zero account                          #
# --------------------------------------------------------------------------- #
def test_status_without_genesis_explains_itself_and_exits_cleanly(tmp_path: Path) -> None:
    """No genesis row: say so, do not raise, and do not invent a $0.00 account.

    An unfunded account and an unwritten one are the same picture and neither
    is safe to trade from, so the one thing this must never print is a balance.
    """
    assert runner.invoke(app, ["init-db"]).exit_code == 0

    result = runner.invoke(app, ["portfolio", "status"])

    assert result.exit_code == 0, result.output
    assert "No account of record" in result.output
    assert "--open-account" in result.output
    assert "$" not in result.output.replace("$10,000.00 deposit", "")
    assert result.exception is None


# --------------------------------------------------------------------------- #
# 2b. status: the high-water mark comes back from the marks table               #
# --------------------------------------------------------------------------- #
#: The ledger of record's version, as declared in ``config/settings.yaml``.
LEDGER_VERSION = "portfolio-sim-v1"

#: A peak struck between rebalances. Well above the opening deposit, because the
#: opening deposit is exactly what the folded mark used to return.
PEAK = 12_625.0


def _seed_account(home: Path, *, marked: bool) -> None:
    """A funded ledger holding 10 AAA at $50, optionally with its peak marked.

    Cash is $9,600 and the holding marks at $500, so equity is $10,100 -- below
    ``PEAK`` by exactly 20%, which is the drawdown ``status`` must report once
    the mark is restored, and 0% if it is not.
    """
    open_day, buy_day, mark_day = DAYS[0], DAYS[1], DAYS[-1]
    with database.connection(_db_path(home)) as con:
        database.init_db(con)
        con.execute(
            """
            INSERT INTO daily_prices (
                price_id, symbol, price_date, open, high, low, close, adj_close,
                volume, provider, schema_version
            ) VALUES ('p-AAA-mark', 'AAA', ?, 50.0, 50.0, 50.0, 50.0, 50.0, 1000,
                      'test', '1.0.0')
            """,
            [mark_day],
        )
        portfolio_store.write_genesis(
            con, cash=10_000.0, as_of=open_day, simulation_version=LEDGER_VERSION
        )
        buy = portfolio_store.fill_id("AAA", buy_day, Side.BUY, LEDGER_VERSION)
        portfolio_store.write_orders(
            con,
            (
                Order(
                    order_id=portfolio_store.order_id("AAA", buy_day, Side.BUY, LEDGER_VERSION),
                    symbol="AAA",
                    side=Side.BUY,
                    shares=10.0,
                    as_of=buy_day,
                    reason="test",
                ),
            ),
            simulation_version=LEDGER_VERSION,
        )
        portfolio_store.write_fills(
            con,
            (
                Fill(
                    fill_id=buy,
                    order_id=portfolio_store.order_id("AAA", buy_day, Side.BUY, LEDGER_VERSION),
                    symbol="AAA",
                    side=Side.BUY,
                    shares=10.0,
                    price=40.0,
                    commission=0.0,
                    as_of=buy_day,
                ),
            ),
            simulation_version=LEDGER_VERSION,
        )
        if marked:
            portfolio_store.write_account_mark(
                con,
                simulation_version=LEDGER_VERSION,
                as_of=mark_day,
                equity=10_100.0,
                cash=9_600.0,
                high_water_mark=PEAK,
                drawdown=(PEAK - 10_100.0) / PEAK,
            )


def test_status_reports_the_restored_high_water_mark_and_its_drawdown(tmp_path: Path) -> None:
    """The peak the account struck, not the deposit the fold would return.

    The fold cannot rebuild the mark -- ``apply_fill`` lifts it only on a
    ``DEPOSIT`` -- so without the marks table this account reports a $10,000
    mark and a 0% drawdown while sitting 20% below its real peak. That is the
    number the drawdown governor reads on the live ``portfolio step`` path.
    """
    _seed_account(tmp_path, marked=True)

    result = runner.invoke(app, ["portfolio", "status"])

    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert f"${PEAK:,.2f}" in result.output  # the restored peak
    assert "20.00%" in result.output  # measured from it, not from the deposit
    assert DAYS[-1].isoformat() in result.output  # the session it was marked on
    # The broken value must not be what is shown.
    assert "$10,000.00" not in result.output


def test_status_on_an_unmarked_ledger_falls_back_and_still_exits_cleanly(
    tmp_path: Path,
) -> None:
    """No mark yet is a legitimate state, not an error.

    An account that has been funded but never stepped has nothing in
    ``paper_account_marks``. ``status`` must fall back to the folded mark and
    say so, rather than raising or implying a peak it cannot substantiate.
    (An entirely *unwritten* ledger is covered by the test above.)
    """
    _seed_account(tmp_path, marked=False)

    result = runner.invoke(app, ["portfolio", "status"])

    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "$10,000.00" in result.output  # the folded mark, honestly labelled
    assert "never marked" in result.output
    assert "0.00%" in result.output


def _marks(home: Path) -> list[tuple[date, float, float]]:
    with database.connection(_db_path(home)) as con:
        database.init_db(con)
        rows = con.execute(
            """
            SELECT as_of, equity, high_water_mark FROM paper_account_marks
            WHERE simulation_version = ? ORDER BY as_of
            """,
            [LEDGER_VERSION],
        ).fetchall()
    return [(row[0], float(row[1]), float(row[2])) for row in rows]


def test_step_marks_the_account_even_on_a_session_that_traded_nothing(
    tmp_path: Path,
) -> None:
    """The mark moves on marking days, not fill days. This is why it is its own table.

    ``DAYS[1]`` is not a scheduled rebalance (``rebalance_days`` is 5) and no
    view fires, so this session writes no fills at all -- and a high-water mark
    recorded only alongside fills would lose every peak struck on a session like
    this one. Two steps are taken so the second proves the write is per-session
    rather than once per account.
    """
    _seed_feed(tmp_path)

    first = runner.invoke(
        app, ["portfolio", "step", "--as-of", DAYS[1].isoformat(), "--open-account", "--no-notify"]
    )
    assert first.exit_code == 0, first.output

    marks = _marks(tmp_path)
    assert [mark[0] for mark in marks] == [DAYS[1]]
    # Nothing traded, and the mark was written anyway.
    assert _row_count(tmp_path, "paper_fills") == 1  # the genesis deposit only
    assert marks[0][2] == pytest.approx(10_000.0)

    second = runner.invoke(
        app, ["portfolio", "step", "--as-of", DAYS[2].isoformat(), "--no-notify"]
    )
    assert second.exit_code == 0, second.output
    assert [mark[0] for mark in _marks(tmp_path)] == [DAYS[1], DAYS[2]]

    # Re-running a session rewrites its own mark rather than adding one.
    repeat = runner.invoke(
        app, ["portfolio", "step", "--as-of", DAYS[2].isoformat(), "--no-notify"]
    )
    assert repeat.exit_code == 0, repeat.output
    assert [mark[0] for mark in _marks(tmp_path)] == [DAYS[1], DAYS[2]]


# --------------------------------------------------------------------------- #
# 3. holdout-read: refusing is the default, success is the exception            #
# --------------------------------------------------------------------------- #
def test_holdout_read_refuses_without_a_chain_and_with_a_failed_gate(tmp_path: Path) -> None:
    """Two refusals, each naming what is missing rather than just saying no."""
    _seed_feed(tmp_path)

    missing = runner.invoke(app, ["portfolio", "holdout-read"])

    assert missing.exit_code != 0
    assert "Refusing to read the sealed holdout" in missing.output
    assert "no preregistered experiment chain" in missing.output

    # A chain now exists -- but its recorded verdict failed every leg, and the
    # existence of a preregistration is not itself permission to read.
    _write_failing_chain(tmp_path, family="portfolio-sim-v1")

    failed = runner.invoke(app, ["portfolio", "holdout-read"])

    assert failed.exit_code != 0
    assert "Refusing to read the sealed holdout" in failed.output
    assert "no preregistered experiment chain" not in failed.output
    assert "DSR" in failed.output
    assert "PBO" in failed.output
    assert "stress suite" in failed.output

    # The same seal guards the split flag, so `replay --split holdout` cannot
    # be used as a side door around `holdout-read`.
    side_door = runner.invoke(app, ["portfolio", "replay", "--split", "holdout"])
    assert side_door.exit_code != 0
    assert "Refusing to read the sealed holdout" in side_door.output


# --------------------------------------------------------------------------- #
# 4. the index: every command wired up, every command documented                #
# --------------------------------------------------------------------------- #
def test_every_portfolio_command_is_registered_and_documented() -> None:
    """Catches the failure mode where a command exists but was never registered."""
    portfolio_group = get_command(app).commands["portfolio"]
    registered = set(portfolio_group.commands)

    assert registered >= EXPECTED_COMMANDS, f"missing: {sorted(EXPECTED_COMMANDS - registered)}"
    assert registered == EXPECTED_COMMANDS, f"unexpected: {sorted(registered - EXPECTED_COMMANDS)}"

    for name, command in sorted(portfolio_group.commands.items()):
        assert (command.help or "").strip(), f"{name} has no help string"
        assert runner.invoke(app, ["portfolio", name, "--help"]).exit_code == 0, name
