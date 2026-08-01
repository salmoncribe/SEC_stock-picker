"""Tests for grading open trade alerts in split-adjusted price space.

Offline throughout: ``memory_db`` seeds ``trade_alerts`` rows directly (via
``duckdb_store.insert_new_trade_alerts``, so first-firing-wins semantics still
apply) and ``daily_prices`` bars via ``duckdb_store.upsert_daily_prices``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import duckdb
import pytest

from market_intelligence import database
from market_intelligence.autopilot import orchestrator
from market_intelligence.config import Config
from market_intelligence.signals import grading
from market_intelligence.storage import duckdb as duckdb_store

FIRED = datetime(2026, 6, 1, tzinfo=UTC)
FIRED_DATE = FIRED.date()
EXIT_DATE = FIRED_DATE + timedelta(days=20)


def _alert_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "alert_id": "a1",
        "kind": "reaction_lag",
        "ticker": "NKE",
        "trigger_key": "ev1:self:20",
        "fired_at": FIRED,
        "direction": 1,
        "entry_ref": 100.0,
        "stop": 96.0,
        "target": 103.0,
        "shares": 10,
        "notional": 1000.0,
        "risk_amount": 40.0,
        "time_exit_date": EXIT_DATE,
        "confidence": 70,
        "evidence": "{}",
        "event_id": "ev1",
        "edge_id": "self",
        "delivered": False,
        "delivery_note": None,
        "outcome": "open",
        "outcome_return": None,
        "graded_at": None,
        "unsizeable": False,
        "schema_version": "1.0.0",
    }
    row.update(overrides)
    return row


def _seed_alert(con: duckdb.DuckDBPyConnection, **overrides: Any) -> dict[str, Any]:
    row = _alert_row(**overrides)
    inserted = duckdb_store.insert_new_trade_alerts(con, [row])
    assert len(inserted) == 1
    return row


def _price_row(
    symbol: str,
    price_date: date,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    adj_close: float | None = None,
) -> dict[str, Any]:
    return {
        "price_id": f"{symbol}-{price_date.isoformat()}",
        "symbol": symbol,
        "price_date": price_date,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": adj_close if adj_close is not None else close,
        "volume": 1_000_000,
        "provider": "yfinance",
        "validation_status": "valid",
        "source": "market",
    }


def _seed_prices(con: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> None:
    duckdb_store.upsert_daily_prices(con, rows)


def _row(con: duckdb.DuckDBPyConnection, alert_id: str = "a1") -> tuple[Any, ...]:
    result = con.execute(
        "SELECT outcome, outcome_return, graded_at FROM trade_alerts WHERE alert_id = ?",
        [alert_id],
    ).fetchone()
    assert result is not None
    return result


# --------------------------------------------------------------------------- #
# long: stop / target / conservative tie-break / expiry                      #
# --------------------------------------------------------------------------- #
class TestLongAlert:
    def test_stop_hit_grades_hit_stop_with_negative_return(self, memory_db):
        _seed_alert(memory_db)
        # Fired-day bar: no adjustment (f0 = 1.0).
        _seed_prices(
            memory_db,
            [
                _price_row("NKE", FIRED_DATE, open_=100, high=101, low=99, close=100),
                # Low pierces the stop (96); high stays well under target.
                _price_row(
                    "NKE", FIRED_DATE + timedelta(days=1), open_=99, high=99.5, low=95, close=97
                ),
            ],
        )

        counts = grading.grade_open_alerts(memory_db, as_of=FIRED_DATE + timedelta(days=2))

        assert counts == {"hit_stop": 1, "hit_target": 0, "expired": 0, "still_open": 0}
        outcome, outcome_return, graded_at = _row(memory_db)
        assert outcome == "hit_stop"
        assert outcome_return == pytest.approx((96.0 - 100.0) / 100.0)
        assert outcome_return < 0
        assert graded_at is not None

    def test_target_hit_with_no_stop_touch_grades_hit_target(self, memory_db):
        _seed_alert(memory_db)
        _seed_prices(
            memory_db,
            [
                _price_row("NKE", FIRED_DATE, open_=100, high=101, low=99, close=100),
                _price_row(
                    "NKE", FIRED_DATE + timedelta(days=1), open_=101, high=104, low=100, close=103
                ),
            ],
        )

        counts = grading.grade_open_alerts(memory_db, as_of=FIRED_DATE + timedelta(days=2))

        assert counts == {"hit_stop": 0, "hit_target": 1, "expired": 0, "still_open": 0}
        outcome, outcome_return, graded_at = _row(memory_db)
        assert outcome == "hit_target"
        assert outcome_return == pytest.approx((103.0 - 100.0) / 100.0)
        assert outcome_return > 0
        assert graded_at is not None

    def test_bar_touching_both_grades_hit_stop_conservatively(self, memory_db):
        _seed_alert(memory_db)
        _seed_prices(
            memory_db,
            [
                _price_row("NKE", FIRED_DATE, open_=100, high=101, low=99, close=100),
                # Single bar's range spans both the stop (96) and target (103).
                _price_row(
                    "NKE", FIRED_DATE + timedelta(days=1), open_=100, high=104, low=95, close=100
                ),
            ],
        )

        counts = grading.grade_open_alerts(memory_db, as_of=FIRED_DATE + timedelta(days=2))

        assert counts["hit_stop"] == 1
        assert counts["hit_target"] == 0
        outcome, _, _ = _row(memory_db)
        assert outcome == "hit_stop"

    def test_no_touch_through_exit_date_expires_from_exit_date_close(self, memory_db):
        _seed_alert(memory_db)
        rows = [_price_row("NKE", FIRED_DATE, open_=100, high=101, low=99, close=100)]
        # Every bar through the exit date stays inside the stop/target band.
        d = FIRED_DATE + timedelta(days=1)
        while d <= EXIT_DATE:
            rows.append(_price_row("NKE", d, open_=100, high=101, low=99, close=101))
            d += timedelta(days=1)
        _seed_prices(memory_db, rows)

        counts = grading.grade_open_alerts(memory_db, as_of=EXIT_DATE)

        assert counts == {"hit_stop": 0, "hit_target": 0, "expired": 1, "still_open": 0}
        outcome, outcome_return, graded_at = _row(memory_db)
        assert outcome == "expired"
        assert outcome_return == pytest.approx((101.0 - 100.0) / 100.0)
        assert graded_at is not None

    def test_no_bars_yet_stays_open(self, memory_db):
        _seed_alert(memory_db)

        counts = grading.grade_open_alerts(memory_db, as_of=FIRED_DATE + timedelta(days=5))

        assert counts == {"hit_stop": 0, "hit_target": 0, "expired": 0, "still_open": 1}
        outcome, outcome_return, graded_at = _row(memory_db)
        assert outcome == "open"
        assert outcome_return is None
        assert graded_at is None

    def test_partial_window_before_exit_date_stays_open_not_expired(self, memory_db):
        # Bars stop 3 days before the exit date -- a mid-hold delisting or a
        # provider gap in the tail. as_of is well past the exit date, but
        # nothing in daily_prices proves the ticker's history ever reached
        # time_exit_date, so grading must not fabricate an expiry from a
        # stale last-known close.
        _seed_alert(memory_db)
        rows = [_price_row("NKE", FIRED_DATE, open_=100, high=101, low=99, close=100)]
        d = FIRED_DATE + timedelta(days=1)
        last_bar_date = EXIT_DATE - timedelta(days=3)
        while d <= last_bar_date:
            rows.append(_price_row("NKE", d, open_=100, high=101, low=99, close=101))
            d += timedelta(days=1)
        _seed_prices(memory_db, rows)

        counts = grading.grade_open_alerts(memory_db, as_of=EXIT_DATE + timedelta(days=5))

        assert counts == {"hit_stop": 0, "hit_target": 0, "expired": 0, "still_open": 1}
        outcome, outcome_return, graded_at = _row(memory_db)
        assert outcome == "open"
        assert outcome_return is None
        assert graded_at is None


# --------------------------------------------------------------------------- #
# split-adjusted grading: the whole point of this module                    #
# --------------------------------------------------------------------------- #
class TestSplitAdjustedGrading:
    def test_post_split_raw_lows_do_not_falsely_trigger_a_stop(self, memory_db):
        # Long entry 100 / stop 96 / target 103, fired the day before a future
        # 2:1 split. adj_close on the fired-day bar already reflects that
        # future split (factor f0 = 50/100 = 0.5), so stop_adj = 48,
        # target_adj = 51.5, entry_adj = 50.
        _seed_alert(memory_db, time_exit_date=FIRED_DATE + timedelta(days=20))
        rows = [
            # Fired day: pre-split raw scale, adjusted for the future split.
            _price_row(
                "NKE", FIRED_DATE, open_=100, high=101, low=99, close=100, adj_close=50
            ),
            # Two more pre-split days, same scale/factor, no real touch.
            _price_row(
                "NKE",
                FIRED_DATE + timedelta(days=1),
                open_=100,
                high=101,
                low=98,
                close=100,
                adj_close=50,
            ),
            _price_row(
                "NKE",
                FIRED_DATE + timedelta(days=2),
                open_=100,
                high=101,
                low=98,
                close=100,
                adj_close=50,
            ),
            # Split day: raw prices halve. adj_close == close post-split (this
            # is now the current scale). Raw low ~49 would look like a stop-out
            # against the raw stop (96) with no adjustment at all -- but in
            # adjusted space (factor 1.0) it is 49, still above stop_adj (48).
            _price_row(
                "NKE",
                FIRED_DATE + timedelta(days=3),
                open_=50,
                high=50.5,
                low=49,
                close=50,
                adj_close=50,
            ),
        ]
        _seed_prices(memory_db, rows)

        counts = grading.grade_open_alerts(memory_db, as_of=FIRED_DATE + timedelta(days=3))

        assert counts["hit_stop"] == 0
        outcome, _, _ = _row(memory_db)
        assert outcome != "hit_stop"
        assert outcome == "open"  # no real touch, exit date not yet reached


# --------------------------------------------------------------------------- #
# short alerts: mirrored stop/target logic                                   #
# --------------------------------------------------------------------------- #
class TestShortAlert:
    def test_high_piercing_stop_grades_hit_stop(self, memory_db):
        # Short: entry 100, stop 104 (above), target 97 (below).
        _seed_alert(memory_db, direction=-1, stop=104.0, target=97.0)
        _seed_prices(
            memory_db,
            [
                _price_row("NKE", FIRED_DATE, open_=100, high=101, low=99, close=100),
                _price_row(
                    "NKE", FIRED_DATE + timedelta(days=1), open_=101, high=105, low=100, close=104
                ),
            ],
        )

        counts = grading.grade_open_alerts(memory_db, as_of=FIRED_DATE + timedelta(days=2))

        assert counts == {"hit_stop": 1, "hit_target": 0, "expired": 0, "still_open": 0}
        outcome, outcome_return, _ = _row(memory_db)
        assert outcome == "hit_stop"
        assert outcome_return == pytest.approx(-1 * (104.0 - 100.0) / 100.0)
        assert outcome_return < 0

    def test_low_piercing_target_grades_hit_target(self, memory_db):
        _seed_alert(memory_db, direction=-1, stop=104.0, target=97.0)
        _seed_prices(
            memory_db,
            [
                _price_row("NKE", FIRED_DATE, open_=100, high=101, low=99, close=100),
                _price_row(
                    "NKE", FIRED_DATE + timedelta(days=1), open_=99, high=100, low=96, close=97
                ),
            ],
        )

        counts = grading.grade_open_alerts(memory_db, as_of=FIRED_DATE + timedelta(days=2))

        assert counts == {"hit_stop": 0, "hit_target": 1, "expired": 0, "still_open": 0}
        outcome, outcome_return, _ = _row(memory_db)
        assert outcome == "hit_target"
        assert outcome_return == pytest.approx(-1 * (97.0 - 100.0) / 100.0)
        assert outcome_return > 0


# --------------------------------------------------------------------------- #
# already-graded rows are never re-graded                                    #
# --------------------------------------------------------------------------- #
class TestAlreadyGraded:
    def test_already_graded_row_is_untouched_on_rerun(self, memory_db):
        stamp = datetime(2026, 6, 5, tzinfo=UTC)
        _seed_alert(
            memory_db,
            outcome="hit_stop",
            outcome_return=-0.04,
            graded_at=stamp,
        )
        # Bars that, if this row were (wrongly) re-graded, would flip it to
        # hit_target -- proof the row is genuinely skipped, not coincidentally
        # unchanged.
        _seed_prices(
            memory_db,
            [
                _price_row("NKE", FIRED_DATE, open_=100, high=101, low=99, close=100),
                _price_row(
                    "NKE", FIRED_DATE + timedelta(days=1), open_=101, high=110, low=100, close=105
                ),
            ],
        )

        counts = grading.grade_open_alerts(memory_db, as_of=FIRED_DATE + timedelta(days=2))

        assert counts == {"hit_stop": 0, "hit_target": 0, "expired": 0, "still_open": 0}
        outcome, outcome_return, graded_at = _row(memory_db)
        assert outcome == "hit_stop"
        assert outcome_return == pytest.approx(-0.04)
        assert graded_at == stamp


# --------------------------------------------------------------------------- #
# malformed rows: not enough on the row to grade against                    #
# --------------------------------------------------------------------------- #
class TestMalformedRow:
    def test_null_stop_counts_still_open_and_is_untouched(self, memory_db):
        _seed_alert(memory_db, stop=None)
        _seed_prices(
            memory_db,
            [_price_row("NKE", FIRED_DATE, open_=100, high=101, low=99, close=100)],
        )

        counts = grading.grade_open_alerts(memory_db, as_of=FIRED_DATE + timedelta(days=5))

        assert counts == {"hit_stop": 0, "hit_target": 0, "expired": 0, "still_open": 1}
        outcome, outcome_return, graded_at = _row(memory_db)
        assert outcome == "open"
        assert outcome_return is None
        assert graded_at is None


# --------------------------------------------------------------------------- #
# the grade() pipeline-step wrapper                                          #
# --------------------------------------------------------------------------- #
class TestGradeWrapper:
    def test_updated_counts_only_the_terminal_rows(self, tmp_config: Config) -> None:
        with database.connection(tmp_config.paths.database_path) as con:
            database.init_db(con)
            duckdb_store.insert_new_trade_alerts(con, [_alert_row(alert_id="a1")])
            duckdb_store.insert_new_trade_alerts(
                con,
                [
                    _alert_row(
                        alert_id="a2", ticker="AAPL", trigger_key="ev2:self:20"
                    )
                ],
            )
            _seed_prices(
                con,
                [
                    _price_row("NKE", FIRED_DATE, open_=100, high=101, low=99, close=100),
                    # a1's stop (96) is pierced; a2 (ticker AAPL) has no bars at all yet.
                    _price_row(
                        "NKE",
                        FIRED_DATE + timedelta(days=1),
                        open_=99,
                        high=99.5,
                        low=95,
                        close=97,
                    ),
                ],
            )

        summary = grading.grade(tmp_config, as_of=FIRED_DATE + timedelta(days=2))

        assert summary.collected == 2  # both open alerts examined
        assert summary.updated == 1  # only a1 resolved and was written
        assert summary.stage == {"hit_stop": 1, "hit_target": 0, "expired": 0, "still_open": 1}


# --------------------------------------------------------------------------- #
# orchestrator wiring                                                        #
# --------------------------------------------------------------------------- #
class TestOrchestratorWiring:
    def test_grade_trade_alerts_runs_after_compute_returns(self) -> None:
        names = [step.name for step in orchestrator.default_steps()]

        assert "grade-trade-alerts" in names
        assert "compute-returns" in names
        assert names.index("grade-trade-alerts") > names.index("compute-returns")

    @pytest.mark.skip(
        reason=(
            "build-propagation pulled from default_steps 2026-07-29: an N+1 query "
            "hung the daily run for 3h40m, and the fix uncovered a duplicate-key "
            "collision on insert. Re-enable this test alongside the step once both "
            "are fixed; see orchestrator.default_steps."
        )
    )
    def test_propagation_samples_build_before_evaluate(self) -> None:
        names = [step.name for step in orchestrator.default_steps()]

        assert "build-propagation" in names
        assert names.index("build-dataset") < names.index("build-propagation")
        assert names.index("build-propagation") < names.index("evaluate")

    def test_calibrate_gap_confidence_runs_right_after_grading_and_before_filing_events(
        self,
    ) -> None:
        """Calibration needs today's freshly-graded outcomes (grade-trade-alerts)
        and must land before anything else so a same-run price_gap alert can see
        it -- see gap_calibration.py's module docstring."""
        names = [step.name for step in orchestrator.default_steps()]

        assert "calibrate-gap-confidence" in names
        assert names.index("grade-trade-alerts") < names.index("calibrate-gap-confidence")
        assert names.index("calibrate-gap-confidence") < names.index("sync-filing-events")
