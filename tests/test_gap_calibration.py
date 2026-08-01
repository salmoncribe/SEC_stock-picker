"""Tests for gap-alert calibration: bucketing, aggregation, and lookup.

Offline throughout: ``memory_db`` (in-memory DuckDB, schema initialized) seeds
``trade_alerts`` rows directly via ``duckdb_store.insert_new_trade_alerts`` --
same seeding style as ``test_grading.py``, since calibration reads exactly the
rows grading writes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from market_intelligence.signals import gap_calibration
from market_intelligence.storage import duckdb as duckdb_store

FIRED = datetime(2026, 7, 20, tzinfo=UTC)

_NO_CATALYST_5PCT = {"gap_pct": 0.05, "catalyst": "no catalyst"}


def _gap_alert_row(**overrides: Any) -> dict[str, Any]:
    evidence = overrides.pop("evidence", _NO_CATALYST_5PCT)
    row: dict[str, Any] = {
        "alert_id": "g1",
        "kind": "price_gap",
        "ticker": "GAPX",
        "trigger_key": "2026-07-20",
        "fired_at": FIRED,
        "direction": 1,
        "entry_ref": 105.0,
        "stop": 100.0,
        "target": 115.0,
        "shares": 10,
        "notional": 1050.0,
        "risk_amount": 50.0,
        "time_exit_date": None,
        "confidence": 50,
        "evidence": json.dumps(evidence) if evidence is not None else None,
        "event_id": None,
        "edge_id": None,
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


def _seed(con: Any, **overrides: Any) -> None:
    row = _gap_alert_row(**overrides)
    inserted = duckdb_store.insert_new_trade_alerts(con, [row])
    assert len(inserted) == 1


# --------------------------------------------------------------------------- #
# bucketing primitives                                                        #
# --------------------------------------------------------------------------- #
class TestBucketing:
    def test_gap_size_bucket_thresholds(self) -> None:
        assert gap_calibration.gap_size_bucket(0.01) == "small"
        assert gap_calibration.gap_size_bucket(0.0299) == "small"
        assert gap_calibration.gap_size_bucket(0.03) == "medium"
        assert gap_calibration.gap_size_bucket(0.0599) == "medium"
        assert gap_calibration.gap_size_bucket(0.06) == "large"
        assert gap_calibration.gap_size_bucket(0.20) == "large"

    def test_gap_size_bucket_is_magnitude_only(self) -> None:
        assert gap_calibration.gap_size_bucket(-0.01) == gap_calibration.gap_size_bucket(0.01)
        assert gap_calibration.gap_size_bucket(-0.10) == gap_calibration.gap_size_bucket(0.10)

    def test_bucket_id_is_deterministic_and_direction_sensitive(self) -> None:
        a = gap_calibration.bucket_id(1, "small", True)
        assert gap_calibration.bucket_id(1, "small", True) == a
        assert gap_calibration.bucket_id(-1, "small", True) != a
        assert gap_calibration.bucket_id(1, "medium", True) != a
        assert gap_calibration.bucket_id(1, "small", False) != a


# --------------------------------------------------------------------------- #
# recompute                                                                   #
# --------------------------------------------------------------------------- #
class TestRecompute:
    def test_no_graded_rows_is_a_correct_no_op(self, memory_db: Any) -> None:
        """The exact state of production today: 93 open, 0 graded price_gap rows."""
        _seed(memory_db, outcome="open", graded_at=None)

        n = gap_calibration.recompute(memory_db)

        assert n == 0
        assert memory_db.execute(
            "SELECT count(*) FROM gap_calibration_stats"
        ).fetchone()[0] == 0

    def test_hit_target_and_hit_stop_aggregate_into_one_bucket(self, memory_db: Any) -> None:
        _seed(
            memory_db,
            alert_id="g1",
            ticker="A",
            trigger_key="t1",
            direction=1,
            outcome="hit_target",
            outcome_return=0.06,
            graded_at=FIRED,
        )
        _seed(
            memory_db,
            alert_id="g2",
            ticker="B",
            trigger_key="t2",
            direction=1,
            outcome="hit_stop",
            outcome_return=-0.03,
            graded_at=FIRED,
        )

        n = gap_calibration.recompute(memory_db)
        assert n == 1

        rows = gap_calibration.load_all(memory_db)
        assert len(rows) == 1
        row = rows[0]
        assert row.bucket_id == gap_calibration.bucket_id(1, "medium", False)
        assert row.direction == 1
        assert row.gap_size_bucket == "medium"
        assert row.catalyst_present is False
        assert row.n_decisive == 2
        assert row.n_expired == 0
        assert row.hit_rate == pytest.approx(0.5)
        assert row.mean_return == pytest.approx((0.06 - 0.03) / 2)

    def test_expired_counted_separately_and_folded_into_mean_return(
        self, memory_db: Any
    ) -> None:
        _seed(
            memory_db,
            alert_id="g1",
            direction=1,
            outcome="expired",
            outcome_return=0.01,
            graded_at=FIRED,
            evidence={"gap_pct": 0.02, "catalyst": "no catalyst"},
        )

        gap_calibration.recompute(memory_db)
        row = gap_calibration.load_all(memory_db)[0]

        assert row.gap_size_bucket == "small"
        assert row.n_decisive == 0
        assert row.n_expired == 1
        assert row.hit_rate is None  # no decisive outcomes -> NULL, not 0.0
        assert row.mean_return == pytest.approx(0.01)

    def test_catalyst_present_and_absent_bucket_separately(self, memory_db: Any) -> None:
        with_catalyst = {
            "gap_pct": 0.05,
            "catalyst": {"event_id": "e1", "event_type": "insider_transaction"},
        }
        _seed(
            memory_db,
            alert_id="g1",
            trigger_key="t1",
            direction=1,
            outcome="hit_target",
            outcome_return=0.05,
            graded_at=FIRED,
            evidence=_NO_CATALYST_5PCT,
        )
        _seed(
            memory_db,
            alert_id="g2",
            trigger_key="t2",
            direction=1,
            outcome="hit_target",
            outcome_return=0.05,
            graded_at=FIRED,
            evidence=with_catalyst,
        )

        gap_calibration.recompute(memory_db)
        rows = {r.bucket_id: r for r in gap_calibration.load_all(memory_db)}

        assert len(rows) == 2
        assert rows[gap_calibration.bucket_id(1, "medium", False)].n_decisive == 1
        assert rows[gap_calibration.bucket_id(1, "medium", True)].n_decisive == 1

    def test_rerun_replaces_rather_than_accumulates(self, memory_db: Any) -> None:
        _seed(
            memory_db,
            alert_id="g1",
            direction=1,
            outcome="hit_target",
            outcome_return=0.05,
            graded_at=FIRED,
        )
        gap_calibration.recompute(memory_db)
        assert gap_calibration.load_all(memory_db)[0].n_decisive == 1

        # Re-running against an unchanged ledger must not double-count.
        gap_calibration.recompute(memory_db)
        rows = gap_calibration.load_all(memory_db)
        assert len(rows) == 1
        assert rows[0].n_decisive == 1

    def test_still_open_rows_are_never_read(self, memory_db: Any) -> None:
        """outcome='open' rows are excluded by the WHERE clause, not just uncounted."""
        _seed(memory_db, alert_id="g1", trigger_key="t1", outcome="open", graded_at=None)
        _seed(
            memory_db,
            alert_id="g2",
            trigger_key="t2",
            outcome="hit_target",
            outcome_return=0.05,
            graded_at=FIRED,
        )

        gap_calibration.recompute(memory_db)
        rows = gap_calibration.load_all(memory_db)

        assert len(rows) == 1
        assert rows[0].n_decisive == 1  # only g2 counted

    def test_malformed_evidence_is_skipped_not_raised(self, memory_db: Any) -> None:
        _seed(
            memory_db,
            alert_id="g1",
            outcome="hit_target",
            outcome_return=0.05,
            graded_at=FIRED,
        )
        memory_db.execute(
            "UPDATE trade_alerts SET evidence = 'not valid json' WHERE alert_id = 'g1'"
        )

        n = gap_calibration.recompute(memory_db)

        assert n == 0  # nothing to bucket, and nothing raised


# --------------------------------------------------------------------------- #
# lookup_track_record                                                         #
# --------------------------------------------------------------------------- #
class TestLookupTrackRecord:
    def test_empty_table_returns_none(self, memory_db: Any) -> None:
        assert (
            gap_calibration.lookup_track_record(
                memory_db, direction=1, gap_pct=0.05, catalyst_present=False
            )
            is None
        )

    def test_bucket_with_zero_decisive_outcomes_returns_none(self, memory_db: Any) -> None:
        _seed(
            memory_db,
            alert_id="g1",
            direction=1,
            outcome="expired",
            outcome_return=0.01,
            graded_at=FIRED,
        )
        gap_calibration.recompute(memory_db)

        assert (
            gap_calibration.lookup_track_record(
                memory_db, direction=1, gap_pct=0.05, catalyst_present=False
            )
            is None
        )

    def test_bucket_with_decisive_outcomes_returns_hit_rate_and_n(
        self, memory_db: Any
    ) -> None:
        _seed(
            memory_db,
            alert_id="g1",
            trigger_key="t1",
            direction=1,
            outcome="hit_target",
            outcome_return=0.05,
            graded_at=FIRED,
        )
        _seed(
            memory_db,
            alert_id="g2",
            trigger_key="t2",
            direction=1,
            outcome="hit_stop",
            outcome_return=-0.02,
            graded_at=FIRED,
        )
        _seed(
            memory_db,
            alert_id="g3",
            trigger_key="t3",
            direction=1,
            outcome="hit_target",
            outcome_return=0.05,
            graded_at=FIRED,
        )
        gap_calibration.recompute(memory_db)

        result = gap_calibration.lookup_track_record(
            memory_db, direction=1, gap_pct=0.05, catalyst_present=False
        )

        assert result is not None
        hit_rate, n_decisive = result
        assert hit_rate == pytest.approx(2 / 3)
        assert n_decisive == 3

    def test_lookup_is_bucket_specific_not_global(self, memory_db: Any) -> None:
        """A different direction/size/catalyst combination must not leak in."""
        _seed(
            memory_db,
            alert_id="g1",
            direction=1,
            outcome="hit_target",
            outcome_return=0.05,
            graded_at=FIRED,
            evidence=_NO_CATALYST_5PCT,
        )
        gap_calibration.recompute(memory_db)

        # Opposite direction, same size/catalyst -- a different bucket.
        assert (
            gap_calibration.lookup_track_record(
                memory_db, direction=-1, gap_pct=0.05, catalyst_present=False
            )
            is None
        )
        # Same direction/size, catalyst flipped -- also a different bucket.
        assert (
            gap_calibration.lookup_track_record(
                memory_db, direction=1, gap_pct=0.05, catalyst_present=True
            )
            is None
        )
