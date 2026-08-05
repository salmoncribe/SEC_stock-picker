"""The four defects that kept ``build-propagation`` out of the daily loop.

Each test here pins one of them, and each defect is one that fails silently
rather than loudly: a batch the database refuses aborts a multi-hour build, a
full-table scan per flush turns a linear job quadratic, and the remaining two --
edge-dimension lookahead and a ticker-keyed edge join -- produce *answers*,
which is worse than producing nothing.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import duckdb
import pytest

from market_intelligence import database
from market_intelligence.autopilot import briefing as briefing_builder
from market_intelligence.config import Config
from market_intelligence.signals import dataset
from market_intelligence.signals.promotion import LadderStatus
from market_intelligence.storage import duckdb as duckdb_store

NKE_CIK = "0000320187"
ONON_CIK = "0001858681"
LULU_CIK = "0001397187"


def _returns(symbol: str, start: date, values: list[float]) -> list[dict[str, Any]]:
    """A daily abnormal-return series on consecutive calendar days."""
    return [
        {
            "return_id": f"r-{symbol}-{start + timedelta(days=i)}",
            "symbol": symbol,
            "price_date": start + timedelta(days=i),
            "total_return": value,
            "abnormal_return": value,
            "source": "market",
        }
        for i, value in enumerate(values)
    ]


def _event(
    event_id: str,
    ticker: str,
    available: datetime,
    *,
    cik: str | None = None,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_type": "insider_transaction",
        "event_key": f"{ticker}:{event_id}",
        "event_subtype": "P",
        "ticker": ticker,
        "cik": cik,
        "available_time": available,
        "magnitude": 100_000.0,
        "direction": 1,
        "extraction_confidence": 0.9,
    }


def _edge(
    *,
    edge_id: str,
    source_cik: str,
    source_ticker: str,
    target_ticker: str,
    target_cik: str,
    edge_type: str = "competitor",
    report_date: date | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    token = target if target is not None else target_ticker
    return {
        "edge_id": edge_id,
        "edge_key": f"{source_cik}|{token}|{edge_type}",
        "source_cik": source_cik,
        "source_ticker": source_ticker,
        "target": token,
        "target_name": token,
        "target_cik": target_cik,
        "target_ticker": target_ticker,
        "edge_type": edge_type,
        "resolution_status": "resolved",
        "report_date": report_date,
        "times_asserted": 1,
        "source": "derived",
    }


def _seed_propagation(
    config: Config,
    *,
    edges: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    targets: tuple[str, ...] = ("ONON",),
) -> None:
    """NKE (source) events plus a clean return series for each target."""
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_daily_returns(con, _returns("NKE", date(2020, 3, 2), [0.0, 0.0, 0.0]))
        for target in targets:
            duckdb_store.upsert_daily_returns(
                con, _returns(target, date(2020, 3, 2), [0.01, 0.02, -0.01, 0.03])
            )
        duckdb_store.upsert_events(
            con,
            events
            if events is not None
            else [_event("evt-nke-1", "NKE", datetime(2020, 3, 2, tzinfo=UTC), cik=NKE_CIK)],
        )
        duckdb_store.upsert_company_edges(con, edges)


# --------------------------------------------------------------------------- #
# 1. duplicate-key collision                                                   #
# --------------------------------------------------------------------------- #
def test_colliding_sample_rows_collapse_instead_of_reaching_the_database() -> None:
    """Two rows on one natural key become one, and the drop is counted."""
    shared = {"event_id": "evt-1", "edge_id": "customer", "horizon_days": 20}
    rows = [
        {**shared, "target_ticker": "ONON", "sample_id": "a", "forward_abnormal_return": 0.01},
        {**shared, "target_ticker": "ONON", "sample_id": "b", "forward_abnormal_return": 0.02},
        {**shared, "target_ticker": "LULU", "sample_id": "c", "forward_abnormal_return": 0.03},
    ]

    deduped, dropped = dataset._dedupe_samples(rows)

    assert dropped == 1
    assert [row["sample_id"] for row in deduped] == ["b", "c"]  # last wins


def test_dedupe_key_is_the_table_constraint_not_a_copy_of_it() -> None:
    """The collapse key and the upsert key are the same object, so can't drift."""
    assert duckdb_store.EVENT_SAMPLE_KEY == (
        "event_id",
        "edge_id",
        "horizon_days",
        "target_ticker",
    )


def test_duplicate_edges_do_not_raise_a_duplicate_key_error(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two edge rows for one relationship write one sample, not a collision.

    ``_validatable_edges`` is stubbed rather than seeded around: the point is
    that the *build path* survives a duplicated edge list, whatever produced it.
    Left to the database this is a UNIQUE-constraint abort part-way through a
    multi-hour run, with every target after it unwritten.
    """
    _seed_propagation(
        tmp_config,
        edges=[
            _edge(
                edge_id="e1",
                source_cik=NKE_CIK,
                source_ticker="NKE",
                target_ticker="ONON",
                target_cik=ONON_CIK,
                report_date=date(2019, 6, 30),
            )
        ],
    )
    monkeypatch.setattr(
        dataset,
        "_validatable_edges",
        lambda con: [
            ("NKE", "ONON", "competitor", date(2019, 6, 30)),
            ("NKE", "ONON", "competitor", date(2018, 6, 30)),
        ],
    )

    summary = dataset.build_propagation(tmp_config, horizons=(1,))

    assert summary.status == "success"
    assert summary.inserted == 1
    assert summary.deduped == 1
    with database.connection(tmp_config.paths.database_path) as con:
        assert con.execute("SELECT count(*) FROM event_samples").fetchone()[0] == 1


def test_one_relationship_asserted_under_two_target_spellings_is_one_edge(
    tmp_config: Config,
) -> None:
    """``company_edges`` is keyed on the target *name*; the ticker is the edge."""
    _seed_propagation(
        tmp_config,
        edges=[
            _edge(
                edge_id="e1",
                source_cik=NKE_CIK,
                source_ticker="NKE",
                target_ticker="ONON",
                target_cik=ONON_CIK,
                target="On Holding AG",
                report_date=date(2019, 6, 30),
            ),
            _edge(
                edge_id="e2",
                source_cik=NKE_CIK,
                source_ticker="NKE",
                target_ticker="ONON",
                target_cik=ONON_CIK,
                target="On Holding A.G.",
                report_date=date(2018, 6, 30),
            ),
        ],
    )

    with database.connection(tmp_config.paths.database_path) as con:
        edges = dataset._validatable_edges(con)

    # One relationship, dated from the earliest filing that asserted it.
    assert edges == [("NKE", "ONON", "competitor", date(2018, 6, 30))]


# --------------------------------------------------------------------------- #
# 2. batch-scoped key lookup                                                   #
# --------------------------------------------------------------------------- #
def test_existing_key_lookup_reads_only_the_batch_not_the_table(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    """A key absent from the batch is never fetched, however large the table."""
    memory_db.execute(
        """
        INSERT INTO event_samples (sample_id, event_id, edge_id, horizon_days, target_ticker)
        VALUES ('s1', 'evt-1', 'customer', 20, 'ONON'),
               ('s2', 'evt-2', 'customer', 20, 'LULU'),
               ('s3', 'evt-3', 'customer', 20, 'AMD')
        """
    )

    existing = duckdb_store._existing_keys(
        memory_db,
        "event_samples",
        duckdb_store.EVENT_SAMPLE_KEY,
        [("evt-1", "customer", 20, "ONON"), ("evt-9", "customer", 20, "NVDA")],
    )

    assert existing == {("evt-1", "customer", 20, "ONON")}


def test_propagation_write_looks_keys_up_once_per_batch(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One scoped lookup per target flush, sized to that flush -- not per row.

    The per-row alternative is what made ``upsert`` a full scan of
    ``event_samples`` per call: fine while the table was small, and the reason
    propagation could not be re-enabled once its own rows were landing in it.
    """
    _seed_propagation(
        tmp_config,
        targets=("ONON", "LULU"),
        edges=[
            _edge(
                edge_id="e1",
                source_cik=NKE_CIK,
                source_ticker="NKE",
                target_ticker="ONON",
                target_cik=ONON_CIK,
                report_date=date(2019, 6, 30),
            ),
            _edge(
                edge_id="e2",
                source_cik=NKE_CIK,
                source_ticker="NKE",
                target_ticker="LULU",
                target_cik=LULU_CIK,
                report_date=date(2019, 6, 30),
            ),
        ],
    )
    real = duckdb_store._existing_keys
    batch_sizes: list[int] = []

    def counting(con: Any, table: str, key_cols: Any, candidates: Any) -> Any:
        if table == "event_samples":
            batch_sizes.append(len(candidates))
        return real(con, table, key_cols, candidates)

    monkeypatch.setattr(duckdb_store, "_existing_keys", counting)

    summary = dataset.build_propagation(tmp_config, horizons=(1, 2))

    # Two targets, one event, two horizons: two flushes of two rows each. One
    # lookup per flush carrying the whole flush -- not one per row (which would
    # be four calls of one), and not one unscoped read of the table.
    assert summary.inserted == 4
    assert batch_sizes == [2, 2]


# --------------------------------------------------------------------------- #
# 3. point-in-time edge filter                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("report_date", "expected"),
    [
        (date(2019, 6, 30), 1),  # disclosed before the event: usable
        (date(2020, 3, 2), 1),   # disclosed the same day: the boundary, usable
        (date(2020, 3, 3), 0),   # disclosed after the event: lookahead
        (date(2024, 6, 30), 0),  # today's graph explaining 2020: lookahead
        (None, 1),               # no date on record: paired, and counted
    ],
)
def test_edge_pairs_only_with_events_it_predates(
    tmp_config: Config, report_date: date | None, expected: int
) -> None:
    """``edge.report_date <= event.available_time``, boundary included."""
    _seed_propagation(
        tmp_config,
        edges=[
            _edge(
                edge_id="e1",
                source_cik=NKE_CIK,
                source_ticker="NKE",
                target_ticker="ONON",
                target_cik=ONON_CIK,
                report_date=report_date,
            )
        ],
    )

    summary = dataset.build_propagation(tmp_config, horizons=(1,))

    assert summary.inserted == expected
    assert summary.stage["pairs_edge_not_yet_known"] == (0 if expected else 1)
    assert summary.stage["edges_undated"] == (1 if report_date is None else 0)


def test_a_later_event_still_pairs_with_an_edge_a_previous_one_could_not(
    tmp_config: Config,
) -> None:
    """The filter is per pair, not per edge: the edge switches on, mid-history."""
    _seed_propagation(
        tmp_config,
        edges=[
            _edge(
                edge_id="e1",
                source_cik=NKE_CIK,
                source_ticker="NKE",
                target_ticker="ONON",
                target_cik=ONON_CIK,
                report_date=date(2020, 3, 3),
            )
        ],
        events=[
            _event("evt-before", "NKE", datetime(2020, 3, 2, tzinfo=UTC), cik=NKE_CIK),
            _event("evt-after", "NKE", datetime(2020, 3, 3, tzinfo=UTC), cik=NKE_CIK),
        ],
    )

    summary = dataset.build_propagation(tmp_config, horizons=(1,))

    assert summary.inserted == 1
    with database.connection(tmp_config.paths.database_path) as con:
        assert con.execute("SELECT event_id FROM event_samples").fetchall() == [("evt-after",)]


# --------------------------------------------------------------------------- #
# 4. briefing's propagation join                                               #
# --------------------------------------------------------------------------- #
AS_OF = date(2026, 7, 22)


def _seed_alert_world(config: Config, *, edge_source_cik: str) -> None:
    """An active ``customer`` cell, an AMD event, and one edge out of ``AMD``.

    ``edge_source_cik`` decides whether the edge really belongs to the company
    that filed the event, or merely to a company that once used its ticker.
    """
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_signal_status(
            con,
            [
                {
                    "signal_id": "sig-P-customer-20",
                    "event_type": "insider_transaction",
                    "event_subtype": "P",
                    "edge_type": "customer",
                    "horizon_days": 20,
                    "status": LadderStatus.ACTIVE.value,
                    "confirm_streak": 2,
                    "fail_streak": 0,
                    "mean_car": 0.04,
                    "hit_rate": 0.66,
                    "n_clusters": 500,
                    "direction": 1,
                    "last_reason": "seeded",
                }
            ],
        )
        duckdb_store.upsert_events(
            con,
            [
                _event(
                    "evt-amd-1",
                    "AMD",
                    datetime(2026, 7, 22, tzinfo=UTC),
                    cik="0000002488",
                )
            ],
        )
        duckdb_store.upsert_company_edges(
            con,
            [
                _edge(
                    edge_id="edge-AMD-NVDA-customer",
                    source_cik=edge_source_cik,
                    source_ticker="AMD",
                    target_ticker="NVDA",
                    target_cik="0001045810",
                    edge_type="customer",
                )
            ],
        )


def test_a_reused_ticker_does_not_borrow_another_company_s_edges(
    tmp_config: Config,
) -> None:
    """Same ticker, different CIK: not the same company, so not the same graph.

    Under the old ``ce.source_ticker = e.ticker`` join this fired a live alert
    for NVDA off a relationship that belongs to whoever else held the symbol.
    """
    _seed_alert_world(tmp_config, edge_source_cik="0009999999")

    with database.connection(tmp_config.paths.database_path) as con:
        alerts = briefing_builder.event_alerts(con, as_of=AS_OF, lookback_days=4)

    assert [alert.ticker for alert in alerts] == []


def test_the_filer_s_own_edges_still_fire(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive control: matching CIKs still produce the target alert.

    This test is about the CIK-based join, not about whether a 'customer'
    edge is business-approved for production (it isn't, as of 2026-08-04 --
    see signals.approved_signals). Monkeypatching is_approved here isolates
    the join-correctness question this test exists to pin from that separate,
    unrelated concern.
    """
    monkeypatch.setattr(briefing_builder, "is_approved", lambda *args: True)
    _seed_alert_world(tmp_config, edge_source_cik="0000002488")

    with database.connection(tmp_config.paths.database_path) as con:
        alerts = briefing_builder.event_alerts(con, as_of=AS_OF, lookback_days=4)

    assert len(alerts) == 1
    assert alerts[0].ticker == "NVDA"
    assert alerts[0].source_ticker == "AMD"
    assert alerts[0].edge_type == "customer"
