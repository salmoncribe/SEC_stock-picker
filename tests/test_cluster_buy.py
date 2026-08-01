"""Tests for insider cluster-buy detection (stage 3c): the pure windowing
function and the end-to-end derivation against seeded ``insider_transaction``
events.

The load-bearing property under test throughout: **one candidate per
(company, window), never one per underlying purchase** -- the exact
clustered-statistics discipline the base signal-graph design's §6 requires
(see the module docstring of ``analytics/cluster_buy.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from market_intelligence import database, hashing
from market_intelligence.analytics import cluster_buy
from market_intelligence.config import Config
from market_intelligence.schemas.common import Source, utcnow
from market_intelligence.schemas.events import EventRecord, EventType
from market_intelligence.storage import duckdb as duckdb_store

COMPANY_X = "company-x"


def _dt(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=UTC)


def _purchase(**overrides: Any) -> cluster_buy.PurchaseObservation:
    base: dict[str, Any] = {
        "owner_cik": "0002000001",
        "accession_number": "acc-1",
        "available_time": _dt("2024-01-15"),
        "shares": 100.0,
        "dollar_value": 1000.0,
        "company_id": COMPANY_X,
        "cik": "0000000001",
        "ticker": "AAA",
    }
    base.update(overrides)
    return cluster_buy.PurchaseObservation(**base)


# --------------------------------------------------------------------------- #
# pure: compute_cluster_buys                                                  #
# --------------------------------------------------------------------------- #
def test_two_insiders_within_window_form_one_cluster() -> None:
    observations = [
        _purchase(owner_cik="cik-1", available_time=_dt("2024-01-01")),
        _purchase(owner_cik="cik-2", available_time=_dt("2024-01-03")),
    ]
    candidates = cluster_buy.compute_cluster_buys(
        observations, window_days=5, min_distinct_insiders=2
    )
    assert len(candidates) == 1
    assert candidates[0].distinct_insiders == 2
    assert set(candidates[0].owner_ciks) == {"cik-1", "cik-2"}


def test_single_insider_is_not_a_cluster() -> None:
    observations = [_purchase(owner_cik="cik-1", available_time=_dt("2024-01-01"))]
    candidates = cluster_buy.compute_cluster_buys(
        observations, window_days=5, min_distinct_insiders=2
    )
    assert candidates == []


def test_purchases_outside_window_form_separate_clusters() -> None:
    observations = [
        _purchase(owner_cik="cik-1", available_time=_dt("2024-01-01")),
        _purchase(owner_cik="cik-2", available_time=_dt("2024-01-03")),
        _purchase(owner_cik="cik-3", available_time=_dt("2024-03-01")),
        _purchase(owner_cik="cik-4", available_time=_dt("2024-03-02")),
    ]
    candidates = cluster_buy.compute_cluster_buys(
        observations, window_days=5, min_distinct_insiders=2
    )
    assert len(candidates) == 2
    windows = sorted((c.window_start, c.window_end) for c in candidates)
    assert windows[0][0] == _dt("2024-01-01")
    assert windows[1][0] == _dt("2024-03-01")


def test_window_end_is_the_last_purchases_available_time() -> None:
    """The emitted event's clock is the moment the full cluster went public."""
    observations = [
        _purchase(owner_cik="cik-1", available_time=_dt("2024-01-01")),
        _purchase(owner_cik="cik-2", available_time=_dt("2024-01-04")),
    ]
    candidates = cluster_buy.compute_cluster_buys(
        observations, window_days=5, min_distinct_insiders=2
    )
    assert candidates[0].window_end == _dt("2024-01-04")


def test_same_insider_repeated_does_not_count_twice() -> None:
    observations = [
        _purchase(owner_cik="cik-1", accession_number="acc-1", available_time=_dt("2024-01-01")),
        _purchase(owner_cik="cik-1", accession_number="acc-2", available_time=_dt("2024-01-02")),
    ]
    candidates = cluster_buy.compute_cluster_buys(
        observations, window_days=5, min_distinct_insiders=2
    )
    assert candidates == []  # one distinct insider, not a cluster


def test_total_shares_and_dollar_value_sum_across_the_window() -> None:
    observations = [
        _purchase(owner_cik="cik-1", shares=100.0, dollar_value=1000.0,
                   available_time=_dt("2024-01-01")),
        _purchase(owner_cik="cik-2", shares=200.0, dollar_value=2500.0,
                   available_time=_dt("2024-01-02")),
    ]  # fmt: skip
    candidates = cluster_buy.compute_cluster_buys(
        observations, window_days=5, min_distinct_insiders=2
    )
    assert candidates[0].total_shares == 300.0
    assert candidates[0].total_dollar_value == 3500.0


def test_companies_are_clustered_independently() -> None:
    observations = [
        _purchase(company_id="company-x", cik="1", ticker="AAA", owner_cik="cik-1",
                   available_time=_dt("2024-01-01")),
        _purchase(company_id="company-x", cik="1", ticker="AAA", owner_cik="cik-2",
                   available_time=_dt("2024-01-02")),
        _purchase(company_id="company-y", cik="2", ticker="BBB", owner_cik="cik-3",
                   available_time=_dt("2024-01-01")),
        _purchase(company_id="company-y", cik="2", ticker="BBB", owner_cik="cik-4",
                   available_time=_dt("2024-01-02")),
    ]  # fmt: skip
    candidates = cluster_buy.compute_cluster_buys(
        observations, window_days=5, min_distinct_insiders=2
    )
    assert len(candidates) == 2
    assert {c.company_id for c in candidates} == {"company-x", "company-y"}


def test_min_distinct_insiders_threshold_is_configurable() -> None:
    observations = [
        _purchase(owner_cik="cik-1", available_time=_dt("2024-01-01")),
        _purchase(owner_cik="cik-2", available_time=_dt("2024-01-02")),
    ]
    # Two insiders clear a threshold of 2 but not 3.
    assert (
        len(cluster_buy.compute_cluster_buys(observations, window_days=5, min_distinct_insiders=2))
        == 1
    )
    assert (
        cluster_buy.compute_cluster_buys(observations, window_days=5, min_distinct_insiders=3) == []
    )


def test_missing_shares_are_excluded_from_total_but_do_not_crash() -> None:
    observations = [
        _purchase(owner_cik="cik-1", shares=None, dollar_value=None,
                   available_time=_dt("2024-01-01")),
        _purchase(owner_cik="cik-2", shares=100.0, dollar_value=1000.0,
                   available_time=_dt("2024-01-02")),
    ]  # fmt: skip
    candidates = cluster_buy.compute_cluster_buys(
        observations, window_days=5, min_distinct_insiders=2
    )
    assert candidates[0].total_shares == 100.0
    assert candidates[0].total_dollar_value == 1000.0


# --------------------------------------------------------------------------- #
# collector: detect() end to end against seeded events                       #
# --------------------------------------------------------------------------- #
def _insider_purchase_event(
    *,
    event_key: str,
    owner_cik: str,
    available_time: datetime,
    shares: float = 100.0,
    price: float = 10.0,
    company_id: str = COMPANY_X,
    cik: str = "0000000001",
    ticker: str = "AAA",
    code: str = "P",
) -> dict[str, Any]:
    magnitude = shares * price if code == "P" else None
    record = EventRecord(
        event_id=hashing.content_hash("event", EventType.INSIDER_TRANSACTION, event_key),
        event_type=EventType.INSIDER_TRANSACTION,
        event_key=event_key,
        company_id=company_id,
        cik=cik,
        ticker=ticker,
        event_subtype=code,
        accession_number=f"acc-{event_key}",
        event_time=available_time,
        available_time=available_time,
        magnitude=magnitude,
        direction=1,
        payload={"owner_cik": owner_cik, "shares": shares, "price_per_share": price},
        source=Source.SEC,
        schema_version="1.0.0",
        collected_time=utcnow(),
    )
    return record.to_row()


def _seed(config: Config, rows: list[dict[str, Any]]) -> None:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_events(con, rows)


def _cluster_events(config: Config) -> list[dict[str, Any]]:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        rows = con.execute(
            "SELECT event_key, ticker, magnitude, direction, payload FROM events "
            "WHERE event_type = ?",
            [EventType.INSIDER_CLUSTER_BUY],
        ).fetchall()
    cols = ("event_key", "ticker", "magnitude", "direction", "payload")
    return [dict(zip(cols, r, strict=True)) for r in rows]


def test_detect_emits_one_event_for_a_cluster(tmp_config: Config) -> None:
    rows = [
        _insider_purchase_event(
            event_key="SK-1", owner_cik="cik-1", available_time=_dt("2024-01-01")
        ),
        _insider_purchase_event(
            event_key="SK-2", owner_cik="cik-2", available_time=_dt("2024-01-03")
        ),
    ]
    _seed(tmp_config, rows)

    summary = cluster_buy.detect(tmp_config)

    assert summary.status == "success"
    events = _cluster_events(tmp_config)
    assert len(events) == 1
    assert events[0]["ticker"] == "AAA"
    assert events[0]["direction"] == 1


def test_detect_ignores_non_purchase_codes(tmp_config: Config) -> None:
    rows = [
        _insider_purchase_event(
            event_key="SK-1", owner_cik="cik-1", available_time=_dt("2024-01-01"), code="S"
        ),
        _insider_purchase_event(
            event_key="SK-2", owner_cik="cik-2", available_time=_dt("2024-01-02"), code="A"
        ),
    ]
    _seed(tmp_config, rows)

    cluster_buy.detect(tmp_config)

    assert _cluster_events(tmp_config) == []


def test_detect_does_not_inflate_one_cluster_into_many_samples(tmp_config: Config) -> None:
    """The regression this whole module exists to prevent: N purchases in one
    window must become exactly one event, never N events/rows.
    """
    rows = [
        _insider_purchase_event(
            event_key=f"SK-{i}", owner_cik=f"cik-{i}", available_time=_dt("2024-01-01")
        )
        for i in range(10)
    ]
    _seed(tmp_config, rows)

    cluster_buy.detect(tmp_config)

    events = _cluster_events(tmp_config)
    assert len(events) == 1
    assert events[0]["ticker"] == "AAA"


def test_detect_is_idempotent(tmp_config: Config) -> None:
    rows = [
        _insider_purchase_event(
            event_key="SK-1", owner_cik="cik-1", available_time=_dt("2024-01-01")
        ),
        _insider_purchase_event(
            event_key="SK-2", owner_cik="cik-2", available_time=_dt("2024-01-03")
        ),
    ]
    _seed(tmp_config, rows)

    first = cluster_buy.detect(tmp_config)
    second = cluster_buy.detect(tmp_config)

    assert (first.inserted, first.updated) == (1, 0)
    assert (second.inserted, second.updated) == (0, 1)
    assert len(_cluster_events(tmp_config)) == 1


def test_detect_respects_configured_window_and_threshold(tmp_config: Config) -> None:
    rows = [
        _insider_purchase_event(
            event_key="SK-1", owner_cik="cik-1", available_time=_dt("2024-01-01")
        ),
        _insider_purchase_event(
            event_key="SK-2", owner_cik="cik-2", available_time=_dt("2024-01-10")
        ),
    ]
    _seed(tmp_config, rows)

    # 9 days apart: not a cluster under the default 5-day window.
    cluster_buy.detect(tmp_config)
    assert _cluster_events(tmp_config) == []
