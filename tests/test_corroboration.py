"""Tests for counting corroborating insiders on the same ticker."""

from __future__ import annotations

import json
from datetime import date, timedelta

import duckdb

from market_intelligence.signals.corroboration import count_corroborating_people

DAY_ZERO = date(2024, 6, 3)  # a Monday


def _insert_event(
    con: duckdb.DuckDBPyConnection,
    *,
    event_id: str,
    ticker: str,
    t0: date,
    event_subtype: str,
    owner_cik: str,
    owner_relationship: str = "Officer",
) -> None:
    con.execute(
        "INSERT INTO events (event_id, event_type, event_subtype, event_key, payload) "
        "VALUES (?, 'insider_transaction', ?, ?, ?)",
        [
            event_id,
            event_subtype,
            f"key-{event_id}",
            json.dumps({"owner_cik": owner_cik, "owner_relationship": owner_relationship}),
        ],
    )
    con.execute(
        "INSERT INTO event_samples (sample_id, event_id, edge_id, event_type, event_subtype, "
        "target_ticker, horizon_days, t0, split) "
        "VALUES (?, ?, 'self', 'insider_transaction', ?, ?, 20, ?, 'discovery')",
        [f"sample-{event_id}", event_id, event_subtype, ticker, t0],
    )


def test_a_single_qualifying_event_counts_as_one(memory_db):
    _insert_event(memory_db, event_id="e1", ticker="ACME", t0=DAY_ZERO,
                   event_subtype="P", owner_cik="0001")

    assert count_corroborating_people(memory_db, ticker="ACME", t0=DAY_ZERO) == 1


def test_two_distinct_insiders_within_the_window_count_as_two(memory_db):
    _insert_event(memory_db, event_id="e1", ticker="ACME", t0=DAY_ZERO,
                   event_subtype="P", owner_cik="0001")
    _insert_event(memory_db, event_id="e2", ticker="ACME", t0=DAY_ZERO + timedelta(days=5),
                   event_subtype="A", owner_cik="0002")

    t0 = DAY_ZERO + timedelta(days=5)
    assert count_corroborating_people(memory_db, ticker="ACME", t0=t0) == 2


def test_same_insider_transacting_twice_counts_once(memory_db):
    _insert_event(memory_db, event_id="e1", ticker="ACME", t0=DAY_ZERO,
                   event_subtype="P", owner_cik="0001")
    _insert_event(memory_db, event_id="e2", ticker="ACME", t0=DAY_ZERO + timedelta(days=3),
                   event_subtype="M", owner_cik="0001")

    t0 = DAY_ZERO + timedelta(days=3)
    assert count_corroborating_people(memory_db, ticker="ACME", t0=t0) == 1


def test_outside_the_window_does_not_corroborate(memory_db):
    _insert_event(memory_db, event_id="e1", ticker="ACME", t0=DAY_ZERO,
                   event_subtype="P", owner_cik="0001")
    _insert_event(memory_db, event_id="e2", ticker="ACME",
                   t0=DAY_ZERO + timedelta(days=30), event_subtype="A", owner_cik="0002")

    assert count_corroborating_people(
        memory_db, ticker="ACME", t0=DAY_ZERO + timedelta(days=30)
    ) == 1


def test_plain_officer_sale_does_not_corroborate(memory_db):
    """A generic 'S' does not count -- only a TenPercentOwner sale does."""
    _insert_event(memory_db, event_id="e1", ticker="ACME", t0=DAY_ZERO,
                   event_subtype="P", owner_cik="0001")
    _insert_event(memory_db, event_id="e2", ticker="ACME", t0=DAY_ZERO,
                   event_subtype="S", owner_cik="0002", owner_relationship="Officer")

    assert count_corroborating_people(memory_db, ticker="ACME", t0=DAY_ZERO) == 1


def test_ten_percent_owner_sale_does_corroborate(memory_db):
    _insert_event(memory_db, event_id="e1", ticker="ACME", t0=DAY_ZERO,
                   event_subtype="P", owner_cik="0001")
    _insert_event(memory_db, event_id="e2", ticker="ACME", t0=DAY_ZERO,
                   event_subtype="S", owner_cik="0002", owner_relationship="TenPercentOwner")

    assert count_corroborating_people(memory_db, ticker="ACME", t0=DAY_ZERO) == 2


def test_a_different_ticker_does_not_corroborate(memory_db):
    _insert_event(memory_db, event_id="e1", ticker="ACME", t0=DAY_ZERO,
                   event_subtype="P", owner_cik="0001")
    _insert_event(memory_db, event_id="e2", ticker="OTHER", t0=DAY_ZERO,
                   event_subtype="A", owner_cik="0002")

    assert count_corroborating_people(memory_db, ticker="ACME", t0=DAY_ZERO) == 1


def test_no_matching_events_floors_at_one(memory_db):
    assert count_corroborating_people(memory_db, ticker="NOTHING", t0=DAY_ZERO) == 1
