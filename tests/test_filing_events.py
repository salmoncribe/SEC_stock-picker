"""Tests for 8-K item-code events.

The properties that matter here are the ones that silently corrupt a cell if
they break: one event per item code, a neutral direction, and an
``available_time`` that never precedes the moment the filing was public.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from market_intelligence import database
from market_intelligence.collectors import filing_events
from market_intelligence.config import Config
from market_intelligence.schemas.events import DIRECTION_NEUTRAL, EventType
from market_intelligence.storage import duckdb as duckdb_store

CIK = "0001045810"
TICKER = "NVDA"
ACCESSION = "0001045810-24-000100"


def _row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "filing_id": "filing-1",
        "company_id": "company-1",
        "cik": CIK,
        "accession_number": ACCESSION,
        "form": "8-K",
        "filing_date": date(2024, 5, 7),
        "report_date": date(2024, 5, 6),
        "acceptance_time": datetime(2024, 5, 6, 20, 30, tzinfo=UTC),
        "ticker": TICKER,
    }
    base.update(overrides)
    return base


def _build(codes: list[str], **overrides: Any):
    return filing_events.build_events(_row(**overrides), codes, schema_version="1.0.0")


# --------------------------------------------------------------------------- #
# item parsing                                                                 #
# --------------------------------------------------------------------------- #
def test_parses_and_trims_comma_separated_codes() -> None:
    assert filing_events.parse_item_codes(" 1.01, 2.03 ,8.01 ") == ["1.01", "2.03", "8.01"]


def test_blank_item_strings_yield_no_codes() -> None:
    assert filing_events.parse_item_codes("") == []
    assert filing_events.parse_item_codes(None) == []
    assert filing_events.parse_item_codes(" , ") == []


def test_repeated_code_in_one_filing_counts_once() -> None:
    """A formatting quirk must not double a filing's weight in its cell."""
    assert filing_events.parse_item_codes("2.02,2.02,9.01") == ["2.02", "9.01"]


# --------------------------------------------------------------------------- #
# one event per item code                                                      #
# --------------------------------------------------------------------------- #
def test_each_item_code_becomes_its_own_event() -> None:
    records = _build(["1.01", "2.03", "8.01", "9.01"])

    assert len(records) == 4
    assert [r.event_subtype for r in records] == ["1.01", "2.03", "8.01", "9.01"]
    assert all(r.event_type == EventType.FILING_ITEM for r in records)


def test_event_keys_are_unique_and_stable_across_runs() -> None:
    first = _build(["2.02", "9.01"])
    second = _build(["2.02", "9.01"])

    keys = [r.event_key for r in first]
    assert len(set(keys)) == 2
    assert keys == [r.event_key for r in second]
    assert [r.event_id for r in first] == [r.event_id for r in second]


def test_co_items_record_what_else_was_announced() -> None:
    records = _build(["2.02", "9.01"])

    assert records[0].payload["co_items"] == ["9.01"]
    assert records[1].payload["co_items"] == ["2.02"]
    assert records[0].payload["item_count"] == 2


def test_boilerplate_item_is_flagged_but_still_emitted() -> None:
    """9.01 is kept as a co-occurrence diagnostic, not dropped."""
    records = _build(["2.02", "9.01"])

    assert records[0].payload["is_boilerplate"] is False
    assert records[1].payload["is_boilerplate"] is True


def test_unknown_item_code_is_kept_without_a_title() -> None:
    """An unrecognised code is data, not an error -- the gate can judge it."""
    (record,) = _build(["99.99"])

    assert record.event_subtype == "99.99"
    assert record.payload["item_title"] is None


def test_legacy_pre_2004_codes_stay_distinct_from_modern_ones() -> None:
    legacy = _build(["5"])[0]
    modern = _build(["5.02"])[0]

    assert legacy.event_subtype != modern.event_subtype
    assert legacy.event_key != modern.event_key


# --------------------------------------------------------------------------- #
# direction                                                                    #
# --------------------------------------------------------------------------- #
def test_direction_is_neutral_for_every_item_code() -> None:
    """Item codes are topics, not signed actions; the gate discovers the sign.

    Hard-coding a guess (4.02 restatement = bad) would put an unmeasured prior
    into the labels, which is the thing the validation gate exists to prevent.
    """
    records = _build(["4.02", "1.01", "2.02", "5.02"])

    assert {r.direction for r in records} == {DIRECTION_NEUTRAL}
    assert all(r.magnitude is None for r in records)


# --------------------------------------------------------------------------- #
# the two clocks                                                               #
# --------------------------------------------------------------------------- #
def test_available_time_prefers_acceptance_over_filing_date() -> None:
    """An after-hours filing carries the next business day's filing date.

    Using acceptance keeps the event on the day it was actually readable.
    """
    (record,) = _build(
        ["8.01"],
        acceptance_time=datetime(2024, 5, 3, 22, 5, tzinfo=UTC),
        filing_date=date(2024, 5, 6),
    )

    assert record.available_time == datetime(2024, 5, 3, 22, 5, tzinfo=UTC)


def test_available_time_falls_back_to_filing_date() -> None:
    (record,) = _build(["8.01"], acceptance_time=None)

    assert record.available_time == datetime(2024, 5, 7, tzinfo=UTC)


def test_event_time_never_follows_available_time() -> None:
    """The public cannot know something before it happened."""
    for codes in (["2.02"], ["5.02"]):
        (record,) = _build(codes)
        assert record.event_time is not None
        assert record.available_time is not None
        assert record.event_time <= record.available_time


# --------------------------------------------------------------------------- #
# reading the preserved raw store                                              #
# --------------------------------------------------------------------------- #
def _write_submissions(raw_dir: Path, payload: dict[str, Any]) -> None:
    path = raw_dir / "sec" / "submissions" / CIK / "2026-07-22" / "snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _payload(**recent: Any) -> dict[str, Any]:
    return {"filings": {"recent": recent}}


def test_item_index_reads_only_the_requested_forms(tmp_path: Path) -> None:
    _write_submissions(
        tmp_path,
        _payload(
            accessionNumber=[ACCESSION, "0001045810-24-000200"],
            form=["8-K", "10-Q"],
            items=["2.02,9.01", ""],
        ),
    )

    index = filing_events.load_item_index(tmp_path, ("8-K",))

    assert index == {ACCESSION: "2.02,9.01"}


def test_item_index_survives_a_corrupt_snapshot(tmp_path: Path) -> None:
    """One unreadable file must not abort a scan over hundreds of them."""
    bad = tmp_path / "sec" / "submissions" / "0000000000" / "2026-07-22" / "snapshot.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json", encoding="utf-8")
    _write_submissions(
        tmp_path, _payload(accessionNumber=[ACCESSION], form=["8-K"], items=["8.01"])
    )

    assert filing_events.load_item_index(tmp_path, ("8-K",)) == {ACCESSION: "8.01"}


def test_missing_submissions_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    assert filing_events.load_item_index(tmp_path, ("8-K",)) == {}


# --------------------------------------------------------------------------- #
# end to end                                                                   #
# --------------------------------------------------------------------------- #
def _seed(config: Config, *, ticker: str | None = TICKER) -> None:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(
            con,
            [
                {
                    "company_id": "company-1",
                    "cik": CIK,
                    "ticker": ticker,
                    "company_name": "NVIDIA Corporation",
                    "source": "sec",
                }
            ],
        )
        duckdb_store.upsert_filings(
            con,
            [
                {
                    "filing_id": "filing-1",
                    "company_id": "company-1",
                    "cik": CIK,
                    "accession_number": ACCESSION,
                    "form": "8-K",
                    "filing_date": date(2024, 5, 7),
                    "report_date": date(2024, 5, 6),
                    "acceptance_time": datetime(2024, 5, 6, 20, 30, tzinfo=UTC),
                    "validation_status": "valid",
                    "source": "sec",
                }
            ],
        )


def test_sync_writes_one_event_per_item_code(tmp_config: Config) -> None:
    _seed(tmp_config)
    _write_submissions(
        tmp_config.paths.raw_dir,
        _payload(accessionNumber=[ACCESSION], form=["8-K"], items=["2.02,9.01"]),
    )

    summary = filing_events.sync(tmp_config)

    assert summary.status == "success"
    assert summary.collected == 2
    assert summary.inserted == 2
    assert summary.stage["boilerplate_events"] == 1

    with database.connection(tmp_config.paths.database_path) as con:
        rows = con.execute(
            "SELECT event_subtype, ticker, direction FROM events "
            "WHERE event_type = ? ORDER BY event_subtype",
            [EventType.FILING_ITEM],
        ).fetchall()

    assert rows == [("2.02", TICKER, 0), ("9.01", TICKER, 0)]


def test_sync_is_idempotent(tmp_config: Config) -> None:
    _seed(tmp_config)
    _write_submissions(
        tmp_config.paths.raw_dir,
        _payload(accessionNumber=[ACCESSION], form=["8-K"], items=["2.02"]),
    )

    first = filing_events.sync(tmp_config)
    second = filing_events.sync(tmp_config)

    assert (first.inserted, first.updated) == (1, 0)
    assert (second.inserted, second.updated) == (0, 1)

    with database.connection(tmp_config.paths.database_path) as con:
        assert con.execute("SELECT count(*) FROM events").fetchone()[0] == 1


def test_filing_without_item_data_is_counted_not_invented(tmp_config: Config) -> None:
    """No item codes on disk means no events -- never a guessed one."""
    _seed(tmp_config)

    summary = filing_events.sync(tmp_config)

    assert summary.collected == 0
    assert summary.stage["no_item_data"] == 1


def test_issuer_without_a_ticker_is_skipped(tmp_config: Config) -> None:
    """An event the dataset builder can never select is not worth storing."""
    _seed(tmp_config, ticker=None)
    _write_submissions(
        tmp_config.paths.raw_dir,
        _payload(accessionNumber=[ACCESSION], form=["8-K"], items=["2.02"]),
    )

    summary = filing_events.sync(tmp_config)

    assert summary.stage["candidate_filings"] == 0
    assert summary.collected == 0


def test_counts_accumulate_across_flushes(tmp_config: Config) -> None:
    _seed(tmp_config)
    _write_submissions(
        tmp_config.paths.raw_dir,
        _payload(accessionNumber=[ACCESSION], form=["8-K"], items=["1.01,2.02,5.02,8.01,9.01"]),
    )

    summary = filing_events.sync(tmp_config, flush_every=2)

    assert summary.collected == 5
    assert summary.inserted == 5


@pytest.mark.parametrize("code", ["2.02", "5.02", "4.02", "1.05"])
def test_known_codes_carry_a_human_readable_title(code: str) -> None:
    (record,) = _build([code])

    assert record.payload["item_title"] == filing_events.ITEM_TITLES[code]
