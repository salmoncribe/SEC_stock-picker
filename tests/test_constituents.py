"""Offline tests for the S&P 500 constituents vertical: parsing, reconstruction,
client, and the end-to-end collector.

Network is never touched: the client is exercised through ``httpx.MockTransport``
and the parsing/reconstruction tests read the trimmed HTML fixture under
``tests/fixtures/``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from market_intelligence import database
from market_intelligence.clients.constituents import (
    SP500_INCEPTION_DATE,
    ChangeRow,
    ConstituentsClient,
    CurrentConstituentRow,
    MembershipWindow,
    parse_changes,
    parse_current_constituents,
    reconstruct_membership,
)
from market_intelligence.collectors import constituents as constituents_collector
from market_intelligence.storage import duckdb as duckdb_store

FIXTURES = Path(__file__).parent / "fixtures"
HTML = (FIXTURES / "sp500_constituents_trimmed.html").read_bytes()


def _handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=HTML)


# --------------------------------------------------------------------------- #
# 1. parse_current_constituents                                               #
# --------------------------------------------------------------------------- #
def test_parse_current_constituents_reads_every_row() -> None:
    rows = parse_current_constituents(HTML)

    assert {r.ticker for r in rows} == {"MMM", "AOS", "ABT", "FOXA", "FOX", "META", "Q"}
    assert len(rows) == 7


def test_parse_current_constituents_field_values() -> None:
    by_ticker = {r.ticker: r for r in parse_current_constituents(HTML)}

    mmm = by_ticker["MMM"]
    assert mmm.company_name == "3M"
    assert mmm.cik == "0000066740"
    assert mmm.added_date == date(1957, 3, 4)

    q = by_ticker["Q"]
    assert q.company_name == "Qnity Electronics"
    assert q.added_date == date(2025, 11, 3)


def test_parse_current_constituents_missing_table_raises() -> None:
    with pytest.raises(ValueError, match="constituents"):
        parse_current_constituents(b"<html><body>no tables here</body></html>")


# --------------------------------------------------------------------------- #
# 2. parse_changes                                                             #
# --------------------------------------------------------------------------- #
def test_parse_changes_reads_every_dated_row() -> None:
    rows = parse_changes(HTML)

    # 10 rows in the fixture; the trailing "Unknown"-dated synthetic row must be
    # dropped, not crash the parser or appear as a phantom event.
    assert len(rows) == 9
    assert all(row.effective_date is not None for row in rows)
    assert not any(row.added_ticker == "ZZZZ" for row in rows)


def test_parse_changes_strips_citation_markers() -> None:
    rows = parse_changes(HTML)
    cag_row = next(r for r in rows if r.removed_ticker == "CAG")

    # The fixture's CAG reason carries a "[5]" footnote marker inline; it must not
    # leak into the extracted text.
    assert cag_row.removed_security == "Conagra Brands"
    assert cag_row.effective_date == date(2026, 6, 30)
    assert cag_row.added_ticker is None


def test_parse_changes_reads_add_and_remove_on_one_row() -> None:
    rows = parse_changes(HTML)
    foxa_row = next(r for r in rows if r.added_ticker == "FOXA")

    assert foxa_row.effective_date == date(2019, 3, 19)
    assert foxa_row.added_security == "Fox Corporation"
    assert foxa_row.removed_ticker == "FOXA"
    assert foxa_row.removed_security == "21st Century Fox"


def test_parse_changes_missing_table_raises() -> None:
    with pytest.raises(ValueError, match="changes"):
        parse_changes(b"<html><body>no tables here</body></html>")


# --------------------------------------------------------------------------- #
# 3. reconstruct_membership -- against the trimmed fixture                    #
# --------------------------------------------------------------------------- #
def _windows_for(ticker: str) -> list:
    current = parse_current_constituents(HTML)
    changes = parse_changes(HTML)
    windows = reconstruct_membership(current, changes)
    return [w for w in windows if w.ticker == ticker]


def test_simple_ticker_with_no_changes_events_is_one_open_window() -> None:
    windows = _windows_for("MMM")

    assert len(windows) == 1
    assert windows[0].added_date == date(1957, 3, 4)
    assert windows[0].removed_date is None
    assert windows[0].is_estimated is False
    assert windows[0].cik == "0000066740"


def test_remove_then_rejoin_reconstructs_every_window_in_order() -> None:
    """Ticker Q: Qwest -> QuintilesIMS -> Qnity Electronics, three distinct windows.

    This is the required remove-then-rejoin case: the same ticker symbol has been
    reused by three unrelated companies over time, and the source data records two
    full departures and two later re-additions.
    """
    windows = sorted(_windows_for("Q"), key=lambda w: w.added_date)

    assert len(windows) == 3

    qwest, quintiles, qnity = windows

    # Window 1: Qwest predates the changes table's coverage for this ticker, so its
    # start is the estimated inception floor, not an attested date.
    assert qwest.company_name == "Qwest Communications"
    assert qwest.added_date == SP500_INCEPTION_DATE
    assert qwest.removed_date == date(2011, 3, 31)
    assert qwest.is_estimated is True
    assert "floor" in (qwest.note or "")

    # Window 2: QuintilesIMS, a fully attested closed window from two real events.
    assert quintiles.company_name == "QuintilesIMS"
    assert quintiles.added_date == date(2017, 8, 29)
    assert quintiles.removed_date == date(2017, 11, 15)
    assert quintiles.is_estimated is False
    assert quintiles.note is None

    # Window 3: Qnity Electronics, the still-open membership sourced from the
    # current table (with its CIK), matching the collector's live snapshot.
    assert qnity.company_name == "Qnity Electronics"
    assert qnity.added_date == date(2025, 11, 3)
    assert qnity.removed_date is None
    assert qnity.cik == "0002058873"


def test_same_day_close_and_reopen_under_one_ticker() -> None:
    """FOXA: 21st Century Fox closes and Fox Corporation opens on the same date.

    The tie-break rule (close before open) must produce two windows sharing a
    boundary date, never one inverted or zero-length window.
    """
    windows = sorted(_windows_for("FOXA"), key=lambda w: w.added_date)

    assert len(windows) == 2
    old, new = windows

    assert old.company_name == "21st Century Fox"
    assert old.removed_date == date(2019, 3, 19)
    assert old.is_estimated is True  # no earlier "added" event for FOXA in the fixture

    assert new.company_name == "Fox Corporation (Class A)"
    assert new.added_date == date(2019, 3, 19)
    assert new.removed_date is None
    assert new.cik == "0001754301"


def test_current_table_date_wins_over_a_trailing_changes_event() -> None:
    """FOX has both a closed 2015-2019 window and an open window from 2019.

    The open window's ``added_date`` must come from the current table, not be
    duplicated from the same-date "added" event already consumed while closing the
    prior window.
    """
    windows = sorted(_windows_for("FOX"), key=lambda w: w.added_date)

    assert len(windows) == 2
    assert windows[0].removed_date == date(2019, 3, 19)
    assert windows[1].added_date == date(2019, 3, 19)
    assert windows[1].removed_date is None


def test_leading_removed_with_no_prior_added_is_estimated() -> None:
    """CAG: only ever seen leaving the index in this fixture."""
    windows = _windows_for("CAG")

    assert len(windows) == 1
    window = windows[0]
    assert window.added_date == SP500_INCEPTION_DATE
    assert window.removed_date == date(2026, 6, 30)
    assert window.is_estimated is True
    assert "CAG" in (window.note or "")


def test_undocumented_departure_is_flagged_not_silently_trusted() -> None:
    """FB: added once, never removed, and absent from the current table.

    The window must still be emitted (no removal is invented) but must carry a
    note distinguishing it from a genuinely confirmed open membership.
    """
    windows = _windows_for("FB")

    assert len(windows) == 1
    window = windows[0]
    assert window.added_date == date(2013, 12, 23)
    assert window.removed_date is None
    assert window.is_estimated is False
    assert window.note is not None
    assert "absent from the current constituents table" in window.note


def test_ticker_in_current_with_no_note_is_not_flagged() -> None:
    windows = _windows_for("META")

    assert len(windows) == 1
    assert windows[0].note is None


# --------------------------------------------------------------------------- #
# 4. reconstruct_membership -- synthetic unit cases (not exercised by the      #
#    fixture, so built directly)                                              #
# --------------------------------------------------------------------------- #
def test_duplicate_added_event_without_intervening_removal_keeps_the_earlier() -> None:
    changes = [
        ChangeRow(date(2010, 1, 1), "XYZ", "XYZ Co", None, None),
        ChangeRow(date(2012, 1, 1), "XYZ", "XYZ Co Again", None, None),  # inconsistent source data
        ChangeRow(date(2015, 1, 1), None, None, "XYZ", "XYZ Co"),
    ]

    windows = reconstruct_membership([], changes)

    assert len(windows) == 1
    assert windows[0].added_date == date(2010, 1, 1)
    assert windows[0].removed_date == date(2015, 1, 1)


def test_ordinary_closed_window_from_a_matched_pair() -> None:
    changes = [
        ChangeRow(date(2000, 6, 1), "ABC", "ABC Corp", None, None),
        ChangeRow(date(2005, 6, 1), None, None, "ABC", "ABC Corp"),
    ]

    windows = reconstruct_membership([], changes)

    assert len(windows) == 1
    assert windows[0].added_date == date(2000, 6, 1)
    assert windows[0].removed_date == date(2005, 6, 1)
    assert windows[0].is_estimated is False
    assert windows[0].note is None


def test_ticker_present_only_in_current_table_with_no_events() -> None:
    current = [CurrentConstituentRow("SOLO", "Solo Inc", "0000000123", date(1999, 1, 1))]

    windows = reconstruct_membership(current, [])

    assert len(windows) == 1
    assert windows[0].added_date == date(1999, 1, 1)
    assert windows[0].removed_date is None
    assert windows[0].cik == "0000000123"


# --------------------------------------------------------------------------- #
# 5. ConstituentsClient                                                       #
# --------------------------------------------------------------------------- #
def test_fetch_page_returns_raw_bytes_and_url() -> None:
    seen_agents: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_agents.append(request.headers.get("User-Agent"))
        return _handler(request)

    with ConstituentsClient(transport=httpx.MockTransport(handler)) as client:
        result = client.fetch_page()

    assert result.raw == HTML
    assert "List_of_S%26P_500_companies" in result.url
    assert seen_agents and seen_agents[0]  # a descriptive User-Agent was sent


def test_from_config_builds_a_working_client(tmp_config) -> None:
    transport = httpx.MockTransport(_handler)
    with ConstituentsClient.from_config(tmp_config, transport=transport) as client:
        result = client.fetch_page()

    assert result.raw == HTML


# --------------------------------------------------------------------------- #
# 6. collector: sync()                                                        #
# --------------------------------------------------------------------------- #
def _sync(config, handler=_handler):
    return constituents_collector.sync(config, transport=httpx.MockTransport(handler))


def test_sync_persists_every_reconstructed_window(tmp_config) -> None:
    summary = _sync(tmp_config)

    assert summary.status == "success"
    assert summary.collected == 17
    assert summary.rejected == 0
    assert summary.inserted == 17
    assert summary.updated == 0

    with database.connection(tmp_config.paths.database_path) as con:
        count = con.execute("SELECT count(*) FROM index_constituents").fetchone()[0]
    assert count == 17


def test_sync_raw_html_is_preserved_verbatim(tmp_config) -> None:
    _sync(tmp_config)

    raw_dir = tmp_config.paths.raw_dir / "reference" / "sp500_constituents" / "sp500"
    files = list(raw_dir.glob("*/*.html"))
    assert len(files) == 1
    assert files[0].read_bytes() == HTML


def test_sync_is_idempotent(tmp_config) -> None:
    first = _sync(tmp_config)
    second = _sync(tmp_config)

    assert (first.inserted, first.updated) == (17, 0)
    assert (second.inserted, second.updated) == (0, 17)

    with database.connection(tmp_config.paths.database_path) as con:
        count = con.execute("SELECT count(*) FROM index_constituents").fetchone()[0]
    assert count == 17


def test_a_rejected_window_is_counted_and_not_persisted(tmp_config, monkeypatch) -> None:
    real_reconstruct = constituents_collector.reconstruct_membership
    bad_window = MembershipWindow(
        ticker="",  # validate_constituent rejects a blank ticker
        company_name="Bad Row",
        cik=None,
        added_date=date(2020, 1, 1),
        removed_date=None,
        is_estimated=False,
    )

    def fake_reconstruct(current_rows, changes):
        return [*real_reconstruct(current_rows, changes), bad_window]

    monkeypatch.setattr(constituents_collector, "reconstruct_membership", fake_reconstruct)

    summary = _sync(tmp_config)

    assert summary.collected == 18
    assert summary.rejected == 1
    assert summary.inserted == 17  # the bad row never reaches DuckDB

    with database.connection(tmp_config.paths.database_path) as con:
        count = con.execute("SELECT count(*) FROM index_constituents").fetchone()[0]
    assert count == 17


def test_sync_resolves_company_id_and_cik_by_ticker_join(tmp_config) -> None:
    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_companies(
            con,
            [
                {
                    "company_id": "company-mmm",
                    "ticker": "MMM",
                    "cik": "0000066740",
                    "validation_status": "valid",
                    "source": "sec",
                }
            ],
        )

    _sync(tmp_config)

    with database.connection(tmp_config.paths.database_path) as con:
        company_id, cik = con.execute(
            "SELECT company_id, cik FROM index_constituents WHERE ticker = 'MMM'"
        ).fetchone()
        unmatched_company_id = con.execute(
            "SELECT company_id FROM index_constituents WHERE ticker = 'AOS'"
        ).fetchone()[0]

    assert company_id == "company-mmm"
    assert cik == "0000066740"
    assert unmatched_company_id is None  # no companies-table row for AOS: left NULL


def test_sync_flags_estimated_and_undocumented_windows_in_a_note(tmp_config) -> None:
    summary = _sync(tmp_config)

    assert any("approximated added_date" in note for note in summary.notes)

    with database.connection(tmp_config.paths.database_path) as con:
        status, errors = con.execute(
            "SELECT validation_status, validation_errors FROM index_constituents "
            "WHERE ticker = 'CAG'"
        ).fetchone()
    assert status == "warning"
    assert "floor" in errors
