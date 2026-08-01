"""Tests for the Obsidian projection of a daily briefing.

What matters here is the reader-facing contract: percentages that read exactly
right, activated cells that lead, a note that survives a quiet day and surfaces a
failed run's notes, and a write that a re-run corrects rather than duplicates.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from market_intelligence.autopilot import obsidian
from market_intelligence.autopilot.types import (
    Briefing,
    ChangeKind,
    EventAlert,
    RunStatus,
    SignalChange,
)

AS_OF = date(2026, 7, 22)


def _change(kind: str = ChangeKind.ACTIVATED, **overrides: Any) -> SignalChange:
    base: dict[str, Any] = {
        "event_type": "insider_transaction",
        "event_subtype": "P",
        "edge_type": "self",
        "horizon_days": 20,
        "kind": kind,
        "mean_car": 0.0162,
        "hit_rate": 0.588,
        "n_clusters": 12,
        "reason": "3rd confirmation; holdout held",
    }
    base.update(overrides)
    return SignalChange(**base)


def _alert(**overrides: Any) -> EventAlert:
    base: dict[str, Any] = {
        "ticker": "NVDA",
        "event_type": "filing_item",
        "event_subtype": "2.02",
        "available_on": AS_OF,
        "horizon_days": 20,
        "direction": 1,
        "predicted_car": 0.0162,
        "basis": "active, holdout hit 58.8%",
    }
    base.update(overrides)
    return EventAlert(**base)


def _briefing(**overrides: Any) -> Briefing:
    base: dict[str, Any] = {
        "as_of": AS_OF,
        "run_status": RunStatus.SUCCESS,
        "ingest": {"prices": 127, "filings": 14, "events": 42, "matured": 8},
        "changes": [],
        "alerts": [],
        "active_signals": [],
        "notes": [],
    }
    base.update(overrides)
    return Briefing(**base)


# --------------------------------------------------------------------------- #
# structure and formatting                                                     #
# --------------------------------------------------------------------------- #
def test_render_note_has_every_section_and_frontmatter() -> None:
    note = obsidian.render_note(
        _briefing(changes=[_change()], alerts=[_alert()], active_signals=[_change()])
    )

    assert note.startswith("---\n")
    assert "date: 2026-07-22" in note
    assert "run_status: success" in note
    assert "tags:" in note
    assert "# Briefing -- 2026-07-22" in note
    for heading in ("## Signal changes", "## Alerts", "## Active signals", "## Notes"):
        assert heading in note


def test_percentages_format_a_positive_mean_and_a_hit_rate() -> None:
    note = obsidian.render_note(_briefing(changes=[_change(mean_car=0.0162, hit_rate=0.588)]))

    assert "+1.62%" in note  # 0.0162 -> signed, two decimals
    assert "58.8%" in note  # 0.588 -> one decimal, unsigned


def test_negative_mean_is_signed_and_never_negative_zero() -> None:
    note = obsidian.render_note(
        _briefing(changes=[_change(mean_car=-0.023), _change(mean_car=-0.00001)])
    )

    assert "-2.30%" in note
    assert "-0.00%" not in note
    assert "+0.00%" in note


def test_what_moved_line_mentions_only_present_ingest_keys() -> None:
    note = obsidian.render_note(_briefing(ingest={"prices": 127, "filings": 14}))

    assert "127 new prices" in note
    assert "14 new filings" in note
    assert "events" not in note.split("## Signal changes", 1)[0].split("What moved", 1)[1]


def test_alert_shows_ticker_direction_horizon_and_basis() -> None:
    note = obsidian.render_note(
        _briefing(alerts=[_alert(ticker="NVDA", direction=1, predicted_car=0.0162)])
    )

    assert "**NVDA**" in note
    assert "predicted +1.62% over 20d" in note
    assert "basis: active, holdout hit 58.8%" in note


def test_bearish_alert_uses_direction_for_its_sign() -> None:
    note = obsidian.render_note(_briefing(alerts=[_alert(direction=-1, predicted_car=0.031)]))

    assert "predicted -3.10% over 20d" in note


def test_propagation_alert_names_source_and_edge_type() -> None:
    note = obsidian.render_note(
        _briefing(
            alerts=[
                _alert(
                    ticker="NVDA",
                    source_ticker="AMD",
                    edge_type="customer",
                    basis="active customer edge from AMD, holdout hit 58.8%",
                )
            ]
        )
    )

    assert "**NVDA** from AMD via customer" in note


# --------------------------------------------------------------------------- #
# activated cells lead                                                         #
# --------------------------------------------------------------------------- #
def test_activated_changes_appear_before_other_kinds() -> None:
    note = obsidian.render_note(
        _briefing(
            changes=[
                _change(ChangeKind.DEMOTED, event_subtype="S"),
                _change(ChangeKind.NEW_CANDIDATE, event_subtype="A"),
                _change(ChangeKind.ACTIVATED, event_subtype="P"),
            ]
        )
    )

    assert "### Activated" in note
    assert note.index("Activated") < note.index("Demoted") < note.index("New candidates")
    # ...and the activated cell's label leads the demoted one in the body.
    assert note.index("insider_transaction/P/") < note.index("insider_transaction/S/")


# --------------------------------------------------------------------------- #
# quiet day and failed runs                                                    #
# --------------------------------------------------------------------------- #
def test_quiet_day_still_renders_a_valid_note() -> None:
    briefing = _briefing(
        ingest={}, changes=[], alerts=[], active_signals=[], notes=[], run_status=RunStatus.SUCCESS
    )
    assert briefing.has_news is False

    note = obsidian.render_note(briefing)

    assert note.startswith("---\n")
    assert "# Briefing -- 2026-07-22" in note
    assert "Quiet day" in note
    assert "_No signal changes today._" in note
    assert "_No events fired an active signal today._" in note


def test_partial_run_surfaces_its_failure_notes() -> None:
    notes = ["step prices.sync failed: HTTP 503", "step returns.compute skipped"]
    note = obsidian.render_note(_briefing(run_status=RunStatus.PARTIAL, notes=notes))

    assert "run_status: partial" in note
    assert "Run status: partial" in note  # the warning banner
    for line in notes:
        assert line in note


# --------------------------------------------------------------------------- #
# writing to the vault                                                         #
# --------------------------------------------------------------------------- #
def test_write_daily_note_writes_to_the_dated_path(tmp_path: Path) -> None:
    vault = tmp_path / "vault"

    path = obsidian.write_daily_note(vault, _briefing())

    assert path == vault / "briefings" / "2026-07-22.md"
    assert path.exists()
    assert path.read_text(encoding="utf-8") == obsidian.render_note(_briefing())


def test_write_daily_note_is_idempotent(tmp_path: Path) -> None:
    """Re-running the loop for a day corrects the note, never duplicates it."""
    vault = tmp_path / "vault"

    first = obsidian.write_daily_note(vault, _briefing(notes=["first pass"]))
    second_briefing = _briefing(notes=["second pass"])
    second = obsidian.write_daily_note(vault, second_briefing)

    assert first == second
    md_files = list((vault / "briefings").glob("*.md"))
    assert md_files == [first]
    content = second.read_text(encoding="utf-8")
    assert "second pass" in content
    assert "first pass" not in content
    assert content == obsidian.render_note(second_briefing)
