"""Project the daily briefing into a dated Obsidian note.

One of two read-only projections of a :class:`Briefing` -- the other nudges via
Telegram. An Obsidian vault is just a folder of plain Markdown, so writing the
day's changes, alerts and standing roster there lands them somewhere a human
already reads: searchable, backlinkable, and with no app or database to keep
running. The note is keyed by date and rewritten in place, so re-running the
loop corrects the day rather than leaving a second copy behind.

Pure by construction: :func:`render_note` turns a briefing into Markdown with no
I/O, so the exact text is testable without a filesystem. Only
:func:`write_daily_note` touches disk, and it depends on nothing but this note's
shape -- never on the orchestrator that produced it.
"""

from __future__ import annotations

from pathlib import Path

from market_intelligence.autopilot.types import (
    Briefing,
    ChangeKind,
    EventAlert,
    RunStatus,
    SignalChange,
)

#: Signal-change groups, most consequential first. A cell that newly *fires*
#: (activated) is what a reader acts on, so it leads; a broken cell (demoted,
#: retired) comes next; a fresh candidate is context, not action.
_KIND_ORDER: tuple[str, ...] = (
    ChangeKind.ACTIVATED,
    ChangeKind.DEMOTED,
    ChangeKind.RETIRED,
    ChangeKind.NEW_CANDIDATE,
)

_KIND_HEADINGS: dict[str, str] = {
    ChangeKind.ACTIVATED: "Activated -- now firing",
    ChangeKind.DEMOTED: "Demoted",
    ChangeKind.RETIRED: "Retired",
    ChangeKind.NEW_CANDIDATE: "New candidates",
}

#: Preferred order and human labels for the "what moved" line. Unknown ingest
#: keys are kept -- a new counter should surface, not vanish -- using the raw
#: key with underscores softened to spaces.
_INGEST_ORDER: tuple[str, ...] = ("prices", "filings", "events", "matured")
_INGEST_LABELS: dict[str, str] = {
    "prices": "new prices",
    "filings": "new filings",
    "events": "new events",
    "matured": "matured samples",
}


def render_note(briefing: Briefing) -> str:
    """Render a briefing to a full Obsidian Markdown note (no I/O).

    Always returns a complete, valid note -- frontmatter, title, and every
    section -- even on a quiet day or a failed run, so the vault holds one
    uniform entry per date rather than a ragged mix of shapes.
    """
    blocks: list[str] = [
        _frontmatter(briefing),
        f"# Briefing -- {briefing.as_of.isoformat()}",
    ]
    if briefing.run_status != RunStatus.SUCCESS:
        blocks.append(_status_banner(briefing.run_status))
    blocks.append(_summary_line(briefing.ingest))
    if not briefing.has_news:
        blocks.append("_Quiet day: no signal changes or alerts. Model roster unchanged._")
    blocks.append(_signal_changes_section(briefing.changes))
    blocks.append(_alerts_section(briefing.alerts))
    blocks.append(_active_signals_section(briefing.active_signals))
    blocks.append(_notes_section(briefing.notes))
    return "\n\n".join(blocks) + "\n"


def write_daily_note(vault_dir: Path, briefing: Briefing) -> Path:
    """Render the briefing and write it to ``<vault>/briefings/<date>.md``.

    Idempotent by design: the path is fixed by ``as_of``, so re-running the loop
    for a day overwrites that day's note in place -- never appends, never
    duplicates. Returns the path written so a caller can link or log it.
    """
    note_dir = vault_dir / "briefings"
    note_dir.mkdir(parents=True, exist_ok=True)
    path = note_dir / f"{briefing.as_of.isoformat()}.md"
    path.write_text(render_note(briefing), encoding="utf-8")
    return path


def _frontmatter(briefing: Briefing) -> str:
    """Obsidian YAML frontmatter: date, run status, and query-able tags."""
    tags = ["briefing", f"run/{briefing.run_status}"]
    if briefing.alerts:
        tags.append("briefing/alerts")
    if not briefing.has_news:
        tags.append("briefing/quiet")
    tag_lines = "\n".join(f"  - {tag}" for tag in tags)
    return (
        "---\n"
        f"date: {briefing.as_of.isoformat()}\n"
        f"run_status: {briefing.run_status}\n"
        "tags:\n"
        f"{tag_lines}\n"
        "---"
    )


def _status_banner(run_status: str) -> str:
    """A callout flagging a non-clean run, pointing the reader at the notes."""
    detail = {
        RunStatus.PARTIAL: "some steps failed; earlier work still landed. See Notes below.",
        RunStatus.FAILED: "the run could not produce a usable briefing. See Notes below.",
    }.get(run_status, "see Notes below.")
    return f"> [!warning] Run status: {run_status}\n> {detail}"


def _summary_line(ingest: dict[str, int]) -> str:
    """The one-line "what moved" tally, mentioning only counters that are present."""
    if not ingest:
        return "**What moved:** no new inputs recorded."
    known = [key for key in _INGEST_ORDER if key in ingest]
    extra = [key for key in ingest if key not in _INGEST_ORDER]
    parts = [
        f"{ingest[key]:,} {_INGEST_LABELS.get(key, key.replace('_', ' '))}"
        for key in [*known, *extra]
    ]
    return "**What moved:** " + ", ".join(parts) + "."


def _signal_changes_section(changes: list[SignalChange]) -> str:
    """Signal changes grouped by kind, activated cells foremost."""
    lines = ["## Signal changes"]
    if not changes:
        lines.append("_No signal changes today._")
        return "\n".join(lines)
    for kind in _kinds_in_order(changes):
        lines.append("")
        lines.append(f"### {_KIND_HEADINGS.get(kind, kind)}")
        lines.extend(_change_bullet(change) for change in changes if change.kind == kind)
    return "\n".join(lines)


def _kinds_in_order(changes: list[SignalChange]) -> list[str]:
    """Kinds actually present, known ones first in priority order, extras after."""
    present = list(dict.fromkeys(change.kind for change in changes))
    known = [kind for kind in _KIND_ORDER if kind in present]
    extra = [kind for kind in present if kind not in _KIND_ORDER]
    return [*known, *extra]


def _change_bullet(change: SignalChange) -> str:
    return (
        f"- **{change.label}** -- {_signed_pct(change.mean_car)} mean CAR, "
        f"{_rate_pct(change.hit_rate)} hit, {change.n_clusters} clusters -- {change.reason}"
    )


def _alerts_section(alerts: list[EventAlert]) -> str:
    """Today's events that fired an active cell, one bullet each."""
    lines = ["## Alerts"]
    if not alerts:
        lines.append("_No events fired an active signal today._")
        return "\n".join(lines)
    lines.extend(_alert_bullet(alert) for alert in alerts)
    return "\n".join(lines)


def _alert_bullet(alert: EventAlert) -> str:
    fired = f"{alert.event_type}/{alert.event_subtype or '-'}"
    move = f"{_direction_sign(alert.direction)}{abs(alert.predicted_car) * 100.0:.2f}%"
    return (
        f"- **{alert.ticker}** {fired} -- predicted {move} over "
        f"{alert.horizon_days}d -- basis: {alert.basis}"
    )


def _active_signals_section(active: list[SignalChange]) -> str:
    """The standing trusted roster as a compact table, for context."""
    lines = ["## Active signals"]
    if not active:
        lines.append("_No active signals in the roster._")
        return "\n".join(lines)
    lines.append("")
    lines.append("| Signal | Mean CAR | Hit | Clusters |")
    lines.append("| --- | --- | --- | --- |")
    for sig in active:
        lines.append(
            f"| {sig.label} | {_signed_pct(sig.mean_car)} | "
            f"{_rate_pct(sig.hit_rate)} | {sig.n_clusters} |"
        )
    return "\n".join(lines)


def _notes_section(notes: list[str]) -> str:
    """Free-form run detail, including which steps failed on a partial run."""
    lines = ["## Notes"]
    if not notes:
        lines.append("_No notes._")
        return "\n".join(lines)
    lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines)


def _signed_pct(value: float) -> str:
    """A decimal return as a signed percentage: ``0.0162`` -> ``"+1.62%"``.

    A move that rounds to zero prints ``"+0.00%"``, never a jarring ``"-0.00%"``.
    """
    return f"{value * 100.0:+.2f}%".replace("-0.00%", "+0.00%")


def _rate_pct(value: float) -> str:
    """A decimal rate as a percentage: ``0.588`` -> ``"58.8%"``."""
    return f"{value * 100.0:.1f}%"


def _direction_sign(direction: int) -> str:
    """The leading ``+``/``-`` for a predicted move; empty when neutral."""
    if direction > 0:
        return "+"
    if direction < 0:
        return "-"
    return ""


__all__ = [
    "render_note",
    "write_daily_note",
]
