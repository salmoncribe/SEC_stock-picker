"""Regenerable Obsidian projection for relationship decision records.

The renderer intentionally accepts an already-sanitized decision summary.  It
does not receive raw broker/locate payloads, credentials, or account state;
DuckDB remains the source of truth for the full audit packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpportunityDecisionNote:
    opportunity_id: str
    ticker: str
    status: str
    evidence_score: float
    strategy_version: str
    evidence_snapshot_hash: str
    candidate_scored_at: str
    first_public_state: str
    suppression_reasons: tuple[str, ...]
    verdict: str | None = None
    why_now: str | None = None
    risk_flags: tuple[str, ...] = ()
    maximum_entry_condition: str | None = None
    invalidation: str | None = None
    time_limit: str | None = None


def render_opportunity_decision_note(note: OpportunityDecisionNote) -> str:
    """Render a concise, non-executable research note."""
    lines = [
        "---",
        "type: relationship-opportunity",
        f"opportunity_id: {note.opportunity_id}",
        f"ticker: {note.ticker}",
        f"status: {note.status}",
        f"strategy_version: {note.strategy_version}",
        f"evidence_snapshot_hash: {note.evidence_snapshot_hash}",
        "---",
        "",
        f"# Relationship opportunity — {note.ticker}",
        "",
        "> Research candidate only. This note is not an order instruction;",
        "> the system never trades.",
        "",
        f"- Evidence score: {note.evidence_score:g}",
        f"- First-public state: {note.first_public_state}",
        f"- Candidate scored: {note.candidate_scored_at}",
        f"- LLM verdict: {note.verdict or 'not requested'}",
    ]
    if note.why_now:
        lines.extend(["", "## Why now", "", note.why_now])
    if note.suppression_reasons:
        lines.extend(["", "## Suppressed / blocked because", ""])
        lines.extend(f"- {reason}" for reason in note.suppression_reasons)
    if note.risk_flags:
        lines.extend(["", "## Risks / counterevidence", ""])
        lines.extend(f"- {flag}" for flag in note.risk_flags)
    if note.maximum_entry_condition or note.invalidation or note.time_limit:
        lines.extend(["", "## Candidate constraints", ""])
        if note.maximum_entry_condition:
            lines.append(f"- Maximum entry condition: {note.maximum_entry_condition}")
        if note.invalidation:
            lines.append(f"- Invalidation: {note.invalidation}")
        if note.time_limit:
            lines.append(f"- Time limit: {note.time_limit}")
    return "\n".join(lines) + "\n"


def write_opportunity_decision_note(vault_dir: Path, note: OpportunityDecisionNote) -> Path:
    """Write the safe projection atomically under the regenerable vault tree."""
    directory = vault_dir / "relationship-opportunities"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{note.opportunity_id}.md"
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text(render_opportunity_decision_note(note), encoding="utf-8")
    temporary.replace(target)
    return target


__all__ = [
    "OpportunityDecisionNote",
    "render_opportunity_decision_note",
    "write_opportunity_decision_note",
]
