"""Project the company relationship graph into a navigable Obsidian vault.

The ``company_edges`` table is the truth: typed, deduplicated relationship
claims extracted from filings. This module is one *read-only projection* of that
truth -- a folder of interlinked Markdown notes, one per company, that Obsidian's
graph view draws as an actual graph. Nothing here is a second source of record:
every note is regenerated from the database, so the vault can be deleted and
rebuilt at any time, and a stale edge in a note is a bug in the last projection,
never a fact the database has lost.

The links are what make it a graph. A *resolved* target -- a company we track and
price -- becomes an Obsidian wiki-link ``[[TICKER]]`` that Obsidian resolves to
that company's own note, so a path through the graph is a path through the
files. An *unresolved* target (a foreign or private firm with no note of its
own) stays plain text: there is nothing to link to, and inventing a note would
imply we track a company we do not.

Pure by construction: :func:`render_company_note` turns one company's edges into
Markdown with no I/O, so the exact text is testable without a filesystem or a
database. Only :func:`project_graph` reads the connection and writes files, and
it writes each note at a fixed path so re-running overwrites in place -- the
vault is corrected, never appended to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from market_intelligence.schemas.edges import EdgeType, ResolutionStatus

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb

#: Outgoing edges grouped most-predictive first: the economic-links literature
#: (and ``edges.py``) ranks customer/supplier above competitor/partner.
_EDGE_ORDER: tuple[str, ...] = (
    EdgeType.CUSTOMER,
    EdgeType.SUPPLIER,
    EdgeType.COMPETITOR,
    EdgeType.PARTNER,
)

#: Headings for a company's OWN claims (it is the source of the edge).
_OUTGOING_HEADINGS: dict[str, str] = {
    EdgeType.CUSTOMER: "Sells to / Customers",
    EdgeType.SUPPLIER: "Buys from / Suppliers",
    EdgeType.COMPETITOR: "Competitors",
    EdgeType.PARTNER: "Partners",
}

#: Sub-headings for edges that named this company (it is the resolved target).
_INCOMING_HEADINGS: dict[str, str] = {
    EdgeType.CUSTOMER: "Named as a customer by",
    EdgeType.SUPPLIER: "Named as a supplier by",
    EdgeType.COMPETITOR: "Named as a competitor by",
    EdgeType.PARTNER: "Named as a partner by",
}

#: Evidence quotes are one sentence but can run long; trim to keep a note
#: scannable. ASCII ``...`` (not an ellipsis glyph) satisfies ruff RUF001.
_EVIDENCE_MAX_CHARS: int = 160

#: Columns pulled from ``company_edges`` to build both sides of every note.
_EDGE_COLUMNS: tuple[str, ...] = (
    "source_ticker", "source_cik", "target_ticker", "target_name", "target_cik",
    "edge_type", "resolution_status", "evidence", "extraction_confidence",
    "report_date", "times_asserted",
)  # fmt: skip

Edge = dict[str, Any]


def render_company_note(ticker: str, outgoing: list[Edge], incoming: list[Edge]) -> str:
    """Render one company's relationship note to Markdown (no I/O).

    ``outgoing`` are edges this company filed (it is the source); ``incoming``
    are resolved edges that named this company (it is the target). Resolved
    counterparties become ``[[TICKER]]`` wiki-links so Obsidian's graph view
    connects the notes; an unresolved target stays plain text, since no note
    exists for a company we do not track.

    Always returns a complete note -- frontmatter and title -- even for a company
    with only outgoing or only incoming edges, so the vault holds one uniform
    shape per company rather than a ragged mix.
    """
    blocks: list[str] = [
        _frontmatter(ticker, outgoing, incoming),
        f"# {ticker}",
    ]
    blocks.extend(_outgoing_sections(outgoing))
    inbound = _inbound_section(incoming)
    if inbound is not None:
        blocks.append(inbound)
    if not outgoing and not incoming:
        blocks.append("_No relationships recorded._")
    return "\n\n".join(blocks) + "\n"


def project_graph(con: duckdb.DuckDBPyConnection, vault_dir: Path) -> int:
    """Read ``company_edges`` and write one note per company under ``companies/``.

    Generates a note for every ticker that appears as a source or as a resolved
    target, at ``<vault>/companies/<TICKER>.md``. Idempotent: the path is fixed
    by ticker, so regenerating overwrites each note in place -- never appends,
    never leaves a second copy. Returns the number of notes written.
    """
    outgoing, incoming = _read_edges(con)
    note_dir = vault_dir / "companies"
    note_dir.mkdir(parents=True, exist_ok=True)
    tickers = sorted(set(outgoing) | set(incoming))
    for ticker in tickers:
        note = render_company_note(ticker, outgoing.get(ticker, []), incoming.get(ticker, []))
        (note_dir / f"{ticker}.md").write_text(note, encoding="utf-8")
    return len(tickers)


# --------------------------------------------------------------------------- #
# Reading the graph out of the database                                        #
# --------------------------------------------------------------------------- #
def _read_edges(
    con: duckdb.DuckDBPyConnection,
) -> tuple[dict[str, list[Edge]], dict[str, list[Edge]]]:
    """Load every edge once, bucketed by the source and the resolved target.

    A row lands in ``outgoing[source_ticker]`` when its filer is a company we
    track, and in ``incoming[target_ticker]`` only when it resolved to a tracked
    company -- an unresolved target has no note to receive an inbound link.
    """
    sql = f"SELECT {', '.join(_EDGE_COLUMNS)} FROM company_edges"
    outgoing: dict[str, list[Edge]] = {}
    incoming: dict[str, list[Edge]] = {}
    for row in con.execute(sql).fetchall():
        edge: Edge = dict(zip(_EDGE_COLUMNS, row, strict=True))
        source = edge.get("source_ticker")
        if source:
            outgoing.setdefault(str(source), []).append(edge)
        if _is_resolved(edge):
            incoming.setdefault(str(edge["target_ticker"]), []).append(edge)
    return outgoing, incoming


# --------------------------------------------------------------------------- #
# Frontmatter and sections                                                     #
# --------------------------------------------------------------------------- #
def _frontmatter(ticker: str, outgoing: list[Edge], incoming: list[Edge]) -> str:
    """Obsidian YAML frontmatter: the ticker and a tag per edge kind present."""
    edge_tags = (f"edge/{edge_type}" for edge_type in _edge_types_present(outgoing, incoming))
    tag_lines = "\n".join(f"  - {tag}" for tag in ("company", *edge_tags))
    return f"---\nticker: {ticker}\ntags:\n{tag_lines}\n---"


def _outgoing_sections(outgoing: list[Edge]) -> list[str]:
    """One ``##`` section per outgoing edge kind actually present, in order."""
    grouped = _group_by_type(outgoing)
    sections: list[str] = []
    for edge_type in _types_in_order(grouped):
        heading = _OUTGOING_HEADINGS.get(edge_type, edge_type.title())
        lines = [f"## {heading}"]
        lines.extend(_outgoing_bullet(edge) for edge in _sorted_outgoing(grouped[edge_type]))
        sections.append("\n".join(lines))
    return sections


def _inbound_section(incoming: list[Edge]) -> str | None:
    """The "named by" section, grouped by edge kind. ``None`` when there is none."""
    if not incoming:
        return None
    grouped = _group_by_type(incoming)
    lines = ["## Inbound (named by)"]
    for edge_type in _types_in_order(grouped):
        heading = _INCOMING_HEADINGS.get(edge_type, edge_type.title())
        lines.append("")
        lines.append(f"### {heading}")
        lines.extend(_inbound_bullet(edge) for edge in _sorted_inbound(grouped[edge_type]))
    return "\n".join(lines)


def _outgoing_bullet(edge: Edge) -> str:
    """A target line: link (or plain name), then evidence and confidence."""
    return f"- {_target_link(edge)}{_evidence_suffix(edge)}"


def _inbound_bullet(edge: Edge) -> str:
    """A source line: the naming company as a ``[[TICKER]]`` link back."""
    source = edge.get("source_ticker")
    link = f"[[{source}]]" if source else str(edge.get("source_cik") or "?")
    return f"- {link}{_evidence_suffix(edge)}"


# --------------------------------------------------------------------------- #
# Small pure helpers                                                           #
# --------------------------------------------------------------------------- #
def _edge_types_present(outgoing: list[Edge], incoming: list[Edge]) -> list[str]:
    """Edge kinds appearing on either side, known ones first, extras sorted."""
    all_edges = (*outgoing, *incoming)
    present = {str(edge.get("edge_type")) for edge in all_edges if edge.get("edge_type")}
    known = [edge_type for edge_type in _EDGE_ORDER if edge_type in present]
    extra = sorted(edge_type for edge_type in present if edge_type not in _EDGE_ORDER)
    return [*known, *extra]


def _group_by_type(edges: list[Edge]) -> dict[str, list[Edge]]:
    """Bucket edges by ``edge_type`` (falsy types fold into one empty-key group)."""
    grouped: dict[str, list[Edge]] = {}
    for edge in edges:
        grouped.setdefault(str(edge.get("edge_type") or ""), []).append(edge)
    return grouped


def _types_in_order(grouped: dict[str, list[Edge]]) -> list[str]:
    """Present edge kinds, the known four first in priority order, extras after."""
    known = [edge_type for edge_type in _EDGE_ORDER if edge_type in grouped]
    extra = sorted(edge_type for edge_type in grouped if edge_type not in _EDGE_ORDER)
    return [*known, *extra]


def _sorted_outgoing(edges: list[Edge]) -> list[Edge]:
    """Stable order for a target group: by display label, then report date."""
    return sorted(
        edges,
        key=lambda edge: (_target_label(edge).lower(), str(edge.get("report_date") or "")),
    )


def _sorted_inbound(edges: list[Edge]) -> list[Edge]:
    """Stable order for a "named by" group: by the naming ticker."""
    return sorted(edges, key=lambda edge: str(edge.get("source_ticker") or ""))


def _is_resolved(edge: Edge) -> bool:
    """True when the target is a tracked company we can link to by ticker."""
    return edge.get("resolution_status") == ResolutionStatus.RESOLVED.value and bool(
        edge.get("target_ticker")
    )


def _target_link(edge: Edge) -> str:
    """A ``[[TICKER]]`` wiki-link for a resolved target, else its plain name."""
    if _is_resolved(edge):
        return f"[[{edge['target_ticker']}]]"
    return _target_label(edge)


def _target_label(edge: Edge) -> str:
    """The human name to show for a target (also the outgoing sort key)."""
    return str(edge.get("target_name") or edge.get("target_ticker") or edge.get("target") or "?")


def _evidence_suffix(edge: Edge) -> str:
    """The trailing ``-- "quote" (conf 90%)``; each part omitted when absent."""
    detail = ""
    evidence = edge.get("evidence")
    if evidence:
        detail += f' -- "{_truncate(str(evidence))}"'
    confidence = edge.get("extraction_confidence")
    if confidence is not None:
        detail += f" (conf {_conf_pct(float(confidence))})"
    return detail


def _truncate(text: str) -> str:
    """Collapse whitespace and cap length, ending in ASCII ``...`` when cut."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _EVIDENCE_MAX_CHARS:
        return collapsed
    return collapsed[: _EVIDENCE_MAX_CHARS - 3].rstrip() + "..."


def _conf_pct(value: float) -> str:
    """An extraction confidence as a whole percentage: ``0.9`` -> ``"90%"``."""
    return f"{value * 100.0:.0f}%"


__all__ = [
    "project_graph",
    "render_company_note",
]
