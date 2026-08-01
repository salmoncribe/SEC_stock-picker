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

**Inclusion is earned, not assumed (docs/specs/2026-07-25-people-insider-graph-
design.md §9, the 2026-07-26 amendment).** The first build rendered a page for
every person in ``role_memberships`` -- 106,265 notes, including institutional
10%-owner funds that hold no board/executive relationship at all. A ``Person``
now earns a dedicated note only if they (a) interlock (sit as a director on 2+
tracked companies -- :func:`~market_intelligence.analytics.interlocks.interlocking_person_ids`)
or (b) are tied to a currently-live ``insider_cluster_buy`` event; a person who
is *only ever* a 10%-owner never qualifies regardless of (a)/(b). Everyone who
doesn't clear the bar stays in DuckDB but out of Obsidian -- the vault is the
AI's reasoning surface, not a roster dump. See :func:`_qualifying_person_ids`.

**Relevance is measured signal state, not existence (§9.2).** An
``insider_cluster_buy`` event is rendered while live, or after resolution only
when it has a graded outcome worth using as precedent. Raw old cluster windows
with no grade stay in DuckDB. Propagation edges (including
``shared_board_member`` interlocks) render if they either touch an open trade
alert ticker or the signal ladder has an ``active`` cell for that edge type;
until then they are extraction/search space, not AI context for "why this stock
might move."

**The vault writer deletes, not just adds (§9.3).** Because inclusion criteria
can tighten -- as they did here -- :func:`project_graph` and
:func:`project_person_notes` each diff the note set they are about to write
against what already exists on disk in their own subdirectory
(``companies/``/``people/``) and remove anything that no longer qualifies. This
is scoped to exactly those two subdirectories: the vault can hold files this
projector never wrote (``briefings/``, ``.obsidian/``), and only ``.md`` files
directly inside the relevant subdirectory are ever candidates for deletion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from market_intelligence.analytics import interlocks
from market_intelligence.analytics.interlocks import SHARED_BOARD_MEMBER_EDGE_TYPE
from market_intelligence.schemas.edges import EdgeType, ResolutionStatus
from market_intelligence.schemas.events import EventType
from market_intelligence.schemas.samples import SELF_EDGE
from market_intelligence.signals.dataset import DEFAULT_HORIZONS

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb

#: Outgoing edges grouped most-predictive first: the economic-links literature
#: (and ``edges.py``) ranks customer/supplier above competitor/partner.
#: ``shared_board_member`` (Decision C of docs/specs/2026-07-25-people-
#: insider-graph-design.md) is not an ``EdgeType`` member -- that enum is
#: scoped to "what the LLM extractor may assert" -- but it is a real,
#: first-class edge in ``company_edges`` and belongs in the same allow-list
#: so it renders with a proper heading instead of falling into the generic
#: "extra, title-cased" bucket.
_EDGE_ORDER: tuple[str, ...] = (
    EdgeType.CUSTOMER,
    EdgeType.SUPPLIER,
    EdgeType.COMPETITOR,
    EdgeType.PARTNER,
    SHARED_BOARD_MEMBER_EDGE_TYPE,
)

#: Headings for a company's OWN claims (it is the source of the edge).
_OUTGOING_HEADINGS: dict[str, str] = {
    EdgeType.CUSTOMER: "Sells to / Customers",
    EdgeType.SUPPLIER: "Buys from / Suppliers",
    EdgeType.COMPETITOR: "Competitors",
    EdgeType.PARTNER: "Partners",
    SHARED_BOARD_MEMBER_EDGE_TYPE: "Shared board members",
}

#: Sub-headings for edges that named this company (it is the resolved target).
_INCOMING_HEADINGS: dict[str, str] = {
    EdgeType.CUSTOMER: "Named as a customer by",
    EdgeType.SUPPLIER: "Named as a supplier by",
    EdgeType.COMPETITOR: "Named as a competitor by",
    EdgeType.PARTNER: "Named as a partner by",
    # A shared-board-member edge is symmetric, so the same heading reads
    # naturally from either side ("Shared board members" with company X).
    SHARED_BOARD_MEMBER_EDGE_TYPE: "Shared board members",
}

#: Evidence quotes are one sentence but can run long; trim to keep a note
#: scannable. ASCII ``...`` (not an ellipsis glyph) satisfies ruff RUF001.
_EVIDENCE_MAX_CHARS: int = 160

#: Columns pulled from ``company_edges`` to build both sides of every note.
_EDGE_COLUMNS: tuple[str, ...] = (
    "edge_id", "source_ticker", "source_cik", "target_ticker", "target_name", "target_cik",
    "edge_type", "resolution_status", "evidence", "extraction_confidence",
    "report_date", "times_asserted",
)  # fmt: skip

Edge = dict[str, Any]
#: A row describing one person's role at one company, oriented either way:
#: from a company note it carries ``canonical_name``/``person_link``; from a
#: person note it carries ``ticker``/``company_name``. Both shapes always
#: carry the flag/title fields ``_role_label`` reads.
RoleEdge = dict[str, Any]
#: One ``insider_cluster_buy`` event, already classified live/resolved for
#: rendering -- see ``_read_cluster_buy_facts``. Always carries ``ticker``
#: (a fact with no ticker is dropped at the read boundary, the same rule
#: ``_read_role_rows`` applies to roles) so both a company's own note and a
#: person's ``[[TICKER]]`` link can use it with no null check.
ClusterBuyFact = dict[str, Any]
#: One persisted trade-alert ledger row, trimmed to fields an AI needs for a
#: "why might this stock move?" note.
TradeAlertFact = dict[str, Any]

#: Columns read out of ``role_memberships`` (joined to ``companies``/``people``)
#: to build both the company-side "People" section and the person notes.
_ROLE_ROW_COLUMNS: tuple[str, ...] = (
    "person_id", "canonical_name", "ticker", "company_name",
    "is_officer", "is_director", "is_ten_pct_owner", "latest_officer_title",
)  # fmt: skip

#: Mirrors ``config.PeopleGraphConfig.interlock_max_companies_per_person``'s
#: own default. Production call sites (``cli.py``) always pass the real
#: configured value explicitly; this only covers a caller that does not.
DEFAULT_INTERLOCK_CAP: int = 15

#: The longest forward horizon any event is actually graded against, read
#: from the real dataset builder rather than hardcoded. Design doc §9.2 cites
#: "the 20-day horizon per the base dataset builder" -- computing it from
#: ``signals.dataset.DEFAULT_HORIZONS`` means that claim stays verified
#: against the source it describes instead of silently drifting from it.
LONGEST_EVALUATED_HORIZON_DAYS: int = max(DEFAULT_HORIZONS)

#: Shared board members are real, but weak and numerous. Around open trade
#: alerts they should add color to the graph, not drown out commercial edges.
_MAX_ANCHOR_BOARD_EDGES_PER_SIDE: int = 5


@dataclass(frozen=True)
class VaultProjectionResult:
    """How many notes one projection run wrote, and how many stale ones it pruned."""

    written: int
    deleted: int


def render_company_note(
    ticker: str,
    outgoing: list[Edge],
    incoming: list[Edge],
    roles: list[RoleEdge] | None = None,
    cluster_buys: list[ClusterBuyFact] | None = None,
    trade_alerts: list[TradeAlertFact] | None = None,
) -> str:
    """Render one company's relationship note to Markdown (no I/O).

    ``outgoing`` are edges this company filed (it is the source); ``incoming``
    are resolved edges that named this company (it is the target). Resolved
    counterparties become ``[[TICKER]]`` wiki-links so Obsidian's graph view
    connects the notes; an unresolved target stays plain text, since no note
    exists for a company we do not track. ``roles`` (optional, default none)
    lists the officers/directors this company's ``role_memberships`` name only
    when that person clears the vault-inclusion bar (interlock or currently-live
    cluster-buy participation). Owner-only rows and ordinary titles remain
    queryable in DuckDB but stay out of Obsidian, so the vault stays focused on
    facts that could explain a stock move. ``cluster_buys`` (optional, default
    none) lists this company's ``insider_cluster_buy`` events, split into
    "Watching" (live) and "Resolved" (graded) per §9.2. ``trade_alerts`` are
    already-fired predictions from the alert ledger and are the strongest
    vault-inclusion anchor.

    Always returns a complete note -- frontmatter and title -- even for a company
    with only outgoing or only incoming edges, so the vault holds one uniform
    shape per company rather than a ragged mix.
    """
    roles = roles or []
    cluster_buys = cluster_buys or []
    trade_alerts = trade_alerts or []
    blocks: list[str] = [
        _frontmatter(
            ticker,
            outgoing,
            incoming,
            has_cluster_buys=bool(cluster_buys),
            has_trade_alerts=bool(trade_alerts),
        ),
        f"# {ticker}",
    ]
    trade_alerts_section = _trade_alerts_section(trade_alerts)
    if trade_alerts_section is not None:
        blocks.append(trade_alerts_section)
    people_section = _people_section(roles)
    if people_section is not None:
        blocks.append(people_section)
    cluster_section = _cluster_buy_section(cluster_buys, link=False)
    if cluster_section is not None:
        blocks.append(cluster_section)
    blocks.extend(_outgoing_sections(outgoing))
    inbound = _inbound_section(incoming)
    if inbound is not None:
        blocks.append(inbound)
    if not outgoing and not incoming and not roles and not cluster_buys and not trade_alerts:
        blocks.append("_No relationships recorded._")
    return "\n\n".join(blocks) + "\n"


def render_person_note(
    display_name: str,
    roles: list[RoleEdge],
    cluster_buys: list[ClusterBuyFact] | None = None,
) -> str:
    """Render one person's note to Markdown (no I/O) -- the company-note mirror.

    ``roles`` lists the officer/director company ties that made this person
    useful to the graph, each carrying ``ticker``/``company_name`` plus the same
    flag/title fields ``render_company_note``'s People section reads. A resolved
    ticker becomes a ``[[TICKER]]`` wiki-link back to that company's own note,
    so the graph is navigable in both directions. ``cluster_buys`` (optional,
    default none) lists ``insider_cluster_buy`` events this person was a
    purchaser in, split into "Watching"/"Resolved" per §9.2, each linking back
    to the company.
    """
    cluster_buys = cluster_buys or []
    blocks: list[str] = [
        _person_frontmatter(roles, has_cluster_buys=bool(cluster_buys)),
        f"# {display_name}",
    ]
    if roles:
        lines = ["## Roles"]
        for role in sorted(roles, key=lambda r: str(r.get("ticker") or "")):
            ticker = role.get("ticker")
            link = f"[[{ticker}]]" if ticker else str(role.get("company_name") or "?")
            lines.append(f"- {link} -- {_role_label(role)}")
        blocks.append("\n".join(lines))
    elif not cluster_buys:
        blocks.append("_No company roles recorded._")
    cluster_section = _cluster_buy_section(cluster_buys, link=True)
    if cluster_section is not None:
        blocks.append(cluster_section)
    return "\n\n".join(blocks) + "\n"


def project_graph(
    con: duckdb.DuckDBPyConnection,
    vault_dir: Path,
    *,
    max_companies_per_person: int = DEFAULT_INTERLOCK_CAP,
    longest_horizon_days: int = LONGEST_EVALUATED_HORIZON_DAYS,
    today: date | None = None,
) -> VaultProjectionResult:
    """Read ``company_edges``/``role_memberships``/``events`` and write one note per company.

    Generates a note for every ticker with an open trade alert, every one-hop
    company connected to those alert tickers, every ticker that appears as a
    source or resolved target of an active-signal company edge, and every
    ticker with a live/graded ``insider_cluster_buy`` event, at
    ``<vault>/companies/<TICKER>.md``. Idempotent: the path is fixed by
    ticker, so regenerating overwrites each note in place. Also deletes any
    ``companies/*.md`` file on disk for a ticker that no longer qualifies
    (§9.3) -- e.g. a ticker that only ever had roles/edges that have since
    been removed upstream. ``today`` is injectable (defaults to the real date)
    so live/resolved classification of cluster-buy events is deterministic in
    tests.
    """
    as_of = today or date.today()
    active_edge_types = _active_propagation_edge_types(con)
    trade_alert_facts = _read_trade_alert_facts(con)
    alert_tickers = _trade_alert_tickers(trade_alert_facts)
    outgoing, incoming = _read_edges(
        con,
        active_edge_types=active_edge_types,
        anchor_tickers=alert_tickers,
    )
    role_rows = _read_role_rows(con)
    by_person = _group_by_person(role_rows)
    filenames = _person_filenames(by_person)

    interlocking_ids = (
        interlocks.interlocking_person_ids(
            con, max_companies_per_person=max_companies_per_person
        )
        if SHARED_BOARD_MEMBER_EDGE_TYPE in active_edge_types
        else set()
    )
    cluster_buy_facts = _read_cluster_buy_facts(
        con, today=as_of, longest_horizon_days=longest_horizon_days
    )
    qualifying = _qualifying_person_ids(
        role_rows, interlocking_ids, _live_event_person_ids(cluster_buy_facts)
    )
    roles_by_ticker = _group_by_ticker(role_rows, by_person, filenames, qualifying)
    cluster_buys_by_ticker = _group_cluster_buys_by_ticker(cluster_buy_facts)
    trade_alerts_by_ticker = _group_trade_alerts_by_ticker(trade_alert_facts)

    note_dir = vault_dir / "companies"
    note_dir.mkdir(parents=True, exist_ok=True)
    tickers = sorted(
        set(outgoing)
        | set(incoming)
        | set(roles_by_ticker)
        | set(cluster_buys_by_ticker)
        | set(trade_alerts_by_ticker)
    )
    for ticker in tickers:
        note = render_company_note(
            ticker,
            outgoing.get(ticker, []),
            incoming.get(ticker, []),
            roles_by_ticker.get(ticker),
            cluster_buys_by_ticker.get(ticker),
            trade_alerts_by_ticker.get(ticker),
        )
        (note_dir / f"{ticker}.md").write_text(note, encoding="utf-8")

    deleted = _prune_stale_notes(note_dir, keep_stems=set(tickers))
    return VaultProjectionResult(written=len(tickers), deleted=deleted)


def project_person_notes(
    con: duckdb.DuckDBPyConnection,
    vault_dir: Path,
    *,
    max_companies_per_person: int = DEFAULT_INTERLOCK_CAP,
    longest_horizon_days: int = LONGEST_EVALUATED_HORIZON_DAYS,
    today: date | None = None,
) -> VaultProjectionResult:
    """Read ``role_memberships``/``people``/``events`` and write one note per qualifying person.

    A person earns a note at ``<vault>/people/<name>.md`` only if they clear
    the design's §9.1 bar: they interlock (sit as a director on 2+ tracked
    companies) or are tied to a currently-live ``insider_cluster_buy`` event,
    and are not *purely* a 10%-owner. Two different real people can share a
    name (identity is CIK-keyed, not name-keyed -- see
    ``entities.PersonIndex``), so filenames are disambiguated with a short
    ``person_id`` suffix whenever a plain name is not unique among *all*
    known people (not just qualifying ones), so a later run that promotes a
    second "Jane Doe" to qualifying never collides with an already-written
    file. Idempotent for the same reason ``project_graph`` is, and prunes any
    ``people/*.md`` file on disk for a person who no longer qualifies (§9.3)
    -- the exact fix for the 106,265-note over-generation this amendment
    exists to correct.
    """
    as_of = today or date.today()
    role_rows = _read_role_rows(con)
    by_person = _group_by_person(role_rows)
    filenames = _person_filenames(by_person)

    active_edge_types = _active_propagation_edge_types(con)
    interlocking_ids = (
        interlocks.interlocking_person_ids(
            con, max_companies_per_person=max_companies_per_person
        )
        if SHARED_BOARD_MEMBER_EDGE_TYPE in active_edge_types
        else set()
    )
    cluster_buy_facts = _read_cluster_buy_facts(
        con, today=as_of, longest_horizon_days=longest_horizon_days
    )
    qualifying = _qualifying_person_ids(
        role_rows, interlocking_ids, _live_event_person_ids(cluster_buy_facts)
    )
    cluster_buys_by_person = _group_cluster_buys_by_person(cluster_buy_facts)

    note_dir = vault_dir / "people"
    note_dir.mkdir(parents=True, exist_ok=True)
    written_stems: set[str] = set()
    for person_id, (display_name, roles) in by_person.items():
        if person_id not in qualifying:
            continue
        note = render_person_note(
            display_name,
            [role for role in roles if _is_connection_role(role)],
            cluster_buys_by_person.get(person_id),
        )
        stem = filenames[person_id]
        (note_dir / f"{stem}.md").write_text(note, encoding="utf-8")
        written_stems.add(stem)

    deleted = _prune_stale_notes(note_dir, keep_stems=written_stems)
    return VaultProjectionResult(written=len(written_stems), deleted=deleted)


# --------------------------------------------------------------------------- #
# Vault-inclusion rule (docs/specs/2026-07-25-people-insider-graph-design.md   #
# §9.1) and stale-file pruning (§9.3)                                          #
# --------------------------------------------------------------------------- #
def _qualifying_person_ids(
    role_rows: list[dict[str, Any]],
    interlocking_person_ids: set[str],
    live_event_person_ids: set[str],
) -> set[str]:
    """Person ids earning a dedicated vault node.

    A person qualifies only if (1) they hold at least one officer or director
    role *somewhere* -- a person who is only ever a 10%-owner is institutional
    ownership-flow data, never a graph node, full stop, even if they happen to
    turn up as a purchaser in a live cluster-buy event -- and (2) they either
    interlock or are tied to a currently-live event. Everyone else remains in
    ``role_memberships`` but is omitted from Obsidian, so the vault does not
    mirror every database fact into the AI context.

    Interlocks are passed in only when ``shared_board_member`` is an active
    signal edge type. That keeps raw role graph exploration out of the AI vault
    until the validation layer says it has earned attention.
    """
    has_officer_or_director: dict[str, bool] = {}
    for row in role_rows:
        person_id = str(row["person_id"])
        flag = bool(row["is_officer"]) or bool(row["is_director"])
        has_officer_or_director[person_id] = has_officer_or_director.get(person_id, False) or flag

    return {
        person_id
        for person_id, has_role in has_officer_or_director.items()
        if has_role
        and (person_id in interlocking_person_ids or person_id in live_event_person_ids)
    }


def _prune_stale_notes(note_dir: Path, keep_stems: set[str]) -> int:
    """Delete ``.md`` files in ``note_dir`` whose stem is not in ``keep_stems``.

    Scoped to exactly one vault subdirectory (``companies/`` or ``people/``)
    by the caller -- the vault can hold files this projector never wrote
    (``briefings/``, ``.obsidian/``), and a blanket delete rooted any higher
    would destroy them. Only ``.md`` files directly inside ``note_dir`` are
    ever candidates -- the exact shape this projector itself writes -- and
    only regular files are removed, so an unexpected subdirectory is left
    alone rather than deleted.
    """
    if not note_dir.is_dir():
        return 0
    deleted = 0
    for path in note_dir.glob("*.md"):
        if path.is_file() and path.stem not in keep_stems:
            path.unlink()
            deleted += 1
    return deleted


# --------------------------------------------------------------------------- #
# Reading the graph out of the database                                        #
# --------------------------------------------------------------------------- #
def _active_propagation_edge_types(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Edge types whose propagation cell is active enough to spend AI context on."""
    rows = con.execute(
        """
        SELECT DISTINCT edge_type
        FROM signal_status
        WHERE status = 'active' AND edge_type <> ?
        """,
        [SELF_EDGE],
    ).fetchall()
    return {str(row[0]) for row in rows if row[0]}


def _read_edges(
    con: duckdb.DuckDBPyConnection,
    *,
    active_edge_types: set[str],
    anchor_tickers: set[str],
) -> tuple[dict[str, list[Edge]], dict[str, list[Edge]]]:
    """Load relevant edges once, bucketed by source and resolved target.

    A row lands in ``outgoing[source_ticker]`` when its filer is a company we
    track, and in ``incoming[target_ticker]`` only when it resolved to a tracked
    company -- an unresolved target has no note to receive an inbound link.
    Relevance means either an active propagation edge type, or a one-hop edge
    touching a ticker with an open trade alert.
    """
    if not active_edge_types and not anchor_tickers:
        return {}, {}
    sql = f"SELECT {', '.join(_EDGE_COLUMNS)} FROM company_edges"
    outgoing: dict[str, list[Edge]] = {}
    incoming: dict[str, list[Edge]] = {}
    edges = [
        dict(zip(_EDGE_COLUMNS, row, strict=True))
        for row in con.execute(sql).fetchall()
    ]
    anchor_board_edge_ids = _selected_anchor_board_edge_ids(edges, anchor_tickers)
    for edge in edges:
        if not _is_relevant_edge(
            edge,
            active_edge_types,
            anchor_tickers,
            anchor_board_edge_ids,
        ):
            continue
        source = edge.get("source_ticker")
        if source:
            outgoing.setdefault(str(source), []).append(edge)
        if _is_resolved(edge):
            incoming.setdefault(str(edge["target_ticker"]), []).append(edge)
    return outgoing, incoming


def _is_relevant_edge(
    edge: Edge,
    active_edge_types: set[str],
    anchor_tickers: set[str],
    anchor_board_edge_ids: set[str],
) -> bool:
    edge_type = str(edge.get("edge_type") or "")
    if edge_type in active_edge_types:
        return True
    if edge_type == SHARED_BOARD_MEMBER_EDGE_TYPE:
        return str(edge.get("edge_id") or "") in anchor_board_edge_ids
    source = str(edge.get("source_ticker") or "")
    target = str(edge.get("target_ticker") or "")
    return source in anchor_tickers or target in anchor_tickers


def _selected_anchor_board_edge_ids(edges: list[Edge], anchor_tickers: set[str]) -> set[str]:
    """Top shared-board edges per alert ticker/direction, capped to avoid hairballs."""
    groups: dict[tuple[str, str], list[Edge]] = {}
    for edge in edges:
        if edge.get("edge_type") != SHARED_BOARD_MEMBER_EDGE_TYPE:
            continue
        source = str(edge.get("source_ticker") or "")
        target = str(edge.get("target_ticker") or "")
        if source in anchor_tickers:
            groups.setdefault((source, "out"), []).append(edge)
        if target in anchor_tickers:
            groups.setdefault((target, "in"), []).append(edge)

    selected: set[str] = set()
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda edge: (
                -int(edge.get("times_asserted") or 0),
                _target_label(edge).lower(),
                str(edge.get("source_ticker") or ""),
            ),
        )
        selected.update(
            str(edge.get("edge_id"))
            for edge in ordered[:_MAX_ANCHOR_BOARD_EDGES_PER_SIDE]
            if edge.get("edge_id")
        )
    return selected


def _read_role_rows(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Load every ``role_memberships`` row joined to its company and person.

    Restricted to ticker-bearing companies, the same "unusable as a link
    target otherwise" rule ``CompanyIndex`` applies elsewhere: a company with
    no ticker has no note of its own to link to or from.
    """
    sql = """
        SELECT r.person_id, p.canonical_name, c.ticker, c.company_name,
               r.is_officer, r.is_director, r.is_ten_pct_owner, r.latest_officer_title
        FROM role_memberships r
        JOIN companies c ON c.company_id = r.company_id
        LEFT JOIN people p ON p.person_id = r.person_id
        WHERE c.ticker IS NOT NULL AND c.ticker <> ''
    """
    return [
        dict(zip(_ROLE_ROW_COLUMNS, row, strict=True)) for row in con.execute(sql).fetchall()
    ]


def _parse_window_start(raw: str | None) -> date | None:
    """Best-effort parse of a cluster-buy payload's ISO ``window_start`` string."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return None


def _load_cluster_buy_outcomes(
    con: duckdb.DuckDBPyConnection, event_ids: list[str], horizon_days: int
) -> dict[str, float]:
    """``event_id -> forward_abnormal_return`` for resolved cluster-buy events.

    Reads ``event_samples`` -- the exact table every other cell's hit rate is
    computed from (see ``analytics/impact.py::summarize``) -- rather than a
    second grading implementation. ``insider_cluster_buy`` events are always a
    positive-direction call (open-market purchases only, per
    ``analytics/cluster_buy.py``), so "hit" is the same ``car > 0`` check
    ``summarize`` already uses for every other cell, with no direction
    adjustment needed.
    """
    if not event_ids:
        return {}
    placeholders = ", ".join("?" * len(event_ids))
    rows = con.execute(
        f"""
        SELECT event_id, forward_abnormal_return
        FROM event_samples
        WHERE event_id IN ({placeholders})
          AND edge_id = ?
          AND horizon_days = ?
          AND forward_abnormal_return IS NOT NULL
        """,
        [*event_ids, SELF_EDGE, horizon_days],
    ).fetchall()
    return {str(event_id): float(car) for event_id, car in rows}


def _read_cluster_buy_facts(
    con: duckdb.DuckDBPyConnection, *, today: date, longest_horizon_days: int
) -> list[ClusterBuyFact]:
    """Load every ``insider_cluster_buy`` event, classified live/resolved (§9.2).

    Joined through ``companies`` on ``cik`` for its ticker, the same "unusable
    as a link target otherwise" rule ``_read_role_rows`` applies -- deliberately
    *not* the event's own ``ticker`` column, which is copied verbatim from the
    filer's self-reported (and occasionally malformed -- dual-class strings
    like ``"WLY/WLYB"``, or in a few real rows a stray date) issuer symbol on
    the underlying Form 4, and was never validated the way ``companies.ticker``
    is. ``cik`` is SEC's own stable numeric identifier and is trustworthy.

    A fact is **live** while ``window_end + longest_horizon_days >= today`` --
    the forward-return horizon has not yet closed -- and **resolved** once it
    has, at which point its graded outcome (if ``signals build-dataset`` has
    run since) is looked up from ``event_samples`` via
    :func:`_load_cluster_buy_outcomes`. Resolved-but-ungraded facts are omitted
    from the vault: they are historical raw material, not reasoning context.
    """
    rows = con.execute(
        """
        SELECT e.event_id, c.ticker, e.available_time, e.magnitude, e.payload
        FROM events e
        JOIN companies c ON c.cik = e.cik
        WHERE e.event_type = ? AND c.ticker IS NOT NULL AND c.ticker <> ''
        """,
        [EventType.INSIDER_CLUSTER_BUY],
    ).fetchall()

    drafts: list[ClusterBuyFact] = []
    for event_id, ticker, available_time, magnitude, payload in rows:
        data: dict[str, Any] = json.loads(payload) if payload else {}
        window_end = available_time.date()
        window_start = _parse_window_start(data.get("window_start")) or window_end
        person_ids = tuple(str(pid) for pid in data.get("person_ids", []))
        drafts.append(
            {
                "event_id": str(event_id),
                "ticker": str(ticker),
                "window_start": window_start,
                "window_end": window_end,
                "distinct_insiders": int(data.get("distinct_insiders") or len(person_ids)),
                "total_dollar_value": float(magnitude) if magnitude is not None else None,
                "person_ids": person_ids,
                "live": (window_end + timedelta(days=longest_horizon_days)) >= today,
                "outcome": None,
                "forward_abnormal_return": None,
            }
        )

    resolved_ids = [fact["event_id"] for fact in drafts if not fact["live"]]
    outcomes = _load_cluster_buy_outcomes(con, resolved_ids, longest_horizon_days)
    facts: list[ClusterBuyFact] = []
    for fact in drafts:
        if fact["live"]:
            facts.append(fact)
            continue
        car = outcomes.get(fact["event_id"])
        fact["forward_abnormal_return"] = car
        fact["outcome"] = None if car is None else ("hit" if car > 0 else "miss")
        if fact["outcome"] is not None:
            facts.append(fact)
    return facts


def _group_cluster_buys_by_ticker(
    facts: list[ClusterBuyFact],
) -> dict[str, list[ClusterBuyFact]]:
    grouped: dict[str, list[ClusterBuyFact]] = {}
    for fact in facts:
        grouped.setdefault(str(fact["ticker"]), []).append(fact)
    return grouped


def _group_cluster_buys_by_person(
    facts: list[ClusterBuyFact],
) -> dict[str, list[ClusterBuyFact]]:
    grouped: dict[str, list[ClusterBuyFact]] = {}
    for fact in facts:
        for person_id in fact["person_ids"]:
            grouped.setdefault(str(person_id), []).append(fact)
    return grouped


def _live_event_person_ids(facts: list[ClusterBuyFact]) -> set[str]:
    """Person ids tied to at least one currently-live ``insider_cluster_buy`` event."""
    ids: set[str] = set()
    for fact in facts:
        if fact["live"]:
            ids.update(fact["person_ids"])
    return ids


def _read_trade_alert_facts(con: duckdb.DuckDBPyConnection) -> list[TradeAlertFact]:
    """Open trade alerts from the ledger, trimmed for vault reasoning."""
    rows = con.execute(
        """
        SELECT kind, ticker, fired_at, direction, entry_ref, stop, target,
               time_exit_date, confidence, evidence, outcome
        FROM trade_alerts
        WHERE outcome = 'open'
        ORDER BY fired_at DESC
        """
    ).fetchall()
    columns = (
        "kind", "ticker", "fired_at", "direction", "entry_ref", "stop", "target",
        "time_exit_date", "confidence", "evidence", "outcome",
    )  # fmt: skip
    return [dict(zip(columns, row, strict=True)) for row in rows if row[1]]


def _group_trade_alerts_by_ticker(
    facts: list[TradeAlertFact],
) -> dict[str, list[TradeAlertFact]]:
    grouped: dict[str, list[TradeAlertFact]] = {}
    for fact in facts:
        grouped.setdefault(str(fact["ticker"]), []).append(fact)
    return grouped


def _trade_alert_tickers(facts: list[TradeAlertFact]) -> set[str]:
    return {str(fact["ticker"]) for fact in facts if fact.get("ticker")}


def _group_by_person(role_rows: list[dict[str, Any]]) -> dict[str, tuple[str, list[RoleEdge]]]:
    """``person_id -> (display_name, [{ticker, company_name, flags...}, ...])``.

    ``display_name`` falls back to a short, stable label built from the
    ``person_id`` when a person has no ``canonical_name`` on file, so a person
    note is never silently skipped for lacking a name.
    """
    by_person: dict[str, tuple[str, list[RoleEdge]]] = {}
    for row in role_rows:
        person_id = str(row["person_id"])
        display = row.get("canonical_name") or f"Unknown ({person_id[:8]})"
        _, roles = by_person.setdefault(person_id, (str(display), []))
        roles.append(
            {
                "ticker": row["ticker"],
                "company_name": row["company_name"],
                "is_officer": bool(row["is_officer"]),
                "is_director": bool(row["is_director"]),
                "is_ten_pct_owner": bool(row["is_ten_pct_owner"]),
                "latest_officer_title": row["latest_officer_title"],
            }
        )
    return by_person


def _sanitize_filename(name: str) -> str:
    """A filesystem-safe note stem: collapse whitespace, drop unsafe characters."""
    collapsed = " ".join(name.split())
    cleaned = "".join(ch if (ch.isalnum() or ch in " -_.,'()") else "_" for ch in collapsed)
    return cleaned.strip() or "unknown"


def _person_filenames(by_person: dict[str, tuple[str, list[RoleEdge]]]) -> dict[str, str]:
    """``person_id -> disambiguated vault filename stem``.

    Two different real people can share a name -- identity here is
    ``person_id`` (CIK-keyed), never the name (see ``entities.PersonIndex``'s
    own docstring on this). The vault is files on disk, though, so writing
    every "Jane Doe" to the same ``Jane Doe.md`` would let one person's note
    silently overwrite another's. A short ``person_id`` suffix is added only
    when the plain sanitized name is not unique across all known people.

    Collisions are counted case-*insensitively* (``.casefold()``) even though
    the returned stem preserves original casing: the vault lives on macOS's
    default APFS volumes, which are case-insensitive, so "Jane Doe" and
    "JANE DOE" collide as the same file on disk regardless of what two Python
    strings look like. Comparing with the filesystem's own notion of equality
    is what actually prevents the silent overwrite this function exists to
    avoid -- a case-sensitive-only check was found (2026-07-26) to still lose
    a handful of notes this way against the real vault.
    """
    name_counts: dict[str, int] = {}
    for name, _roles in by_person.values():
        key = _sanitize_filename(name).casefold()
        name_counts[key] = name_counts.get(key, 0) + 1

    filenames: dict[str, str] = {}
    for person_id, (name, _roles) in by_person.items():
        base = _sanitize_filename(name)
        is_collision = name_counts[base.casefold()] > 1
        filenames[person_id] = f"{base} ({person_id[:8]})" if is_collision else base
    return filenames


def _group_by_ticker(
    role_rows: list[dict[str, Any]],
    by_person: dict[str, tuple[str, list[RoleEdge]]],
    filenames: dict[str, str],
    qualifying_person_ids: set[str],
) -> dict[str, list[RoleEdge]]:
    """``ticker -> [{canonical_name, person_link, flags...}, ...]`` for company notes.

    ``person_link`` is the exact, already-disambiguated vault filename stem to
    wiki-link to -- set only when that ``person_id`` clears the §9.1 bar
    (``qualifying_person_ids``). Non-qualifying people and owner-only rows are
    skipped outright: they remain fully queryable in DuckDB, but they are not
    useful enough to spend Obsidian/AI context on.
    """
    roles_by_ticker: dict[str, list[RoleEdge]] = {}
    for row in role_rows:
        ticker = row.get("ticker")
        person_id = str(row["person_id"])
        if (
            not ticker
            or person_id not in qualifying_person_ids
            or not _is_connection_role(row)
        ):
            continue
        display_name, _ = by_person[person_id]
        roles_by_ticker.setdefault(str(ticker), []).append(
            {
                "canonical_name": display_name,
                "person_link": filenames[person_id],
                "is_officer": bool(row["is_officer"]),
                "is_director": bool(row["is_director"]),
                "is_ten_pct_owner": bool(row["is_ten_pct_owner"]),
                "latest_officer_title": row["latest_officer_title"],
            }
        )
    return roles_by_ticker


def _is_connection_role(role: RoleEdge | dict[str, Any]) -> bool:
    """True for roles worth rendering as graph context: officer or director."""
    return bool(role.get("is_officer")) or bool(role.get("is_director"))


# --------------------------------------------------------------------------- #
# Frontmatter and sections                                                     #
# --------------------------------------------------------------------------- #
def _frontmatter(
    ticker: str,
    outgoing: list[Edge],
    incoming: list[Edge],
    *,
    has_cluster_buys: bool = False,
    has_trade_alerts: bool = False,
) -> str:
    """Obsidian YAML frontmatter: the ticker and a tag per edge/event kind present."""
    edge_tags = [f"edge/{edge_type}" for edge_type in _edge_types_present(outgoing, incoming)]
    tags = ["company", *edge_tags]
    if has_cluster_buys:
        tags.append("event/insider_cluster_buy")
    if has_trade_alerts:
        tags.append("alert/trade")
    tag_lines = "\n".join(f"  - {tag}" for tag in tags)
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


def _people_section(roles: list[RoleEdge]) -> str | None:
    """A company note's relevant people section. ``None`` when there are none.

    Only people who already earned a graph node are listed, so this section
    contains active interlocks or live insider-cluster participants rather than
    a full SEC role roster. Not split into Watching/Resolved sub-sections:
    every current role is "live" until Phase 4 ships a departure signal (§9.2),
    so a "Resolved" sub-heading here would always be empty today.
    """
    if not roles:
        return None
    lines = ["## People to watch"]
    for role in sorted(roles, key=lambda r: str(r.get("canonical_name") or "").lower()):
        lines.append(f"- {_person_link(role)} -- {_role_label(role)}")
    return "\n".join(lines)


def _person_link(role: RoleEdge) -> str:
    """A ``[[disambiguated-name]]`` wiki-link to that person's note."""
    link_target = role.get("person_link")
    if link_target:
        return f"[[{link_target}]]"
    return str(role.get("canonical_name") or "?")


def _person_frontmatter(roles: list[RoleEdge], *, has_cluster_buys: bool = False) -> str:
    """Obsidian YAML frontmatter for a person note: a tag per role/event kind held."""
    tags = ["person"]
    if any(role.get("is_director") for role in roles):
        tags.append("role/director")
    if any(role.get("is_officer") for role in roles):
        tags.append("role/officer")
    if any(role.get("is_ten_pct_owner") for role in roles):
        tags.append("role/ten_pct_owner")
    if has_cluster_buys:
        tags.append("event/insider_cluster_buy")
    tag_lines = "\n".join(f"  - {tag}" for tag in tags)
    return f"---\ntags:\n{tag_lines}\n---"


def _role_label(role: RoleEdge) -> str:
    """A human label combining every relationship flag set on one role.

    E.g. ``"Director"``, ``"Officer (CFO)"``, or ``"Director, Officer (CEO)"``
    for someone who has held both relationships (Phase 3 does not attempt to
    say when each applied -- see the design doc's title-staleness note).
    Falls back to ``"Insider"`` for a role with no flag set, which
    ``validators/people.py`` already flags as a warning rather than silently
    hiding.
    """
    parts: list[str] = []
    if role.get("is_director"):
        parts.append("Director")
    if role.get("is_officer"):
        title = role.get("latest_officer_title")
        parts.append(f"Officer ({title})" if title else "Officer")
    if role.get("is_ten_pct_owner"):
        parts.append("10% owner")
    return ", ".join(parts) if parts else "Insider"


def _date_span(start: date, end: date) -> str:
    """A human date range, collapsed to one date when the window is a single day."""
    if start == end:
        return start.isoformat()
    return f"{start.isoformat()} to {end.isoformat()}"


def _cluster_buy_bullet(fact: ClusterBuyFact, *, link_prefix: str = "") -> str:
    """One cluster-buy fact line: insiders, dollar value, window, and (if
    resolved) its graded outcome."""
    detail = f"{fact['distinct_insiders']} insiders"
    value = fact.get("total_dollar_value")
    if value:
        detail += f", ${value:,.0f}"
    detail += f" ({_date_span(fact['window_start'], fact['window_end'])})"
    if not fact["live"]:
        outcome = fact.get("outcome")
        if outcome:
            car = fact.get("forward_abnormal_return") or 0.0
            detail += f" -- {outcome} ({car:+.1%} CAR, {LONGEST_EVALUATED_HORIZON_DAYS}d)"
        else:
            detail += " -- not yet graded"
    return f"- {link_prefix}{detail}"


def _cluster_buy_section(facts: list[ClusterBuyFact], *, link: bool) -> str | None:
    """The "Insider cluster buys" section, split into Watching/Resolved (§9.2).

    ``link=True`` (person notes) prefixes each bullet with a ``[[TICKER]]``
    link back to the company; ``link=False`` (company notes) omits it, since
    the note is already that company's own.
    """
    if not facts:
        return None
    watching = [fact for fact in facts if fact["live"]]
    resolved = [fact for fact in facts if not fact["live"]]
    lines = ["## Insider cluster buys"]
    for label, group in (("Watching", watching), ("Resolved", resolved)):
        if not group:
            continue
        lines.append("")
        lines.append(f"### {label}")
        ordered = sorted(group, key=lambda fact: fact["window_end"], reverse=True)
        lines.extend(
            _cluster_buy_bullet(fact, link_prefix=f"[[{fact['ticker']}]] -- " if link else "")
            for fact in ordered
        )
    return "\n".join(lines)


def _trade_alerts_section(alerts: list[TradeAlertFact]) -> str | None:
    """Open trade-alert ledger rows, compact enough for an AI context window."""
    if not alerts:
        return None
    lines = ["## Trade alerts"]
    for alert in alerts:
        lines.append(_trade_alert_bullet(alert))
    return "\n".join(lines)


def _trade_alert_bullet(alert: TradeAlertFact) -> str:
    direction = _direction_word(int(alert.get("direction") or 0))
    fired_at = alert.get("fired_at")
    fired = fired_at.date().isoformat() if hasattr(fired_at, "date") else str(fired_at or "?")
    detail = (
        f"- **{alert['kind']}** {direction}, confidence {alert.get('confidence') or '?'} "
        f"(fired {fired})"
    )
    plan = _trade_plan_summary(alert)
    if plan:
        detail += f" -- {plan}"
    basis = _trade_alert_basis(alert.get("evidence"))
    if basis:
        detail += f" -- basis: {basis}"
    return detail


def _trade_plan_summary(alert: TradeAlertFact) -> str:
    parts: list[str] = []
    for label, key in (("entry", "entry_ref"), ("stop", "stop"), ("target", "target")):
        value = alert.get(key)
        if value is not None:
            parts.append(f"{label} {_money(float(value))}")
    exit_date = alert.get("time_exit_date")
    if exit_date:
        parts.append(f"exit {exit_date}")
    return ", ".join(parts)


def _trade_alert_basis(raw: Any) -> str:
    if not raw:
        return ""
    try:
        evidence = json.loads(str(raw))
    except json.JSONDecodeError:
        return _truncate(str(raw))

    parts: list[str] = []
    gap_pct = evidence.get("gap_pct")
    if gap_pct is not None:
        parts.append(f"{float(gap_pct):+.1%} gap")
    catalyst = evidence.get("catalyst")
    if isinstance(catalyst, dict):
        label = catalyst.get("event_type") or catalyst.get("event_id") or "catalyst"
        parts.append(str(label))
    elif catalyst and catalyst != "no catalyst":
        parts.append(str(catalyst))
    note = evidence.get("note") or evidence.get("basis")
    if note:
        parts.append(str(note))
    return _truncate("; ".join(parts) if parts else str(raw))


def _direction_word(direction: int) -> str:
    if direction > 0:
        return "bullish"
    if direction < 0:
        return "bearish"
    return "neutral"


def _money(value: float) -> str:
    if abs(value) >= 1:
        return f"${value:,.2f}"
    return f"${value:.4f}"


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
    "DEFAULT_INTERLOCK_CAP",
    "LONGEST_EVALUATED_HORIZON_DAYS",
    "VaultProjectionResult",
    "project_graph",
    "project_person_notes",
    "render_company_note",
    "render_person_note",
]
