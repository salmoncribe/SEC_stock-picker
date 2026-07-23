"""The connection engine: read filing prose, propose typed company edges.

For each company's most recent 10-K, the Business (Item 1) and Risk Factors
(Item 1A) sections are handed to a local LLM, which returns the named business
relationships it can find -- who the filer sells to, buys from, competes with,
or partners with. Each becomes a :class:`CompanyEdgeRecord`: a *hypothesis*, not
a fact. The event-study harness decides later whether an event at one end
actually moves the other, which is what makes an imperfect extractor safe.

Three cheap guards run before an edge is stored, because they catch the model's
known failure modes without spending a validation slot:

* **self-edges are dropped** -- the model sometimes names the filer as its own
  counterparty when a relationship is mentioned but the other party is unnamed
  ("our largest supplier accounted for 11%"). An edge from a company to itself
  is never a propagation link.
* **targets are resolved conservatively** -- a name that matches no company we
  track, or matches more than one, is kept in the graph but left unresolved, so
  a guess never invents a relationship between two real companies.
* **edges dedupe by relationship** -- the same claim across several filings is
  one row whose ``times_asserted`` counts up, not four rows.

Durability mirrors the document collector: results flush per batch and the
candidate query subtracts filings already extracted, so a multi-hour run over
the universe is resumable and an interruption costs at most one batch.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.entities import CompanyIndex, normalize_company_name
from market_intelligence.llm.base import LLMError, LLMProvider
from market_intelligence.llm.ollama import OllamaProvider
from market_intelligence.schemas.common import Source, utcnow
from market_intelligence.schemas.edges import CompanyEdgeRecord, EdgeType
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config

EXTRACTION_METHOD = "llm_filing_relationships"

#: Sections whose prose actually names counterparties. Item 1 (Business) carries
#: customers/suppliers/competitors; Item 1A (Risk Factors) names concentration
#: risks ("sales to <X> were 18% of revenue"). Nothing else is worth the tokens.
SECTION_ITEMS: tuple[str, ...] = ("1", "1A")

#: Cap on characters sent to the model. A full Item 1 fits comfortably inside
#: the default context window; beyond this the marginal relationship is rare and
#: the latency is not.
MAX_SECTION_CHARS = 40_000

DEFAULT_FLUSH_EVERY = 50

_VALID_TYPES = frozenset(t.value for t in EdgeType)

#: Schema handed to the model so decoding is constrained to this exact shape.
#: The spike proved bare "json mode" is not enough -- without a schema the model
#: will occasionally drop the ``edges`` key entirely.
EDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "type": {"type": "string", "enum": list(_VALID_TYPES)},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["target", "type", "evidence", "confidence"],
            },
        }
    },
    "required": ["edges"],
}

_SYSTEM_PROMPT = (
    "You extract explicit inter-company business relationships from SEC 10-K text. "
    "The filing is by the SOURCE company named below. Extract ONLY relationships to "
    "other SPECIFICALLY NAMED companies (never vague groups like 'our customers', and "
    "never the source company itself). Types:\n"
    "- customer: the source company SELLS to the target\n"
    "- supplier: the source company BUYS from the target\n"
    "- competitor: the target competes with the source\n"
    "- partner: joint venture, collaboration, or strategic alliance\n"
    "For each relationship quote the EXACT sentence from the text as evidence and give "
    "a confidence from 0.0 to 1.0. If no named relationships exist, return an empty list."
)


def select_sections(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str] | None = None,
    items: tuple[str, ...] = SECTION_ITEMS,
    priced_only: bool = True,
    latest_only: bool = False,
    limit: int | None = None,
    skip_extracted: bool = True,
) -> list[dict[str, Any]]:
    """Candidate sections to extract from, most recent filing first.

    ``priced_only`` restricts the *source* to companies with a return series, so
    the edges' source end is measurable. ``latest_only`` keeps just each
    company's single most recent filing -- current relationships, not twenty
    years of re-extracting the same graph, which is the difference between an
    overnight run and a week-long one. ``skip_extracted`` subtracts filings
    already present in ``company_edges`` -- the resumability guard.
    """
    params: list[Any] = list(items)
    placeholders = ", ".join(["?"] * len(items))
    sql = f"""
        SELECT s.cik, c.ticker, c.company_name, c.company_id,
               s.accession_number, s.item_code, s.report_date, s.text_path
        FROM filing_sections s
        JOIN companies c ON c.cik = s.cik
        WHERE s.item_code IN ({placeholders})
          AND c.ticker IS NOT NULL AND c.ticker <> ''
          AND s.text_path IS NOT NULL
    """
    if priced_only:
        sql += " AND c.ticker IN (SELECT DISTINCT symbol FROM daily_returns)"
    if latest_only:
        # Restrict to the single most recent filing per company: the accession
        # with the latest report_date among that company's candidate sections.
        sql += f"""
          AND s.accession_number = (
              SELECT s2.accession_number FROM filing_sections s2
              WHERE s2.cik = s.cik AND s2.item_code IN ({placeholders})
              ORDER BY s2.report_date DESC NULLS LAST, s2.accession_number
              LIMIT 1
          )
        """
        params.extend(items)
    if tickers:
        uppered = [t.upper() for t in tickers]
        sql += f" AND upper(c.ticker) IN ({', '.join(['?'] * len(uppered))})"
        params.extend(uppered)
    if skip_extracted:
        sql += """
          AND NOT EXISTS (
              SELECT 1 FROM company_edges e
              WHERE e.accession_number = s.accession_number
          )
          AND NOT EXISTS (
              SELECT 1 FROM processed_relationship_sections p
              WHERE p.accession_number = s.accession_number
                AND p.item_code = s.item_code
          )
        """
    sql += " ORDER BY s.report_date DESC NULLS LAST, s.cik, s.item_code"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    columns = (
        "cik", "ticker", "company_name", "company_id",
        "accession_number", "item_code", "report_date", "text_path",
    )  # fmt: skip
    return [dict(zip(columns, row, strict=True)) for row in con.execute(sql, params).fetchall()]


def extract_edges(
    provider: LLMProvider,
    *,
    source_name: str,
    source_ticker: str,
    text: str,
) -> list[dict[str, Any]]:
    """Ask the model for the relationships in one section's text.

    Returns the raw edge dicts (target/type/evidence/confidence). Malformed
    entries are skipped rather than trusted; the model is a proposer, and a
    proposal missing a target or carrying an unknown type is not actionable.
    """
    user = (
        f"SOURCE company: {source_name} (ticker {source_ticker})\n\n"
        f"10-K section text:\n\n{text[:MAX_SECTION_CHARS]}"
    )
    result = provider.complete_json(system=_SYSTEM_PROMPT, user=user, schema=EDGE_SCHEMA)
    raw = result.get("edges")
    if not isinstance(raw, list):
        return []

    edges: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target") or "").strip()
        etype = str(item.get("type") or "").strip().lower()
        if not target or etype not in _VALID_TYPES:
            continue
        edges.append(
            {
                "target": target,
                "type": etype,
                "evidence": str(item.get("evidence") or "").strip() or None,
                "confidence": _as_float(item.get("confidence")),
            }
        )
    return edges


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_self_edge(
    source_name: str, source_ticker: str, target: str, resolved_cik: str | None, source_cik: str
) -> bool:
    """True when the target is really the filer itself.

    Caught two ways: a resolved target CIK equal to the source, or -- before
    resolution even runs -- a target whose normalized name matches the source's.
    The second catches the "unnamed counterparty filled in with the filer"
    hallucination the spike surfaced.
    """
    if resolved_cik is not None and resolved_cik == source_cik:
        return True
    target_norm = normalize_company_name(target)
    return target_norm != "" and target_norm in (
        normalize_company_name(source_name),
        normalize_company_name(source_ticker),
    )


def _build_edge(
    candidate: dict[str, Any],
    raw: dict[str, Any],
    index: CompanyIndex,
    *,
    model: str,
    schema_version: str,
    prior_counts: dict[str, int],
) -> CompanyEdgeRecord | None:
    """Resolve, self-edge-filter, and assemble one edge record.

    Returns ``None`` for a self-edge, which is dropped outright.
    """
    source_cik = str(candidate["cik"])
    source_name = str(candidate["company_name"] or "")
    source_ticker = str(candidate["ticker"] or "")
    target_name = str(raw["target"])
    edge_type = str(raw["type"])

    resolution = index.resolve(target_name)
    if _is_self_edge(source_name, source_ticker, target_name, resolution.cik, source_cik):
        return None

    # The dedup token is the resolved ticker when known, so two spellings of the
    # same company collapse; otherwise the normalized name.
    token = resolution.ticker or normalize_company_name(target_name) or target_name
    edge_key = f"{source_cik}|{token}|{edge_type}"
    prior = prior_counts.get(edge_key, 0)
    now = utcnow()

    return CompanyEdgeRecord(
        edge_id=hashing.content_hash("edge", edge_key),
        edge_key=edge_key,
        source_cik=source_cik,
        source_ticker=source_ticker or None,
        source_company_id=candidate.get("company_id"),
        target=token,
        target_name=target_name,
        target_cik=resolution.cik,
        target_ticker=resolution.ticker,
        edge_type=edge_type,
        resolution_status=resolution.status,
        resolution_confidence=resolution.confidence,
        evidence=raw.get("evidence"),
        extraction_confidence=raw.get("confidence"),
        extraction_method=EXTRACTION_METHOD,
        extraction_model=model,
        accession_number=candidate.get("accession_number"),
        report_date=_coerce_date(candidate.get("report_date")),
        times_asserted=prior + 1,
        first_seen_time=now,
        last_seen_time=now,
        source=Source.DERIVED,
        content_hash=hashing.content_hash(edge_key, edge_type, str(resolution.cik)),
        schema_version=schema_version,
        collected_time=now,
    )


def _coerce_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _read_text(text_path: Any) -> str:
    try:
        return Path(str(text_path)).read_text(encoding="utf-8")
    except OSError:
        return ""


def _flush(
    config: Config,
    con: duckdb.DuckDBPyConnection,
    summary: RunSummary,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    result = duckdb_store.upsert_company_edges(con, rows)
    summary.inserted += result.inserted
    summary.updated += result.updated
    summary.deduped += result.deduped
    parquet.write_records(config.paths.parquet_dir, "company_edges", rows, ["edge_key"])


def _prior_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    rows = con.execute("SELECT edge_key, times_asserted FROM company_edges").fetchall()
    return {str(r[0]): int(r[1] or 0) for r in rows}


def _mark_processed(
    con: duckdb.DuckDBPyConnection,
    *,
    candidate: dict[str, Any],
    edge_count: int,
    schema_version: str,
) -> None:
    con.execute(
        """
        INSERT INTO processed_relationship_sections
          (accession_number, item_code, source_ticker, processed_time, edge_count, schema_version)
        VALUES (?, ?, ?, current_timestamp, ?, ?)
        ON CONFLICT (accession_number, item_code) DO UPDATE SET
          source_ticker = excluded.source_ticker,
          processed_time = excluded.processed_time,
          edge_count = excluded.edge_count,
          schema_version = excluded.schema_version
        """,
        [
            candidate.get("accession_number"),
            candidate.get("item_code"),
            candidate.get("ticker"),
            edge_count,
            schema_version,
        ],
    )


def sync(
    config: Config,
    *,
    tickers: list[str] | None = None,
    items: tuple[str, ...] = SECTION_ITEMS,
    limit: int | None = None,
    priced_only: bool = True,
    latest_only: bool = False,
    provider: LLMProvider | None = None,
    flush_every: int = DEFAULT_FLUSH_EVERY,
) -> RunSummary:
    """Extract relationship edges from filing sections into ``company_edges``."""
    llm = provider or OllamaProvider.from_config(config)

    with pipeline_run(config, "signals.relationships") as (con, summary):
        schema_version = config.settings.app.schema_version
        index = CompanyIndex.from_connection(con)
        prior_counts = _prior_counts(con)

        candidates = select_sections(
            con,
            tickers=tickers,
            items=items,
            priced_only=priced_only,
            latest_only=latest_only,
            limit=limit,
        )
        summary.bump("candidate_sections", len(candidates))

        pending: list[dict[str, Any]] = []
        for candidate in candidates:
            text = _read_text(candidate["text_path"])
            if not text:
                summary.bump("empty_sections")
                _mark_processed(
                    con,
                    candidate=candidate,
                    edge_count=0,
                    schema_version=schema_version,
                )
                continue

            try:
                raw_edges = extract_edges(
                    llm,
                    source_name=str(candidate["company_name"] or ""),
                    source_ticker=str(candidate["ticker"] or ""),
                    text=text,
                )
            except LLMError as exc:  # one bad section must not end the run
                summary.bump("extraction_failed")
                summary.note(f"extraction_failed:{candidate['accession_number']}:{exc}")
                continue

            summary.bump("sections_processed")
            stored_for_section = 0
            for raw in raw_edges:
                edge = _build_edge(
                    candidate,
                    raw,
                    index,
                    model=llm.model,
                    schema_version=schema_version,
                    prior_counts=prior_counts,
                )
                if edge is None:
                    summary.bump("self_edges_dropped")
                    continue
                if edge.is_validatable:
                    summary.bump("validatable_edges")
                else:
                    summary.bump("unresolved_edges")
                prior_counts[edge.edge_key] = edge.times_asserted
                summary.collected += 1
                stored_for_section += 1
                pending.append(edge.to_row())

            _mark_processed(
                con,
                candidate=candidate,
                edge_count=stored_for_section,
                schema_version=schema_version,
            )

            if len(pending) >= flush_every:
                _flush(config, con, summary, pending)
                pending = []

        _flush(config, con, summary, pending)
    return summary


__all__ = [
    "EDGE_SCHEMA",
    "EXTRACTION_METHOD",
    "SECTION_ITEMS",
    "extract_edges",
    "select_sections",
    "sync",
]
