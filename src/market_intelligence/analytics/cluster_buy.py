"""Insider cluster-buy detection: several distinct insiders, one company, one window.

Stage 3c of ``docs/specs/2026-07-25-people-insider-graph-design.md``, Decision
D. A pure derivation over ``insider_transaction`` events already stored:
filters to open-market purchases (Form 4 code ``P`` -- the one transaction
code the literature finds informative, see
``schemas.events.InsiderTransactionCode``), groups them by company into
non-overlapping windows, and emits one candidate ``events`` row per
(company, window) with ``event_type = insider_cluster_buy``. No schema
change -- this is a new ``event_type`` string in the table
``signals/dataset.py`` and ``signals/impact.py`` already consume, per
Decision D.

**Why one row per window, never one per transaction.** The base signal-graph
design's §6 documents the exact failure this must not repeat: one vesting
event produced 327 Form 4 line items on a single company-day, and counting
each as an independent draw inflated the naive t-statistic to roughly 3x the
correctly clustered one. Emitting one ``insider_cluster_buy`` event per window
-- not one per underlying purchase -- means the clustered-statistics
discipline is enforced in this producer's own natural key, before
``signals/impact.py`` ever has a chance to get it wrong.

**Windowing.** Purchases for one company are sorted by ``available_time``
(the filing date -- never the trade date, keeping this on the same
point-in-time clock as every other event in the platform) and greedily
bucketed: a window opens at the first not-yet-assigned purchase and absorbs
every subsequent purchase within ``window_days`` of that opening purchase.
The window's own ``available_time`` is the *last* purchase's filing date --
the moment the full cluster, not just its first member, became public -- so
the emitted event never claims knowledge of insiders who had not yet filed.
A window with fewer than ``min_distinct_insiders`` distinct people is not a
cluster and is dropped; that is just an ordinary single Form 4 purchase,
already covered by the existing self-control.

This is a first-cut heuristic, not a claim that these are the *right*
window/threshold values -- per the design's risk table, both are meant to be
tuned against the discovery split like every other admission threshold in
this platform, not chosen by inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.entities import person_id_for_cik
from market_intelligence.schemas.common import Source, utcnow
from market_intelligence.schemas.events import (
    DIRECTION_POSITIVE,
    EventRecord,
    EventType,
    InsiderTransactionCode,
)
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet
from market_intelligence.validators.events import validate_event

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config

EXTRACTION_METHOD = "insider_cluster_buy_v1"


@dataclass(frozen=True)
class PurchaseObservation:
    """One open-market purchase (code P), the raw input to clustering."""

    owner_cik: str
    accession_number: str
    available_time: datetime
    shares: float | None
    #: The event's already-computed dollar magnitude (shares * price), reused
    #: verbatim from ``collectors/insider.py::_build_event`` rather than
    #: re-derived from a separately re-parsed price -- one source of truth for
    #: "what did this purchase cost" instead of two that could silently drift.
    dollar_value: float | None
    company_id: str
    cik: str | None
    ticker: str | None


@dataclass(frozen=True)
class ClusterBuyCandidate:
    """One candidate ``insider_cluster_buy`` event: one company, one window."""

    company_id: str
    cik: str | None
    ticker: str | None
    window_start: datetime
    window_end: datetime
    owner_ciks: tuple[str, ...]
    total_shares: float
    total_dollar_value: float
    distinct_insiders: int
    accession_numbers: tuple[str, ...]


def _finalize_window(
    window: list[PurchaseObservation], *, min_distinct_insiders: int
) -> ClusterBuyCandidate | None:
    if not window:
        return None
    distinct_ciks = sorted({obs.owner_cik for obs in window})
    if len(distinct_ciks) < min_distinct_insiders:
        return None
    total_shares = sum(obs.shares for obs in window if obs.shares is not None)
    total_dollar = sum(obs.dollar_value for obs in window if obs.dollar_value is not None)
    first = window[0]
    return ClusterBuyCandidate(
        company_id=first.company_id,
        cik=first.cik,
        ticker=first.ticker,
        window_start=window[0].available_time,
        window_end=window[-1].available_time,
        owner_ciks=tuple(distinct_ciks),
        total_shares=float(total_shares),
        total_dollar_value=float(total_dollar),
        distinct_insiders=len(distinct_ciks),
        accession_numbers=tuple(sorted({obs.accession_number for obs in window})),
    )


def compute_cluster_buys(
    observations: list[PurchaseObservation],
    *,
    window_days: int,
    min_distinct_insiders: int,
) -> list[ClusterBuyCandidate]:
    """Group open-market purchases into non-overlapping per-company clusters.

    Pure: takes plain observations (already filtered to code ``P``), returns
    plain candidates. See the module docstring for the windowing rule.
    """
    by_company: dict[str, list[PurchaseObservation]] = {}
    for obs in observations:
        by_company.setdefault(obs.company_id, []).append(obs)

    span = timedelta(days=window_days)
    candidates: list[ClusterBuyCandidate] = []
    for obs_list in by_company.values():
        ordered = sorted(obs_list, key=lambda o: o.available_time)
        window: list[PurchaseObservation] = []
        window_start: datetime | None = None
        for obs in ordered:
            if window_start is None or obs.available_time - window_start <= span:
                if window_start is None:
                    window_start = obs.available_time
                window.append(obs)
            else:
                candidate = _finalize_window(window, min_distinct_insiders=min_distinct_insiders)
                if candidate is not None:
                    candidates.append(candidate)
                window = [obs]
                window_start = obs.available_time
        candidate = _finalize_window(window, min_distinct_insiders=min_distinct_insiders)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _load_purchases(con: duckdb.DuckDBPyConnection) -> list[PurchaseObservation]:
    rows = con.execute(
        """
        SELECT
            json_extract_string(payload, '$.owner_cik') AS owner_cik,
            accession_number,
            available_time,
            json_extract_string(payload, '$.shares') AS shares,
            magnitude,
            company_id,
            cik,
            ticker
        FROM events
        WHERE event_type = ?
          AND event_subtype = ?
          AND company_id IS NOT NULL
          AND available_time IS NOT NULL
          AND json_extract_string(payload, '$.owner_cik') IS NOT NULL
          AND json_extract_string(payload, '$.owner_cik') <> ''
        """,
        [EventType.INSIDER_TRANSACTION, InsiderTransactionCode.PURCHASE],
    ).fetchall()
    observations: list[PurchaseObservation] = []
    for row in rows:
        owner_cik, accession, available_time, shares, magnitude, company_id, cik, ticker = row
        observations.append(
            PurchaseObservation(
                owner_cik=str(owner_cik),
                accession_number=str(accession),
                available_time=available_time,
                shares=float(shares) if shares not in (None, "") else None,
                dollar_value=float(magnitude) if magnitude is not None else None,
                company_id=str(company_id),
                cik=cik,
                ticker=ticker,
            )
        )
    return observations


def _event_row(
    candidate: ClusterBuyCandidate, *, window_days: int, schema_version: str
) -> dict[str, Any]:
    event_key = hashing.content_hash(
        candidate.company_id,
        candidate.window_start.isoformat(),
        candidate.window_end.isoformat(),
        sorted(candidate.owner_ciks),
    )
    person_ids = sorted(person_id_for_cik(cik) for cik in candidate.owner_ciks)
    payload: dict[str, Any] = {
        "owner_ciks": list(candidate.owner_ciks),
        "person_ids": person_ids,
        "total_shares": candidate.total_shares,
        "distinct_insiders": candidate.distinct_insiders,
        "window_days": window_days,
        "window_start": candidate.window_start.isoformat(),
        "accession_numbers": list(candidate.accession_numbers),
        "magnitude_unit": "usd",
    }
    now = utcnow()
    record = EventRecord(
        event_id=hashing.content_hash("event", EventType.INSIDER_CLUSTER_BUY, event_key),
        event_type=EventType.INSIDER_CLUSTER_BUY,
        event_key=event_key,
        company_id=candidate.company_id,
        cik=candidate.cik,
        ticker=candidate.ticker,
        event_subtype=None,
        accession_number=None,
        filing_id=None,
        event_time=candidate.window_end,
        available_time=candidate.window_end,
        magnitude=candidate.total_dollar_value or None,
        direction=DIRECTION_POSITIVE,
        payload=payload,
        extraction_method=EXTRACTION_METHOD,
        source=Source.DERIVED,
        content_hash=hashing.content_hash(
            event_key, candidate.total_shares, candidate.distinct_insiders
        ),
        schema_version=schema_version,
        collected_time=now,
    )
    validate_event(record, today=now)
    return record.to_row()


def detect(
    config: Config,
    *,
    window_days: int | None = None,
    min_distinct_insiders: int | None = None,
) -> RunSummary:
    """Derive ``insider_cluster_buy`` candidate events from stored Form 4 purchases.

    Writes into ``events`` alongside every other producer, so
    ``signals/dataset.py``/``signals/impact.py`` measure this cell through the
    existing gate unchanged. ``window_days``/``min_distinct_insiders`` override
    ``config.settings.people_graph`` for one run (mainly for tests).
    """
    settings = config.settings.people_graph
    window = window_days or settings.cluster_buy_window_days
    min_insiders = min_distinct_insiders or settings.cluster_buy_min_insiders

    with pipeline_run(config, "people.detect-cluster-buys") as (con, summary):
        schema_version = config.settings.app.schema_version
        observations = _load_purchases(con)
        summary.bump("purchase_observations", len(observations))

        candidates = compute_cluster_buys(
            observations, window_days=window, min_distinct_insiders=min_insiders
        )
        summary.bump("candidate_windows", len(candidates))

        rows: list[dict[str, Any]] = []
        rejected = 0
        for candidate in candidates:
            row = _event_row(candidate, window_days=window, schema_version=schema_version)
            if row["validation_status"] == "rejected":
                rejected += 1
                continue
            rows.append(row)

        summary.collected = len(candidates)
        summary.rejected = rejected
        if rows:
            result = duckdb_store.upsert_events(con, rows)
            summary.inserted = result.inserted
            summary.updated = result.updated
            summary.deduped = result.deduped
            parquet.write_records(
                config.paths.parquet_dir, "events", rows, ["event_type", "event_key"]
            )

    return summary


__all__ = [
    "EXTRACTION_METHOD",
    "ClusterBuyCandidate",
    "PurchaseObservation",
    "compute_cluster_buys",
    "detect",
]
