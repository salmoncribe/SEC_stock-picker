"""Dataset builder: join stored events to the forward returns that grade them.

Reads ``events`` and ``daily_returns`` and writes ``event_samples`` -- one row
per (event, target company, horizon). Nothing here fetches; it is pure
computation over data already collected.

**Everything is keyed on ``available_time``, never ``event_time``.** An insider
trades on day D and the Form 4 lands up to two business days later; scoring the
trade from D would measure a move against information nobody had. The window
then starts on the first trading day *strictly after* the filing date, because
filing dates carry no time of day and an after-hours filing is not tradeable
that session. See ``analytics.eventstudy`` for the full argument.

Work is flushed per symbol so an interrupted build keeps what it finished,
matching the price collector. The alternative -- buffering several million rows
for one terminal write -- would make a long build all-or-nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.analytics.eventstudy import ReturnSeries, assign_split
from market_intelligence.analytics.returns import ReturnPoint
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.schemas.common import utcnow
from market_intelligence.schemas.samples import SELF_EDGE, EventSampleRecord
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet

if TYPE_CHECKING:
    import duckdb

    from market_intelligence.config import Config

#: Horizons measured by default. Short enough that a days-to-weeks diffusion
#: effect is still visible, spread enough to show whether an effect decays
#: (real, slowly priced in) or reverses (noise or liquidity, not information).
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20)

#: Chronological boundary between the half that decides what is admitted and
#: the half that reports the track record. Roughly a 70/30 split of the price
#: history, late enough that discovery covers a full market cycle.
DEFAULT_SPLIT_DATE = date(2023, 1, 1)


def _load_returns(con: duckdb.DuckDBPyConnection, symbol: str) -> list[ReturnPoint]:
    """One symbol's abnormal-return series, oldest first."""
    rows = con.execute(
        """
        SELECT price_date, total_return, abnormal_return
        FROM daily_returns
        WHERE symbol = ?
        ORDER BY price_date
        """,
        [symbol],
    ).fetchall()
    return [
        ReturnPoint(
            symbol=symbol,
            price_date=row[0],
            total_return=row[1] if row[1] is not None else 0.0,
            abnormal_return=row[2],
        )
        for row in rows
    ]


def _type_subtype_clauses(
    event_types: list[str] | None, subtypes: list[str] | None, *, prefix: str = ""
) -> tuple[list[str], list[Any]]:
    """WHERE clauses + params for the optional event_type / event_subtype filter.

    Shared by every query that reads ``events`` in this module -- the row-fetch
    queries and the fingerprint queries alike -- so a filter change can never
    drift between "what gets fetched" and "what gets fingerprinted". ``prefix``
    is a table-alias dot-prefix (e.g. ``"e."``) for queries that alias the
    table; unaliased callers leave it empty.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if event_types:
        clauses.append(f"{prefix}event_type IN ({', '.join('?' * len(event_types))})")
        params.extend(event_types)
    if subtypes:
        clauses.append(f"{prefix}event_subtype IN ({', '.join('?' * len(subtypes))})")
        params.extend(subtypes)
    return clauses, params


def _symbols_with_events(
    con: duckdb.DuckDBPyConnection, event_types: list[str] | None, subtypes: list[str] | None
) -> list[str]:
    """Tickers that have both events and a return series, so a join is possible."""
    type_clauses, type_params = _type_subtype_clauses(event_types, subtypes, prefix="e.")
    clauses = ["e.ticker IS NOT NULL", *type_clauses]
    params: list[Any] = [*type_params]

    rows = con.execute(
        f"""
        SELECT DISTINCT e.ticker
        FROM events e
        WHERE {" AND ".join(clauses)}
          AND EXISTS (SELECT 1 FROM daily_returns r WHERE r.symbol = e.ticker)
        ORDER BY e.ticker
        """,
        params,
    ).fetchall()
    return [str(row[0]) for row in rows]


def _events_for_symbol(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    event_types: list[str] | None,
    subtypes: list[str] | None,
) -> list[tuple]:
    type_clauses, type_params = _type_subtype_clauses(event_types, subtypes)
    clauses = ["ticker = ?", "available_time IS NOT NULL", *type_clauses]
    params: list[Any] = [symbol, *type_params]

    return con.execute(
        f"""
        SELECT event_id, event_type, event_subtype, available_time, magnitude, direction
        FROM events
        WHERE {" AND ".join(clauses)}
        """,
        params,
    ).fetchall()


def _rows_for_symbol(
    symbol: str,
    returns: ReturnSeries,
    events: list[tuple],
    horizons: tuple[int, ...],
    split_date: date,
    schema_version: str,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Score one symbol's events against its own return series.

    Pure function of its arguments (no I/O) -- the single place that turns
    ``(events, returns) -> event_samples rows``. Both :func:`build` and
    :func:`build_incremental` call this same function, so "an incremental run
    writes the same rows a full rebuild would" is a structural property (one
    implementation, same inputs in -> same rows out) rather than something two
    hand-synchronized copies merely try to guarantee.

    Returns ``(rows, collected, unmeasurable, purged)``.
    """
    rows: list[dict[str, Any]] = []
    collected = 0
    unmeasurable = 0
    purged = 0

    for event_id, etype, subtype, available_time, magnitude, direction in events:
        available_on = available_time.date()

        for horizon in horizons:
            collected += 1
            window = returns.forward(available_on, horizon)
            if window is None:
                # Not enough measured trading days after the event -- near the
                # end of the series, or a gap in returns.
                unmeasurable += 1
                continue

            split = assign_split(window, split_date)
            if split is None:
                # Straddles the boundary: its label draws on prices from both
                # halves, so it belongs to neither.
                purged += 1
                continue

            record = EventSampleRecord(
                sample_id=hashing.content_hash("sample", event_id, SELF_EDGE, horizon),
                event_id=event_id,
                edge_id=SELF_EDGE,
                event_type=etype,
                event_subtype=subtype,
                source_ticker=symbol,
                target_ticker=symbol,
                horizon_days=horizon,
                available_on=available_on,
                t0=window.t0,
                window_end=window.window_end,
                forward_abnormal_return=window.cumulative_abnormal_return,
                magnitude=magnitude,
                direction=direction,
                split=split.value,
                features={"magnitude_usd": magnitude, "direction": direction},
                schema_version=schema_version,
                collected_time=utcnow(),
            )
            rows.append(record.to_row())

    return rows, collected, unmeasurable, purged


def build(
    config: Config,
    *,
    event_types: list[str] | None = None,
    subtypes: list[str] | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    split_date: date = DEFAULT_SPLIT_DATE,
) -> RunSummary:
    """Build (event, target, horizon) samples for every event with a return series.

    Each event is scored against its own issuer via the ``self`` edge -- the
    positive control. Propagation edges reuse this same builder and table once
    the relationship graph exists.
    """
    with pipeline_run(config, "signals.dataset") as (con, summary):
        schema_version = config.settings.app.schema_version
        symbols = _symbols_with_events(con, event_types, subtypes)
        summary.note(f"{len(symbols)} symbols have both events and returns")

        collected = 0
        inserted = 0
        updated = 0
        unmeasurable = 0
        purged = 0

        for symbol in symbols:
            points = _load_returns(con, symbol)
            if not points:
                continue
            # Indexed once per symbol; the inner loop issues thousands of
            # lookups against it.
            returns = ReturnSeries(points)
            events = _events_for_symbol(con, symbol, event_types, subtypes)

            rows, c, u, p = _rows_for_symbol(
                symbol, returns, events, horizons, split_date, schema_version
            )
            collected += c
            unmeasurable += u
            purged += p

            if rows:
                result = duckdb_store.upsert_event_samples(con, rows)
                inserted += result.inserted
                updated += result.updated
                parquet.write_records(
                    config.paths.parquet_dir,
                    "event_samples",
                    rows,
                    ["sample_id"],
                    partition_col="event_type",
                )

        summary.collected = collected
        summary.inserted = inserted
        summary.updated = updated
        summary.bump("unmeasurable", unmeasurable)
        summary.bump("purged_at_split", purged)
        summary.note(
            f"{unmeasurable} unmeasurable (no complete forward window), "
            f"{purged} purged at the {split_date} split boundary"
        )

    return summary


# --------------------------------------------------------------------------
# Incremental build
#
# build() recomputes every symbol's entire (events x horizons) cross product
# on every run -- cost scales with the *total* size of the dataset, not with
# what changed since yesterday, and it gets slower every day forever. The
# functions below let a symbol be skipped entirely when nothing it depends on
# has changed, while staying provably equivalent to a full rebuild's output.
#
# Watermark signal: content checksums, not collected_time. ``collected_time``
# is stamped fresh (``utcnow()``) on every row a producer *offers* to upsert,
# not just on rows whose value actually changed -- and
# ``collectors.returns.compute()`` recomputes and re-offers a symbol's entire
# return history, unscoped, on every autopilot run. Reading collected_time as
# "has this symbol's data changed" would therefore never skip anything: every
# symbol's daily_returns watermark would advance every single day regardless
# of whether any value moved. A checksum over each row's own ``content_hash``
# (itself a hash of the row's real fields, set once by the producer that
# derived it) only changes when the underlying data does, whether that's a
# brand-new row or a correction to an existing one.
#
# Per-symbol, not per-event: a bounded "reprocess events within N trading
# days of the newest return" window was considered and rejected. It would be
# unsound unless a correction to daily_returns can be proven to only ever
# land near the tail of a symbol's series -- and compute_abnormal_returns
# recomputing full history on every run does not guarantee that (a corporate
# action adjustment, e.g., can retroactively revise an old adjusted close).
# Skipping at the symbol granularity sidesteps that question entirely: for
# ANY symbol whose returns or events checksum moved, at all, for any reason,
# every one of its events is rescored against its complete current return
# series -- identical work to what build() already does for that symbol, so
# equivalence for a reprocessed symbol is exact by construction rather than
# by a lookback-window boundary being wide enough. This also resolves the
# "previously unmeasurable event becomes measurable later" trap without
# needing a persisted per-event retry marker: an event stays unmeasurable
# only because its symbol's return series has not yet grown enough trading
# days past it. The only way that stops being true is new/changed rows
# landing in daily_returns for that symbol -- which is exactly the condition
# that moves the returns checksum and forces the whole symbol (that event
# included) to be rescored. There is no path by which an event's
# measurability can change without its symbol's watermark also changing.
# --------------------------------------------------------------------------

#: IN-list batch size for the fingerprint/watermark queries below. Mirrors
#: storage/duckdb.py's _BATCH_CHUNK_ROWS pattern (bound memory/plan size
#: regardless of how large the symbol universe grows) without reaching into
#: that module's private constant.
_FINGERPRINT_CHUNK_SIZE = 5_000

#: (row_count, content_checksum) -- the fingerprint of one symbol's slice of
#: either daily_returns or events.
_Fingerprint = tuple[int, int]


def _chunked(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _build_fingerprint(
    event_types: list[str] | None,
    subtypes: list[str] | None,
    horizons: tuple[int, ...],
    split_date: date,
    schema_version: str,
) -> str:
    """Identity of every parameter that affects what build() writes.

    A watermark is only meaningful for the exact parameter set it was
    recorded under: horizons, the split date, the type/subtype filter, and
    the schema version all change what "correctly built" means for a symbol.
    Folding them into the watermark's key means a parameter change (e.g.
    adding a horizon) can never be misread as "this symbol's raw data didn't
    change" -- a fingerprint mismatch forces every symbol to be reprocessed
    under the new parameters, exactly as a fresh full build() would.
    """
    return hashing.sha256_json(
        {
            "event_types": sorted(event_types) if event_types else None,
            "subtypes": sorted(subtypes) if subtypes else None,
            "horizons": list(horizons),
            "split_date": split_date.isoformat(),
            "schema_version": schema_version,
        }
    )


def _returns_fingerprints(
    con: duckdb.DuckDBPyConnection, symbols: list[str]
) -> dict[str, _Fingerprint]:
    """COUNT + content checksum of each symbol's full ``daily_returns`` slice.

    One GROUP BY scan per chunk instead of one query per symbol -- cheap
    relative to the full per-symbol row materialization ``_load_returns``
    does, and the whole point is to decide whether that heavier read is even
    necessary. ``COALESCE(content_hash, return_id)`` so a row somehow missing
    a content_hash still contributes a stable, distinguishing value instead
    of silently dropping out of the checksum (it still can't hide from
    COUNT(*) either way).
    """
    out: dict[str, _Fingerprint] = {}
    for chunk in _chunked(symbols, _FINGERPRINT_CHUNK_SIZE):
        if not chunk:
            continue
        placeholders = ", ".join("?" * len(chunk))
        rows = con.execute(
            f"""
            SELECT symbol, COUNT(*), COALESCE(SUM(hash(COALESCE(content_hash, return_id))), 0)
            FROM daily_returns
            WHERE symbol IN ({placeholders})
            GROUP BY symbol
            """,
            list(chunk),
        ).fetchall()
        for symbol, count, checksum in rows:
            out[str(symbol)] = (int(count), int(checksum))
    return out


def _events_fingerprints(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str],
    event_types: list[str] | None,
    subtypes: list[str] | None,
) -> dict[str, _Fingerprint]:
    """COUNT + content checksum of each symbol's ``_events_for_symbol`` slice.

    Uses the same :func:`_type_subtype_clauses` filter as the real fetch, so
    this can never fingerprint a different row set than the one
    ``_events_for_symbol`` would actually read.
    """
    out: dict[str, _Fingerprint] = {}
    type_clauses, type_params = _type_subtype_clauses(event_types, subtypes)
    for chunk in _chunked(symbols, _FINGERPRINT_CHUNK_SIZE):
        if not chunk:
            continue
        placeholders = ", ".join("?" * len(chunk))
        clauses = [f"ticker IN ({placeholders})", "available_time IS NOT NULL", *type_clauses]
        params = [*chunk, *type_params]
        rows = con.execute(
            f"""
            SELECT ticker, COUNT(*), COALESCE(SUM(hash(COALESCE(content_hash, event_id))), 0)
            FROM events
            WHERE {" AND ".join(clauses)}
            GROUP BY ticker
            """,
            params,
        ).fetchall()
        for ticker, count, checksum in rows:
            out[str(ticker)] = (int(count), int(checksum))
    return out


def _load_build_watermarks(
    con: duckdb.DuckDBPyConnection, symbols: list[str], fingerprint: str
) -> dict[str, tuple[_Fingerprint, _Fingerprint]]:
    """Each symbol's watermark as of the last successful build under ``fingerprint``.

    Returns ``{symbol: (returns_fingerprint, events_fingerprint)}``. A symbol
    absent from the result has never been built under this exact parameter
    set (first build, or the parameters changed since it was last built) and
    must be processed unconditionally.
    """
    out: dict[str, tuple[_Fingerprint, _Fingerprint]] = {}
    for chunk in _chunked(symbols, _FINGERPRINT_CHUNK_SIZE):
        if not chunk:
            continue
        placeholders = ", ".join("?" * len(chunk))
        rows = con.execute(
            f"""
            SELECT symbol, returns_row_count, returns_checksum, events_row_count, events_checksum
            FROM dataset_build_watermarks
            WHERE build_fingerprint = ? AND symbol IN ({placeholders})
            """,
            [fingerprint, *chunk],
        ).fetchall()
        for symbol, r_count, r_sum, e_count, e_sum in rows:
            out[str(symbol)] = ((int(r_count), int(r_sum)), (int(e_count), int(e_sum)))
    return out


def _save_build_watermark(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    fingerprint: str,
    returns_fp: _Fingerprint,
    events_fp: _Fingerprint,
    built_time: datetime,
) -> None:
    """Record what this symbol looked like at the end of a successful flush.

    Delete-then-insert rather than an UPDATE: a symbol may have no prior row
    (first build) or one under a different, now-superseded fingerprint
    (parameters changed) -- both are handled the same way by replacing
    whatever is there for this exact ``(symbol, fingerprint)`` pair. Called
    immediately after this symbol's samples are flushed, inside the same
    per-symbol loop iteration that does the flush, so an interrupted run
    leaves the watermark table consistent with what was actually written:
    only symbols that finished are marked done, matching the "flush per
    symbol so an interrupted build keeps what it finished" invariant this
    module already relies on for event_samples/parquet.
    """
    con.execute(
        "DELETE FROM dataset_build_watermarks WHERE symbol = ? AND build_fingerprint = ?",
        [symbol, fingerprint],
    )
    con.execute(
        """
        INSERT INTO dataset_build_watermarks
            (symbol, build_fingerprint, returns_row_count, returns_checksum,
             events_row_count, events_checksum, last_built_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [symbol, fingerprint, returns_fp[0], returns_fp[1], events_fp[0], events_fp[1], built_time],
    )


def build_incremental(
    config: Config,
    *,
    event_types: list[str] | None = None,
    subtypes: list[str] | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    split_date: date = DEFAULT_SPLIT_DATE,
) -> RunSummary:
    """Incremental equivalent of :func:`build`.

    Skips any symbol whose ``daily_returns`` and ``events`` rows are both
    byte-for-byte unchanged (by content checksum, see the module comment
    above) since the last successful build under these exact parameters. Any
    symbol that IS reprocessed gets the identical full treatment build() would
    give it -- same :func:`_rows_for_symbol` call, same complete return
    series, same complete event list -- so its output is exactly what a full
    rebuild would write, not an approximation of it. The final
    ``event_samples`` state after repeated incremental runs is therefore
    identical to running :func:`build` once, as long as every run that
    touched a symbol completed (see :func:`_save_build_watermark`: an
    interrupted run simply leaves that symbol unwatermarked, so the next run
    reprocesses it rather than skipping it).

    Not wired into the default pipeline; call this explicitly once its output
    has been checked against :func:`build`.
    """
    with pipeline_run(config, "signals.dataset.incremental") as (con, summary):
        schema_version = config.settings.app.schema_version
        fingerprint = _build_fingerprint(
            event_types, subtypes, horizons, split_date, schema_version
        )
        symbols = _symbols_with_events(con, event_types, subtypes)
        summary.note(f"{len(symbols)} symbols have both events and returns")

        returns_fps = _returns_fingerprints(con, symbols)
        events_fps = _events_fingerprints(con, symbols, event_types, subtypes)
        prior_watermarks = _load_build_watermarks(con, symbols, fingerprint)

        collected = 0
        inserted = 0
        updated = 0
        unmeasurable = 0
        purged = 0
        symbols_skipped = 0
        symbols_reprocessed = 0

        for symbol in symbols:
            returns_fp = returns_fps.get(symbol, (0, 0))
            events_fp = events_fps.get(symbol, (0, 0))
            prior = prior_watermarks.get(symbol)

            if prior is not None and prior == (returns_fp, events_fp):
                symbols_skipped += 1
                continue

            symbols_reprocessed += 1
            points = _load_returns(con, symbol)
            if not points:
                # No return series (should not happen given _symbols_with_events's
                # own EXISTS check, but mirrors build()'s guard). Nothing to
                # write, but still record the watermark so this symbol is not
                # rechecked from scratch every single run.
                _save_build_watermark(con, symbol, fingerprint, returns_fp, events_fp, utcnow())
                continue

            # Indexed once per symbol; the inner loop issues thousands of
            # lookups against it.
            returns = ReturnSeries(points)
            events = _events_for_symbol(con, symbol, event_types, subtypes)

            rows, c, u, p = _rows_for_symbol(
                symbol, returns, events, horizons, split_date, schema_version
            )
            collected += c
            unmeasurable += u
            purged += p

            if rows:
                result = duckdb_store.upsert_event_samples(con, rows)
                inserted += result.inserted
                updated += result.updated
                parquet.write_records(
                    config.paths.parquet_dir,
                    "event_samples",
                    rows,
                    ["sample_id"],
                    partition_col="event_type",
                )

            _save_build_watermark(con, symbol, fingerprint, returns_fp, events_fp, utcnow())

        summary.collected = collected
        summary.inserted = inserted
        summary.updated = updated
        summary.bump("unmeasurable", unmeasurable)
        summary.bump("purged_at_split", purged)
        summary.bump("symbols_skipped", symbols_skipped)
        summary.bump("symbols_reprocessed", symbols_reprocessed)
        summary.note(
            f"{symbols_skipped} symbols unchanged and skipped, "
            f"{symbols_reprocessed} reprocessed "
            f"({unmeasurable} unmeasurable, {purged} purged at the {split_date} split boundary)"
        )

    return summary


def _validatable_edges(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str, date | None]]:
    """Resolved edges whose both ends have a return series, grouped-ready.

    Returns ``(source_ticker, target_ticker, edge_type, known_from)``. Only
    resolved edges with a real target and a source distinct from the target can
    be measured -- the same is-validatable test the collector records,
    re-checked here against the return series so a target with no prices is
    never queued.

    ``known_from`` is the earliest ``report_date`` on which this relationship is
    on the record, and is what :func:`build_propagation` uses to keep a pair
    point-in-time. It comes from a GROUP BY rather than the ``SELECT DISTINCT``
    this query used before carrying a date: ``company_edges`` is keyed on
    ``(source_cik, target, edge_type)``, so one *ticker*-level relationship can
    hold several rows -- two source CIKs behind one ticker, two spellings of the
    same target name -- each with its own report date, and selecting the date
    without grouping would turn one relationship back into several. ``MIN``
    takes the earliest of them because each of those dates is a real filing that
    really did assert this relationship.
    """
    rows = con.execute(
        """
        SELECT e.source_ticker, e.target_ticker, e.edge_type, MIN(e.report_date)
        FROM company_edges e
        WHERE e.resolution_status = 'resolved'
          AND e.target_ticker IS NOT NULL AND e.target_ticker <> ''
          AND e.source_ticker IS NOT NULL AND e.source_ticker <> ''
          AND e.source_ticker <> e.target_ticker
          AND EXISTS (SELECT 1 FROM daily_returns r WHERE r.symbol = e.source_ticker)
          AND EXISTS (SELECT 1 FROM daily_returns r WHERE r.symbol = e.target_ticker)
        GROUP BY e.source_ticker, e.target_ticker, e.edge_type
        ORDER BY e.target_ticker, e.source_ticker, e.edge_type
        """
    ).fetchall()
    return [(str(r[0]), str(r[1]), str(r[2]), r[3]) for r in rows]


def _dedupe_samples(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Collapse sample rows sharing ``event_samples``' natural key; last wins.

    Returns ``(rows, dropped)``. Keyed on
    :data:`storage.duckdb.EVENT_SAMPLE_KEY`, i.e. the table's own UNIQUE
    constraint, so a batch that reaches the upsert can no longer contain two
    rows the database would refuse. Doing this here rather than leaving it to
    the storage layer keeps the collapse where the fan-out happens -- a
    propagation batch is assembled from many edges pointing into one target, and
    ``dropped`` is reported so a collapse shows up as a counter instead of as an
    unexplained shortfall between rows built and rows written.
    """
    collapsed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        collapsed[tuple(row.get(col) for col in duckdb_store.EVENT_SAMPLE_KEY)] = row
    return list(collapsed.values()), len(rows) - len(collapsed)


def build_propagation(
    config: Config,
    *,
    event_types: list[str] | None = None,
    subtypes: list[str] | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    split_date: date = DEFAULT_SPLIT_DATE,
) -> RunSummary:
    """Build propagation samples: an event at the source, the target's return.

    For every validatable edge, each of the *source's* events is scored against
    the *target's* forward return, with ``edge_id`` set to the relationship type.
    Those samples land in the same ``event_samples`` table the self-control uses,
    so the gate and the promotion ladder judge propagation cells (e.g.
    ``insider_transaction/P/customer/20d``) with no additional machinery.

    Indexed by target: each target's return series is built once and reused for
    every edge pointing into it, and rows flush per target so an interrupted
    build keeps what it finished.

    **A pair is only formed once the edge was on the record.** An (event, edge)
    pair requires ``edge.report_date <= event.available_time``; a relationship
    first disclosed in 2024 may not be used to explain a 2016 return. Without
    that test the graph is used at full present-day density across all of
    history, which is lookahead in the *edge* dimension -- invisible in the
    result, and enough on its own to inflate every propagation number here.

    The early years are consequently sparse, and that sparsity is left as it is
    rather than patched around. ``collectors.relationships`` extracts from each
    company's single most recent filing (its ``latest_only`` candidate rule), so
    ``company_edges`` holds one row per relationship dated at the *newest*
    filing that asserted it -- an edge that has existed since 2016 is typically
    on the record here only from its 2024 report date. The filter therefore
    claims fewer relationships than really existed at any past moment. That
    error has one direction: it withholds true edges, never invents them, so a
    cell that clears the gate under it would only have cleared it by more with
    complete history. Widening the graph backwards means re-extracting older
    filings, not loosening this test.

    An edge with no ``report_date`` at all carries no point-in-time evidence in
    either direction and is paired unconditionally; ``edges_undated`` counts
    them so the size of that exception is visible in the run rather than
    assumed to be zero.
    """
    with pipeline_run(config, "signals.dataset.propagation") as (con, summary):
        schema_version = config.settings.app.schema_version
        edges = _validatable_edges(con)
        by_target: dict[str, list[tuple[str, str, date | None]]] = {}
        for source, target, edge_type, known_from in edges:
            by_target.setdefault(target, []).append((source, edge_type, known_from))
        summary.note(f"{len(edges)} validatable edges into {len(by_target)} targets")
        undated = sum(1 for edge in edges if edge[3] is None)

        collected = inserted = updated = unmeasurable = purged = 0
        deduped = not_yet_known = 0

        # A source's events don't depend on which target it's paired with, but
        # the same source recurs across many targets (a hub like HPQ or PG
        # sits on 60-80 edges) -- fetching fresh per edge turned a ~500-source
        # job into ~4,600 full scans of a multi-million-row table. Cached by
        # source ticker instead, since that's the only key the query varies on.
        events_by_source: dict[str, list[tuple]] = {}

        for target, incoming in by_target.items():
            points = _load_returns(con, target)
            if not points:
                continue
            returns = ReturnSeries(points)

            rows: list[dict[str, Any]] = []
            for source, edge_type, known_from in incoming:
                if source not in events_by_source:
                    events_by_source[source] = _events_for_symbol(
                        con, source, event_types, subtypes
                    )
                for (
                    event_id,
                    etype,
                    subtype,
                    available_time,
                    magnitude,
                    direction,
                ) in events_by_source[source]:
                    available_on = available_time.date()
                    if known_from is not None and known_from > available_on:
                        # The edge was not yet on the record when this event
                        # became actionable -- pairing them would be
                        # edge-dimension lookahead. Not counted in `collected`:
                        # this pair was never a candidate observation.
                        not_yet_known += 1
                        continue
                    for horizon in horizons:
                        collected += 1
                        window = returns.forward(available_on, horizon)
                        if window is None:
                            unmeasurable += 1
                            continue
                        split = assign_split(window, split_date)
                        if split is None:
                            purged += 1
                            continue

                        record = EventSampleRecord(
                            # Target is in the id: one source event points at
                            # several targets, each a distinct sample.
                            sample_id=hashing.content_hash(
                                "prop", event_id, edge_type, target, horizon
                            ),
                            event_id=event_id,
                            edge_id=edge_type,
                            event_type=etype,
                            event_subtype=subtype,
                            source_ticker=source,
                            target_ticker=target,
                            horizon_days=horizon,
                            available_on=available_on,
                            t0=window.t0,
                            window_end=window.window_end,
                            forward_abnormal_return=window.cumulative_abnormal_return,
                            magnitude=magnitude,
                            direction=direction,
                            split=split.value,
                            features={"edge_type": edge_type, "source": source},
                            schema_version=schema_version,
                            collected_time=utcnow(),
                        )
                        rows.append(record.to_row())

            rows, dropped = _dedupe_samples(rows)
            deduped += dropped

            if rows:
                result = duckdb_store.upsert_event_samples(con, rows)
                inserted += result.inserted
                updated += result.updated
                parquet.write_records(
                    config.paths.parquet_dir,
                    "event_samples",
                    rows,
                    ["sample_id"],
                    partition_col="event_type",
                )

        summary.collected = collected
        summary.inserted = inserted
        summary.updated = updated
        summary.deduped = deduped
        summary.bump("unmeasurable", unmeasurable)
        summary.bump("purged_at_split", purged)
        summary.bump("edges", len(edges))
        summary.bump("edges_undated", undated)
        summary.bump("pairs_edge_not_yet_known", not_yet_known)
        summary.note(
            f"{not_yet_known} (event, edge) pairs skipped as not-yet-disclosed, "
            f"{undated} edges carry no report_date and were paired unconditionally"
        )
    return summary


__all__ = [
    "DEFAULT_HORIZONS",
    "DEFAULT_SPLIT_DATE",
    "build",
    "build_incremental",
    "build_propagation",
]
