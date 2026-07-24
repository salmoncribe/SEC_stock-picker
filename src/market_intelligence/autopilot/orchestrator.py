"""The daily loop: sequence the pipeline, then brief a human.

One entry point runs the whole self-checking loop as an ordered set of steps.
It is a Python orchestrator rather than a shell script for one reason that
matters in production: partial-failure handling. A later step failing must not
discard an earlier step's work, and the run must still produce a briefing that
says plainly what failed. Silence must never look like success.

Each step is an existing collector that manages its own database connection and
``pipeline_runs`` row; the orchestrator only sequences them (DuckDB is
single-writer, so they run one at a time by nature) and folds their counts into
the briefing. The ladder snapshot is taken before the gate runs, so the diff
afterwards is exactly this run's promotions and demotions. Once the briefing is
assembled, its fired alerts are projected into the trade-alert ledger (plan,
confidence, persistence) and the confident ones are pushed to Telegram as a
best-effort tail step that can never sink the briefing itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from market_intelligence import database
from market_intelligence.autopilot import briefing as briefing_builder
from market_intelligence.autopilot import notify, obsidian, ram_guard
from market_intelligence.autopilot.types import Briefing, RunStatus
from market_intelligence.collectors import filing_events as filing_events_collector
from market_intelligence.collectors import prices as price_collector
from market_intelligence.collectors import returns as returns_collector
from market_intelligence.logging_config import get_logger
from market_intelligence.signals import dataset as dataset_builder
from market_intelligence.signals import impact as impact_gate
from market_intelligence.signals import trade_alerts as trade_alerts_builder

if TYPE_CHECKING:
    from market_intelligence.collectors import RunSummary
    from market_intelligence.config import Config

logger = get_logger(__name__)


@dataclass(frozen=True)
class Step:
    """One named unit of the daily loop.

    ``critical`` marks a step the briefing cannot be trusted without: if the
    gate itself fails, the run is FAILED, not merely PARTIAL, because the
    promotion decisions the briefing reports never happened. A non-critical step
    (a data pull) failing degrades to PARTIAL -- the loop still re-gates and
    briefs on the data it already had.
    """

    name: str
    run: Callable[[Config], RunSummary]
    critical: bool = False


def default_steps() -> list[Step]:
    """The phase-1 loop: refresh data, mature samples, then re-gate.

    Ordering matters. Prices must land before returns, returns before the
    dataset that reads them, and the dataset before the gate that judges it.
    The gate is the only critical step -- everything upstream is best-effort
    enrichment of what it will measure.
    """
    return [
        Step("sync-prices", lambda c: price_collector.sync(c)),
        Step("compute-returns", lambda c: returns_collector.compute(c)),
        Step("sync-filing-events", lambda c: filing_events_collector.sync(c)),
        Step("build-dataset", lambda c: dataset_builder.build(c)),
        Step("evaluate", lambda c: impact_gate.evaluate(c), critical=True),
    ]


def _ingest_counts(results: dict[str, RunSummary]) -> dict[str, int]:
    """The 'what moved' line: a few honest counts from the step summaries.

    Only steps that ran contribute; a failed step is absent here and named in
    the notes instead, so a zero never masquerades as a successful nothing.
    """
    counts: dict[str, int] = {}
    if "sync-prices" in results:
        counts["new_prices"] = results["sync-prices"].inserted
    if "sync-filing-events" in results:
        counts["new_events"] = results["sync-filing-events"].inserted
    if "build-dataset" in results:
        counts["samples_built"] = results["build-dataset"].inserted
    if "evaluate" in results:
        counts["cells_evaluated"] = results["evaluate"].stage.get("signals_tracked", 0)
    return counts


def _check_ram(config: Config) -> list[str]:
    """Run the RAM guard before the loop starts; fold what it did into notes.

    Best-effort like every other step here: a guard that can't read the
    machine (missing ``ps``/``memory_pressure``, a locked-down sandbox) must
    not sink the whole daily loop over a safety check failing to run. Silent
    otherwise -- a healthy machine with nothing to report adds no note, so the
    briefing isn't cluttered with "everything's fine" every single day.
    """
    try:
        result = ram_guard.run_guard(database_path=config.paths.database_path)
    except Exception as exc:  # a failing safety check must not abort the loop
        logger.error("autopilot_ram_guard_failed", error=str(exc))
        return []
    logger.info("autopilot_ram_guard", free_pct=result.free_pct, killed=len(result.killed))
    if not result.killed and result.free_pct >= ram_guard.DEFAULT_ACT_BELOW_FREE_PCT:
        return []
    return [result.summary_line]


def run(
    config: Config,
    *,
    as_of: date | None = None,
    steps: list[Step] | None = None,
    now: datetime | None = None,
) -> Briefing:
    """Run the daily loop and return the briefing it produced.

    The briefing is always produced, even on failure: a run that cannot gate
    still emits a FAILED briefing naming what broke, because a loop that goes
    quiet on failure is indistinguishable from one that had nothing to say.
    ``as_of`` and ``now`` are injectable so the loop is testable without the
    real clock.
    """
    config.paths.ensure()
    plan = steps if steps is not None else default_steps()
    briefing_date = as_of or (now or datetime.now(tz=UTC)).date()

    # Snapshot the ladder before the gate runs, so the post-run diff is exactly
    # this run's transitions.
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        before = briefing_builder.snapshot_statuses(con)

    results: dict[str, RunSummary] = {}
    notes: list[str] = _check_ram(config)
    run_status = RunStatus.SUCCESS

    for step in plan:
        try:
            summary = step.run(config)
        except Exception as exc:  # a failing step must not abort the loop
            logger.error("autopilot_step_failed", step=step.name, error=str(exc))
            notes.append(f"step {step.name} FAILED: {exc}")
            if step.critical:
                run_status = RunStatus.FAILED
            elif run_status is RunStatus.SUCCESS:
                run_status = RunStatus.PARTIAL
            continue
        results[step.name] = summary
        if summary.status != "success" and run_status is RunStatus.SUCCESS:
            run_status = RunStatus.PARTIAL
            notes.append(f"step {step.name} reported status={summary.status}")

    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        # The pre-run snapshot captured above is diffed against the ladder now.
        # On a FAILED run the gate never advanced it, so the diff is empty and
        # the briefing carries only the failure notes -- the honest report.
        briefing = briefing_builder.build(
            con,
            as_of=briefing_date,
            run_status=run_status,
            ingest=_ingest_counts(results),
            before=before,
            notes=notes,
        )

        # Project today's fired alerts into the trade-alert ledger: build a
        # plan + confidence for each, persist the first-firing-wins rows, and
        # gate anything under the confidence floor. Best-effort like every
        # other tail of the loop -- a broken plan/persist step must not cost
        # the briefing that already succeeded, it only earns a note.
        sendable_records: list[trade_alerts_builder.TradeAlertRecord] = []
        try:
            records, ta_notes = trade_alerts_builder.build_records(
                con, briefing.alerts, config, as_of=briefing_date
            )
            briefing.notes.extend(ta_notes)
            new_records = trade_alerts_builder.persist_new(
                con, records, schema_version=config.settings.app.schema_version
            )
            min_conf = config.settings.trading.min_confidence
            sendable_records = trade_alerts_builder.sendable(new_records, min_conf)
            trade_alerts_builder.mark_gated(
                con, trade_alerts_builder.gated(new_records, min_conf)
            )
        except Exception as exc:  # the ledger must not sink the briefing
            logger.error("autopilot_trade_alerts_failed", error=str(exc))
            briefing.notes.append(f"step trade-alerts FAILED: {exc}")

    _deliver(config, briefing)

    if sendable_records:
        sent = notify.send_trade_alerts(config, sendable_records)
        try:
            with database.connection(config.paths.database_path) as con:
                trade_alerts_builder.mark_delivered(con, sendable_records, delivered=sent)
        except Exception as exc:  # a failed stamp must not affect the return value
            logger.error("autopilot_trade_alerts_stamp_failed", error=str(exc))

    return briefing


def _deliver(config: Config, briefing: Briefing) -> None:
    """Write the Obsidian note and push the Telegram nudge; never raise.

    Delivery is best-effort: a down notifier or an unwritable vault must not
    fail a run whose real work -- gating and recording -- already succeeded.
    """
    try:
        path = obsidian.write_daily_note(config.paths.obsidian_vault_dir, briefing)
        logger.info("autopilot_note_written", path=str(path))
    except OSError as exc:
        logger.error("autopilot_note_failed", error=str(exc))

    sent = notify.send(config, briefing)
    logger.info("autopilot_notified", telegram_sent=sent)


__all__ = ["Step", "default_steps", "run"]
