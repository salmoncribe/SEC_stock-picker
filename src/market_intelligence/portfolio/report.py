"""Rendering. Pure functions that never take a database connection.

Purity here is a testability decision and a lock decision at once: a renderer
that could query would be a renderer that could block on the 7.2 GB database
while formatting a Telegram message. These take already-computed results and
return strings or write files, nothing more.

Telegram messages own their 3,900-character budget by truncating themselves --
the notify layer never raises, so an over-long message would simply vanish.
Artifacts are written atomically (temp file, then rename) so a crashed run
leaves either the previous artifact or the new one, never a half-written file
that the next run reads as real.

Truncation drops *whole trailing lines* and counts what it dropped, in the same
"cap and note the rest" shape as ``autopilot.notify``. The head -- equity, the
gate verdict, the propagation line, the holdout state -- is built to a bounded
length (the title is clipped) and is never a candidate for dropping, so the
facts a reader acts on cannot be the part that falls off the end.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from market_intelligence.portfolio.metrics import PortfolioMetrics
from market_intelligence.portfolio.simulator import ReplayResult, StressOutcome

TELEGRAM_MAX_CHARS = 3900

#: The two anti-overfitting bars, both INCLUSIVE. They live here as named
#: constants rather than as literals inside ``gate_passed`` so that a report
#: can print the bar it was judged against next to the number it scored.
DSR_THRESHOLD = 0.95
PBO_THRESHOLD = 0.20

#: Appended when whole lines were dropped, so the reader knows the message is
#: partial and by how much. A *counted* omission, never a silent one.
_OMITTED_TEMPLATE = "\n… ({dropped} more line(s) omitted)"

#: Last-resort guard for the pathological case where the head alone overflows.
#: Normal operation never reaches it -- if it appears in a message, the head
#: grew unbounded and that is a bug, not a long report.
_TRUNCATION_MARKER = "\n… (truncated)"

#: A caller-supplied title is clipped to this, so no title can crowd out the
#: numbers underneath it.
_MAX_TITLE_CHARS = 96

#: Artifact filenames, in the order ``write_artifacts`` returns them.
_WEIGHTS_FILE = "weights.parquet"
_DECISIONS_FILE = "decisions.parquet"
_FILLS_FILE = "fills.parquet"
_EQUITY_FILE = "equity.parquet"
_SUMMARY_FILE = "summary.json"


@dataclass(frozen=True)
class GateReport:
    """The honesty surface: what was measured, and what was not.

    ``propagation_cells_admitted == 0`` is a publishable result, rendered as
    "propagation gate: 0 admitted" rather than omitted. A report that only
    mentions the graph when the graph worked is a report that overstates.
    """

    dsr: float | None
    pbo: float | None
    n_trials: int
    propagation_cells_admitted: int
    stress_outcomes: tuple[StressOutcome, ...]
    holdout_unsealed: bool

    @property
    def gate_passed(self) -> bool:
        """DSR >= 0.95 AND PBO <= 0.20 AND every stress scenario passed.

        Both bars are inclusive: a DSR of exactly 0.95 clears, as does a PBO of
        exactly 0.20. ``None`` is *not* a pass -- an unmeasured statistic is an
        unmeasured statistic, and defaulting it to "fine" is how a gate ends up
        certifying a run it never judged.

        **The empty-tuple guard is deliberate; do not "simplify" this method to
        a bare ``all(...)``.** ``all(())`` is ``True``, so dropping the guard
        would let a stress suite that never ran certify the gate -- the exact
        failure this class exists to prevent, and the same reasoning that makes
        ``dsr=None`` a failure. A suite that never ran is not a suite that
        passed. ``test_portfolio_report`` pins this.

        The per-scenario verdict is delegated to ``StressOutcome.passed`` rather
        than restated here. Those criteria are preregistered risk-control rules
        owned by ``simulator``; a second copy inside a renderer is a second copy
        to drift.
        """
        if self.dsr is None or self.pbo is None:
            return False
        if self.dsr < DSR_THRESHOLD or self.pbo > PBO_THRESHOLD:
            return False
        if not self.stress_outcomes:
            return False
        return all(outcome.passed for outcome in self.stress_outcomes)


def render_telegram(
    metrics: PortfolioMetrics,
    gate: GateReport,
    *,
    title: str = "Portfolio",
    ending_equity: float | None = None,
) -> str:
    """<= 3,900 chars, self-truncating. Own message, never appended to another.

    ``ending_equity`` is the account's closing balance in dollars. It is a
    parameter rather than a field of ``metrics``, which carries only returns:
    this is a $10,000 account and Telegram is the surface a human actually
    reads daily, so "1.2450x start" is a conversion the reader should not have
    to do in their head. When it is ``None`` -- or non-finite -- the line falls
    back to the multiple of starting capital rather than inventing a balance.
    """
    return _fit_to_budget(
        _telegram_head(metrics, gate, title=title, ending_equity=ending_equity),
        _telegram_body(metrics, gate),
        budget=TELEGRAM_MAX_CHARS,
    )


def render_obsidian(
    result: ReplayResult, metrics: PortfolioMetrics, gate: GateReport, *, title: str
) -> str:
    """Markdown note body. Pure -- the caller decides where it lands.

    Unlike the Telegram nudge this is unbounded and states equity in dollars,
    because it has the ``ReplayResult`` and a vault has no character budget.
    Every section is always emitted, empty or not, so the vault holds one
    uniform shape per run rather than a ragged mix.
    """
    blocks = [
        _note_frontmatter(result, gate),
        f"# {_clip(title)}",
        _note_equity_line(result, metrics),
        _note_gate_section(gate),
        _note_stress_section(gate),
        _note_performance_section(metrics),
        _note_trading_section(result),
    ]
    return "\n\n".join(blocks) + "\n"


def write_artifacts(
    result: ReplayResult, metrics: PortfolioMetrics, *, out_dir: Path
) -> tuple[Path, ...]:
    """Parquet (weights, decisions, fills) + atomic ``summary.json``.

    Lands under ``MARKET_INTELLIGENCE_HOME`` on the SSD, not in the database.

    ``equity.parquet`` is written alongside the three named tables: the daily
    equity curve is the primary evidence of a replay, and neither the summary
    (aggregates only) nor the other tables carry it, so without this file it
    would exist nowhere on disk.

    Every file, Parquet included, goes through the same temp-then-``os.replace``
    publish, so an interrupted run leaves each artifact either wholly previous
    or wholly new. Nothing here is timestamped: an identical replay produces
    identical bytes, which is what makes a re-run verifiable by comparison.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    return (
        _write_table(out_dir / _WEIGHTS_FILE, _weights_table(result)),
        _write_table(out_dir / _DECISIONS_FILE, _decisions_table(result)),
        _write_table(out_dir / _FILLS_FILE, _fills_table(result)),
        _write_table(out_dir / _EQUITY_FILE, _equity_table(result)),
        _write_json(out_dir / _SUMMARY_FILE, _summary(result, metrics)),
    )


# --------------------------------------------------------------------------- #
# Telegram                                                                     #
# --------------------------------------------------------------------------- #
def _telegram_head(
    metrics: PortfolioMetrics,
    gate: GateReport,
    *,
    title: str,
    ending_equity: float | None = None,
) -> list[str]:
    """The lines truncation may never drop. Bounded by construction."""
    return [
        f"📈 {_clip(title)}",
        _equity_line(metrics, ending_equity),
        _gate_line(gate),
        _propagation_line(gate),
        _holdout_line(gate),
        _stress_summary_line(gate),
    ]


def _equity_line(metrics: PortfolioMetrics, ending_equity: float | None) -> str:
    """Dollars when the balance is known, a multiple of the start when it is not.

    Never both, and never a dollar figure derived from the return: the balance
    is a fact the caller either has or does not, and reconstructing it from a
    rounded percentage would print a number no ledger agrees with.
    """
    if ending_equity is not None and math.isfinite(ending_equity):
        opening = f"${ending_equity:,.2f}"
    else:
        opening = f"{1.0 + metrics.total_return:.4f}x start"
    return (
        f"Equity: {opening} ({_signed_pct(metrics.total_return)} total, "
        f"{_signed_pct(metrics.annualized_return)}/yr)"
    )


def _telegram_body(metrics: PortfolioMetrics, gate: GateReport) -> list[str]:
    """Droppable detail, most valuable first: ratios, then per-scenario lines."""
    lines = [
        "",
        f"Sharpe {_ratio(metrics.sharpe)} · Sortino {_ratio(metrics.sortino)} "
        f"· Calmar {_ratio(metrics.calmar)}",
        f"Vol {_pct(metrics.annualized_vol)}/yr · MaxDD {_pct(metrics.max_drawdown)} "
        f"· CVaR95 {_pct(metrics.cvar_95)}",
        f"Turnover {_pct(metrics.turnover_annual)}/yr · {metrics.n_days:,} day(s) "
        f"· {gate.n_trials} trial(s)",
    ]
    if metrics.beta is not None:
        lines.append(
            f"Alpha {_signed_pct(metrics.alpha or 0.0)}/yr · Beta {_ratio(metrics.beta)} "
            f"· IR {_ratio(metrics.information_ratio or 0.0)}"
        )
    if gate.stress_outcomes:
        lines.append("")
        lines.extend(f"  {_stress_line(outcome)}" for outcome in gate.stress_outcomes)
    return lines


def _gate_line(gate: GateReport) -> str:
    """``Gate: NOT PASSED — DSR 0.4120 < 0.95; PBO ...`` -- verdict plus every leg.

    Each leg prints its number against the bar it was judged by, in both
    directions, so a pass is as auditable as a failure.
    """
    verdict = "PASSED" if gate.gate_passed else "NOT PASSED"
    return f"Gate: {verdict} — " + "; ".join(_gate_legs(gate))


def _gate_legs(gate: GateReport) -> list[str]:
    """One phrase per leg: DSR, PBO, and the stress suite."""
    return [
        _threshold_leg("DSR", gate.dsr, DSR_THRESHOLD, at_least=True),
        _threshold_leg("PBO", gate.pbo, PBO_THRESHOLD, at_least=False),
        f"stress {_stress_tally(gate)}",
    ]


def _threshold_leg(name: str, value: float | None, bar: float, *, at_least: bool) -> str:
    """``DSR 0.9720 >= 0.95`` / ``DSR 0.4120 < 0.95`` / ``DSR not computed``."""
    if value is None:
        return f"{name} not computed"
    met, missed = (">=", "<") if at_least else ("<=", ">")
    passed = value >= bar if at_least else value <= bar
    return f"{name} {value:.4f} {met if passed else missed} {bar:.2f}"


def _stress_tally(gate: GateReport) -> str:
    """``3/4 passed``, or that the suite has not run at all."""
    if not gate.stress_outcomes:
        return "not run"
    passed = sum(1 for outcome in gate.stress_outcomes if outcome.passed)
    return f"{passed}/{len(gate.stress_outcomes)} passed"


def _stress_summary_line(gate: GateReport) -> str:
    return f"Stress: {_stress_tally(gate)}"


def _stress_line(outcome: StressOutcome) -> str:
    """One scenario: verdict, the two measured numbers, and the governor."""
    governor = "governor fired" if outcome.governor_fired else "governor never fired"
    return (
        f"{outcome.scenario} {'PASS' if outcome.passed else 'FAIL'} "
        f"vol {_pct(outcome.realized_vol)} · dd {_pct(outcome.max_drawdown)} · {governor}"
    )


def _propagation_line(gate: GateReport) -> str:
    """Unconditional. Zero is the expected answer and it gets said out loud."""
    return f"propagation gate: {gate.propagation_cells_admitted} admitted"


def _holdout_line(gate: GateReport) -> str:
    state = "UNSEALED (read)" if gate.holdout_unsealed else "sealed (not read)"
    return f"Holdout: {state}"


def _fit_to_budget(head: Sequence[str], body: Sequence[str], *, budget: int) -> str:
    """Keep the head, then as many whole body lines as fit; count the rest.

    A line is added only if the message so far, plus that line, plus room for an
    omission note covering everything after it, still fits -- so the note itself
    can never be the thing that overflows.
    """
    kept: list[str] = []
    for index, line in enumerate(body):
        remaining_after = len(body) - index - 1
        reserve = len(_omitted_note(remaining_after)) if remaining_after else 0
        candidate = "\n".join([*head, *kept, line])
        if len(candidate) + reserve > budget:
            break
        kept.append(line)

    dropped = len(body) - len(kept)
    # A blank separator as the final surviving line reads as a mistake; drop it
    # (it is counted like any other omitted line).
    while kept and not kept[-1].strip():
        kept.pop()
        dropped += 1

    message = "\n".join([*head, *kept])
    if dropped:
        message += _omitted_note(dropped)
    if len(message) > budget:  # pragma: no cover -- bounded head makes this dead
        message = message[: budget - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
    return message


def _omitted_note(dropped: int) -> str:
    return _OMITTED_TEMPLATE.format(dropped=dropped)


# --------------------------------------------------------------------------- #
# Obsidian                                                                     #
# --------------------------------------------------------------------------- #
def _note_frontmatter(result: ReplayResult, gate: GateReport) -> str:
    """YAML frontmatter, with the gate verdict as a query-able tag."""
    tags = [
        "portfolio",
        "portfolio/replay",
        f"gate/{'passed' if gate.gate_passed else 'not-passed'}",
    ]
    if gate.propagation_cells_admitted == 0:
        tags.append("propagation/none-admitted")
    if gate.holdout_unsealed:
        tags.append("holdout/unsealed")
    lines = ["---"]
    last_day = _last_day(result)
    if last_day is not None:
        lines.append(f"date: {last_day.isoformat()}")
    lines.extend(
        [
            f"gate_passed: {str(gate.gate_passed).lower()}",
            f"propagation_cells_admitted: {gate.propagation_cells_admitted}",
            "tags:",
            *(f"  - {tag}" for tag in tags),
            "---",
        ]
    )
    return "\n".join(lines)


def _note_equity_line(result: ReplayResult, metrics: PortfolioMetrics) -> str:
    """Dollars, not multiples -- the note has the curve, so it can say so."""
    equity = _series(result.equity_curve)
    if equity.size == 0:
        return "**Equity:** not recorded."
    span = ""
    first_day, last_day = _first_day(result), _last_day(result)
    if first_day is not None and last_day is not None:
        span = f", {first_day.isoformat()} -> {last_day.isoformat()}"
    return (
        f"**Equity:** ${equity[0]:,.2f} -> ${equity[-1]:,.2f} "
        f"({_signed_pct(metrics.total_return)}) over {metrics.n_days:,} day(s){span}."
    )


def _note_gate_section(gate: GateReport) -> str:
    """The verdict, each leg with its bar, and the two facts that get buried."""
    verdict = "PASSED" if gate.gate_passed else "NOT PASSED"
    lines = [
        "## Gate",
        "",
        f"**Verdict: {verdict}** -- " + "; ".join(_gate_legs(gate)) + ".",
        "",
        f"- DSR: {_measured(gate.dsr)} (bar: >= {DSR_THRESHOLD:.2f}, inclusive)",
        f"- PBO: {_measured(gate.pbo)} (bar: <= {PBO_THRESHOLD:.2f}, inclusive)",
        f"- Trials searched: {gate.n_trials}",
        f"- {_propagation_line(gate)}"
        + (
            " -- the machinery shipped and the data declined; no coefficient is"
            " hand-waived in to make the graph look useful."
            if gate.propagation_cells_admitted == 0
            else ""
        ),
        f"- Sealed holdout: {'READ (unsealed)' if gate.holdout_unsealed else 'not read'}",
    ]
    return "\n".join(lines)


def _note_stress_section(gate: GateReport) -> str:
    lines = ["## Stress"]
    if not gate.stress_outcomes:
        lines.append("_Stress suite has not run -- the gate cannot pass on an unrun suite._")
        return "\n".join(lines)
    lines.extend(
        [
            "",
            "| Scenario | Verdict | Realized vol | Max drawdown | Governor fired |"
            " Vol within target | Drawdown within halt |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for outcome in gate.stress_outcomes:
        lines.append(
            f"| {outcome.scenario} | {'pass' if outcome.passed else 'FAIL'} "
            f"| {_pct(outcome.realized_vol)} | {_pct(outcome.max_drawdown)} "
            f"| {_yes_no(outcome.governor_fired)} | {_yes_no(outcome.vol_within_target)} "
            f"| {_yes_no(outcome.drawdown_within_halt)} |"
        )
    return "\n".join(lines)


def _note_performance_section(metrics: PortfolioMetrics) -> str:
    rows: list[tuple[str, str]] = [
        ("Total return", _signed_pct(metrics.total_return)),
        ("Annualized return", _signed_pct(metrics.annualized_return)),
        ("Annualized volatility", _pct(metrics.annualized_vol)),
        ("Sharpe", _ratio(metrics.sharpe)),
        ("Sortino", _ratio(metrics.sortino)),
        ("Calmar", _ratio(metrics.calmar)),
        ("Max drawdown", _pct(metrics.max_drawdown)),
        ("CVaR 95", _pct(metrics.cvar_95)),
        ("Turnover (annual)", _pct(metrics.turnover_annual)),
        ("Days", f"{metrics.n_days:,}"),
    ]
    if metrics.beta is None:
        # Benchmark-relative fields are None without SPY. Said plainly rather
        # than left as absent rows a reader could mistake for zeros.
        rows.append(("Benchmark-relative", "not computed (no benchmark)"))
    else:
        rows.extend(
            [
                ("Alpha (annual)", _signed_pct(metrics.alpha or 0.0)),
                ("Beta", _ratio(metrics.beta)),
                ("Information ratio", _ratio(metrics.information_ratio or 0.0)),
            ]
        )
    lines = ["## Performance", "", "| Metric | Value |", "| --- | --- |"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    return "\n".join(lines)


def _note_trading_section(result: ReplayResult) -> str:
    rebalanced = sum(1 for decision in result.decisions if decision.rebalanced)
    orders = sum(len(decision.orders) for decision in result.decisions)
    state = result.final_state
    return "\n".join(
        [
            "## Trading",
            f"- Decisions: {len(result.decisions):,} (rebalanced on {rebalanced:,})",
            f"- Orders proposed: {orders:,}",
            f"- Fills: {len(result.fills):,}",
            f"- Final account: ${state.cash:,.2f} cash across "
            f"{len(state.positions):,} position(s), as of {_as_date(state.as_of).isoformat()}",
        ]
    )


# --------------------------------------------------------------------------- #
# Artifacts                                                                    #
# --------------------------------------------------------------------------- #
def _weights_table(result: ReplayResult) -> Any:
    """Long form ``(as_of, symbol, weight)``, one row per symbol per day.

    Long rather than wide because the tradeable universe changes across a
    2,500-day replay, and a wide table would have to invent a column for every
    symbol that ever appeared and a zero for every day it did not.
    """
    days: list[date] = []
    symbols: list[str] = []
    weights: list[float] = []
    for decision in result.decisions:
        row = _series(decision.target_weights)
        if row.size != len(decision.symbols):
            raise ValueError(
                f"decision {decision.as_of} has {row.size} weights for "
                f"{len(decision.symbols)} symbols"
            )
        days.extend([_as_date(decision.as_of)] * row.size)
        symbols.extend(decision.symbols)
        weights.extend(float(value) for value in row)
    return pa.table(
        {
            "as_of": pa.array(days, type=pa.date32()),
            "symbol": pa.array(symbols, type=pa.string()),
            "weight": pa.array(weights, type=pa.float64()),
        }
    )


def _decisions_table(result: ReplayResult) -> Any:
    """One row per decided day: the knobs that produced that day's weights."""
    decisions = result.decisions
    return pa.table(
        {
            "as_of": pa.array([_as_date(d.as_of) for d in decisions], type=pa.date32()),
            "optimizer_status": pa.array([d.optimizer_status for d in decisions], pa.string()),
            "governor_scale": pa.array([float(d.governor_scale) for d in decisions], pa.float64()),
            "vol_scale": pa.array([float(d.vol_scale) for d in decisions], pa.float64()),
            "rebalanced": pa.array([bool(d.rebalanced) for d in decisions], pa.bool_()),
            "n_symbols": pa.array([len(d.symbols) for d in decisions], pa.int64()),
            "n_views": pa.array([len(d.views) for d in decisions], pa.int64()),
            "n_orders": pa.array([len(d.orders) for d in decisions], pa.int64()),
            "gross_exposure": pa.array(
                [float(np.abs(_series(d.target_weights)).sum()) for d in decisions], pa.float64()
            ),
        }
    )


def _fills_table(result: ReplayResult) -> Any:
    """The executed ledger, the same shape ``store`` persists to DuckDB."""
    fills = result.fills
    return pa.table(
        {
            "fill_id": pa.array([f.fill_id for f in fills], pa.string()),
            "order_id": pa.array([f.order_id for f in fills], pa.string()),
            "symbol": pa.array([f.symbol for f in fills], pa.string()),
            "side": pa.array([str(f.side) for f in fills], pa.string()),
            "shares": pa.array([float(f.shares) for f in fills], pa.float64()),
            "price": pa.array([float(f.price) for f in fills], pa.float64()),
            "commission": pa.array([float(f.commission) for f in fills], pa.float64()),
            "as_of": pa.array([_as_date(f.as_of) for f in fills], pa.date32()),
        }
    )


def _equity_table(result: ReplayResult) -> Any:
    """``(as_of, equity, daily_return, benchmark_return)``, one row per day.

    ``daily_returns`` is allowed to be one shorter than the calendar -- the
    first mark has no prior close to have moved from, and that day's return is
    null rather than a fabricated zero. Any other length mismatch raises: a
    silently mis-aligned equity curve is worse than no file.
    """
    days = [_as_date(value) for value in list(result.calendar)]
    equity = _series(result.equity_curve)
    if equity.size != len(days):
        raise ValueError(f"equity_curve has {equity.size} marks for {len(days)} calendar days")
    return pa.table(
        {
            "as_of": pa.array(days, type=pa.date32()),
            "equity": pa.array([float(value) for value in equity], pa.float64()),
            "daily_return": pa.array(_right_aligned(result.daily_returns, len(days)), pa.float64()),
            "benchmark_return": pa.array(
                _right_aligned(result.benchmark_returns, len(days)), pa.float64()
            ),
        }
    )


def _right_aligned(values: np.ndarray | None, n_days: int) -> list[float | None]:
    """Pad a per-day series to the calendar, nulls first. See ``_equity_table``."""
    if values is None:
        return [None] * n_days
    series = _series(values)
    if series.size == n_days:
        return [float(value) for value in series]
    if series.size == n_days - 1:
        return [None, *(float(value) for value in series)]
    raise ValueError(f"a per-day series of {series.size} cannot align to {n_days} days")


def _summary(result: ReplayResult, metrics: PortfolioMetrics) -> dict[str, Any]:
    """The small, human-readable index of a run. Aggregates only, no curves."""
    equity = _series(result.equity_curve)
    first_day, last_day = _first_day(result), _last_day(result)
    return {
        "version": 1,
        "start": first_day.isoformat() if first_day else None,
        "end": last_day.isoformat() if last_day else None,
        "starting_equity": _json_number(equity[0]) if equity.size else None,
        "ending_equity": _json_number(equity[-1]) if equity.size else None,
        "metrics": {key: _json_number(value) for key, value in asdict(metrics).items()},
        "counts": {
            "decisions": len(result.decisions),
            "orders": sum(len(decision.orders) for decision in result.decisions),
            "fills": len(result.fills),
            "rebalances": sum(1 for decision in result.decisions if decision.rebalanced),
            "positions_held": len(result.final_state.positions),
        },
        "account": {
            "as_of": _as_date(result.final_state.as_of).isoformat(),
            "cash": _json_number(result.final_state.cash),
            "high_water_mark": _json_number(result.final_state.high_water_mark),
        },
        "artifacts": [_WEIGHTS_FILE, _DECISIONS_FILE, _FILLS_FILE, _EQUITY_FILE],
    }


def _write_table(path: Path, table: Any) -> Path:
    return _publish(path, lambda temporary: pq.write_table(table, temporary))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    return _publish(path, write)


def _publish(path: Path, write: Callable[[Path], None]) -> Path:
    """Write to a sibling temp file, then ``os.replace`` it into place.

    The rename is atomic within a filesystem, so a reader sees the whole old
    file or the whole new one. The temp file is a *sibling* for exactly that
    reason -- a rename across filesystems is a copy, and a copy is not atomic.
    On any failure the temp file is removed rather than left as debris the next
    run would have to distinguish from a real artifact.
    """
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        write(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


# --------------------------------------------------------------------------- #
# Formatting                                                                   #
# --------------------------------------------------------------------------- #
def _clip(text: str, *, limit: int = _MAX_TITLE_CHARS) -> str:
    """Collapse whitespace and cap length, so a title cannot blow the budget."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _pct(value: float, *, digits: int = 2) -> str:
    """``0.142`` -> ``"14.20%"``. Non-finite renders as ``n/a``, never ``nan%``."""
    if not math.isfinite(value):
        return "n/a"
    return f"{value * 100.0:.{digits}f}%"


def _signed_pct(value: float) -> str:
    """``0.0162`` -> ``"+1.62%"``; a move rounding to zero never prints ``-0.00%``."""
    if not math.isfinite(value):
        return "n/a"
    return f"{value * 100.0:+.2f}%".replace("-0.00%", "+0.00%")


def _ratio(value: float) -> str:
    return f"{value:.2f}" if math.isfinite(value) else "n/a"


def _measured(value: float | None) -> str:
    """A statistic, or the plain statement that it was never computed."""
    if value is None:
        return "not computed"
    return f"{value:.4f}" if math.isfinite(value) else "n/a"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _series(values: np.ndarray) -> np.ndarray:
    """A flat float64 view of an array-like, for formatting and Parquet."""
    return np.asarray(values, dtype=float).ravel()


def _first_day(result: ReplayResult) -> date | None:
    calendar = list(result.calendar)
    return _as_date(calendar[0]) if calendar else None


def _last_day(result: ReplayResult) -> date | None:
    calendar = list(result.calendar)
    return _as_date(calendar[-1]) if calendar else None


def _as_date(value: Any) -> date:
    """Normalize a calendar entry to ``datetime.date``.

    The calendar arrives as a numpy array, so an entry may be a ``date``, a
    ``datetime64``, or an ISO string depending on how it was built. Parquet
    wants one type, and guessing per row is how a date column ends up half
    date32 and half string.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.datetime64):
        converted = value.astype("datetime64[D]").astype(object)
        if isinstance(converted, datetime):
            return converted.date()
        if isinstance(converted, date):
            return converted
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"cannot read {value!r} ({type(value).__name__}) as a calendar date")


def _json_number(value: Any) -> Any:
    """JSON-safe scalar: NaN/inf become ``None`` rather than invalid JSON.

    ``json.dump`` writes bare ``NaN``/``Infinity`` by default, which no strict
    JSON reader accepts -- so ``allow_nan=False`` is set at the call site and
    non-finite values are made explicitly absent here instead.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    number = float(value)
    return number if math.isfinite(number) else None


__all__ = [
    "DSR_THRESHOLD",
    "PBO_THRESHOLD",
    "TELEGRAM_MAX_CHARS",
    "GateReport",
    "render_obsidian",
    "render_telegram",
    "write_artifacts",
]
