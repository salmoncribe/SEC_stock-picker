"""Rendering, and the honesty the renderers exist to enforce.

Five tests, one per property that can silently fail in production:

1. **Truncation is graceful.** ``notify._post_text`` never raises, so an
   over-long Telegram message does not error -- it *vanishes*. The renderer
   therefore owns its own budget, and it must spend that budget on the facts a
   reader acts on (equity, return, gate verdict) rather than on the tail of a
   scenario table.
2. **Purity.** A renderer that could query is a renderer that could block on
   the 7.2 GB single-writer database while formatting a phone nudge.
3. **The gate is an AND.** A partial pass rendered as a pass is the exact
   failure this platform's statistics exist to prevent.
4. **Zero admitted is published, not omitted.** No propagation coefficient has
   ever been measured here; a report that mentions the graph only when the
   graph worked systematically overstates.
5. **Artifacts are atomic.** A half-written ``summary.json`` is read by the
   next run as real.

These tests run against the real ``simulator.StressOutcome.passed``. While that
property was being written concurrently, a temporary autouse fixture stood in
for it with the criteria its own docstring preregisters; the fixture was
removed once the simulator landed, and every test here passed unchanged against
the real property. That agreement is the point worth recording: the renderer's
understanding of the gate and the simulator's implementation of it were written
independently and matched.

``report.GateReport.gate_passed`` deliberately *consults* ``StressOutcome
.passed`` rather than restating its criteria. Preregistered risk-control rules
belong to the simulator, and a second copy inside a renderer is a second copy to
drift.
"""

from __future__ import annotations

import builtins
import inspect
import json
import os
import re
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from market_intelligence.portfolio import report as report_module
from market_intelligence.portfolio.account import AccountState, Fill, Position, Side
from market_intelligence.portfolio.metrics import PortfolioMetrics
from market_intelligence.portfolio.report import (
    TELEGRAM_MAX_CHARS,
    GateReport,
    render_obsidian,
    render_telegram,
    write_artifacts,
)
from market_intelligence.portfolio.simulator import DayDecision, ReplayResult, StressOutcome

PROPAGATION_ZERO_LINE = "propagation gate: 0 admitted"

#: A whole stress line, as the Telegram renderer emits it. Used to prove the
#: last surviving line of a truncated message is intact rather than sliced.
_STRESS_LINE = re.compile(r" {2}\S+ (PASS|FAIL) .* governor (fired|never fired)")
_OMITTED_NOTE = re.compile(r"… \((\d+) more line\(s\) omitted\)")


def test_stress_outcome_criteria_match_what_the_renderer_was_built_against():
    """The renderer's assumption about the gate and the simulator's rule agree.

    ``report`` was written against the criteria ``StressOutcome.passed``
    preregisters in its docstring -- vol within target AND drawdown within halt
    AND the governor observed firing -- before that property existed. This pins
    the agreement so a later change to the simulator's criteria surfaces here as
    a failure rather than as a renderer that quietly reports the wrong verdict.
    """
    assert _stress(passed=True).passed is True
    for leg in ("vol_within_target", "drawdown_within_halt", "governor_fired"):
        broken = replace(_stress(passed=True), **{leg: False})
        assert broken.passed is False, f"{leg} must be load-bearing in StressOutcome.passed"


def _stress(scenario: str = "bootstrap", *, passed: bool = True) -> StressOutcome:
    """One scenario outcome; ``passed=False`` fails exactly the vol leg."""
    return StressOutcome(
        scenario=scenario,  # type: ignore[arg-type]
        realized_vol=0.121,
        max_drawdown=0.09,
        governor_fired=True,
        vol_within_target=passed,
        drawdown_within_halt=True,
    )


def _metrics(**overrides: Any) -> PortfolioMetrics:
    fields: dict[str, Any] = {
        "total_return": 0.245,
        "annualized_return": 0.1162,
        "annualized_vol": 0.142,
        "sharpe": 0.84,
        "sortino": 1.10,
        "calmar": 0.62,
        "max_drawdown": 0.185,
        "cvar_95": 0.031,
        "turnover_annual": 2.4,
        "n_days": 2500,
    }
    return PortfolioMetrics(**{**fields, **overrides})


def _gate(**overrides: Any) -> GateReport:
    fields: dict[str, Any] = {
        "dsr": 0.97,
        "pbo": 0.12,
        "n_trials": 18,
        "propagation_cells_admitted": 0,
        "stress_outcomes": (_stress("bootstrap"), _stress("vol_x2")),
        "holdout_unsealed": False,
    }
    return GateReport(**{**fields, **overrides})


def _replay_result(n_days: int = 3) -> ReplayResult:
    """A tiny but structurally complete replay: 3 days, 2 symbols, 1 fill."""
    calendar = np.array([date(2026, 1, 5 + i) for i in range(n_days)], dtype=object)
    equity = np.array([10_000.0 + 100.0 * i for i in range(n_days)])
    daily_returns = np.array([0.0, *[0.01] * (n_days - 1)])
    decisions = tuple(
        DayDecision(
            as_of=calendar[i],
            target_weights=np.array([0.6, 0.4]),
            symbols=("AAPL", "MSFT"),
            views=(),
            orders=(),
            optimizer_status="optimal",
            governor_scale=1.0,
            vol_scale=0.8,
            rebalanced=i == 0,
        )
        for i in range(n_days)
    )
    fills = (
        Fill(
            fill_id="pf:AAPL:2026-01-05:BUY:v1",
            order_id="po:AAPL:2026-01-05:BUY:v1",
            symbol="AAPL",
            side=Side.BUY,
            shares=10.0,
            price=150.0,
            commission=0.0,
            as_of=date(2026, 1, 5),
        ),
    )
    final_state = AccountState(
        as_of=date(2026, 1, 5 + n_days - 1),
        cash=1204.55,
        positions=MappingProxyType({"AAPL": Position("AAPL", 10.0, 1500.0)}),
        high_water_mark=10_500.0,
    )
    return ReplayResult(
        equity_curve=equity,
        calendar=calendar,
        decisions=decisions,
        fills=fills,
        final_state=final_state,
        daily_returns=daily_returns,
        benchmark_returns=None,
    )


# --------------------------------------------------------------------------- #
# 1. Telegram truncation                                                       #
# --------------------------------------------------------------------------- #
def test_telegram_truncation_drops_whole_lines_and_keeps_the_headline_facts() -> None:
    ordinary = render_telegram(_metrics(), _gate(), title="Portfolio")
    assert len(ordinary) <= TELEGRAM_MAX_CHARS
    # A normal report is not truncated at all -- no omission note, no marker.
    assert _OMITTED_NOTE.search(ordinary) is None
    assert "(truncated)" not in ordinary

    n_scenarios = 500
    oversized = render_telegram(
        _metrics(),
        _gate(
            dsr=0.412,
            pbo=0.31,
            stress_outcomes=tuple(
                _stress(f"scenario_{i:03d}", passed=i % 2 == 0) for i in range(n_scenarios)
            ),
        ),
        title="A deliberately absurd title " * 20,
    )

    assert len(oversized) <= TELEGRAM_MAX_CHARS
    lines = oversized.split("\n")

    # Graceful, not a blunt slice: the raw last-resort marker never fired, the
    # message ends on a counted omission note, and the last surviving content
    # line is a WHOLE stress line rather than a fragment of one.
    assert "(truncated)" not in oversized
    note = _OMITTED_NOTE.fullmatch(lines[-1])
    assert note is not None, f"message ends mid-thought: {lines[-1]!r}"
    assert _STRESS_LINE.fullmatch(lines[-2]) is not None

    # Nothing vanished silently: every scenario is either rendered whole or
    # counted in the omission note.
    rendered = sum(1 for line in lines if _STRESS_LINE.fullmatch(line))
    assert rendered > 0
    assert rendered + int(note.group(1)) == n_scenarios

    # The facts a reader acts on survive; they are never the part cut off.
    assert "1.2450x start" in oversized  # equity, as a multiple of the start
    assert "+24.50%" in oversized  # total return
    assert "NOT PASSED" in oversized  # gate verdict
    assert PROPAGATION_ZERO_LINE in oversized
    # An absurd title is clipped rather than allowed to crowd them out.
    assert len(lines[0]) <= 100
    assert lines[0].endswith("…")


# --------------------------------------------------------------------------- #
# 1b. Equity in dollars when the caller knows the balance                       #
# --------------------------------------------------------------------------- #
def test_telegram_states_dollars_when_ending_equity_is_supplied() -> None:
    # Keyword-only and last, so no existing positional call shifts.
    parameters = inspect.signature(render_telegram).parameters
    assert list(parameters) == ["metrics", "gate", "title", "ending_equity"]
    assert parameters["ending_equity"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["ending_equity"].default is None

    with_dollars = render_telegram(_metrics(), _gate(), ending_equity=12_450.0)
    assert "Equity: $12,450.00 (+24.50% total, +11.62%/yr)" in with_dollars
    assert "x start" not in with_dollars

    # Omitted: the multiple-of-start form is unchanged, and no dollar figure is
    # reconstructed from the return -- an invented balance is worse than a ratio.
    without = render_telegram(_metrics(), _gate())
    assert "Equity: 1.2450x start (+24.50% total, +11.62%/yr)" in without
    assert "$" not in without

    # The balance lives in the non-droppable head, so truncation cannot eat it.
    oversized = render_telegram(
        _metrics(),
        _gate(stress_outcomes=tuple(_stress(f"scenario_{i:03d}") for i in range(500))),
        title="Portfolio replay",
        ending_equity=12_450.0,
    )
    assert len(oversized) <= TELEGRAM_MAX_CHARS
    assert "$12,450.00" in oversized


# --------------------------------------------------------------------------- #
# 2. Purity                                                                    #
# --------------------------------------------------------------------------- #
def test_renderers_take_no_connection_and_touch_no_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for renderer in (render_telegram, render_obsidian):
        parameters = inspect.signature(renderer).parameters
        assert not {"con", "connection", "cursor", "db", "database"} & set(parameters)
        annotations = " ".join(str(p.annotation) for p in parameters.values()).lower()
        assert "duckdb" not in annotations
        assert "connection" not in annotations
        assert "path" not in annotations

    monkeypatch.chdir(tmp_path)

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a pure renderer must perform no I/O")

    monkeypatch.setattr(builtins, "open", _explode)
    monkeypatch.setattr(os, "replace", _explode)
    telegram = render_telegram(_metrics(), _gate())
    obsidian = render_obsidian(_replay_result(), _metrics(), _gate(), title="Replay")
    monkeypatch.undo()

    assert telegram and obsidian
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# 3. gate_passed is an AND, and an unmeasured leg is not a pass                 #
# --------------------------------------------------------------------------- #
def test_gate_passed_requires_every_leg_including_the_boundary_values() -> None:
    assert _gate().gate_passed is True
    # Both thresholds are inclusive per the spec: exactly at the bar passes.
    assert _gate(dsr=0.95, pbo=0.20).gate_passed is True

    # ... and one ulp outside it does not.
    assert _gate(dsr=0.9499).gate_passed is False
    assert _gate(pbo=0.2001).gate_passed is False

    # Each leg failing on its own is enough to fail the whole gate.
    assert _gate(dsr=0.40).gate_passed is False
    assert _gate(pbo=0.60).gate_passed is False
    assert _gate(
        stress_outcomes=(_stress("bootstrap"), _stress("vol_x2", passed=False))
    ).gate_passed is False

    # Not yet computed is not a pass -- in either statistic, or in the suite.
    assert _gate(dsr=None).gate_passed is False
    assert _gate(pbo=None).gate_passed is False
    assert _gate(dsr=None, pbo=None).gate_passed is False
    assert _gate(stress_outcomes=()).gate_passed is False

    # A failing gate must render as failing in both projections.
    failing = _gate(dsr=0.40, pbo=0.60)
    telegram = render_telegram(_metrics(), failing)
    obsidian = render_obsidian(_replay_result(), _metrics(), failing, title="Replay")
    assert "NOT PASSED" in telegram
    assert "NOT PASSED" in obsidian
    assert "DSR 0.4000 < 0.95" in telegram
    assert "PBO 0.6000 > 0.20" in telegram


# --------------------------------------------------------------------------- #
# 4. Zero-admitted propagation is rendered, never hidden                        #
# --------------------------------------------------------------------------- #
def test_zero_admitted_propagation_is_stated_in_both_projections() -> None:
    zero = _gate(propagation_cells_admitted=0)
    telegram = render_telegram(_metrics(), zero, title="Portfolio")
    obsidian = render_obsidian(_replay_result(), _metrics(), zero, title="Portfolio replay")

    assert PROPAGATION_ZERO_LINE in telegram
    assert PROPAGATION_ZERO_LINE in obsidian

    # The line is unconditional -- the same slot carries a non-zero count, so
    # the graph cannot be mentioned only on the days it worked.
    admitted = _gate(propagation_cells_admitted=3)
    assert "propagation gate: 3 admitted" in render_telegram(_metrics(), admitted)
    assert "propagation gate: 3 admitted" in render_obsidian(
        _replay_result(), _metrics(), admitted, title="Portfolio replay"
    )


# --------------------------------------------------------------------------- #
# 5. Artifacts land atomically                                                  #
# --------------------------------------------------------------------------- #
def test_write_artifacts_are_atomic_and_rerunnable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "run"
    result = _replay_result()

    renames: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def _spy(source: Any, destination: Any) -> None:
        renames.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(report_module.os, "replace", _spy)
    written = write_artifacts(result, _metrics(), out_dir=out_dir)
    monkeypatch.undo()

    expected = {
        "weights.parquet",
        "decisions.parquet",
        "fills.parquet",
        "equity.parquet",
        "summary.json",
    }
    assert {path.name for path in written} == expected
    assert all(path.exists() for path in written)
    # Every file arrived by rename from a *different* path in the same dir --
    # temp-then-rename, so a crashed run leaves the old file or the new one.
    assert {destination for _, destination in renames} == set(written)
    assert all(source != destination for source, destination in renames)
    # Nothing but the artifacts is left behind: no ".portfolio-*" temp debris.
    assert {path.name for path in out_dir.iterdir()} == expected

    weights = pq.read_table(out_dir / "weights.parquet")
    assert weights.num_rows == 3 * 2  # 3 days x 2 symbols
    assert weights.column("symbol").to_pylist()[:2] == ["AAPL", "MSFT"]
    assert pq.read_table(out_dir / "fills.parquet").num_rows == 1
    assert pq.read_table(out_dir / "equity.parquet").num_rows == 3

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["total_return"] == pytest.approx(0.245)
    assert summary["counts"]["decisions"] == 3

    # Re-running overwrites cleanly: same paths, new content, no debris.
    again = write_artifacts(result, _metrics(total_return=0.5), out_dir=out_dir)
    assert again == written
    assert {path.name for path in out_dir.iterdir()} == expected
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["total_return"] == pytest.approx(0.5)

    # A crash during the rename leaves the PREVIOUS summary intact, never half
    # of the new one, and drops no temp file on the floor.
    def _fail_on_summary(source: Any, destination: Any) -> None:
        if Path(destination).name == "summary.json":
            raise RuntimeError("simulated crash mid-publish")
        real_replace(source, destination)

    monkeypatch.setattr(report_module.os, "replace", _fail_on_summary)
    with pytest.raises(RuntimeError, match="simulated crash"):
        write_artifacts(result, _metrics(total_return=0.99), out_dir=out_dir)
    monkeypatch.undo()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["total_return"] == pytest.approx(0.5)
    assert {path.name for path in out_dir.iterdir()} == expected
