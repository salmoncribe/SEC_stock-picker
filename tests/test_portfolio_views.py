"""Return-view tests: the drag correction, trading-day expiry, Black-Litterman.

The two claims worth failing a build over are (1) a view's expected return is a
*compound* return, so the volatility drag comes off the measured sum, and (2)
the posterior this module produces is the textbook Black-Litterman posterior --
checked against an independently written matrix expression, not against itself.
"""

from __future__ import annotations

import sys
from datetime import date
from typing import Any

import numpy as np
import pytest

from market_intelligence.analytics.backtest import BacktestSignal
from market_intelligence.analytics.backtest_data import AdmittedCell
from market_intelligence.portfolio import views as views_module
from market_intelligence.portfolio.views import (
    View,
    bl_posterior,
    build_view_matrices,
    closed_form_bl,
    drag_adjusted_edge,
    equal_weight_prior,
    idzorek_omega,
    views_from_propagation,
    views_from_signals,
)

# Weekdays only, so calendar-day and trading-day arithmetic genuinely differ.
_SPAN = np.arange(np.datetime64("2026-01-05"), np.datetime64("2026-02-14"), dtype="datetime64[D]")
CALENDAR = _SPAN[np.is_busday(_SPAN)]

FRIDAY = date(2026, 1, 9)


def _signal(
    symbol: str,
    *,
    horizon_days: int = 10,
    direction: int = 1,
    available_on: date = FRIDAY,
    confidence: int = 100,
    predicted_move: float = 0.05,
) -> BacktestSignal:
    return BacktestSignal(
        symbol=symbol,
        available_on=available_on,
        direction=direction,
        predicted_move=predicted_move,
        horizon_days=horizon_days,
        confidence=confidence,
    )


def _cell(
    *,
    horizon_days: int = 10,
    direction: int = 1,
    event_type: str = "insider_transaction",
    event_subtype: str | None = "P",
    mean_car: float = 0.05,
) -> AdmittedCell:
    return AdmittedCell(
        event_type=event_type,
        event_subtype=event_subtype,
        horizon_days=horizon_days,
        direction=direction,
        mean_car=mean_car,
        hit_rate=0.58,
        n_clusters=40,
    )


def _sigma_3() -> np.ndarray:
    return np.array(
        [
            [0.0400, 0.0060, 0.0020],
            [0.0060, 0.0900, 0.0120],
            [0.0020, 0.0120, 0.0625],
        ],
        dtype=np.float64,
    )


class _RecordingLogger:
    """Stands in for the module's structlog logger so a test can assert on what
    got logged without depending on structlog's process-wide configuration.
    """

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **kwargs: Any) -> None:
        self.warnings.append((event, kwargs))


# --------------------------------------------------------------------------- #
# 1. the drag adjustment
# --------------------------------------------------------------------------- #
def test_drag_adjustment_golden_including_a_negative_result() -> None:
    """``hedged_edge - 0.5 * vol^2 * H``, hand-computed, sign preserved.

    A negative result is the whole point of the correction: a 1% measured edge
    on a 5%-a-day name is a losing trade once compounding is accounted for. The
    function must return it, not clamp it -- the caller declines to issue the
    view, and the report can say by how much it missed.
    """
    # 0.05 - 0.5 * 0.03^2 * 20 = 0.05 - 0.009
    assert drag_adjusted_edge(0.05, volatility=0.03, horizon_days=20) == pytest.approx(0.041)

    # 0.01 - 0.5 * 0.05^2 * 20 = 0.01 - 0.025.  Returned honestly, not clamped.
    thin = drag_adjusted_edge(0.01, volatility=0.05, horizon_days=20)
    assert thin == pytest.approx(-0.015)
    assert thin < 0.0

    # A riskless name has no drag, so the edge survives untouched.
    assert drag_adjusted_edge(0.02, volatility=0.0, horizon_days=60) == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# 2. expiry is counted in trading days
# --------------------------------------------------------------------------- #
def test_expiry_counts_trading_days_not_calendar_days() -> None:
    """A Friday signal with a 3-day horizon exits Wednesday, not Monday.

    Friday 2026-01-09 + 3 *calendar* days is Monday the 12th; + 3 *trading*
    days is Wednesday the 14th. Every day in between the two answers is a day
    the position is still supposed to be on.
    """
    signals = [_signal("AAA", horizon_days=3)]
    cells = [_cell(horizon_days=3)]
    edges = {cells[0].key: 0.030}
    vols = {"AAA": 0.01}

    # Tuesday the 13th: past the calendar-day answer, inside the trading-day one.
    live = views_from_signals(
        signals, cells, edges, vols, as_of=date(2026, 1, 13), calendar=CALENDAR
    )
    assert [view.symbol for view in live] == ["AAA"]
    assert live[0].expires_on == date(2026, 1, 14)

    # The expiry day itself still holds the position.
    assert views_from_signals(
        signals, cells, edges, vols, as_of=date(2026, 1, 14), calendar=CALENDAR
    )

    # One trading day past it there is no reason to hold, so there is no view.
    assert (
        views_from_signals(signals, cells, edges, vols, as_of=date(2026, 1, 15), calendar=CALENDAR)
        == ()
    )

    # A signal that has not fired yet is not a view either.
    assert (
        views_from_signals(signals, cells, edges, vols, as_of=date(2026, 1, 8), calendar=CALENDAR)
        == ()
    )


# --------------------------------------------------------------------------- #
# 3. closed-form Black-Litterman against an independent expression
# --------------------------------------------------------------------------- #
def test_closed_form_bl_matches_independent_matrix_expression() -> None:
    """Textbook 3-asset case, posterior computed two algebraically distinct ways.

    The implementation uses the precision form
    ``[(tS)^-1 + P'O^-1 P]^-1 [(tS)^-1 pi + P'O^-1 Q]``. This test uses the
    equivalent "solve" form ``pi + tSP'(PtSP' + O)^-1 (Q - P pi)``, written out
    here rather than imported, so agreement is evidence rather than tautology.
    """
    sigma = _sigma_3()
    pi = np.array([0.030, 0.050, 0.040])
    picks = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]])  # one absolute, one relative
    views = np.array([0.060, 0.010])
    omega = np.diag([0.001, 0.002])
    tau = 0.05

    tau_sigma_p = tau * sigma @ picks.T
    expected = pi + tau_sigma_p @ np.linalg.solve(picks @ tau_sigma_p + omega, views - picks @ pi)

    posterior = closed_form_bl(sigma, pi, picks, views, omega, tau=tau)
    assert posterior.shape == (3,)
    assert np.allclose(posterior, expected, rtol=0.0, atol=1e-10)

    # Sanity anchors, independent of the algebra: the view pulls asset 0 up from
    # its prior toward 6%, and asset 2 (untouched by any absolute view but
    # correlated with 1) does not run off.
    assert pi[0] < posterior[0] < 0.060
    assert abs(posterior[2] - pi[2]) < 0.02


# --------------------------------------------------------------------------- #
# 4. no views is a normal day
# --------------------------------------------------------------------------- #
def test_zero_views_returns_the_prior_unchanged() -> None:
    """A day with no live signal is normal; the posterior is exactly the prior."""
    sigma = _sigma_3()
    symbols = ("AAA", "BBB", "CCC")

    posterior = bl_posterior(sigma, symbols, (), engine="numpy")

    assert posterior.n_views == 0
    assert posterior.symbols == symbols
    assert np.array_equal(posterior.mu, equal_weight_prior(sigma))
    assert np.array_equal(posterior.sigma, sigma)

    # And the prior is the reverse optimization it claims to be: gamma * S * w_eq.
    hand = 2.5 * (sigma @ np.full(3, 1.0 / 3.0))
    assert np.allclose(equal_weight_prior(sigma), hand, rtol=0.0, atol=1e-15)


# --------------------------------------------------------------------------- #
# 5. Idzorek confidence behaviour
# --------------------------------------------------------------------------- #
def test_idzorek_confidence_orders_omega_and_the_posterior_tilt() -> None:
    """More confidence => smaller omega => posterior nearer the view.

    Asserted as an ordering rather than against a constant: the mapping is
    ``omega = tau * (1-c)/c * P S P'``, and what has to hold for the portfolio
    to behave is monotonicity, not any particular number.
    """
    sigma = _sigma_3()
    symbols = ("AAA", "BBB", "CCC")
    picks = np.array([[1.0, 0.0, 0.0]])
    view_return = np.array([0.20])
    pi = equal_weight_prior(sigma)

    high = idzorek_omega(picks, sigma, np.array([0.90]))
    low = idzorek_omega(picks, sigma, np.array([0.10]))
    faint = idzorek_omega(picks, sigma, np.array([1e-4]))

    assert high[0, 0] < low[0, 0] < faint[0, 0]

    mu_high = closed_form_bl(sigma, pi, picks, view_return, high)
    mu_low = closed_form_bl(sigma, pi, picks, view_return, low)
    mu_faint = closed_form_bl(sigma, pi, picks, view_return, faint)

    assert abs(mu_high[0] - 0.20) < abs(mu_low[0] - 0.20) < abs(mu_faint[0] - 0.20)
    # Near-zero confidence leaves the prior essentially untouched.
    assert abs(mu_faint[0] - pi[0]) < 0.01 * abs(0.20 - pi[0])

    # Full confidence is allowed and must stay invertible: the posterior lands
    # on the view instead of raising on a singular omega.
    certain = idzorek_omega(picks, sigma, np.array([1.0]))
    mu_certain = closed_form_bl(sigma, pi, picks, view_return, certain)
    assert mu_certain[0] == pytest.approx(0.20, abs=1e-6)

    # A confidence outside (0, 1] is not a confidence.
    with pytest.raises(ValueError):
        idzorek_omega(picks, sigma, np.array([0.0]))
    with pytest.raises(ValueError):
        idzorek_omega(picks, sigma, np.array([1.5]))

    # And build_view_matrices places each absolute view on its own symbol.
    built_p, built_q, built_c = build_view_matrices(
        (
            View("BBB", 0.03, 0.5, 10, date(2026, 1, 23), "insider"),
            View("AAA", 0.02, 0.8, 10, date(2026, 1, 23), "insider"),
        ),
        symbols,
    )
    assert np.array_equal(built_p, np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    assert np.array_equal(built_q, np.array([0.03, 0.02]))
    assert np.array_equal(built_c, np.array([0.5, 0.8]))


# --------------------------------------------------------------------------- #
# 6. the two engines agree
# --------------------------------------------------------------------------- #
def test_pypfopt_and_numpy_engines_agree() -> None:
    """Same inputs, both engines, agreement far tighter than 1e-6.

    Skipped rather than failed if pypfopt is ever dropped: the numpy engine is
    the one the platform can always run.
    """
    black_litterman = pytest.importorskip("pypfopt.black_litterman")

    sigma = _sigma_3()
    symbols = ("AAA", "BBB", "CCC")
    views = (
        View("AAA", 0.040, 0.75, 10, date(2026, 1, 23), "insider"),
        View("CCC", 0.015, 0.25, 20, date(2026, 2, 6), "insider"),
    )

    from_numpy = bl_posterior(sigma, symbols, views, engine="numpy")
    from_pypfopt = bl_posterior(sigma, symbols, views, engine="pypfopt")

    assert from_numpy.engine == "numpy"
    assert from_pypfopt.engine == "pypfopt"
    assert from_numpy.n_views == from_pypfopt.n_views == 2
    assert np.allclose(from_numpy.mu, from_pypfopt.mu, rtol=0.0, atol=1e-6)

    # The hand-rolled Idzorek mapping is pypfopt's, not merely similar to it.
    picks, quantities, confidences = build_view_matrices(views, symbols)
    theirs = black_litterman.BlackLittermanModel.idzorek_method(
        confidences.reshape(-1, 1), sigma, equal_weight_prior(sigma), quantities, picks, 0.05
    )
    assert np.allclose(idzorek_omega(picks, sigma, confidences), theirs, rtol=0.0, atol=1e-15)


# --------------------------------------------------------------------------- #
# 7. the fallback is loud
# --------------------------------------------------------------------------- #
def test_pypfopt_fallback_reports_numpy_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable pypfopt downgrades to numpy *on the record*.

    A silent switch would leave two runs incomparable with nothing in the log
    to say why, so the reported engine -- not just the numbers -- is asserted.
    """
    recorder = _RecordingLogger()
    monkeypatch.setattr(views_module, "_log", recorder)
    # None in sys.modules is a genuine ImportError at the real import site.
    monkeypatch.setitem(sys.modules, "pypfopt", None)
    monkeypatch.setitem(sys.modules, "pypfopt.black_litterman", None)

    sigma = _sigma_3()
    symbols = ("AAA", "BBB", "CCC")
    views = (View("AAA", 0.040, 0.75, 10, date(2026, 1, 23), "insider"),)

    posterior = bl_posterior(sigma, symbols, views, engine="pypfopt")

    assert posterior.engine == "numpy"
    assert np.array_equal(posterior.mu, bl_posterior(sigma, symbols, views, engine="numpy").mu)
    assert [event for event, _ in recorder.warnings] == ["bl_engine_fallback"]
    assert recorder.warnings[0][1]["requested"] == "pypfopt"
    assert recorder.warnings[0][1]["engine"] == "numpy"


# --------------------------------------------------------------------------- #
# 8. end to end
# --------------------------------------------------------------------------- #
def test_views_from_signals_end_to_end() -> None:
    """Signals + cells + hedged edges + vols -> drag-adjusted, positive views only.

    Three signals fire. Only one survives: the second is a thin edge on a
    volatile name whose drag eats it, and the third is an insider *sale*, whose
    hedged edge is negative because shorting an index cannot remove the alpha
    term.
    """
    purchase_10 = _cell(horizon_days=10, event_subtype="P", mean_car=0.06)
    purchase_20 = _cell(horizon_days=20, event_subtype="P", mean_car=0.03)
    sale_10 = _cell(horizon_days=10, event_subtype="S", direction=-1, mean_car=-0.04)
    cells = [purchase_10, purchase_20, sale_10]
    hedged_edges = {
        purchase_10.key: 0.030,
        purchase_20.key: 0.010,
        sale_10.key: -0.020,  # inverts under hedging: the alpha term is unhedgeable
    }
    volatilities = {"AAA": 0.02, "BBB": 0.05, "CCC": 0.02}
    signals = [
        _signal("AAA", horizon_days=10, confidence=80),
        _signal("BBB", horizon_days=20),
        _signal("CCC", horizon_days=10, direction=-1),
    ]

    issued = views_from_signals(
        signals,
        cells,
        hedged_edges,
        volatilities,
        as_of=date(2026, 1, 12),
        calendar=CALENDAR,
    )

    assert [view.symbol for view in issued] == ["AAA"]
    only = issued[0]
    # 0.030 - 0.5 * 0.02^2 * 10 = 0.030 - 0.002
    assert only.expected_excess_return == pytest.approx(0.028)
    assert only.confidence == pytest.approx(0.80)
    assert only.horizon_days == 10
    assert only.source == "insider"
    # Friday the 9th + 10 trading days = Friday the 23rd.
    assert only.expires_on == date(2026, 1, 23)

    # BBB's drag-adjusted edge is genuinely negative; the caller declines it.
    assert drag_adjusted_edge(0.010, volatility=0.05, horizon_days=20) < 0.0

    # A signal whose symbol has no volatility estimate cannot be priced at all:
    # there is no compound return to state without one.
    assert (
        views_from_signals(
            [_signal("AAA", horizon_days=10)],
            cells,
            hedged_edges,
            {},
            as_of=date(2026, 1, 12),
            calendar=CALENDAR,
        )
        == ()
    )

    # No propagation coefficient has ever been measured on this platform. An
    # empty admitted list is the expected production case and a legitimate
    # outcome -- not an error, and not a reason to hand-waive a cell in.
    graph_edges = (("AAA", "DDD", "supplier", date(2025, 6, 1)),)
    assert (
        views_from_propagation(
            [],
            signals,
            graph_edges,
            volatilities,
            as_of=date(2026, 1, 12),
            calendar=CALENDAR,
        )
        == ()
    )
