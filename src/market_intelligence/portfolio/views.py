"""Return views and the Black-Litterman posterior.

The single most important line in this module is the drag adjustment:

    mu_view = hedged_edge - 0.5 * sigma^2 * H

A cell's measured label is a *sum* of daily abnormal returns; what an account
earns is a *compound* return. The gap between them is volatility drag, and on
the volatile small-caps insider signals fire on, that drag is worth roughly the
entire measured edge. Feeding raw label CAR into an optimizer produces a
portfolio that looks excellent and earns nothing. Never use raw label CAR.

The edge must also be beta-hedged: ``analytics.returns`` defines abnormal
return as ``total - (alpha + beta * market)``, and shorting an index removes
only the beta term. The unhedgeable alpha term is why every insider-*sale*
cell inverts under hedging and why the tradeable set is all purchases.

Prior: reverse-optimized from equal weight. There is no market-cap data in this
platform, so an equal-weight prior is the honest neutral -- not a claim that
equal weight is optimal, but a refusal to invent capitalizations.

``bl_engine`` verified 2026-07-31: PyPortfolioOpt's posterior matches the
hand-rolled closed form to 1.4e-17 on pandas 3.0.3, so "pypfopt" is the default
and "numpy" is a tested substitute. Neither ever enters the optimizer's solve.

``BLPosterior.sigma`` is the covariance it was handed, not BL's ``S + M``. The
``M`` inflation is uncertainty about the *mean*, and letting it through would
quietly scale every risk budget by ``(1 + tau)`` -- the optimizer, the position
caps and the volatility target would all be reading a covariance that no test in
``risk.py`` ever measured. The estimate that was measured is the one that ships.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np

from market_intelligence.analytics.backtest import BacktestSignal
from market_intelligence.analytics.backtest_data import AdmittedCell
from market_intelligence.logging_config import get_logger

_log = get_logger("portfolio.views")

#: Edge types a *return* may travel along, if the measurement gate ever admits
#: one. ``competitor`` is absent because its sign is unknown -- a rival's good
#: news is plausibly this firm's bad news, and nothing here has measured which.
#: ``shared_board_member`` is absent for the same reason ``risk.py`` refuses to
#: cluster on it: a shared director is a channel, not a measured relationship.
_PROPAGATION_EDGE_TYPES = frozenset({"customer", "partner", "supplier"})

#: A confidence of exactly 1.0 implies ``omega = 0``, which is singular. Scaling
#: the view variance by this instead keeps the posterior on the view to ~1e-10
#: while leaving the matrix invertible.
_CERTAINTY_FLOOR = 1e-10

#: Absolute floor for an omega entry, for the degenerate zero-variance view.
_OMEGA_FLOOR = 1e-16


@dataclass(frozen=True)
class View:
    """One probabilistic return forecast, with an expiry.

    ``expires_on`` is the signal's natural horizon exit. Past it the view is
    dropped and the optimizer trims the position on its own -- there is no
    separate exit rule, the absence of a reason to hold *is* the exit.
    """

    symbol: str
    expected_excess_return: float
    confidence: float  # in (0, 1]; feeds Idzorek's omega
    horizon_days: int
    expires_on: date
    source: str  # "insider" | "gap" | "propagation:<edge_type>"


@dataclass(frozen=True)
class BLPosterior:
    """Posterior mean and covariance, plus which engine produced them."""

    mu: np.ndarray
    sigma: np.ndarray
    symbols: tuple[str, ...]
    engine: Literal["pypfopt", "numpy"]
    n_views: int


def _trading_days(calendar: np.ndarray) -> np.ndarray:
    """Validate the session calendar and normalize it to ``datetime64[D]``."""
    days = np.asarray(calendar).astype("datetime64[D]")
    if days.ndim != 1:
        raise ValueError(f"calendar must be one-dimensional, got shape {days.shape}")
    if days.size == 0:
        raise ValueError("calendar is empty")
    if days.size > 1 and not bool((np.diff(days) > np.timedelta64(0, "D")).all()):
        raise ValueError("calendar must be strictly ascending")
    return days


def _session_index(days: np.ndarray, day: date) -> int:
    """Index of ``day`` in the calendar, or of the next session after it.

    A non-session date reads as the session that follows it: a signal released
    on a Saturday is actionable on Monday, and that is the day the horizon
    starts counting from.
    """
    position = int(np.searchsorted(days, np.datetime64(day, "D"), side="left"))
    if position >= days.size:
        raise ValueError(f"{day.isoformat()} is past the end of the calendar")
    return position


def _cell_order(cell: AdmittedCell) -> tuple[str, str, int]:
    """Total order over cells. ``event_subtype`` is nullable, so None sorts first."""
    return (cell.event_type, cell.event_subtype or "", cell.horizon_days)


def _source_for(cell: AdmittedCell) -> str:
    """``"insider"`` / ``"gap"``: the family of the cell's event type.

    ``event_type`` is ``insider_transaction``, ``insider_cluster_buy``, ``gap``
    and so on; the leading token is the family and is what ``View.source``
    names. Deriving it beats a lookup table that a new event type would silently
    fall out of.
    """
    return cell.event_type.split("_", 1)[0].lower()


def _matchable_cells(
    cells: list[AdmittedCell], hedged_edges: dict[tuple[str, str | None, int], float]
) -> dict[tuple[int, int], tuple[AdmittedCell, float]]:
    """``(horizon, direction) -> (cell, hedged_edge)``, most conservative edge wins.

    ``BacktestSignal`` carries no cell key -- only a symbol, a horizon and a
    direction -- so a signal can be matched to the cell that fired it *only* by
    ``(horizon_days, direction)``. When several admitted cells share that pair,
    the smallest hedged edge is taken: the signal does not say which cell fired,
    and taking the largest would let an unrelated strong cell price this trade,
    which is upward-biased selection dressed as a match.

    A cell with no entry in ``hedged_edges`` is skipped entirely. Its raw
    ``mean_car`` is an unhedged label, and substituting it here is precisely the
    mistake this module exists to prevent.
    """
    index: dict[tuple[int, int], tuple[AdmittedCell, float]] = {}
    for cell in cells:
        edge = hedged_edges.get(cell.key)
        if edge is None:
            continue
        key = (int(cell.horizon_days), int(cell.direction))
        current = index.get(key)
        if current is None or (float(edge), _cell_order(cell)) < (
            current[1],
            _cell_order(current[0]),
        ):
            index[key] = (cell, float(edge))
    return index


def _one_per_symbol(candidates: list[View]) -> tuple[View, ...]:
    """Collapse to one view per symbol, keeping the most conservative.

    The account holds one position per symbol, so it can act on one forecast per
    symbol. Two views on the same name would otherwise enter the posterior as
    two absolute rows arguing with each other, and the optimizer would size the
    argument's midpoint without anything recording that there was one. Smallest
    expected return wins, for the same reason as in ``_matchable_cells``.
    Returned in symbol order so a replay is byte-reproducible.
    """
    best: dict[str, tuple[float, date, str, View]] = {}
    for view in candidates:
        rank = (view.expected_excess_return, view.expires_on, view.source)
        current = best.get(view.symbol)
        if current is None or rank < current[:3]:
            best[view.symbol] = (*rank, view)
    return tuple(best[symbol][3] for symbol in sorted(best))


def drag_adjusted_edge(hedged_edge: float, *, volatility: float, horizon_days: int) -> float:
    """``hedged_edge - 0.5 * vol^2 * H``. The compounding correction.

    ``volatility`` is the symbol's daily return standard deviation; ``H`` is the
    horizon in trading days. Result may be negative -- that is informative, not
    an error, and such a view should simply not be issued.
    """
    if not np.isfinite(hedged_edge):
        raise ValueError(f"hedged_edge must be finite, got {hedged_edge}")
    if not np.isfinite(volatility) or volatility < 0.0:
        raise ValueError(f"volatility must be finite and non-negative, got {volatility}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be at least 1, got {horizon_days}")
    return float(hedged_edge) - 0.5 * float(volatility) ** 2 * float(horizon_days)


def views_from_signals(
    signals: list[BacktestSignal],
    cells: list[AdmittedCell],
    hedged_edges: dict[tuple[str, str | None, int], float],
    volatilities: dict[str, float],
    *,
    as_of: date,
    calendar: np.ndarray,
) -> tuple[View, ...]:
    """Live views on ``as_of``: fired, not expired, drag-adjusted, positive.

    Expiry is counted in *trading* days off ``calendar``, matching the
    validated backtest geometry. A day with no live views is normal and the
    account simply rides its existing positions.

    Four reasons a fired signal produces nothing, all of them ordinary:
    its horizon has elapsed; no admitted cell matches its horizon and
    direction; its symbol has no volatility estimate, so no compound return can
    be stated for it; or the drag-adjusted edge is not positive -- which is the
    whole tradeable set of insider *sales*, whose hedged edge is negative.
    """
    days = _trading_days(calendar)
    as_of_index = _session_index(days, as_of)
    matchable = _matchable_cells(cells, hedged_edges)

    candidates: list[View] = []
    for signal in signals:
        if signal.available_on > as_of:
            continue
        matched = matchable.get((int(signal.horizon_days), int(signal.direction)))
        if matched is None:
            continue
        volatility = volatilities.get(signal.symbol)
        if volatility is None:
            continue
        start = _session_index(days, signal.available_on)
        exit_index = start + int(signal.horizon_days)
        if as_of_index > exit_index:
            continue
        cell, hedged_edge = matched
        expected = drag_adjusted_edge(
            hedged_edge, volatility=volatility, horizon_days=signal.horizon_days
        )
        if expected <= 0.0:
            continue
        confidence = min(max(float(signal.confidence) / 100.0, 0.0), 1.0)
        if confidence <= 0.0:
            # A view held with no confidence moves no posterior; issuing it
            # would only add a row of noise to P.
            continue
        candidates.append(
            View(
                symbol=signal.symbol,
                expected_excess_return=expected,
                confidence=confidence,
                horizon_days=int(signal.horizon_days),
                expires_on=_expiry_date(days, exit_index),
                source=_source_for(cell),
            )
        )
    return _one_per_symbol(candidates)


def _expiry_date(days: np.ndarray, exit_index: int) -> date:
    """Calendar date of the horizon exit, clamped to the last known session.

    A forward-running calendar ends at today, so the exit session is often not
    yet nameable. Clamping reports the last day the calendar can vouch for while
    liveness keeps being decided by elapsed sessions, which preserves the
    invariant that matters: an issued view always has ``as_of <= expires_on``.
    """
    exit_day: date = days[min(exit_index, days.size - 1)].astype(object)
    return exit_day


def views_from_propagation(
    admitted_propagation_cells: list[AdmittedCell],
    signals: list[BacktestSignal],
    edges: tuple[tuple[str, str, str, date], ...],
    volatilities: dict[str, float],
    *,
    as_of: date,
    calendar: np.ndarray,
) -> tuple[View, ...]:
    """Graph-propagated views -- ONLY for cells that passed the measurement gate.

    An empty result is a legitimate, expected outcome. No propagation
    coefficient has ever been measured; if zero cells pass the gate this returns
    empty, ``graph_return_views_enabled`` stays False, and the report says
    "propagation gate: 0 admitted". The machinery ships either way. No cell is
    ever hand-waived in.

    Competitor edges are risk-structure only regardless of the gate: no signed
    return semantics have been measured for them, so their direction is unknown.

    An admitted propagation cell's ``mean_car`` is read as an already-hedged
    edge, because the gate that admits it measures on the same hedged
    abnormal-return basis as every other cell. The drag correction is applied on
    top, against the *neighbour's* volatility -- the neighbour is the name being
    held. If that ever stops being true, the gate is where it gets fixed; a
    correction applied here would be invisible to the measurement.
    """
    if not admitted_propagation_cells:
        # The expected production case. No coefficient has ever been measured,
        # so there is nothing to propagate and that is a result, not a failure.
        return ()

    days = _trading_days(calendar)
    as_of_index = _session_index(days, as_of)
    neighbours = _propagation_neighbours(edges, as_of=as_of)
    by_horizon: dict[tuple[int, int], AdmittedCell] = {}
    for cell in admitted_propagation_cells:
        key = (int(cell.horizon_days), int(cell.direction))
        current = by_horizon.get(key)
        if current is None or cell.predicted_move < current.predicted_move:
            by_horizon[key] = cell

    candidates: list[View] = []
    for signal in signals:
        if signal.available_on > as_of:
            continue
        matched = by_horizon.get((int(signal.horizon_days), int(signal.direction)))
        if matched is None:
            continue
        start = _session_index(days, signal.available_on)
        exit_index = start + int(signal.horizon_days)
        if as_of_index > exit_index:
            continue
        confidence = min(max(float(signal.confidence) / 100.0, 0.0), 1.0)
        if confidence <= 0.0:
            continue
        signed_edge = int(matched.direction) * matched.predicted_move
        for neighbour, edge_type in neighbours.get(signal.symbol, ()):
            volatility = volatilities.get(neighbour)
            if volatility is None:
                continue
            expected = drag_adjusted_edge(
                signed_edge, volatility=volatility, horizon_days=signal.horizon_days
            )
            if expected <= 0.0:
                continue
            candidates.append(
                View(
                    symbol=neighbour,
                    expected_excess_return=expected,
                    confidence=confidence,
                    horizon_days=int(signal.horizon_days),
                    expires_on=_expiry_date(days, exit_index),
                    source=f"propagation:{edge_type}",
                )
            )
    return _one_per_symbol(candidates)


def _propagation_neighbours(
    edges: tuple[tuple[str, str, str, date], ...], *, as_of: date
) -> dict[str, tuple[tuple[str, str], ...]]:
    """``symbol -> ((neighbour, edge_type), ...)``, point-in-time and undirected.

    Admits an edge only when ``report_date <= as_of``: today's graph must not
    explain last year's returns. Traversed in both directions because the
    measured relationship is symmetric even though the economics are not --
    nothing here has established which way a shock travels, and pretending the
    stored ``(source, target)`` order encodes that would be an invention.
    """
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for source, target, edge_type, report_date in edges:
        kind = str(edge_type).strip().lower()
        if kind not in _PROPAGATION_EDGE_TYPES:
            continue
        if report_date > as_of or source == target:
            continue
        adjacency.setdefault(source, []).append((target, kind))
        adjacency.setdefault(target, []).append((source, kind))
    return {symbol: tuple(sorted(set(links))) for symbol, links in adjacency.items()}


def _covariance(sigma: np.ndarray) -> np.ndarray:
    """Validate a square, finite covariance and normalize it to float64."""
    matrix = np.asarray(sigma, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"sigma must be square, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("sigma must cover at least one symbol")
    if not np.isfinite(matrix).all():
        raise ValueError("sigma contains non-finite values")
    return matrix


def equal_weight_prior(sigma: np.ndarray, *, risk_aversion: float = 2.5) -> np.ndarray:
    """Reverse-optimized implied returns from equal weight: ``pi = gamma * S * w_eq``."""
    matrix = _covariance(sigma)
    if risk_aversion <= 0.0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
    size = matrix.shape[0]
    return risk_aversion * (matrix @ np.full(size, 1.0 / size))


def build_view_matrices(
    views: tuple[View, ...], symbols: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(P, Q, confidences)`` -- one absolute-view row per view, in symbol order."""
    position = {symbol: index for index, symbol in enumerate(symbols)}
    if len(position) != len(symbols):
        raise ValueError("symbols contains duplicates")
    picks = np.zeros((len(views), len(symbols)), dtype=np.float64)
    quantities = np.zeros(len(views), dtype=np.float64)
    confidences = np.zeros(len(views), dtype=np.float64)
    for row, view in enumerate(views):
        if view.symbol not in position:
            # Fail closed: a view on a name the universe cannot hold would
            # otherwise tilt the posterior of whatever happened to be row 0.
            raise ValueError(f"view on {view.symbol!r} is outside the universe")
        if not 0.0 < view.confidence <= 1.0:
            raise ValueError(f"confidence must be in (0, 1], got {view.confidence}")
        picks[row, position[view.symbol]] = 1.0
        quantities[row] = view.expected_excess_return
        confidences[row] = view.confidence
    return picks, quantities, confidences


def idzorek_omega(
    P: np.ndarray, sigma: np.ndarray, confidences: np.ndarray, *, tau: float = 0.05
) -> np.ndarray:
    """Idzorek (2005): map view confidence in ``(0, 1]`` to an omega diagonal.

    Confidence 1.0 means the posterior moves all the way to the view; low
    confidence leaves the prior nearly untouched. Hand-rolled so the mapping is
    inspectable rather than inherited.

    ``omega_kk = tau * (1 - c) / c * P_k S P_k'`` -- Idzorek's formula (41) with
    alpha from (44), the closed form Walters (2014) gives. Identical to
    PyPortfolioOpt's ``idzorek_method``, which the tests assert against.
    """
    matrix = _covariance(sigma)
    picks = np.asarray(P, dtype=np.float64)
    if picks.ndim != 2 or picks.shape[1] != matrix.shape[0]:
        raise ValueError(f"P must be (K, {matrix.shape[0]}), got shape {picks.shape}")
    weights = np.asarray(confidences, dtype=np.float64).ravel()
    if weights.size != picks.shape[0]:
        raise ValueError(f"{picks.shape[0]} views but {weights.size} confidences")
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    if weights.size and (not np.isfinite(weights).all() or weights.min() <= 0.0):
        raise ValueError("confidence must be in (0, 1]")
    if weights.size and weights.max() > 1.0:
        raise ValueError("confidence must be in (0, 1]")

    view_variance = np.einsum("kn,nm,km->k", picks, matrix, picks)
    # A certain view is one whose alpha is zero, i.e. omega = 0, which cannot be
    # inverted. Treating certainty as "certain to 1e-10" keeps the posterior on
    # the view without asking the closed form to invert a singular matrix.
    alpha = np.where(weights >= 1.0, _CERTAINTY_FLOOR, (1.0 - weights) / weights)
    diagonal = tau * alpha * view_variance
    return np.diag(np.maximum(diagonal, _OMEGA_FLOOR))


def closed_form_bl(
    sigma: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    *,
    tau: float = 0.05,
) -> np.ndarray:
    """``mu = [(tS)^-1 + P'O^-1 P]^-1 [(tS)^-1 pi + P'O^-1 Q]``. ~30-line reference.

    The precision form, written out because it is the one every textbook states
    and therefore the one a reader can check. PyPortfolioOpt evaluates the
    algebraically identical ``pi + tSP'(PtSP' + O)^-1 (Q - P pi)``; the two
    agreeing to 1e-17 is what makes either engine's output trustworthy.
    """
    matrix = _covariance(sigma)
    prior = np.asarray(pi, dtype=np.float64).ravel()
    picks = np.asarray(P, dtype=np.float64)
    quantities = np.asarray(Q, dtype=np.float64).ravel()
    uncertainty = np.asarray(omega, dtype=np.float64)
    size = matrix.shape[0]
    if prior.size != size:
        raise ValueError(f"pi must have {size} entries, got {prior.size}")
    if picks.ndim != 2 or picks.shape[1] != size:
        raise ValueError(f"P must be (K, {size}), got shape {picks.shape}")
    count = picks.shape[0]
    if quantities.size != count:
        raise ValueError(f"{count} views but {quantities.size} entries in Q")
    if uncertainty.shape != (count, count):
        raise ValueError(f"omega must be ({count}, {count}), got {uncertainty.shape}")
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    if count == 0:
        # No views is not a degenerate case to guard against; it is most days.
        return prior.copy()

    tau_sigma_inverse = np.linalg.inv(tau * matrix)
    omega_inverse = np.linalg.inv(uncertainty)
    precision = tau_sigma_inverse + picks.T @ omega_inverse @ picks
    weighted = tau_sigma_inverse @ prior + picks.T @ omega_inverse @ quantities
    return np.asarray(np.linalg.solve(precision, weighted), dtype=np.float64)


def bl_posterior(
    sigma: np.ndarray,
    symbols: tuple[str, ...],
    views: tuple[View, ...],
    *,
    engine: Literal["pypfopt", "numpy"] = "pypfopt",
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> BLPosterior:
    """Posterior from prior + views. Zero views must return the prior unchanged.

    Falls back from "pypfopt" to "numpy" LOUDLY -- a silent engine switch would
    make two runs incomparable without any record of why.

    The engine is resolved before the view count is looked at, so a day with no
    live views still records which engine was in play. "pypfopt" on a day it
    could not have imported would be a false entry in that record.
    """
    matrix = _covariance(sigma)
    names = tuple(symbols)
    if matrix.shape[0] != len(names):
        raise ValueError(f"sigma is {matrix.shape[0]}x{matrix.shape[0]} but {len(names)} symbols")
    if engine not in ("pypfopt", "numpy"):
        raise ValueError(f"unknown bl engine {engine!r}")

    resolved = _resolve_engine(engine)
    prior = equal_weight_prior(matrix, risk_aversion=risk_aversion)
    if not views:
        return BLPosterior(mu=prior, sigma=matrix, symbols=names, engine=resolved, n_views=0)

    picks, quantities, confidences = build_view_matrices(views, names)
    omega = idzorek_omega(picks, matrix, confidences, tau=tau)
    if resolved == "pypfopt":
        mu = _pypfopt_returns(matrix, names, prior, picks, quantities, omega, tau=tau)
    else:
        mu = closed_form_bl(matrix, prior, picks, quantities, omega, tau=tau)
    return BLPosterior(mu=mu, sigma=matrix, symbols=names, engine=resolved, n_views=len(views))


def _black_litterman_model() -> Any:
    """Import PyPortfolioOpt's model, or raise ``ImportError``. One import site."""
    from pypfopt.black_litterman import BlackLittermanModel  # type: ignore[import-untyped]

    return BlackLittermanModel


def _resolve_engine(engine: Literal["pypfopt", "numpy"]) -> Literal["pypfopt", "numpy"]:
    """Which engine will actually run, announcing any downgrade."""
    if engine == "numpy":
        return "numpy"
    try:
        _black_litterman_model()
    except ImportError as error:
        _log.warning(
            "bl_engine_fallback",
            requested="pypfopt",
            engine="numpy",
            reason=str(error),
        )
        return "numpy"
    return "pypfopt"


def _pypfopt_returns(
    sigma: np.ndarray,
    symbols: tuple[str, ...],
    pi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    *,
    tau: float,
) -> np.ndarray:
    """PyPortfolioOpt's posterior. The only place pandas touches this package.

    pypfopt labels its output by the covariance frame's columns, so it wants a
    DataFrame at its boundary. Convert at the call, convert straight back: the
    array that leaves here is the same numpy the numpy engine would have
    returned, and nothing downstream can tell which one produced it.
    """
    import pandas as pd

    model = _black_litterman_model()(
        pd.DataFrame(sigma, index=list(symbols), columns=list(symbols)),
        pi=pd.Series(pi, index=list(symbols)),
        P=P,
        Q=Q,
        omega=omega,
        tau=tau,
    )
    return np.asarray(model.bl_returns().to_numpy(), dtype=np.float64)


__all__ = [
    "BLPosterior",
    "View",
    "bl_posterior",
    "build_view_matrices",
    "closed_form_bl",
    "drag_adjusted_edge",
    "equal_weight_prior",
    "idzorek_omega",
    "views_from_propagation",
    "views_from_signals",
]
