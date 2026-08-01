"""Portfolio optimization. Direct cvxpy; PyPortfolioOpt never enters here.

Two objectives:

  * ``max_sharpe``: ``max  mu'w - gamma*w'Sw - kappa*||w - w_cur||_1``
    The L1 turnover term is what makes a swap decision *net of costs* rather
    than a fresh optimization that happens to churn.
  * ``min_cvar``: Rockafellar-Uryasev's LP over scenario returns. Optimizes the
    average of the worst ``1-alpha`` tail rather than variance, which is the
    right objective when the concern is a drawdown, not a wiggle.

Constraints: ``sum(w) <= max_gross``, ``0 <= w <= max_position`` (long-only),
per-cluster sums ``<= max_cluster``.

**This module must never raise mid-replay.** A 2,500-day replay cannot stop on
day 1,400 to debug a solver. Escalation is Clarabel -> SCS -> deterministic
inverse-vol fallback, and every day carries its ``optimizer_status`` into the
result so a degraded day is visible rather than silently averaged in.

Verified 2026-07-31 on Python 3.14.6: both Clarabel and SCS solve this exact
objective over a rank-2 covariance across 7 names (min eigenvalue -3.6e-18)
without raising, returning identical weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cvxpy as cp
import numpy as np

OptimizerStatus = Literal["clarabel", "scs", "fallback_inverse_vol"]

# Escalation order. ECOS is deliberately absent -- it is not installed on this
# machine, and a solver name cvxpy does not know raises instead of degrading.
_SOLVER_ORDER: tuple[str, ...] = ("CLARABEL", "SCS")

_ACCEPTED_STATUSES = frozenset({cp.OPTIMAL, cp.OPTIMAL_INACCURATE})

# Weights below this are interior-point residue, not positions. Zeroing them
# keeps "long-only" literally true (0.0, not -1e-13) and keeps the simulator
# from generating sub-penny orders.
_DUST = 1e-8

# Tolerance for calling a cap "binding" when annotating a trade reason.
_CAP_TOLERANCE = 1e-8

# ``optimize`` exposes no alpha parameter, so the CVaR tail level is fixed
# here. Callers needing a different tail call ``solve_min_cvar`` directly.
_DEFAULT_CVAR_ALPHA = 0.95


@dataclass(frozen=True, eq=False)
class OptimizerResult:
    """Target weights plus the provenance needed to audit a degraded day.

    ``eq=False`` is deliberate: a generated ``__eq__`` would compare ``weights``
    with ``==``, which returns an array and raises ValueError at the call site
    of an innocuous-looking assertion. Identity comparison is the honest answer
    -- two results are not meaningfully "equal" -- so compare ``weights`` with
    ``np.allclose`` and say so.
    """

    weights: np.ndarray
    symbols: tuple[str, ...]
    status: OptimizerStatus
    objective_value: float | None
    trade_reasons: dict[str, str]


def _as_vector(values: object, size: int) -> np.ndarray:
    """Coerce to a finite float vector of exactly ``size``. Never raises."""
    try:
        vector = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.zeros(size)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    if vector.size == size:
        return vector
    padded = np.zeros(size)
    usable = min(size, vector.size)
    padded[:usable] = vector[:usable]
    return padded


def _as_covariance(values: object, size: int) -> np.ndarray | None:
    """Coerce to a finite symmetric ``[size, size]`` matrix, or None."""
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (size, size):
        return None
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    # Symmetrize before psd_wrap: an asymmetric input makes the quadratic form
    # ambiguous, and estimated covariances drift asymmetric at the ULP level.
    return 0.5 * (matrix + matrix.T)


def _bounds_and_groups(
    size: int,
    cluster_indices: tuple[tuple[int, ...], ...],
    max_position: float,
    max_cluster: float,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Per-name upper bounds plus the cluster groups that need a constraint.

    A name in no supplied cluster is its own singleton cluster, and a singleton
    cluster cap is just a tighter box bound -- folding it into ``ub`` keeps the
    problem at O(clusters) constraints instead of O(names).
    """
    multi: list[tuple[int, ...]] = []
    in_multi: set[int] = set()
    tightened: set[int] = set()
    for group in cluster_indices:
        members = tuple(sorted({int(i) for i in group if 0 <= int(i) < size}))
        if len(members) >= 2:
            multi.append(members)
            in_multi.update(members)
        elif members:
            tightened.update(members)
    tightened.update(index for index in range(size) if index not in in_multi)

    upper = np.full(size, max(float(max_position), 0.0))
    if tightened and max_cluster < max_position:
        upper[np.fromiter(sorted(tightened), dtype=int, count=len(tightened))] = max(
            float(max_cluster), 0.0
        )
    return upper, tuple(multi)


def _project(
    raw: np.ndarray,
    *,
    upper: np.ndarray,
    max_gross: float,
    groups: tuple[tuple[int, ...], ...],
    max_cluster: float,
) -> np.ndarray:
    """Snap a solver answer onto the feasible set. Only ever reduces weights.

    Interior-point solvers land a hair outside a binding constraint. Returning
    that hair would let a 15% cap hold 15.0000001% for 2,500 days, so the
    caps -- not the solver's tolerance -- decide what the account holds.
    """
    weights = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, upper)
    weights[weights < _DUST] = 0.0
    for group in groups:
        index = np.fromiter(group, dtype=int, count=len(group))
        total = float(weights[index].sum())
        if max_cluster > 0.0 and total > max_cluster:
            weights[index] *= max_cluster / total
    gross = float(weights.sum())
    if max_gross > 0.0 and gross > max_gross:
        weights *= max_gross / gross
    return weights


def _run_solvers(problem: cp.Problem) -> str | None:
    """Try each solver in escalation order. Returns the one that answered."""
    for solver in _SOLVER_ORDER:
        try:
            problem.solve(solver=solver)
        except Exception:  # a failed solver is data, not an error
            continue
        if problem.status in _ACCEPTED_STATUSES:
            return solver.lower()
    return None


def _solved_weights(variable: cp.Variable) -> np.ndarray | None:
    """The variable's value as a finite float vector, or None."""
    if variable.value is None:
        return None
    raw = np.asarray(variable.value, dtype=float).ravel()
    if not np.isfinite(raw).all():
        return None
    return raw


def solve_max_sharpe(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_current: np.ndarray,
    *,
    risk_aversion: float,
    turnover_penalty: float,
    max_gross: float,
    max_position: float,
    cluster_indices: tuple[tuple[int, ...], ...] = (),
    max_cluster: float = 1.0,
) -> tuple[np.ndarray | None, str, float | None]:
    """Mean-variance with an L1 turnover penalty. Returns ``(w, status, value)``.

    ``w`` is None when every solver failed; the caller falls back. Does not
    raise.
    """
    try:
        raw_mu = np.asarray(mu, dtype=float).ravel()
        size = raw_mu.size
        if size == 0:
            return None, "empty_universe", None
        covariance = _as_covariance(sigma, size)
        if covariance is None:
            return None, "malformed_covariance", None
        # A name whose own variance is unmeasurable -- NaN, inf, or negative --
        # is held at zero rather than repaired. Zeroing sigma's NaNs would
        # price it as riskless and hand it the *largest* position exactly when
        # least is known about it. Excluding the name instead lets the rest of
        # the book solve normally, so one bad estimate does not degrade the day.
        raw_variance = np.diag(np.asarray(sigma, dtype=float))
        measurable = np.isfinite(raw_mu) & np.isfinite(raw_variance) & (raw_variance >= 0.0)

        expected = np.nan_to_num(raw_mu, nan=0.0, posinf=0.0, neginf=0.0)
        current = _as_vector(w_current, size)

        # Negative penalties would flip the objective's curvature and make the
        # problem non-DCP; clamp rather than raise.
        gamma = max(float(risk_aversion), 0.0)
        kappa = max(float(turnover_penalty), 0.0)
        gross = max(float(max_gross), 0.0)
        upper, groups = _bounds_and_groups(size, cluster_indices, max_position, max_cluster)
        if not measurable.all():
            upper = np.where(measurable, upper, 0.0)
            # Blank the excluded rows and columns too. Those weights are pinned
            # to zero, so the quadratic form is unchanged at every feasible
            # point, but a repaired NaN left in place can make the matrix
            # indefinite and cost the whole day its solve.
            keep = measurable.astype(float)
            covariance = covariance * np.outer(keep, keep)

        weights = cp.Variable(size)
        objective = cp.Maximize(
            expected @ weights
            - gamma * cp.quad_form(weights, cp.psd_wrap(covariance))
            - kappa * cp.norm1(weights - current)
        )
        constraints = [cp.sum(weights) <= gross, weights >= 0, weights <= upper]
        constraints += [cp.sum(weights[list(group)]) <= max_cluster for group in groups]

        problem = cp.Problem(objective, constraints)
        solver = _run_solvers(problem)
        if solver is None:
            return None, "no_solver_converged", None
        raw = _solved_weights(weights)
        if raw is None:
            return None, "no_solver_converged", None

        final = _project(
            raw, upper=upper, max_gross=gross, groups=groups, max_cluster=max_cluster
        )
        # Recomputed at the projected weights so the reported value always
        # describes the portfolio actually returned.
        value = float(
            expected @ final
            - gamma * float(final @ covariance @ final)
            - kappa * float(np.abs(final - current).sum())
        )
        return final, solver, value
    except Exception:  # the replay must survive anything cvxpy does
        return None, "exception", None


def solve_min_cvar(
    scenario_returns: np.ndarray,
    w_current: np.ndarray,
    *,
    alpha: float,
    turnover_penalty: float,
    max_gross: float,
    max_position: float,
    cluster_indices: tuple[tuple[int, ...], ...] = (),
    max_cluster: float = 1.0,
) -> tuple[np.ndarray | None, str, float | None]:
    """Rockafellar-Uryasev CVaR LP over ``[S, N]`` scenario returns.

    Maximizes ``mean_return - CVaR_alpha(loss) - kappa*turnover``, the standard
    mean-CVaR application, with the mean taken from the scenarios themselves.

    Do not "correct" this to pure CVaR minimization. Under a long-only book
    with ``sum(w) <= max_gross`` as an *inequality*, minimizing CVaR alone has
    the trivial optimum of holding cash: zero position means zero tail loss, so
    the solver would answer ``w = 0`` on every single day. A risk objective
    whose answer is always "don't play" is not an objective. The return term is
    what makes the tail a price worth paying rather than a thing to avoid
    absolutely.
    """
    try:
        scenarios = np.asarray(scenario_returns, dtype=float)
        if scenarios.ndim != 2 or scenarios.shape[0] < 1 or scenarios.shape[1] < 1:
            return None, "malformed_scenarios", None
        scenarios = np.nan_to_num(scenarios, nan=0.0, posinf=0.0, neginf=0.0)
        draws, size = scenarios.shape
        current = _as_vector(w_current, size)

        # alpha at or beyond the ends divides by zero in the RU tail term.
        tail_alpha = min(max(float(alpha), 1e-6), 1.0 - 1e-6)
        kappa = max(float(turnover_penalty), 0.0)
        gross = max(float(max_gross), 0.0)
        upper, groups = _bounds_and_groups(size, cluster_indices, max_position, max_cluster)
        expected = scenarios.mean(axis=0)

        weights = cp.Variable(size)
        var_level = cp.Variable()
        excess = cp.Variable(draws)
        cvar = var_level + cp.sum(excess) / ((1.0 - tail_alpha) * draws)
        objective = cp.Maximize(
            expected @ weights - cvar - kappa * cp.norm1(weights - current)
        )
        constraints = [
            excess >= 0,
            excess >= -(scenarios @ weights) - var_level,
            cp.sum(weights) <= gross,
            weights >= 0,
            weights <= upper,
        ]
        constraints += [cp.sum(weights[list(group)]) <= max_cluster for group in groups]

        problem = cp.Problem(objective, constraints)
        solver = _run_solvers(problem)
        if solver is None:
            return None, "no_solver_converged", None
        raw = _solved_weights(weights)
        if raw is None:
            return None, "no_solver_converged", None

        final = _project(
            raw, upper=upper, max_gross=gross, groups=groups, max_cluster=max_cluster
        )
        losses = -(scenarios @ final)
        level = (
            float(var_level.value)
            if var_level.value is not None and np.isfinite(var_level.value)
            else float(np.quantile(losses, tail_alpha))
        )
        realized_cvar = level + float(np.maximum(losses - level, 0.0).mean()) / (1.0 - tail_alpha)
        value = float(
            expected @ final - realized_cvar - kappa * float(np.abs(final - current).sum())
        )
        return final, solver, value
    except Exception:  # the replay must survive anything cvxpy does
        return None, "exception", None


def fallback_weights(
    sigma: np.ndarray, mu: np.ndarray, *, max_gross: float, max_position: float
) -> np.ndarray:
    """Deterministic inverse-volatility weights, clipped. Cannot fail.

    Only names with positive expected return participate; the rest go to cash.
    Deterministic because two runs of the same replay must produce the same
    account, including on their degraded days.
    """
    try:
        expected = np.asarray(mu, dtype=float).ravel()
        size = expected.size
        if size == 0:
            return np.zeros(0)
        covariance = _as_covariance(sigma, size)
        variances = np.zeros(size) if covariance is None else np.diag(covariance).copy()

        with np.errstate(invalid="ignore"):
            volatility = np.sqrt(np.where(variances > 0.0, variances, 0.0))
        # A zero-vol name cannot be sized by inverse vol and a non-positive
        # expectation has no business being held; both go to cash.
        eligible = np.isfinite(expected) & (expected > 0.0) & (volatility > 0.0)
        if not eligible.any():
            return np.zeros(size)

        inverse = np.zeros(size)
        inverse[eligible] = 1.0 / volatility[eligible]
        total = float(inverse.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.zeros(size)

        gross = max(float(max_gross), 0.0)
        weights = inverse * (gross / total)
        # Clipping only reduces, so gross stays respected and the shortfall is
        # cash. No redistribution: a second pass would be another place to
        # diverge between two runs of the same replay.
        weights = np.clip(weights, 0.0, max(float(max_position), 0.0))
        weights[weights < _DUST] = 0.0
        return weights
    except Exception:  # this is the last rung; it answers or nothing does
        try:
            return np.zeros(int(np.asarray(mu).size))
        except Exception:
            return np.zeros(0)


def apply_no_trade_band(
    target: np.ndarray, current: np.ndarray, *, band: float, bypass: bool = False
) -> np.ndarray:
    """Suppress weight changes smaller than ``band``; ``bypass`` executes all.

    ``bypass=True`` is passed when the drawdown governor is de-risking
    (Michael's decision, 2026-07-31): if the band suppressed a governor-driven
    sell, the account would stay more exposed than the governor intended
    precisely during a drawdown. The band exists to damp ordinary rebalance
    churn, and de-risking is not ordinary.

    **A full exit is never banded.** A target of zero against a held position
    smaller than ``band`` would otherwise be suppressed every single day, and
    the name would be held forever -- the account cannot sell it because the
    sale is too small, and it never grows into a saleable size because nothing
    is buying it. That is the same absorbing failure the drawdown governor had
    before it gained a re-entry rule: a rule with no way out, reached by a
    position simply being small. Exits execute at any size; the band governs
    only how a position is *adjusted*, never whether it may be closed.
    """
    wanted = np.nan_to_num(np.asarray(target, dtype=float).ravel(), nan=0.0)
    held = _as_vector(current, wanted.size)
    if bypass or not np.isfinite(band) or band <= 0.0:
        return wanted.copy()
    # Strict ``<`` so a change of exactly one band executes.
    banded = np.where(np.abs(wanted - held) < float(band), held, wanted)
    closing = (wanted <= _DUST) & (held > _DUST)
    return np.where(closing, wanted, banded)


def _cluster_indices(
    symbols: tuple[str, ...], clusters: dict[str, tuple[str, ...]] | None
) -> tuple[tuple[int, ...], ...]:
    """Map cluster membership by symbol onto positional indices."""
    if not clusters:
        return ()
    position = {symbol: index for index, symbol in enumerate(symbols)}
    groups: list[tuple[int, ...]] = []
    for members in clusters.values():
        indices = tuple(sorted({position[name] for name in members if name in position}))
        if indices:
            groups.append(indices)
    return tuple(groups)


def _trade_reasons(
    symbols: tuple[str, ...],
    target: np.ndarray,
    current: np.ndarray,
    *,
    upper: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    max_cluster: float,
) -> dict[str, str]:
    """One line per symbol explaining the move and which cap, if any, shaped it."""
    at_cluster_cap: set[int] = set()
    if max_cluster > 0.0:
        for group in groups:
            members = np.fromiter(group, dtype=int, count=len(group))
            if float(target[members].sum()) >= max_cluster - _CAP_TOLERANCE:
                at_cluster_cap.update(int(member) for member in members)

    reasons: dict[str, str] = {}
    for index, symbol in enumerate(symbols):
        held, wanted = float(current[index]), float(target[index])
        delta = wanted - held
        if abs(delta) <= _DUST:
            action = "hold"
        elif held <= _DUST:
            action = "open"
        elif wanted <= _DUST:
            action = "close"
        else:
            action = "add" if delta > 0 else "trim"

        notes: list[str] = []
        if wanted > _DUST:
            if wanted >= float(upper[index]) - _CAP_TOLERANCE:
                notes.append("position cap")
            if index in at_cluster_cap:
                notes.append("cluster cap")
        reasons[symbol] = f"{action} ({', '.join(notes)})" if notes else action
    return reasons


def optimize(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_current: np.ndarray,
    symbols: tuple[str, ...],
    *,
    objective: Literal["max_sharpe", "min_cvar"] = "max_sharpe",
    risk_aversion: float = 2.5,
    turnover_penalty: float = 0.0025,
    max_gross: float = 1.0,
    max_position: float = 0.15,
    max_cluster: float = 0.30,
    clusters: dict[str, tuple[str, ...]] | None = None,
    scenario_returns: np.ndarray | None = None,
    alpha: float = _DEFAULT_CVAR_ALPHA,
) -> OptimizerResult:
    """Solve with escalation. Never raises.

    ``alpha`` is the CVaR tail level and is ignored by ``max_sharpe``. It is
    last so that adding it shifts no existing call site.
    """
    names = tuple(symbols)
    size = len(names)
    current = _as_vector(w_current, size)
    groups: tuple[tuple[int, ...], ...] = ()
    upper = np.full(size, max(float(max_position), 0.0))

    weights: np.ndarray | None = None
    status_name = "not_attempted"
    value: float | None = None
    try:
        groups = _cluster_indices(names, clusters)
        upper, solver_groups = _bounds_and_groups(size, groups, max_position, max_cluster)
        groups = solver_groups
        if objective == "min_cvar":
            if scenario_returns is None:
                status_name = "cvar_scenarios_missing"
            else:
                weights, status_name, value = solve_min_cvar(
                    scenario_returns,
                    current,
                    alpha=alpha,
                    turnover_penalty=turnover_penalty,
                    max_gross=max_gross,
                    max_position=max_position,
                    cluster_indices=groups,
                    max_cluster=max_cluster,
                )
        else:
            weights, status_name, value = solve_max_sharpe(
                mu,
                sigma,
                current,
                risk_aversion=risk_aversion,
                turnover_penalty=turnover_penalty,
                max_gross=max_gross,
                max_position=max_position,
                cluster_indices=groups,
                max_cluster=max_cluster,
            )
        if weights is not None:
            weights = np.asarray(weights, dtype=float).ravel()
            if weights.size != size or not np.isfinite(weights).all():
                weights = None
    except Exception:  # escalate rather than stop the replay
        weights, status_name, value = None, "exception", None

    if weights is None or status_name not in {"clarabel", "scs"}:
        # The fallback is projected through the same caps the solver faced, so
        # a degraded day still cannot breach a position or cluster limit. The
        # length is normalized *before* projecting: this is the last rung, and
        # it has to answer even when the caller's arrays disagree with
        # ``symbols``.
        try:
            raw_fallback = _as_vector(
                fallback_weights(sigma, mu, max_gross=max_gross, max_position=max_position), size
            )
            weights = _project(
                raw_fallback,
                upper=upper,
                max_gross=max(float(max_gross), 0.0),
                groups=groups,
                max_cluster=max_cluster,
            )
        except Exception:  # nothing below this rung; hold cash rather than raise
            weights = np.zeros(size)
        status: OptimizerStatus = "fallback_inverse_vol"
        value = None
    else:
        status = "clarabel" if status_name == "clarabel" else "scs"

    reasons = _trade_reasons(
        names, weights, current, upper=upper, groups=groups, max_cluster=max_cluster
    )
    return OptimizerResult(
        weights=weights,
        symbols=names,
        status=status,
        objective_value=value,
        trade_reasons=reasons,
    )


__all__ = [
    "OptimizerResult",
    "OptimizerStatus",
    "apply_no_trade_band",
    "fallback_weights",
    "optimize",
    "solve_max_sharpe",
    "solve_min_cvar",
]
