from __future__ import annotations

import itertools

import numpy as np
import pytest

from market_intelligence.portfolio import optimizer as opt
from market_intelligence.portfolio.optimizer import (
    OptimizerResult,
    apply_no_trade_band,
    fallback_weights,
    optimize,
    solve_max_sharpe,
    solve_min_cvar,
)


def _sigma(vols: np.ndarray, correlation: float = 0.0) -> np.ndarray:
    """Full-rank covariance from a flat correlation matrix."""
    corr = np.full((vols.size, vols.size), correlation)
    np.fill_diagonal(corr, 1.0)
    return corr * np.outer(vols, vols)


def _rank_two_covariance() -> tuple[np.ndarray, np.ndarray]:
    """Seven names spanned by two factors -- singular, faintly negative eigenvalue."""
    rng = np.random.default_rng(20260731)
    loadings = rng.normal(size=(7, 2))
    return loadings, (loadings @ loadings.T) * 0.01


def _equal_mean_equal_variance_scenarios() -> np.ndarray:
    """Two names, identical mean AND identical variance, opposite tail shape.

    Both carry a +6% drift and a variance of exactly 0.0025.  The pair is
    symmetric under swapping in every moment a mean-variance objective can see
    (same mu, same diagonal, one shared covariance term), so mean-variance must
    weight them equally.  They differ only in shape: the thin name is a
    symmetric +/-5%, while the fat name gives back 25% in 10 of 260 scenarios
    to pay for a steady +1%.  The fat name's losses land on the thin name's bad
    days, so the tail is not cushioned by the other leg.
    """
    thin = np.concatenate([np.full(130, 0.06 + 0.05), np.full(130, 0.06 - 0.05)])
    fat = np.concatenate([np.full(250, 0.06 + 0.01), np.full(10, 0.06 - 0.25)])
    return np.column_stack([thin, fat])


def test_position_cap_binds_for_an_overwhelming_asset() -> None:
    mu = np.array([0.50, 0.001, 0.001, 0.001])
    weights, status, value = solve_max_sharpe(
        mu,
        _sigma(np.full(4, 0.20)),
        np.zeros(4),
        risk_aversion=2.5,
        turnover_penalty=0.0,
        max_gross=1.0,
        max_position=0.15,
    )
    assert weights is not None
    assert status == "clarabel"
    assert value is not None
    # Unconstrained this name wants 2.5x the book; the cap is what stops it.
    assert weights[0] == pytest.approx(0.15, abs=1e-6)
    assert weights.max() <= 0.15


def test_cluster_cap_binds_and_singleton_names_do_not() -> None:
    mu = np.array([0.40, 0.39, 0.38, 0.001, 0.001])
    sigma = _sigma(np.full(5, 0.20))
    kwargs = {
        "risk_aversion": 2.5,
        "turnover_penalty": 0.0,
        "max_gross": 1.0,
        "max_position": 0.15,
        "max_cluster": 0.30,
    }

    clustered, _, _ = solve_max_sharpe(
        mu, sigma, np.zeros(5), cluster_indices=((0, 1, 2),), **kwargs
    )
    assert clustered is not None
    assert clustered[:3].sum() <= 0.30 + 1e-6
    assert clustered[:3].sum() == pytest.approx(0.30, abs=1e-5)

    # With no clusters supplied every name is its own singleton, so the cluster
    # cap cannot bind across names -- only each name's own position cap does.
    free, _, _ = solve_max_sharpe(mu, sigma, np.zeros(5), **kwargs)
    assert free is not None
    assert free[:3].sum() == pytest.approx(0.45, abs=1e-5)
    assert free[:3].sum() > 0.30


def test_long_only_holds_against_a_negative_expected_return() -> None:
    mu = np.array([0.20, -0.40, 0.05])
    weights, _, _ = solve_max_sharpe(
        mu,
        _sigma(np.array([0.20, 0.25, 0.30]), correlation=0.1),
        np.zeros(3),
        risk_aversion=2.5,
        turnover_penalty=0.0,
        max_gross=1.0,
        max_position=0.15,
    )
    assert weights is not None
    assert weights[1] == 0.0
    assert (weights >= 0.0).all()
    # Long-and-cash: thin expectations must be allowed to leave the book idle.
    assert weights.sum() < 1.0


def test_turnover_penalty_monotonically_suppresses_churn() -> None:
    mu = np.array([0.30, 0.05, 0.01])
    sigma = _sigma(np.full(3, 0.25), correlation=0.2)
    w_current = np.array([0.0, 0.12, 0.10])

    distances = []
    for kappa in (0.0, 0.01, 0.05, 0.25, 5.0):
        weights, _, _ = solve_max_sharpe(
            mu,
            sigma,
            w_current,
            risk_aversion=2.5,
            turnover_penalty=kappa,
            max_gross=1.0,
            max_position=0.15,
        )
        assert weights is not None
        distances.append(float(np.abs(weights - w_current).sum()))

    for earlier, later in itertools.pairwise(distances):
        assert later <= earlier + 1e-6
    assert distances[0] > 1e-3
    assert distances[-1] < 1e-4


def test_no_trade_band_suppresses_dust_and_the_governor_bypasses_it() -> None:
    current = np.array([0.10, 0.10, 0.10])
    target = np.array([0.105, 0.13, 0.02])

    banded = apply_no_trade_band(target, current, band=0.01)
    assert banded[0] == pytest.approx(0.10)  # 0.005 < band -> suppressed
    assert banded[1] == pytest.approx(0.13)  # 0.030 > band -> executes
    assert banded[2] == pytest.approx(0.02)

    assert apply_no_trade_band(target, current, band=0.01, bypass=True) == pytest.approx(target)

    # The governor's escape hatch: a sub-band de-risking sell must still fire,
    # otherwise the account stays more exposed than the governor intended.
    derisk = np.array([0.096, 0.10, 0.10])
    assert apply_no_trade_band(derisk, current, band=0.01)[0] == pytest.approx(0.10)
    assert apply_no_trade_band(derisk, current, band=0.01, bypass=True)[0] == pytest.approx(0.096)

    # A full exit is never banded, at any size. A 0.4% position targeted to
    # zero sits inside a 1% band, so banding it would refuse the sale every
    # day forever: too small to sell, and never bought back into a saleable
    # size. That is an absorbing state reached by a position merely being
    # small -- the same shape of failure the drawdown governor had before it
    # gained a re-entry rule.
    dust_held = np.array([0.004, 0.10])
    exit_all = np.array([0.0, 0.10])
    assert apply_no_trade_band(exit_all, dust_held, band=0.01)[0] == pytest.approx(0.0)
    # ...but a sub-band *trim* of the same position is still suppressed, so the
    # exemption is for closing a position, not a licence to ignore the band.
    trim = np.array([0.002, 0.10])
    assert apply_no_trade_band(trim, dust_held, band=0.01)[0] == pytest.approx(0.004)

    # Inputs are never mutated in place.
    assert target[0] == pytest.approx(0.105)
    assert current[0] == pytest.approx(0.10)


def test_degenerate_covariance_never_raises_for_either_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loadings, sigma = _rank_two_covariance()
    assert np.linalg.matrix_rank(sigma) == 2
    assert float(np.linalg.eigvalsh(sigma).min()) < 1e-12

    mu = np.full(7, 0.05)
    symbols = tuple(f"S{index}" for index in range(7))

    def _check(result: OptimizerResult) -> None:
        assert isinstance(result, OptimizerResult)
        assert result.status in {"clarabel", "scs", "fallback_inverse_vol"}
        assert result.weights.shape == (7,)
        assert np.isfinite(result.weights).all()
        assert (result.weights >= 0.0).all()
        assert result.weights.max() <= 0.15 + 1e-9
        assert result.weights.sum() <= 1.0 + 1e-9

    sharpe = optimize(mu, sigma, np.zeros(7), symbols)
    _check(sharpe)
    assert sharpe.status == "clarabel"

    rng = np.random.default_rng(7)
    scenarios = (rng.normal(size=(250, 2)) @ loadings.T) * 0.1 + 0.004
    cvar = optimize(
        mu, sigma, np.zeros(7), symbols, objective="min_cvar", scenario_returns=scenarios
    )
    _check(cvar)
    assert cvar.status == "clarabel"

    # A name whose own variance is unmeasurable is held flat, not repaired to
    # zero risk -- otherwise a NaN would earn the largest position in the book.
    unmeasurable_sigma = sigma.copy()
    unmeasurable_sigma[3, 3] = np.nan
    unmeasurable = optimize(mu, unmeasurable_sigma, np.zeros(7), symbols)
    _check(unmeasurable)
    assert unmeasurable.weights[3] == 0.0
    assert unmeasurable.weights.sum() > 0.0

    # Prove the second rung of the escalation actually solves the same problem.
    monkeypatch.setattr(opt, "_SOLVER_ORDER", ("SCS",))
    scs = optimize(mu, sigma, np.zeros(7), symbols)
    _check(scs)
    assert scs.status == "scs"
    assert scs.weights == pytest.approx(sharpe.weights, abs=1e-4)


def test_fallback_is_deterministic_and_never_raises() -> None:
    sigma = np.diag([0.04, 0.01, 0.09, 0.0])
    mu = np.array([0.05, 0.02, -0.01, 0.10])

    first = fallback_weights(sigma, mu, max_gross=1.0, max_position=0.5)
    second = fallback_weights(sigma, mu, max_gross=1.0, max_position=0.5)
    assert first.tobytes() == second.tobytes()

    assert first[2] == 0.0  # negative expected return never participates
    assert first[3] == 0.0  # a zero-vol name cannot be sized by inverse vol
    assert first[1] > first[0]  # lower vol earns more weight
    assert (first >= 0.0).all()
    assert first.max() <= 0.5
    assert first.sum() <= 1.0

    capped = fallback_weights(sigma, mu, max_gross=0.20, max_position=0.15)
    assert capped.max() <= 0.15
    assert capped.sum() <= 0.20 + 1e-12

    singular = np.full((4, 4), 0.02)
    assert np.isfinite(fallback_weights(singular, mu, max_gross=1.0, max_position=0.5)).all()

    all_cash = fallback_weights(sigma, -np.abs(mu), max_gross=1.0, max_position=0.5)
    assert all_cash.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_cvar_underweights_a_fat_left_tail() -> None:
    scenarios = _equal_mean_equal_variance_scenarios()
    means = scenarios.mean(axis=0)
    variances = scenarios.var(axis=0)
    assert means[0] == pytest.approx(means[1], abs=1e-12)
    assert variances[0] == pytest.approx(variances[1], rel=1e-9)

    # Variance genuinely cannot tell these two apart: fed the same scenarios'
    # moments, mean-variance holds them in equal size. Anything CVaR does
    # differently below is the tail, not the second moment.
    mean_variance, _, _ = solve_max_sharpe(
        means,
        np.cov(scenarios.T, ddof=0),
        np.zeros(2),
        risk_aversion=2.5,
        turnover_penalty=0.0,
        max_gross=1.0,
        max_position=0.15,
    )
    assert mean_variance is not None
    assert mean_variance[0] > 0.0
    assert mean_variance[1] == pytest.approx(mean_variance[0], abs=1e-6)

    weights, status, value = solve_min_cvar(
        scenarios,
        np.zeros(2),
        alpha=0.95,
        turnover_penalty=0.0,
        max_gross=1.0,
        max_position=0.15,
    )
    assert weights is not None
    assert status == "clarabel"
    assert value is not None
    assert weights[0] > 0.0
    assert weights[1] < weights[0]
    assert weights[1] == pytest.approx(0.0, abs=1e-6)


def test_cvar_alpha_selects_how_deep_a_tail_is_priced() -> None:
    """A tighter tail level must actually change the book, not just the number.

    Index 0 takes frequent moderate pain, index 1 takes a rare catastrophe. At
    alpha=0.90 the 100-scenario tail dilutes the 10 crash days and the crash
    name looks fine; at alpha=0.99 the tail *is* the crash and it gets cut.
    """
    steady = np.concatenate([np.full(900, 0.05), np.full(100, -0.02)])
    crash = np.concatenate([np.full(990, 0.04), np.full(10, -0.30)])
    scenarios = np.column_stack([steady, crash])
    symbols = ("STEADY", "CRASH")
    mu = scenarios.mean(axis=0)
    sigma = np.cov(scenarios.T, ddof=0)

    def _solve(alpha: float) -> OptimizerResult:
        return optimize(
            mu,
            sigma,
            np.zeros(2),
            symbols,
            objective="min_cvar",
            scenario_returns=scenarios,
            alpha=alpha,
        )

    moderate = _solve(0.90)
    tail_averse = _solve(0.99)
    assert moderate.status == "clarabel"
    assert tail_averse.status == "clarabel"

    assert not np.allclose(moderate.weights, tail_averse.weights)
    assert tail_averse.weights[1] < moderate.weights[1]
    assert tail_averse.weights[1] == pytest.approx(0.0, abs=1e-6)
    # The name without the catastrophe is untouched -- the tail level cut the
    # crash risk specifically, it did not just de-risk the whole book.
    assert tail_averse.weights[0] == pytest.approx(moderate.weights[0], abs=1e-6)
    assert tail_averse.weights[0] > 0.0


def test_optimize_reports_the_path_that_produced_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mu = np.array([0.20, 0.10, -0.05])
    sigma = _sigma(np.array([0.20, 0.15, 0.30]), correlation=0.1)
    w_current = np.array([0.05, 0.0, 0.10])
    symbols = ("AAA", "BBB", "CCC")

    solved = optimize(mu, sigma, w_current, symbols)
    assert solved.status == "clarabel"
    assert solved.symbols == symbols
    assert solved.objective_value is not None
    assert set(solved.trade_reasons) == set(symbols)
    assert "open" in solved.trade_reasons["BBB"]
    assert "close" in solved.trade_reasons["CCC"]

    def _explode(*args: object, **kwargs: object) -> object:
        raise RuntimeError("solver exploded mid-replay")

    monkeypatch.setattr(opt, "solve_max_sharpe", _explode)
    degraded = optimize(mu, sigma, w_current, symbols)
    assert degraded.status == "fallback_inverse_vol"
    assert degraded.objective_value is None
    assert set(degraded.trade_reasons) == set(symbols)
    assert np.isfinite(degraded.weights).all()
    assert (degraded.weights >= 0.0).all()

    # A solver that merely returns no answer degrades the same visible way.
    monkeypatch.setattr(opt, "solve_max_sharpe", lambda *a, **k: (None, "failed", None))
    assert optimize(mu, sigma, w_current, symbols).status == "fallback_inverse_vol"

    monkeypatch.undo()
    monkeypatch.undo()

    # min_cvar without scenarios is a visible degradation, not a crash.
    missing = optimize(mu, sigma, w_current, symbols, objective="min_cvar")
    assert missing.status == "fallback_inverse_vol"
    assert np.isfinite(missing.weights).all()


# --------------------------------------------------------------------------- #
# max_beta -- enforced in-solve, and again as a fallback/tolerance backstop     #
# --------------------------------------------------------------------------- #
def test_max_beta_binds_inside_solve_max_sharpe() -> None:
    mu = np.full(4, 0.30)
    sigma = _sigma(np.full(4, 0.20))
    beta = np.full(4, 2.0)
    kwargs = {
        "risk_aversion": 2.5,
        "turnover_penalty": 0.0,
        "max_gross": 1.0,
        "max_position": 1.0,
    }

    unconstrained, status, _ = solve_max_sharpe(mu, sigma, np.zeros(4), **kwargs)
    assert status == "clarabel"
    assert unconstrained is not None
    assert float(unconstrained.sum()) > 0.5  # meaningfully invested absent a beta cap

    capped, status2, _ = solve_max_sharpe(mu, sigma, np.zeros(4), beta=beta, max_beta=0.6, **kwargs)
    assert status2 in ("clarabel", "scs")
    assert capped is not None
    assert float(beta @ capped) <= 0.6 + 1e-6
    assert float(capped.sum()) < float(unconstrained.sum())


def test_max_beta_binds_inside_solve_min_cvar() -> None:
    rng = np.random.default_rng(7)
    scenarios = 0.10 + 0.02 * rng.standard_normal((500, 4))
    beta = np.full(4, 2.0)
    kwargs = {"alpha": 0.95, "turnover_penalty": 0.0, "max_gross": 1.0, "max_position": 1.0}

    unconstrained, status, _ = solve_min_cvar(scenarios, np.zeros(4), **kwargs)
    assert status == "clarabel"
    assert unconstrained is not None
    assert float(unconstrained.sum()) > 0.5

    capped, status2, _ = solve_min_cvar(scenarios, np.zeros(4), beta=beta, max_beta=0.6, **kwargs)
    assert status2 in ("clarabel", "scs")
    assert capped is not None
    assert float(beta @ capped) <= 0.6 + 1e-6
    assert float(capped.sum()) < float(unconstrained.sum())


def test_optimize_threads_the_beta_cap_through_to_either_objective() -> None:
    mu = np.full(2, 0.30)
    sigma = _sigma(np.full(2, 0.20))
    beta = np.full(2, 2.0)
    symbols = ("AAA", "BBB")

    max_sharpe_result = optimize(
        mu, sigma, np.zeros(2), symbols, max_gross=1.0, max_position=1.0, beta=beta, max_beta=0.6
    )
    assert max_sharpe_result.status in ("clarabel", "scs")
    assert float(beta @ max_sharpe_result.weights) <= 0.6 + 1e-6

    rng = np.random.default_rng(11)
    scenarios = 0.10 + 0.02 * rng.standard_normal((300, 2))
    cvar_result = optimize(
        mu,
        sigma,
        np.zeros(2),
        symbols,
        objective="min_cvar",
        max_gross=1.0,
        max_position=1.0,
        scenario_returns=scenarios,
        beta=beta,
        max_beta=0.6,
    )
    assert cvar_result.status in ("clarabel", "scs")
    assert float(beta @ cvar_result.weights) <= 0.6 + 1e-6


def test_fallback_weights_respect_the_beta_cap_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """A degraded day (solver failure) must not be the day the beta cap is silently off."""
    mu = np.full(2, 0.30)
    sigma = _sigma(np.full(2, 0.20))
    beta = np.full(2, 2.0)

    monkeypatch.setattr(
        opt, "solve_max_sharpe", lambda *a, **k: (None, "no_solver_converged", None)
    )
    result = optimize(
        mu,
        sigma,
        np.zeros(2),
        ("AAA", "BBB"),
        max_gross=1.0,
        max_position=1.0,
        beta=beta,
        max_beta=0.6,
    )
    assert result.status == "fallback_inverse_vol"
    # inverse-vol alone would put both names at gross 1.0 (beta 2.0, over cap);
    # the fallback's own projection has to be the thing that pulls it back in.
    assert float(beta @ result.weights) <= 0.6 + 1e-6
    assert result.weights.sum() > 0.0  # not just zeroed outright


def test_omitting_beta_reproduces_prior_behaviour_exactly() -> None:
    mu = np.array([0.20, 0.10, -0.05])
    sigma = _sigma(np.array([0.20, 0.15, 0.30]), correlation=0.1)
    symbols = ("AAA", "BBB", "CCC")

    implicit = optimize(mu, sigma, np.zeros(3), symbols)
    explicit_none = optimize(mu, sigma, np.zeros(3), symbols, beta=None, max_beta=None)
    assert np.array_equal(implicit.weights, explicit_none.weights)
    assert implicit.status == explicit_none.status


# --------------------------------------------------------------------------- #
# _project's beta clamp in isolation                                          #
# --------------------------------------------------------------------------- #
def test_project_beta_clamp_scales_uniformly_to_the_cap() -> None:
    weights = np.array([0.5, 0.5])
    upper = np.array([1.0, 1.0])
    beta = np.array([2.0, 2.0])
    projected = opt._project(
        weights, upper=upper, max_gross=1.0, groups=(), max_cluster=1.0, beta=beta, max_beta=0.6
    )
    assert float(beta @ projected) == pytest.approx(0.6, abs=1e-9)
    assert projected[0] == pytest.approx(projected[1])


def test_project_beta_clamp_is_a_noop_when_already_within_cap() -> None:
    weights = np.array([0.1, 0.1])
    upper = np.array([1.0, 1.0])
    beta = np.array([2.0, 2.0])
    projected = opt._project(
        weights, upper=upper, max_gross=1.0, groups=(), max_cluster=1.0, beta=beta, max_beta=0.6
    )
    assert np.array_equal(projected, np.array([0.1, 0.1]))


def test_project_beta_clamp_handles_a_non_positive_cap_without_raising() -> None:
    weights = np.array([0.5, 0.5])
    upper = np.array([1.0, 1.0])
    beta = np.array([2.0, 2.0])
    projected = opt._project(
        weights, upper=upper, max_gross=1.0, groups=(), max_cluster=1.0, beta=beta, max_beta=0.0
    )
    assert np.allclose(projected, 0.0)
