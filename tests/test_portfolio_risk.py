"""Risk primitives: covariance honesty, point-in-time clustering, hard limits.

Offline, no DB, fixed seeds. Every random panel is generated from a stated seed
so a failure is reproducible rather than a coin flip.
"""

from datetime import date
from itertools import pairwise

import numpy as np
import pytest

from market_intelligence.portfolio import account, feed, risk
from market_intelligence.portfolio.risk import (
    UnionFind,
    blend_cov,
    commercial_clusters,
    constant_correlation_target,
    decay_high_water_mark,
    drawdown_governor,
    ewma_cov,
    graph_block_target,
    ledoit_wolf,
    linked_pair_correlation_study,
    nearest_psd,
    risk_contributions,
    targeting_cov,
    vol_target_scale,
)
from market_intelligence.schemas.edges import EdgeType

_FROBENIUS = "fro"


def _linked_panel(
    *, n_pairs: int, n_obs: int, loading: float, seed: int
) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Two symbols per cluster; ``loading`` sets how hard each pair co-moves.

    ``loading=0`` gives independent columns whose clusters are pure labels --
    the null panel where the study must NOT report significance.
    """
    rng = np.random.default_rng(seed)
    symbols: list[str] = []
    clusters: dict[str, tuple[str, ...]] = {}
    columns: list[np.ndarray] = []
    for index in range(n_pairs):
        left, right = f"L{index:02d}", f"R{index:02d}"
        symbols += [left, right]
        clusters[left] = (left, right)
        common = rng.normal(size=n_obs)
        for _ in range(2):
            columns.append(loading * common + rng.normal(size=n_obs))
    return np.column_stack(columns) * 0.01, tuple(symbols), clusters


def test_ewma_cov_weights_the_most_recent_observation_heaviest():
    returns = np.array([[0.01, -0.02], [0.02, 0.00], [0.03, 0.04]])
    covariance = ewma_cov(returns, lam=0.5)

    # Hand computation. Raw weights are 0.5**2, 0.5**1, 0.5**0 = 0.25, 0.5, 1.0,
    # normalized to 1/7, 2/7, 4/7 -- the newest row carries four times the
    # oldest. Reliability correction is 1 / (1 - sum(p^2)) = 1 / (1 - 3/7) = 7/4.
    p = (1 / 7, 2 / 7, 4 / 7)
    correction = 1.0 / (1.0 - sum(weight * weight for weight in p))
    assert correction == pytest.approx(7 / 4)
    mean = [sum(p[t] * returns[t][k] for t in range(3)) for k in range(2)]
    expected = [
        [
            correction
            * sum(p[t] * (returns[t][i] - mean[i]) * (returns[t][j] - mean[j]) for t in range(3))
            for j in range(2)
        ]
        for i in range(2)
    ]
    assert covariance == pytest.approx(np.array(expected))

    # The weighting must be *recency*, not just any decay: nudging the newest
    # observation has to move the estimate further than nudging the oldest.
    newest, oldest = returns.copy(), returns.copy()
    newest[2, 0] += 0.05
    oldest[0, 0] += 0.05
    base = covariance[0, 0]
    assert abs(ewma_cov(newest, lam=0.5)[0, 0] - base) > abs(
        ewma_cov(oldest, lam=0.5)[0, 0] - base
    )


def test_ledoit_wolf_delta_is_a_valid_intensity_and_moves_toward_the_target():
    rng = np.random.default_rng(11)
    panels = [
        rng.normal(scale=0.01, size=(250, 8)),
        rng.normal(scale=0.02, size=(60, 20)),
        rng.normal(scale=0.01, size=(5, 12)),  # T < N: the near-singular case
    ]
    for panel in panels:
        target = constant_correlation_target(panel)
        sigma, delta = ledoit_wolf(panel, target)

        assert 0.0 <= delta <= 1.0
        assert delta > 0.0, "a finite sample always carries estimation error to shrink away"
        assert np.array_equal(sigma, sigma.T)

        sample = np.cov(panel, rowvar=False, ddof=1)
        assert np.linalg.norm(sample - target, _FROBENIUS) > 0.0
        assert np.linalg.norm(sigma - target, _FROBENIUS) < np.linalg.norm(
            sample - target, _FROBENIUS
        )

    # blend_cov's ``lw_weight`` rides on the Ledoit-Wolf side (PortfolioConfig
    # .lw_blend), so 1.0 is all-shrunk-sample and 0.0 is all-EWMA.
    panel = panels[0]
    shrunk, _ = ledoit_wolf(panel, constant_correlation_target(panel))
    fast = ewma_cov(panel, lam=0.97)
    assert blend_cov(shrunk, fast, lw_weight=1.0) == pytest.approx(shrunk)
    assert blend_cov(shrunk, fast, lw_weight=0.0) == pytest.approx(fast)
    assert blend_cov(shrunk, fast, lw_weight=0.25) == pytest.approx(0.25 * shrunk + 0.75 * fast)


def test_constant_correlation_target_keeps_variances_and_average_correlation():
    rng = np.random.default_rng(5)
    common = rng.normal(size=(180, 1))
    panel = (common @ np.array([[0.9, 0.5, 0.1, 0.7]]) + rng.normal(size=(180, 4))) * 0.01

    target = constant_correlation_target(panel)
    sample = np.cov(panel, rowvar=False, ddof=1)
    std = np.sqrt(np.diag(sample))
    correlation = sample / np.outer(std, std)
    off_diagonal = ~np.eye(4, dtype=bool)
    average = correlation[off_diagonal].mean()

    assert np.diag(target) == pytest.approx(np.diag(sample), rel=1e-12)
    implied = (target / np.outer(std, std))[off_diagonal]
    assert implied == pytest.approx(np.full(implied.shape, average))
    assert 0.0 < average < 1.0, "the fixture must actually have correlated names"


def test_graph_block_target_degrades_to_constant_correlation_without_edges():
    panel, symbols, clusters = _linked_panel(n_pairs=3, n_obs=200, loading=1.5, seed=2)
    singletons = {symbol: (symbol,) for symbol in symbols}

    # Every symbol its own singleton -- the expected early-year case, not an error.
    assert np.array_equal(
        graph_block_target(panel, symbols, singletons), constant_correlation_target(panel)
    )
    # No cluster mapping at all must behave identically (fail-closed to singletons).
    assert np.array_equal(
        graph_block_target(panel, symbols, {}), constant_correlation_target(panel)
    )
    # ...but with real blocks the target must actually differ, or the graceful
    # degradation above is indistinguishable from a function that ignores edges.
    blocked = graph_block_target(panel, symbols, clusters)
    assert not np.allclose(blocked, constant_correlation_target(panel))
    assert np.diag(blocked) == pytest.approx(np.diag(np.cov(panel, rowvar=False, ddof=1)))


def test_nearest_psd_floors_eigenvalues_of_a_rank_two_covariance():
    rng = np.random.default_rng(3)
    panel = rng.normal(size=(200, 2)) @ rng.normal(size=(2, 7)) * 0.01
    sample = np.cov(panel, rowvar=False, ddof=1)
    assert np.linalg.matrix_rank(sample) == 2
    assert np.linalg.eigvalsh(sample).min() < 0.0, "fixture must be numerically indefinite"

    fixed = nearest_psd(sample)

    assert np.linalg.eigvalsh(fixed).min() >= 0.0
    assert np.array_equal(fixed, fixed.T)
    assert fixed == pytest.approx(sample, abs=1e-8)


def test_union_find_finds_exactly_the_hand_drawn_components():
    finder = UnionFind(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"))
    finder.union("BBB", "CCC")
    finder.union("AAA", "BBB")
    finder.union("EEE", "DDD")

    components = finder.components()

    assert components == {
        "AAA": ("AAA", "BBB", "CCC"),
        "DDD": ("DDD", "EEE"),
        "FFF": ("FFF",),
    }
    members = [member for group in components.values() for member in group]
    assert sorted(members) == ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    assert len(members) == len(set(members)) == 6
    assert finder.find("CCC") == finder.find("AAA") == "AAA"
    assert finder.find("EEE") != finder.find("AAA")
    with pytest.raises(KeyError):
        finder.find("ZZZ")


def test_clusters_merge_commercial_edges_but_never_interlocks():
    symbols = ("AAA", "BBB", "CCC", "DDD")
    reported = date(2020, 1, 1)
    as_of = date(2021, 1, 1)

    linked = commercial_clusters(
        symbols,
        (
            ("AAA", "BBB", "interlock", reported),
            ("AAA", "BBB", "shared_board_member", reported),
            ("CCC", "DDD", "customer", reported),
        ),
        as_of=as_of,
    )
    assert linked == {"AAA": ("AAA",), "BBB": ("BBB",), "CCC": ("CCC", "DDD")}

    for edge_type in ("customer", "supplier", "competitor", "partner"):
        merged = commercial_clusters(
            symbols, (("AAA", "BBB", edge_type, reported),), as_of=as_of
        )
        assert merged["AAA"] == ("AAA", "BBB"), edge_type

    # An edge naming an unknown symbol cannot carry cluster exposure.
    assert commercial_clusters(
        ("AAA", "BBB"), (("AAA", "ZZZ", "customer", reported),), as_of=as_of
    ) == {"AAA": ("AAA",), "BBB": ("BBB",)}

    # Vocabulary drift guard: the commercial set is exactly the extractor's enum.
    #
    # The list is spelled out in three places on purpose. ``schemas.edges`` is
    # canonical, but ``risk`` and ``feed`` each keep a literal copy because the
    # package layers strictly downward -- feed sits below risk and cannot import
    # it, and risk stays numpy-only so it imports no schema module either. The
    # duplication is structurally forced; this assertion is what keeps it from
    # becoming a divergence. Both copies are checked here rather than one being
    # left to a separate test, because a guard that covers only one copy is a
    # guard that reports success while the other silently rots.
    canonical = {member.value for member in EdgeType}
    assert canonical == risk._COMMERCIAL_EDGE_TYPES
    assert canonical == set(feed.COMMERCIAL_EDGE_TYPES)


def test_clusters_ignore_edges_reported_after_as_of():
    symbols = ("AAA", "BBB", "CCC", "DDD")
    edges = (
        ("AAA", "BBB", "customer", date(2016, 6, 30)),
        ("CCC", "DDD", "supplier", date(2026, 1, 1)),
    )

    # Today's graph must not explain 2016's returns.
    assert commercial_clusters(symbols, edges, as_of=date(2016, 12, 31)) == {
        "AAA": ("AAA", "BBB"),
        "CCC": ("CCC",),
        "DDD": ("DDD",),
    }
    # The threshold is inclusive: an edge reported exactly on as_of is knowable.
    assert commercial_clusters(symbols, edges, as_of=date(2026, 1, 1)) == {
        "AAA": ("AAA", "BBB"),
        "CCC": ("CCC", "DDD"),
    }
    # One day earlier and it is not.
    assert commercial_clusters(symbols, edges, as_of=date(2025, 12, 31))["CCC"] == ("CCC",)


def test_drawdown_governor_boundaries_are_inclusive():
    assert drawdown_governor(np.array([100.0, 95.0])) == 1.0
    assert drawdown_governor(np.array([100.0, 90.01])) == 1.0
    assert drawdown_governor(np.array([100.0, 90.0])) == 0.5  # exactly 10%
    assert drawdown_governor(np.array([100.0, 89.99])) == 0.5
    assert drawdown_governor(np.array([100.0, 85.01])) == 0.5
    assert drawdown_governor(np.array([100.0, 85.0])) == 0.0  # exactly 15%
    assert drawdown_governor(np.array([100.0, 80.0])) == 0.0

    # Drawdown is measured from the running peak, not the first observation.
    assert drawdown_governor(np.array([100.0, 120.0])) == 1.0
    assert drawdown_governor(np.array([100.0, 120.0, 108.0])) == 0.5
    assert drawdown_governor(np.array([100.0])) == 1.0
    assert drawdown_governor(np.array([100.0, 90.0]), halve_scale=0.25) == 0.25
    with pytest.raises(ValueError):
        drawdown_governor(np.array([]))
    with pytest.raises(ValueError):
        drawdown_governor(np.array([100.0, np.nan]))


# --------------------------------------------------------------------------- #
# the high-water mark's decay -- the governor's missing re-entry rule           #
# --------------------------------------------------------------------------- #
def test_decay_high_water_mark_is_the_same_object_the_account_marks_with():
    """One implementation, not two.

    ``account`` maintains the mark and ``risk`` publishes the rule; the layering
    forbids ``account`` importing ``risk``, so the function is defined in
    ``account`` and re-exported here. This pins that decision: a future edit that
    forks a second copy into ``risk`` would leave these tests exercising code the
    simulator does not run.
    """
    assert risk.decay_high_water_mark is account.decay_high_water_mark


def test_a_new_high_ratchets_the_mark_up_immediately_without_decaying():
    """Above the mark there is no drawdown to forget, so nothing decays."""
    assert decay_high_water_mark(100.0, 120.0, halflife_days=63) == 120.0
    assert decay_high_water_mark(100.0, 100.0, halflife_days=63) == 100.0
    # Ratcheting is unconditional on the half-life: a new high is a new high.
    assert decay_high_water_mark(100.0, 120.0, halflife_days=1, periods=1_000) == 120.0


def test_exactly_one_halflife_closes_exactly_half_the_gap():
    """Hand-computed. Mark 100, equity 80: the gap is 20, so one half-life is 90."""
    assert decay_high_water_mark(100.0, 80.0, halflife_days=10, periods=10) == pytest.approx(90.0)
    assert decay_high_water_mark(100.0, 80.0, halflife_days=1, periods=1) == pytest.approx(90.0)
    # Two half-lives leave a quarter of the gap: 80 + 5 = 85.
    assert decay_high_water_mark(100.0, 80.0, halflife_days=10, periods=20) == pytest.approx(85.0)
    # Half a half-life leaves 20 / sqrt(2) of it.
    assert decay_high_water_mark(100.0, 80.0, halflife_days=10, periods=5) == pytest.approx(
        80.0 + 20.0 / np.sqrt(2.0)
    )

    # The simulator applies one period a day, so iterating must equal one call
    # for the whole span -- otherwise a daily loop and a docstring disagree.
    iterated = 100.0
    for _ in range(10):
        iterated = decay_high_water_mark(iterated, 80.0, halflife_days=10)
    assert iterated == pytest.approx(90.0, abs=1e-12)


def test_the_mark_never_decays_below_current_equity():
    """The mark is a *peak*. Below equity it would report a negative drawdown."""
    for periods in (1, 10, 1_000, 10_000_000):
        decayed = decay_high_water_mark(100.0, 80.0, halflife_days=1, periods=periods)
        assert decayed >= 80.0
    # In the limit it converges on equity exactly rather than undershooting it.
    assert decay_high_water_mark(100.0, 80.0, halflife_days=1, periods=10_000_000) == 80.0
    # A zero-equity account is the degenerate case and must not go negative.
    assert decay_high_water_mark(100.0, 0.0, halflife_days=1, periods=10_000_000) == 0.0

    with pytest.raises(ValueError):
        decay_high_water_mark(100.0, 80.0, halflife_days=0)
    with pytest.raises(ValueError):
        decay_high_water_mark(100.0, 80.0, halflife_days=-5)
    with pytest.raises(ValueError):
        decay_high_water_mark(100.0, 80.0, halflife_days=10, periods=-1)


def test_a_longer_halflife_forgets_strictly_more_slowly():
    """Monotone in ``halflife_days`` -- the knob has to mean what it says.

    ``periods=0`` is the identity, which is the boundary the monotone sequence
    approaches: an infinitely long half-life forgets nothing.
    """
    marks = [
        decay_high_water_mark(100.0, 60.0, halflife_days=halflife)
        for halflife in (1, 2, 5, 10, 21, 63, 126, 252)
    ]
    assert all(later > earlier for earlier, later in pairwise(marks))
    assert marks[0] == pytest.approx(80.0)  # a one-day half-life halves it in a day
    assert marks[-1] < 100.0  # but even 252 days forgets something
    assert decay_high_water_mark(100.0, 60.0, halflife_days=63, periods=0) == 100.0


def test_vol_target_scale_hits_target_and_never_levers_up():
    weights = np.array([0.5, 0.5])
    annual = np.array([[0.04, 0.005], [0.005, 0.04]])
    # w'Sw = 0.25*0.04 + 2*0.25*0.005 + 0.25*0.04 = 0.0225 -> vol 0.15 a year.
    assert float(weights @ annual @ weights) == pytest.approx(0.0225)
    assert vol_target_scale(
        weights, annual, target_vol=0.12, periods_per_year=1
    ) == pytest.approx(0.12 / 0.15)

    # The real caller hands in a *daily* covariance. w'Sw = 1e-4 -> 1% a day,
    # which annualizes to 0.01*sqrt(252) = 15.9% against a 12% target, so the
    # governor must actually bind.
    daily = np.array([[1.8e-4, 2.0e-5], [2.0e-5, 1.8e-4]])
    assert float(weights @ daily @ weights) == pytest.approx(1e-4)
    bound = vol_target_scale(weights, daily, target_vol=0.12)
    assert bound == pytest.approx(0.12 / (0.01 * np.sqrt(252)))
    assert bound < 1.0
    # ...and without the annualization it would not: 1% < 12% reads as calm, the
    # scale pins to 1.0, and vol targeting is off while the config says 12%.
    assert vol_target_scale(weights, daily, target_vol=0.12, periods_per_year=1) == 1.0

    # Predicted vol far below target: the residual stays in cash, never leverage.
    assert vol_target_scale(weights, annual, target_vol=0.50, periods_per_year=1) == 1.0
    assert vol_target_scale(weights, annual, target_vol=10.0, periods_per_year=1) == 1.0
    assert vol_target_scale(np.zeros(2), annual, target_vol=0.12) == 1.0
    assert vol_target_scale(weights, annual, target_vol=0.50, periods_per_year=1, cap=0.6) == 0.6
    with pytest.raises(ValueError):
        vol_target_scale(weights, annual, target_vol=0.12, periods_per_year=0)

    # Same two-asset arithmetic, the other half of "is the risk where I think
    # it is": 80/20 weights over equal variances put 94% of variance in one name.
    shares = risk_contributions(
        np.array([0.8, 0.2]), np.diag([0.04, 0.04]), ("AAA", "BBB"), limit=0.35
    )
    assert [item.symbol for item in shares] == ["AAA", "BBB"]
    assert shares[0].risk_share == pytest.approx(0.0256 / 0.0272)
    assert shares[1].risk_share == pytest.approx(0.0016 / 0.0272)
    assert sum(item.risk_share for item in shares) == pytest.approx(1.0)
    assert (shares[0].flagged, shares[1].flagged) == (True, False)


# --------------------------------------------------------------------------- #
# targeting_cov: the scaler's own estimate, and why it is not the blend         #
# --------------------------------------------------------------------------- #
_REGIME_CALM_OBS = 200
_REGIME_VIOLENT_OBS = 10


def _regime_panel(
    *,
    calm_obs: int = _REGIME_CALM_OBS,
    violent_obs: int = _REGIME_VIOLENT_OBS,
    calm_vol: float = 0.004,
    violent_vol: float = 0.04,
    n_symbols: int = 3,
    seed: int = 20260731,
) -> np.ndarray:
    """Calm for ``calm_obs`` sessions, then violent for ``violent_obs``.

    A tenfold jump in daily volatility -- 0.4%/day (6.3% a year) to 4%/day
    (63% a year) -- with the break at a known row, so "how fast did the
    estimate turn" is a question about the estimator and not about the panel.
    """
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.normal(0.0, calm_vol, size=(calm_obs, n_symbols)),
            rng.normal(0.0, violent_vol, size=(violent_obs, n_symbols)),
        ]
    )


def _optimizer_blend(block: np.ndarray) -> np.ndarray:
    """The covariance portfolio construction reads, spelled out rather than imported.

    ``lw_weight=0.5`` and ``lam=0.97`` are ``PortfolioConfig``'s shipped
    ``lw_blend`` / ``ewma_lambda``. Written here in full so this file states
    the thing it is comparing against instead of trusting a helper that the
    change under test could have quietly redefined.
    """
    shrunk, _delta = ledoit_wolf(block, constant_correlation_target(block))
    return blend_cov(shrunk, ewma_cov(block, lam=0.97), lw_weight=0.5)


def _implied_vol(covariance: np.ndarray, weights: np.ndarray) -> float:
    """Per-period portfolio volatility ``sqrt(w' Sigma w)``."""
    return float(np.sqrt(weights @ covariance @ weights))


def test_targeting_cov_turns_faster_than_the_blend_the_optimizer_reads():
    """The entire point of the split: the scaler's estimate must react sooner.

    Measured the way the simulator would feel it -- the *rise* in predicted
    portfolio volatility once the violent rows enter the window, on identical
    weights. The blend is not asked to be blind, only to be slower: it is
    asserted to move at all, so "fast rises more" is a comparison between two
    live estimates rather than one estimate against a corpse.

    A bare ``>`` would pass on a hair. The margin is asserted too, because the
    defect this closes was an estimate that moved but not enough to bind: on
    2020-03-06 the blend read 9.7% against a 12% target while the account was
    already down 11.5%.
    """
    panel = _regime_panel()
    calm_only = panel[:_REGIME_CALM_OBS]
    weights = np.full(panel.shape[1], 1.0 / panel.shape[1])

    fast_rise = _implied_vol(targeting_cov(panel), weights) - _implied_vol(
        targeting_cov(calm_only), weights
    )
    blend_rise = _implied_vol(_optimizer_blend(panel), weights) - _implied_vol(
        _optimizer_blend(calm_only), weights
    )

    assert blend_rise > 0.0  # the slow estimate is slow, not dead
    assert fast_rise > blend_rise
    # Measured 2.04x on this panel; 1.5 leaves room for platform noise while
    # still failing any change that quietly re-slows the scaler.
    assert fast_rise > 1.5 * blend_rise


def test_targeting_cov_is_symmetric_and_positive_semidefinite():
    """``vol_target_scale`` reads ``w' Sigma w`` and calls a non-positive
    variance "nothing to scale", so a negative eigenvalue would disarm the
    control rather than raise. Checked on the violent panel and on a rank-
    deficient one, where the floor is the only thing standing between eigh's
    rounding noise and a negative number."""
    for panel in (_regime_panel(), _regime_panel(n_symbols=5)[:, [0, 1, 0, 1, 0]]):
        covariance = targeting_cov(panel)
        assert covariance.shape == (panel.shape[1], panel.shape[1])
        assert np.array_equal(covariance, covariance.T)
        assert float(np.linalg.eigvalsh(covariance).min()) >= 0.0


def test_a_larger_lambda_reacts_strictly_more_slowly():
    """The parameter means what its name says: bigger lambda, longer memory.

    Monotone across the ladder, not just at the endpoints -- a decay that
    happened to order its extremes correctly while doing something else in the
    middle would not be a decay.
    """
    panel = _regime_panel()
    calm_only = panel[:_REGIME_CALM_OBS]
    weights = np.full(panel.shape[1], 1.0 / panel.shape[1])
    rises = [
        _implied_vol(targeting_cov(panel, lam=lam), weights)
        - _implied_vol(targeting_cov(calm_only, lam=lam), weights)
        for lam in (0.90, 0.94, 0.97, 0.99)
    ]

    assert all(later < earlier for earlier, later in pairwise(rises))
    assert all(rise > 0.0 for rise in rises)


def test_linked_pair_correlation_study_detects_planted_pairs_and_rejects_noise():
    planted, symbols, clusters = _linked_panel(n_pairs=6, n_obs=500, loading=2.0, seed=17)
    found = linked_pair_correlation_study(planted, symbols, clusters, n_bootstrap=400, seed=1)

    assert found.n_linked_pairs == 6
    assert found.n_random_pairs == 60  # C(12, 2) = 66, less the 6 linked
    assert found.linked_mean_corr > 0.7
    assert abs(found.random_mean_corr) < 0.1
    assert found.difference > 0.5
    assert found.ci_low > 0.0
    assert found.significant is True

    # Same cluster labels, independent returns: the graph explains nothing and
    # the study must say so. An implementation that only ever reports
    # significance is indistinguishable from a broken one.
    noise, noise_symbols, noise_clusters = _linked_panel(
        n_pairs=8, n_obs=600, loading=0.0, seed=23
    )
    absent = linked_pair_correlation_study(
        noise, noise_symbols, noise_clusters, n_bootstrap=400, seed=1
    )
    assert abs(absent.difference) < 0.1
    assert absent.ci_low < 0.0 < absent.ci_high
    assert absent.significant is False

    # No linked pairs at all (early years): no evidence is not evidence.
    empty = linked_pair_correlation_study(
        planted, symbols, {symbol: (symbol,) for symbol in symbols}, n_bootstrap=50, seed=1
    )
    assert empty.n_linked_pairs == 0
    assert empty.difference == 0.0
    assert empty.significant is False
