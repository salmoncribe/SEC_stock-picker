"""Covariance estimation, graph clustering, and the risk budget.

Three ideas, in dependency order:

1. **Estimate the covariance honestly.** A 252-day sample covariance over ~600
   names is near-singular and its extreme eigenvalues are mostly noise.
   Ledoit-Wolf shrinks it toward a structured target with an analytically
   optimal intensity; an EWMA blend then lets it respond to a regime change.
   Hand-rolled (~40 lines) so domain code carries no scikit-learn dependency.

2. **Use the graph as risk structure, not as a return forecast.** Two firms
   with a customer/supplier relationship co-move more than two random firms.
   That claim is *measurable* (see ``linked_pair_correlation_study``) and does
   not require any propagation coefficient to have been estimated -- which is
   fortunate, because none ever has. Board interlocks are excluded: a shared
   director is a channel, not a measured relationship.

3. **Split the risk so being wrong is survivable.** Position and cluster caps
   bound single-bet damage, volatility targeting bounds portfolio-level
   damage, and the drawdown governor is the circuit breaker beneath both.
   Volatility targeting reads its own covariance (``targeting_cov``) rather
   than the optimizer's: a stable estimate keeps portfolio construction from
   churning, and a stable estimate is exactly what a regime detector must not
   be. Measured, one estimate for both jobs left ``vol_scale`` binding on 1.5%
   of sessions and inert through all of COVID.

Point-in-time discipline: ``commercial_clusters`` admits an edge only if
``report_date <= as_of``. Today's graph must not explain 2016's returns. Early
years are sparse because extraction kept only the latest filing per pair; that
sparsity is conservative (fewer claimed relationships), and it is documented
rather than patched.

**Units are named, not assumed.** Everything here is horizon-agnostic -- feed
it a daily covariance and it returns daily numbers -- except
``vol_target_scale``, which has no choice but to mix a per-period covariance
with an annual target (``target_vol = 0.12``). Its ``periods_per_year`` makes
that conversion a checkable argument instead of a docstring the caller may not
read. The failure it closes is the quiet one: handed a daily covariance with no
annualization, the function returns 1.0 on essentially every day, so volatility
targeting is off while appearing to be on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

# Re-exported, not re-implemented. The drawdown governor's missing half -- the
# rule that lets a halt end -- belongs to whoever maintains the high-water mark,
# and that is ``account``. Defining it there and publishing it here keeps the
# package's strict downward layering intact (``risk`` may import ``account``;
# ``account`` may import nothing internal) while leaving exactly one
# implementation, so the function these tests exercise is the function the
# simulator runs. See ``portfolio/__init__.py`` for the layering rule.
from market_intelligence.portfolio.account import decay_high_water_mark

#: Edge types that assert a *measured commercial relationship* and may therefore
#: merge two names into one risk cluster. Mirrors ``schemas.edges.EdgeType`` as
#: literals so this layer stays numpy-only; ``tests/test_portfolio_risk.py``
#: asserts the two never drift apart. Board interlocks
#: (``"shared_board_member"``) are absent on purpose: a shared director is an
#: information channel, not a measured economic link.
_COMMERCIAL_EDGE_TYPES = frozenset({"competitor", "customer", "partner", "supplier"})


@dataclass(frozen=True)
class RiskLimits:
    """Hard caps. Fail-closed: an unknown cluster counts as its own cluster."""

    max_position: float = 0.15
    max_cluster: float = 0.30
    max_gross: float = 1.0
    target_vol: float = 0.12
    max_single_name_risk_share: float = 0.35


@dataclass(frozen=True)
class RiskContribution:
    """Per-name share of total portfolio variance, and whether it breached."""

    symbol: str
    weight: float
    risk_share: float
    flagged: bool


class UnionFind:
    """Connected components for graph clustering. ~30 lines, no networkx.

    A component's root is its lexicographically smallest member. That is a
    deliberate choice over union-by-rank: the cluster id must not depend on the
    order edges happened to arrive in, because ``max_cluster`` is applied per
    root and a replay that renamed its clusters day to day would be applying a
    different constraint every day while appearing to apply the same one.
    """

    def __init__(self, items: tuple[str, ...]) -> None:
        self._parent: dict[str, str] = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self._parent
        if item not in parent:
            raise KeyError(f"unknown item: {item!r}")
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:  # path compression, iterative
            parent[item], item = root, parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        # Attach the larger root under the smaller so the survivor is always the
        # component's lexicographic minimum.
        if root_a < root_b:
            self._parent[root_b] = root_a
        else:
            self._parent[root_a] = root_b

    def components(self) -> dict[str, tuple[str, ...]]:
        """Root -> members. Every item appears exactly once."""
        groups: dict[str, list[str]] = {}
        for item in tuple(self._parent):
            groups.setdefault(self.find(item), []).append(item)
        return {root: tuple(sorted(members)) for root, members in sorted(groups.items())}


def _panel(returns: np.ndarray, *, minimum: int = 2) -> np.ndarray:
    """Validate and normalize a ``[T, N]`` returns block to float64."""
    data = np.asarray(returns, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"returns must be [T, N], got shape {data.shape}")
    if data.shape[0] < minimum:
        raise ValueError(f"returns needs at least {minimum} observations, got {data.shape[0]}")
    if data.shape[1] == 0:
        raise ValueError("returns needs at least one symbol")
    if not np.isfinite(data).all():
        raise ValueError("returns contains non-finite values; fill or drop them upstream")
    return data


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Force exact symmetry. Cheap, and eigh/optimizer both assume it."""
    return (matrix + matrix.T) / 2.0


def _sample_cov(data: np.ndarray) -> np.ndarray:
    """Unbiased (ddof=1) sample covariance of a validated panel."""
    centred = data - data.mean(axis=0)
    return _symmetrize(centred.T @ centred / (data.shape[0] - 1))


def _std_and_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviations and the correlation matrix implied by ``cov``.

    A zero-variance column (a halted name) yields zero correlations rather than
    a divide-by-zero: it contributes no covariance to anything, so calling its
    correlation zero is the only statement the data supports.
    """
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    safe = np.where(std > 0.0, std, 1.0)
    corr = cov / np.outer(safe, safe)
    np.fill_diagonal(corr, np.where(std > 0.0, 1.0, 0.0))
    return std, np.clip(corr, -1.0, 1.0)


def _cluster_index(symbols: tuple[str, ...], clusters: dict[str, tuple[str, ...]]) -> np.ndarray:
    """Integer cluster id per symbol. Unknown symbols become their own cluster.

    Fail-closed in the direction that matters: an unclustered name is treated as
    a singleton, which *tightens* nothing but also never lets an unmapped symbol
    silently share another name's cluster budget.
    """
    label: dict[str, str] = {}
    for root, members in clusters.items():
        for member in members:
            if member in label and label[member] != root:
                raise ValueError(f"symbol {member!r} appears in more than one cluster")
            label[member] = root
    ids: dict[str, int] = {}
    out = np.empty(len(symbols), dtype=np.int64)
    for position, symbol in enumerate(symbols):
        key = label.get(symbol, f"\x00singleton\x00{symbol}")
        out[position] = ids.setdefault(key, len(ids))
    return out


def ewma_cov(returns: np.ndarray, *, lam: float = 0.97) -> np.ndarray:
    """Exponentially weighted covariance, most recent row weighted heaviest.

    ``returns`` is ``[T, N]``. Weights are normalized so the estimator is
    unbiased in scale regardless of ``T``: after normalization the reliability
    weights ``p`` carry a ``1 / (1 - sum(p^2))`` correction (numpy's
    ``aweights``/``ddof=1`` rule). That denominator depends on the *shape* of
    the decay, not on the number of rows, so a 252-day and a 500-day window
    produce comparably scaled estimates instead of the longer one looking
    calmer purely because it averaged more zeros in.
    """
    data = _panel(returns)
    if not 0.0 < lam <= 1.0:
        raise ValueError(f"lam must be in (0, 1], got {lam}")
    age = np.arange(data.shape[0] - 1, -1, -1, dtype=np.float64)
    weights = lam**age
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(f"lam={lam} underflows every weight to zero")
    weights = weights / total
    centred = data - weights @ data
    correction = 1.0 - float(weights @ weights)
    if correction <= 0.0:
        raise ValueError(f"lam={lam} concentrates all weight on one row; covariance undefined")
    return _symmetrize((centred * weights[:, None]).T @ centred / correction)


def constant_correlation_target(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf's structured target: sample variances, average correlation.

    Variances are kept exactly as sampled -- they are estimated from ``T``
    observations each and are the part of the sample covariance that is *not*
    mostly noise. Only the ``N(N-1)/2`` correlations, which are, get replaced by
    their common average.
    """
    return _constant_correlation_from(_sample_cov(_panel(returns)))


def _constant_correlation_from(sample: np.ndarray) -> np.ndarray:
    std, corr = _std_and_corr(sample)
    size = sample.shape[0]
    off_diagonal = ~np.eye(size, dtype=bool)
    mean_corr = float(corr[off_diagonal].mean()) if size > 1 else 0.0
    target = mean_corr * np.outer(std, std)
    np.fill_diagonal(target, np.diag(sample))
    return _symmetrize(target)


def graph_block_target(
    returns: np.ndarray, symbols: tuple[str, ...], clusters: dict[str, tuple[str, ...]]
) -> np.ndarray:
    """Block-structured target: within-cluster correlation differs from across.

    Must degrade gracefully to ``constant_correlation_target`` when no edges
    exist -- a universe with no measured relationships is the expected case for
    early years, not an error. Tested explicitly.

    The block structure is two numbers, not a free block matrix: one average
    correlation for pairs inside a cluster and one for pairs across clusters.
    Estimating a separate correlation per cluster would reintroduce exactly the
    small-sample noise the target exists to remove.
    """
    data = _panel(returns)
    if data.shape[1] != len(symbols):
        raise ValueError(f"returns has {data.shape[1]} columns but {len(symbols)} symbols")
    cluster_of = _cluster_index(symbols, clusters)
    size = len(symbols)
    off_diagonal = ~np.eye(size, dtype=bool)
    same_cluster = cluster_of[:, None] == cluster_of[None, :]
    within = same_cluster & off_diagonal
    if not within.any():
        # Every name is a singleton: there are no blocks, and the block target
        # *is* the constant-correlation target. Returning it verbatim keeps the
        # degenerate case bit-identical rather than merely close.
        return constant_correlation_target(returns)

    sample = _sample_cov(data)
    std, corr = _std_and_corr(sample)
    across = off_diagonal & ~same_cluster
    within_corr = float(corr[within].mean())
    across_corr = float(corr[across].mean()) if across.any() else within_corr
    target = np.where(same_cluster, within_corr, across_corr) * np.outer(std, std)
    np.fill_diagonal(target, np.diag(sample))
    return _symmetrize(target)


def ledoit_wolf(returns: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed-form shrinkage toward ``target``. Returns ``(sigma, delta)``.

    ``delta`` must land in ``[0, 1]``; clipping it is correct, silently
    exceeding it means the derivation is wrong.

    The intensity is Ledoit & Wolf's (2003) ``delta* = (pi - rho) / (T * gamma)``
    with the ``rho`` term dropped -- the Schaefer & Strimmer (2005) general-target
    form. ``pi`` is the summed asymptotic variance of the sample covariance
    entries and ``gamma = ||F - S||_F^2`` is the target's misspecification.
    ``rho`` (the covariance between the target's entries and the sample's) is
    target-specific and cannot be written down for an arbitrary ``target``
    argument; dropping it *overstates* ``delta`` slightly, i.e. errs toward the
    structured estimate. That direction is the safe one for an optimizer, and
    naming it here is better than a formula that only looks exact.
    """
    data = _panel(returns)
    n_obs, n_assets = data.shape
    target_matrix = np.asarray(target, dtype=np.float64)
    if target_matrix.shape != (n_assets, n_assets):
        raise ValueError(f"target must be ({n_assets}, {n_assets}), got {target_matrix.shape}")
    if not np.isfinite(target_matrix).all():
        raise ValueError("target contains non-finite values")

    centred = data - data.mean(axis=0)
    sample = _symmetrize(centred.T @ centred / (n_obs - 1))
    # pi_ij = (1/T) sum_t (x_ti*x_tj - s_ij)^2, expanded so it is three matrix
    # products rather than a loop over T*N^2 terms.
    second = centred.T @ centred / n_obs
    fourth = (centred**2).T @ (centred**2) / n_obs
    pi_hat = float((fourth - 2.0 * sample * second + sample**2).sum())
    gamma = float(((target_matrix - sample) ** 2).sum())
    raw = pi_hat / (n_obs * gamma) if gamma > 0.0 else 0.0
    delta = float(np.clip(raw, 0.0, 1.0))
    sigma = _symmetrize(delta * target_matrix + (1.0 - delta) * sample)
    return sigma, delta


def blend_cov(sample_cov: np.ndarray, ewma: np.ndarray, *, lw_weight: float) -> np.ndarray:
    """Convex blend of the shrunk sample estimate and the EWMA estimate.

    Result is ``lw_weight * sample_cov + (1 - lw_weight) * ewma``: the weight
    rides on the **Ledoit-Wolf** side, matching ``PortfolioConfig.lw_blend``
    whose default 0.5 the settings file documents as "Ledoit-Wolf shrunk sample
    blended 50/50 with EWMA".

    The parameter is named for its side rather than called ``weight`` because
    the two arrays are the same shape and dtype, so a flipped blend raises
    nothing and returns a plausible covariance -- an inversion that would show
    up only as a portfolio that reacts to regime changes at the wrong speed.
    """
    if not 0.0 <= lw_weight <= 1.0:
        raise ValueError(f"lw_weight must be in [0, 1], got {lw_weight}")
    first = np.asarray(sample_cov, dtype=np.float64)
    second = np.asarray(ewma, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"shape mismatch: {first.shape} vs {second.shape}")
    return _symmetrize(lw_weight * first + (1.0 - lw_weight) * second)


def nearest_psd(matrix: np.ndarray, *, ridge: float = 1e-10) -> np.ndarray:
    """Eigenvalue clip + ridge. The optimizer must never see a negative eigenvalue.

    Verified 2026-07-31 that both Clarabel and SCS survive a rank-2 covariance
    over 7 names without this, so it is a guarantee rather than a crutch --
    but a guarantee the replay depends on, since it cannot stop to debug.

    The ridge is what makes the guarantee hold *after* reconstruction:
    ``V diag(clip(w)) V'`` reintroduces rounding noise of order 1e-18, so
    clipping alone leaves the output free to be a hair negative again. A 1e-10
    floor sits far above that noise and far below any covariance this platform
    produces (daily variances are ~1e-4).
    """
    data = np.asarray(matrix, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f"matrix must be square, got shape {data.shape}")
    if ridge < 0.0:
        raise ValueError(f"ridge must be non-negative, got {ridge}")
    if not np.isfinite(data).all():
        raise ValueError("matrix contains non-finite values")
    eigenvalues, vectors = np.linalg.eigh(_symmetrize(data))
    rebuilt = _symmetrize((vectors * np.clip(eigenvalues, 0.0, None)) @ vectors.T)
    return rebuilt + ridge * np.eye(data.shape[0])


def targeting_cov(returns: np.ndarray, *, lam: float = 0.94) -> np.ndarray:
    """Fast EWMA covariance for :func:`vol_target_scale`, and for nothing else.

    The optimizer and the volatility scaler want opposite things from a
    covariance, and for a long time they were handed the same one. The
    optimizer wants it **stable**: it re-solves every rebalance, and an estimate
    that jumps re-sorts the book, which costs turnover the annual cap already
    struggles to stay inside. So portfolio construction keeps reading the
    blended ``lw_weight * ledoit_wolf(252-day sample) + (1 - lw_weight) *
    ewma_cov(0.97)`` estimate, and this function must never be substituted
    there. The scaler wants it **fast**: its whole job is to notice that the
    market changed regime and cut gross before the damage lands, and an
    estimate that takes a quarter to turn cannot do that job at all.

    Measured on the ten-year replay, sharing one estimate made volatility
    targeting decorative: ``vol_scale`` bound on 37 of 2,518 sessions (1.5%)
    and returned exactly 1.000 on every day of the COVID crash, including
    2020-03-12. On the account's own weights the blend read 9.7% predicted
    volatility on 2020-03-06 -- against a 12% target, with the book already
    down 11.5% -- because its trailing year was still eleven months of calm.
    The same weights under ``lam=0.94`` read 15.9% and cut gross by 24%. The
    blend was not wrong; it was too slow to turn.

    ``lam`` defaults to the RiskMetrics (1996) short-horizon constant, ~11
    sessions of half-life. It is a published number rather than a fitted one on
    purpose -- see ``PortfolioConfig.vol_estimate_lambda``.

    Reuses :func:`ewma_cov` rather than re-deriving the weighting, so the decay
    and its reliability correction have exactly one implementation, and floors
    the result through :func:`nearest_psd`: ``vol_target_scale`` reads
    ``w' Sigma w`` and treats a non-positive variance as "nothing to scale", so
    a rounding-negative eigenvalue would silently disarm the control it feeds.
    """
    return nearest_psd(ewma_cov(returns, lam=lam))


def commercial_clusters(
    symbols: tuple[str, ...],
    edges: tuple[tuple[str, str, str, date], ...],
    *,
    as_of: date,
) -> dict[str, tuple[str, ...]]:
    """Point-in-time connected components over commercial edges only.

    Admits an edge iff ``report_date <= as_of`` and its type is commercial
    (customer / supplier / competitor / partner). Interlocks never cluster.
    A symbol with no admitted edge is its own singleton cluster.

    Edges naming a symbol outside ``symbols`` are dropped: an unknown name has
    no weight, so it cannot carry cluster exposure. Types are matched
    case-insensitively because the risky failure is the *missed* merge -- two
    genuinely linked names each drawing a full ``max_position`` is exactly the
    concentration the cluster cap exists to prevent.
    """
    known = set(symbols)
    finder = UnionFind(tuple(symbols))
    for source, target, edge_type, report_date in edges:
        if str(edge_type).strip().lower() not in _COMMERCIAL_EDGE_TYPES:
            continue
        if report_date > as_of:
            continue
        if source == target or source not in known or target not in known:
            continue
        finder.union(source, target)
    return finder.components()


def vol_target_scale(
    weights: np.ndarray,
    sigma: np.ndarray,
    *,
    target_vol: float,
    periods_per_year: int = 252,
    cap: float = 1.0,
) -> float:
    """Scalar in ``(0, cap]`` bringing predicted vol to target. Never levers up.

    Capped at 1.0 because this account is long-and-cash: when predicted
    volatility is below target the residual stays in cash rather than becoming
    leverage the broker would not extend.

    ``target_vol`` is annual by definition (``PortfolioConfig`` sets 0.12), so
    this function has to reconcile two horizons no matter what. ``sigma`` is
    read as a **per-period** covariance and its volatility is annualized by
    ``sqrt(periods_per_year)``; pass ``periods_per_year=1`` when handing in an
    already-annualized covariance. Making the periodicity an argument rather
    than a docstring note closes the quiet failure: a daily covariance compared
    against an annual target predicts ~1.6% vol against a 12% target, returns
    1.0 every day, and leaves volatility targeting switched off while the
    config still says 12%.
    """
    weight_vector = np.asarray(weights, dtype=np.float64).ravel()
    covariance = np.asarray(sigma, dtype=np.float64)
    if covariance.shape != (weight_vector.size, weight_vector.size):
        raise ValueError(f"sigma must be ({weight_vector.size}, {weight_vector.size})")
    if target_vol <= 0.0:
        raise ValueError(f"target_vol must be positive, got {target_vol}")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be at least 1, got {periods_per_year}")
    if cap <= 0.0:
        raise ValueError(f"cap must be positive, got {cap}")
    variance = float(weight_vector @ covariance @ weight_vector)
    if variance <= 0.0:
        # An empty or riskless book. There is nothing to scale down, and
        # scaling *up* is the one thing this function refuses to do.
        return float(cap)
    annualized_vol = float(np.sqrt(variance) * np.sqrt(periods_per_year))
    return float(min(cap, target_vol / annualized_vol))


def drawdown_governor(
    equity_curve: np.ndarray,
    *,
    halve_at: float = 0.10,
    halve_scale: float = 0.5,
    flatten_at: float = 0.15,
) -> float:
    """Gross multiplier from peak-to-current drawdown: 1.0 / halve_scale / 0.0.

    Reads equity through *t-1* only. Boundary behavior is tested at exactly
    ``halve_at`` and ``flatten_at``: the threshold is inclusive, so a drawdown
    of precisely 10% halves.

    The drawdown is ``(peak - current) / peak`` rather than ``1 - current/peak``
    on purpose. The two are algebraically identical and numerically are not: an
    account at 90 off a peak of 100 gives 0.1 exactly the first way and
    0.09999999999999998 the second, which would slip under an inclusive 10%
    threshold and leave the breaker armed but never tripping.

    This function reads only the curve it is handed; slicing it to end at *t-1*
    is the caller's no-lookahead responsibility, not something it can verify.

    It names an exit and no re-entry, which on its own is an absorbing state --
    see :func:`decay_high_water_mark` for the missing half and the 2020-03-12
    measurement that found it. The peak this reads is the caller's to decay; the
    rule stated here is unchanged and deliberately so.
    """
    curve = np.asarray(equity_curve, dtype=np.float64).ravel()
    if curve.size == 0:
        raise ValueError("equity_curve is empty")
    if not np.isfinite(curve).all():
        # A NaN silently loses every comparison below, which would return 1.0
        # and disarm the circuit breaker exactly when the book is broken.
        raise ValueError("equity_curve contains non-finite values")
    if not 0.0 < halve_at < flatten_at:
        raise ValueError(f"need 0 < halve_at < flatten_at, got {halve_at} / {flatten_at}")
    if not 0.0 <= halve_scale <= 1.0:
        raise ValueError(f"halve_scale must be in [0, 1], got {halve_scale}")
    peak = float(curve.max())
    if peak <= 0.0:
        return 0.0
    drawdown = (peak - float(curve[-1])) / peak
    if drawdown >= flatten_at:
        return 0.0
    if drawdown >= halve_at:
        return float(halve_scale)
    return 1.0


def risk_contributions(
    weights: np.ndarray, sigma: np.ndarray, symbols: tuple[str, ...], *, limit: float = 0.35
) -> tuple[RiskContribution, ...]:
    """Per-name variance share. Flags concentration the weight caps miss.

    A 15% position in the only volatile name can be 50% of portfolio risk. The
    weight cap does not see that; this does. Reporting-only -- it flags, it
    does not re-optimize.

    The share is Euler's decomposition ``w_i * (Sigma w)_i / (w' Sigma w)``,
    which sums to one across names -- so "35% of risk" is a share of the whole,
    not a standalone volatility.
    """
    weight_vector = np.asarray(weights, dtype=np.float64).ravel()
    covariance = np.asarray(sigma, dtype=np.float64)
    if len(symbols) != weight_vector.size:
        raise ValueError(f"{len(symbols)} symbols but {weight_vector.size} weights")
    if covariance.shape != (weight_vector.size, weight_vector.size):
        raise ValueError(f"sigma must be ({weight_vector.size}, {weight_vector.size})")
    contributions = weight_vector * (covariance @ weight_vector)
    total = float(contributions.sum())
    shares = contributions / total if total > 0.0 else np.zeros_like(contributions)
    return tuple(
        RiskContribution(
            symbol=symbol,
            weight=float(weight_vector[position]),
            risk_share=float(shares[position]),
            flagged=bool(shares[position] > limit),
        )
        for position, symbol in enumerate(symbols)
    )


@dataclass(frozen=True)
class CorrelationStudy:
    """Empirical justification for treating the graph as risk structure."""

    linked_mean_corr: float
    random_mean_corr: float
    difference: float
    ci_low: float
    ci_high: float
    n_linked_pairs: int
    n_random_pairs: int

    @property
    def significant(self) -> bool:
        """True when the bootstrap CI for the difference excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def linked_pair_correlation_study(
    returns: np.ndarray,
    symbols: tuple[str, ...],
    clusters: dict[str, tuple[str, ...]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> CorrelationStudy:
    """Do graph-linked pairs actually co-move more than random pairs?

    This is the honest test of whether the graph earns its place in the
    covariance at all. If the CI includes zero, the graph block target is
    decoration and the report should say so.

    Comparison group: *every* pair that is not linked, rather than a random
    subsample of them. Deterministic and strictly more powerful, and "random
    pair" is what an unlinked pair is -- two names drawn from the universe with
    no measured relationship between them.

    The bootstrap resamples **symbols, not pairs**. Pairwise correlations are
    not independent observations -- the ~600 names generate ~180,000 pairs, and
    every pair touching one volatile name moves together -- so resampling pairs
    would treat that shared structure as independent evidence and report a
    confidence interval several times too narrow. Resampling nodes (the dyadic
    bootstrap, Snijders & Borgatti 1999) keeps a symbol's pairs together;
    pairs formed from a duplicated symbol are dropped, since their correlation
    is a resampling artifact rather than data.

    An empty linked set -- the expected early-year case -- returns a zero
    difference with a degenerate CI, which reads as "not significant". That is
    the correct verdict: no evidence is not evidence.
    """
    data = _panel(returns, minimum=3)
    size = len(symbols)
    if data.shape[1] != size:
        raise ValueError(f"returns has {data.shape[1]} columns but {size} symbols")
    _, corr = _std_and_corr(_sample_cov(data))
    cluster_of = _cluster_index(symbols, clusters)
    same = cluster_of[:, None] == cluster_of[None, :]
    upper = np.triu(np.ones((size, size), dtype=bool), 1)

    linked_mask = upper & same
    random_mask = upper & ~same
    n_linked = int(linked_mask.sum())
    n_random = int(random_mask.sum())
    linked_mean = float(corr[linked_mask].mean()) if n_linked else 0.0
    random_mean = float(corr[random_mask].mean()) if n_random else 0.0
    # With either pool empty there is no comparison, so there is no difference
    # either. Reporting ``0 - random_mean`` would state that linked pairs
    # co-move *less*, which is a claim the absent side never made.
    difference = linked_mean - random_mean if n_linked and n_random else 0.0

    differences: list[float] = []
    if n_linked and n_random and n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_bootstrap):
            draw = rng.integers(0, size, size=size)
            grid = np.ix_(draw, draw)
            distinct = upper & (draw[:, None] != draw[None, :])
            drawn_same = same[grid]
            linked_draw = distinct & drawn_same
            random_draw = distinct & ~drawn_same
            if not linked_draw.any() or not random_draw.any():
                continue
            block = corr[grid]
            differences.append(float(block[linked_draw].mean() - block[random_draw].mean()))

    if differences:
        low, high = np.percentile(np.asarray(differences, dtype=np.float64), (2.5, 97.5))
    else:
        low, high = 0.0, 0.0
    return CorrelationStudy(
        linked_mean_corr=linked_mean,
        random_mean_corr=random_mean,
        difference=difference,
        ci_low=float(low),
        ci_high=float(high),
        n_linked_pairs=n_linked,
        n_random_pairs=n_random,
    )


__all__ = [
    "CorrelationStudy",
    "RiskContribution",
    "RiskLimits",
    "UnionFind",
    "blend_cov",
    "commercial_clusters",
    "constant_correlation_target",
    "decay_high_water_mark",
    "drawdown_governor",
    "ewma_cov",
    "graph_block_target",
    "ledoit_wolf",
    "linked_pair_correlation_study",
    "nearest_psd",
    "risk_contributions",
    "targeting_cov",
    "vol_target_scale",
]
