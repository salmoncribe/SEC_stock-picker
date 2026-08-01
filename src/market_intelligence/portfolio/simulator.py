"""The daily loop, and the stress harness.

One function, ``step_day``, decides everything for a single date; ``replay``
calls it 2,500 times and ``portfolio step --as-of`` calls it once. That is the
whole point: the evidence from the replay applies to the account of record
because they are not two implementations of the same idea, they are one
function called twice.

Ordering within a day, and why each bound is where it is:

  * signals with ``available_on < t``      -- knowable before the open
  * covariance window ``end_exclusive=t``  -- day t's return cannot inform day t
  * edges with ``report_date <= t - 1``    -- yesterday's graph, not today's
  * governor reads equity through ``t-1``  -- today's P&L is not yet realized
  * orders execute at ``t``'s OPEN         -- decided on t-1's information
  * sells settle before buys               -- cash from a sale funds a buy
  * mark at ``t``'s CLOSE

Three leak canaries guard this, and they are not optional:
  1. **t+1 execution** -- a decision on day t fills at t+1's open.
  2. **Covariance canary** -- mutating day *t*'s returns must not change day
     *t*'s weights. If it does, the window is inclusive somewhere.
  3. **Shuffle canary** -- shuffling the signal-to-date mapping must destroy
     the edge. An edge that survives shuffling was never in the signals.

The stress harness resamples *block indices* over the calendar, keeping each
day's cross-section intact, so correlations survive the bootstrap. Resampling
per-symbol independently would destroy the very structure being stress-tested.
Stress criteria are risk-control checks -- volatility stays near target, the
governor is observed firing, drawdown stays inside the halt plus a margin --
and are preregistered. Stress testing NEVER tunes for profit.

Five decisions this module had to make, recorded because none of them is
recoverable from the signatures alone:

* **The investable set is exactly the set of live views.** A held name whose
  view has expired is targeted at zero -- that is the horizon exit
  ``analytics/backtest.py`` replays, expressed as "the absence of a reason to
  hold *is* the exit". It is a rule, not a hope: a viewless name's posterior
  mean is the equal-weight prior ``gamma * S * w_eq``, which is *positive* for
  any real covariance, so a return-seeking objective sees a positive-return
  asset and pays a turnover penalty to sell it. Left to the optimizer it would
  be kept forever. A day with *no* live views at all is different and rides --
  there is no decision to make, so none is made.
* **Portfolio construction happens in HORIZON space, not daily space.** The
  views are horizon-cumulative returns, so the covariance they are priced
  against covers the same span (:func:`_horizon_covariance`) -- otherwise the
  risk term is silently divided by the horizon and the solve degenerates into
  "rank by mu, fill to the caps". The volatility scaler keeps the daily
  estimate; it annualizes on its own.
* **The ledger lives in raw price space, the statistics in adjusted space.**
  Returns, volatility and covariance read ``panel.close`` (adjusted); fills and
  marks read raw prices, because an executed price is what a broker charged.
  The panel carries no raw open, so it is recovered exactly from the row's own
  factor: ``raw_open = open_adj * raw_close / close_adj``.
* **Sizing is on the unslipped open.** A target weight names a market value, so
  the share count is ``dw * equity / raw_open`` and the slippage is a cost paid
  out of cash -- matching ``run_backtest``, which sizes from the plan and then
  fills at the slipped price.
* **Delisting is decided point-in-time.** Nothing here can know a symbol is
  NaN-*forever*; it can only know it has been missing for a full trading week.
  Five consecutive absent sessions force the exit at the last known price.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Literal

import numpy as np

from market_intelligence.analytics.backtest import BacktestSignal
from market_intelligence.config import PortfolioConfig
from market_intelligence.logging_config import get_logger
from market_intelligence.portfolio.account import (
    AccountError,
    AccountState,
    Fill,
    Order,
    Side,
    apply_fill,
    equity_identity_holds,
    genesis,
    mark,
)
from market_intelligence.portfolio.costs import CostModel, commission, fill_price
from market_intelligence.portfolio.feed import FeedBundle, MarketPanel
from market_intelligence.portfolio.metrics import max_drawdown
from market_intelligence.portfolio.optimizer import apply_no_trade_band, optimize
from market_intelligence.portfolio.risk import (
    blend_cov,
    commercial_clusters,
    drawdown_governor,
    ewma_cov,
    graph_block_target,
    ledoit_wolf,
    nearest_psd,
    targeting_cov,
    vol_target_scale,
)
from market_intelligence.portfolio.views import View, bl_posterior, views_from_signals

_log = get_logger("portfolio.simulator")

StressScenario = Literal["bootstrap", "vol_x2", "corr_one", "covid"]

#: Consecutive absent sessions after which a held symbol is treated as delisted
#: and force-closed. One trading week: long enough that a data outage or a
#: trading halt does not eject a live position, short enough that a genuinely
#: dead name is not marked at a ghost price for a month.
_DELISTED_SESSIONS = 5

#: Minimum finite return observations before a name's variance is believed. A
#: name below it enters the optimizer with a NaN diagonal, which pins it to zero
#: rather than letting an unmeasurable variance read as "riskless".
_MIN_COV_OBS = 2

#: Target weights at or below this are a full exit, not a small position. Mirrors
#: ``optimizer._DUST`` so the two agree on what counts as nothing.
_WEIGHT_DUST = 1e-8

#: Trading days per year. ``target_vol`` is annual and the covariance is daily;
#: this is the conversion, passed explicitly on every call.
_PERIODS_PER_YEAR = 252

#: Preregistered stress pass criteria. Realized vol may exceed target by 25%;
#: drawdown may exceed the flatten threshold by 5 percentage points (the
#: governor acts on t-1 information, so one day's damage is always unavoidable).
_STRESS_VOL_MULTIPLE = 1.25
_STRESS_DRAWDOWN_MARGIN = 0.05

#: The COVID crash window, and the fallback width when a panel never reaches it.
_COVID_START = date(2020, 2, 19)
_COVID_END = date(2020, 4, 30)
_CRASH_WINDOW_SESSIONS = 50

#: A doubled -60% day is -120%, which is not a price. Returns are floored here
#: before a synthetic path is compounded, so a stressed panel stays positive.
_MIN_STRESSED_RETURN = -0.99


@dataclass(frozen=True)
class DayDecision:
    """Everything decided on one date, retained so any day can be audited."""

    as_of: date
    target_weights: np.ndarray
    symbols: tuple[str, ...]
    views: tuple[View, ...]
    orders: tuple[Order, ...]
    optimizer_status: str
    governor_scale: float
    vol_scale: float
    rebalanced: bool


@dataclass(frozen=True)
class ReplayResult:
    """The full simulated history: equity, decisions, fills, and diagnostics."""

    equity_curve: np.ndarray
    calendar: np.ndarray
    decisions: tuple[DayDecision, ...]
    fills: tuple[Fill, ...]
    final_state: AccountState
    daily_returns: np.ndarray
    benchmark_returns: np.ndarray | None


@dataclass(frozen=True)
class _Trade:
    """A planned execution: what to trade, which way, and at what price."""

    symbol: str
    side: Side
    shares: float
    price: float
    reason: str


def step_day(
    state: AccountState,
    bundle: FeedBundle,
    config: PortfolioConfig,
    *,
    as_of: date,
    equity_history: np.ndarray,
    costs: CostModel,
    force_rebalance: bool = False,
) -> tuple[DayDecision, AccountState]:
    """Decide and execute one day. THE choke point -- replay and step share it.

    Everything it may read about day ``as_of`` is that day's open (for
    execution) and close (for marking). All *decisions* use data strictly
    before ``as_of``.

    A thin projection of :func:`execute_day`, which additionally hands back the
    day's fills and marked equity.

    **Anything that persists the day must call ``execute_day``, not this.** The
    account of record is a fold over *fills*, so a caller that writes only what
    this returns saves the decision and loses the ledger -- and the account then
    reconstructs to something that never happened. This projection exists for
    callers that want the decision alone: inspection, dry runs, tests.
    """
    decision, next_state, _fills, _equity = execute_day(
        state,
        bundle,
        config,
        as_of=as_of,
        equity_history=equity_history,
        costs=costs,
        force_rebalance=force_rebalance,
    )
    return decision, next_state


def replay(
    bundle: FeedBundle,
    config: PortfolioConfig,
    *,
    start: date | None = None,
    end: date | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ReplayResult:
    """Run ``step_day`` across the calendar. Asserts the equity identity daily.

    The rebalance cadence keys off the *panel's* absolute row index rather than
    a counter local to this call, which is what makes a replay split at any date
    identical to the whole one -- and therefore what makes the Monday-forward
    ``portfolio step`` provably continuous with the replay that preceded it.
    """
    panel = bundle.panel
    calendar = panel.calendar
    if calendar.size == 0:
        raise ValueError("cannot replay an empty calendar")

    low = 0 if start is None else int(np.searchsorted(calendar, np.datetime64(start, "D"), "left"))
    high = (
        calendar.size
        if end is None
        else int(np.searchsorted(calendar, np.datetime64(end, "D"), "right"))
    )
    if high <= low:
        raise ValueError(f"empty replay window: {start} .. {end}")

    days: list[date] = [calendar[index].astype(object) for index in range(low, high)]
    costs = CostModel(
        slippage_bps=config.slippage_bps, commission_per_share=config.commission_per_share
    )
    state = genesis(config.starting_cash, days[0])

    equity: list[float] = []
    decisions: list[DayDecision] = []
    fills: list[Fill] = []
    for position, day in enumerate(days):
        decision, state, day_fills, day_equity = execute_day(
            state,
            bundle,
            config,
            as_of=day,
            equity_history=np.asarray(equity, dtype=np.float64),
            costs=costs,
            force_rebalance=False,
        )
        decisions.append(decision)
        fills.extend(day_fills)
        equity.append(day_equity)
        if progress is not None:
            progress(position + 1, len(days))

    equity_curve = np.asarray(equity, dtype=np.float64)
    return ReplayResult(
        equity_curve=equity_curve,
        calendar=calendar[low:high].copy(),
        decisions=tuple(decisions),
        fills=tuple(fills),
        final_state=state,
        daily_returns=_equity_returns(equity_curve, config.starting_cash),
        benchmark_returns=_benchmark_returns(panel, bundle.spy_index, low, high),
    )


# --------------------------------------------------------------------------- #
# the day                                                                      #
# --------------------------------------------------------------------------- #
def _governor_curve(state: AccountState, history: np.ndarray) -> np.ndarray:
    """The two points the governor needs: the account's peak, and yesterday.

    ``drawdown_governor`` takes the max of the curve as the peak and its last
    element as current, so this is exactly ``(hwm - equity_{t-1}) / hwm`` --
    the account's own drawdown, not a statistic of whatever slice of curve this
    call happened to be handed.

    Two things follow, and both were bugs before it existed:

    * **The decaying mark reaches the governor.** ``history.max()`` is a
      ratchet that nothing can lower, so decaying ``state.high_water_mark`` in
      ``account.mark`` would have healed the reported drawdown while the
      governor went on reading the undecayed peak -- the halt would never have
      lifted, which is the whole defect.
    * **Replay and the live step agree.** ``portfolio step`` already builds
      precisely this two-point curve (it has a stored mark and no stored equity
      curve). The replay was passing its full curve instead, so the "one
      function called twice" guarantee held everywhere except here.

    In a from-genesis replay with the decay disabled the two are the same
    number: the mark starts at the opening cash and ratchets over exactly the
    equities the curve contains.
    """
    return np.array([state.high_water_mark, float(history[-1])], dtype=np.float64)


def execute_day(
    state: AccountState,
    bundle: FeedBundle,
    config: PortfolioConfig,
    *,
    as_of: date,
    equity_history: np.ndarray,
    costs: CostModel,
    force_rebalance: bool,
) -> tuple[DayDecision, AccountState, tuple[Fill, ...], float]:
    """One simulated day, start to finish. Every bound in the module docstring
    is enforced here and nowhere else."""
    panel = bundle.panel
    row = panel.index_of(as_of)

    history = np.asarray(equity_history, dtype=np.float64).ravel()
    governor_scale = (
        drawdown_governor(
            _governor_curve(state, history),
            halve_at=config.governor_halve_drawdown,
            halve_scale=config.governor_halve_scale,
            flatten_at=config.governor_flatten_drawdown,
        )
        if history.size
        else 1.0
    )
    de_risking = governor_scale < 1.0

    # Day t-1 is the whole of the decision's information set: its close prices
    # value the book, and its date bounds the signals and the graph.
    decision_day: date | None = panel.calendar[row - 1].astype(object) if row else None
    held = tuple(sorted(state.positions))
    prior_prices = _mark_prices(panel, row - 1, held) if row else {}
    equity_ref = float(mark(state, prior_prices, stale_ok=True).equity)

    delisted = tuple(symbol for symbol in held if _is_delisted(panel, row, symbol))

    views: tuple[View, ...] = ()
    if decision_day is not None:
        views = _live_views(bundle, config, as_of=as_of, decision_day=decision_day)

    known = set(panel.symbols)
    universe = tuple(
        sorted(({view.symbol for view in views} | set(held)) - set(delisted) & known)
    )
    views = tuple(view for view in views if view.symbol in universe)
    invested = tuple(sorted({view.symbol for view in views}))

    w_current = _current_weights(state, universe, prior_prices, equity_ref)
    w_target = w_current.copy()
    optimizer_status = "not_rebalanced"
    vol_scale = 1.0
    reasons: Mapping[str, str] = {}

    scheduled = row % max(int(config.rebalance_days), 1) == 0
    rebalanced = bool(universe) and (
        de_risking or force_rebalance or (scheduled and bool(views))
    )

    if rebalanced and not invested:
        # Nothing to size, but the governor still has an instruction. De-risk
        # what is held rather than optimizing an empty book.
        w_target = w_current * governor_scale
        optimizer_status = "governor_only"
    elif rebalanced:
        block, counts = _returns_block(panel, invested, as_of=as_of, lookback=config.cov_lookback)
        if block.shape[0] < _MIN_COV_OBS:
            # No covariance can be estimated, so no risk budget can be. Holding
            # is the only honest answer; guessing one is not.
            rebalanced = False
        else:
            clusters = commercial_clusters(
                invested, bundle.edges, as_of=decision_day if decision_day else as_of
            )
            sigma = _estimate_covariance(block, invested, clusters, config)
            # Construction happens in HORIZON space, not daily space. See
            # ``_horizon_covariance``: the views are horizon-cumulative returns,
            # so the covariance they are priced against has to cover the same
            # span or the risk term is silently divided by the horizon.
            horizon_sigma = _horizon_covariance(sigma, invested, views)
            posterior = bl_posterior(
                horizon_sigma,
                invested,
                views,
                engine=config.bl_engine,
                tau=config.tau,
                risk_aversion=config.risk_aversion,
            )
            result = optimize(
                posterior.mu,
                _optimizer_covariance(horizon_sigma, counts),
                _subset(w_current, universe, invested),
                invested,
                objective=config.objective,
                risk_aversion=config.risk_aversion,
                turnover_penalty=config.turnover_penalty_bps / 10_000.0,
                max_gross=config.max_gross * governor_scale,
                max_position=config.max_position,
                max_cluster=config.max_cluster,
                clusters=clusters,
                scenario_returns=block,
                alpha=config.cvar_alpha,
            )
            # The scaler reads its own fast covariance, not ``sigma``. Same
            # weights, same window, different estimator: the optimizer above
            # needs a stable covariance or it churns the book, and a stable
            # covariance is exactly what a regime detector must not be. Sharing
            # one estimate measured as a control that never acted -- 37 of
            # 2,518 sessions, and 1.000 through all of COVID. ``targeting_cov``
            # carries the measurement. (It is emphatically *not*
            # ``posterior.sigma``, and never was Black-Litterman's ``S + M``.)
            vol_scale = vol_target_scale(
                result.weights,
                targeting_cov(block, lam=config.vol_estimate_lambda),
                target_vol=config.target_vol,
                periods_per_year=_PERIODS_PER_YEAR,
            )
            optimizer_status = result.status
            reasons = result.trade_reasons
            w_target = np.zeros(len(universe), dtype=np.float64)
            for column, symbol in enumerate(invested):
                w_target[universe.index(symbol)] = float(result.weights[column]) * vol_scale

    if rebalanced:
        w_target = apply_no_trade_band(
            w_target,
            w_current,
            band=config.no_trade_band,
            bypass=de_risking and bool(config.governor_bypasses_band),
        )

    trades = _planned_trades(
        panel,
        row,
        state,
        costs,
        as_of=as_of,
        universe=universe,
        delisted=delisted,
        w_target=w_target,
        w_current=w_current,
        equity_ref=equity_ref,
        reasons=reasons,
    )
    orders, fills, state = _settle(state, trades, config, costs, as_of=as_of)

    prices = _mark_prices(panel, row, tuple(sorted(state.positions)))
    # The one mark of the day whose state is kept, so the one that advances the
    # high-water mark. The valuation-only mark above deliberately does not carry
    # the half-life: it re-values an already-marked state at t-1's prices, and
    # decaying there would spend a second period of the account's memory on a
    # result that is thrown away except for its ``.equity``.
    marked = mark(
        state, prices, stale_ok=True, hwm_decay_halflife_days=config.hwm_decay_halflife_days
    )
    if not equity_identity_holds(marked):
        raise AccountError(f"equity identity broken on {as_of.isoformat()}")
    state = replace(marked.state, as_of=as_of)

    decision = DayDecision(
        as_of=as_of,
        target_weights=np.asarray(w_target, dtype=np.float64),
        symbols=universe,
        views=views,
        orders=orders,
        optimizer_status=optimizer_status,
        governor_scale=float(governor_scale),
        vol_scale=float(vol_scale),
        rebalanced=bool(rebalanced),
    )
    return decision, state, fills, float(marked.equity)


def _live_views(
    bundle: FeedBundle, config: PortfolioConfig, *, as_of: date, decision_day: date
) -> tuple[View, ...]:
    """Views from signals knowable strictly before ``as_of``.

    ``views_from_signals`` admits ``available_on <= as_of``, so it is handed
    *yesterday* as its as-of date. That is the ``available_on < t`` bound: a
    signal released on day *t* cannot reach day *t*'s open.
    """
    signals = _signal_window(bundle, decision_day)
    if not signals:
        return ()
    known = set(bundle.panel.symbols)
    candidates = tuple(sorted({signal.symbol for signal in signals} & known))
    if not candidates:
        return ()
    raw = bundle.panel.trailing_returns(
        end_exclusive=as_of, lookback=config.cov_lookback, symbols=candidates
    )
    volatilities: dict[str, float] = {}
    for column, symbol in enumerate(candidates):
        observed = raw[:, column][np.isfinite(raw[:, column])]
        if observed.size >= _MIN_COV_OBS:
            volatilities[symbol] = float(observed.std(ddof=1))
    return views_from_signals(
        signals,
        bundle.cells,
        bundle.hedged_edges,
        volatilities,
        as_of=decision_day,
        calendar=bundle.panel.calendar,
    )


def _signal_window(bundle: FeedBundle, decision_day: date) -> list[BacktestSignal]:
    """Signals that could still be inside their horizon on ``decision_day``.

    Bounded by the longest admitted horizon so a 2,500-day replay does not
    re-scan the whole signal history every day. The bound is in calendar days
    with a week of slack, because a signal may be dated on a non-session day.
    """
    if not bundle.signals_by_day:
        return []
    horizon = max((int(cell.horizon_days) for cell in bundle.cells), default=0)
    floor_day = decision_day - timedelta(days=2 * horizon + 14)
    window = [
        signal
        for day, signals in bundle.signals_by_day.items()
        if floor_day <= day <= decision_day
        for signal in signals
    ]
    window.sort(key=lambda signal: (signal.available_on, signal.symbol, signal.horizon_days))
    return window


def _returns_block(
    panel: MarketPanel, symbols: tuple[str, ...], *, as_of: date, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """Finite ``[T, K]`` return block ending strictly before ``as_of``, plus the
    per-symbol count of genuinely observed returns.

    Rows nobody traded on are dropped; a single name's missing day is filled
    with zero rather than dropped, because dropping it would misalign every
    other name's day and the cross-section is the thing being estimated. The
    counts travel alongside so a name whose variance rests on almost nothing can
    be excluded downstream instead of being silently believed.
    """
    raw = panel.trailing_returns(end_exclusive=as_of, lookback=lookback, symbols=symbols)
    finite = np.isfinite(raw)
    keep = finite.any(axis=1)
    block = np.nan_to_num(raw[keep], nan=0.0, posinf=0.0, neginf=0.0)
    return block, finite.sum(axis=0)


def _estimate_covariance(
    block: np.ndarray,
    symbols: tuple[str, ...],
    clusters: dict[str, tuple[str, ...]],
    config: PortfolioConfig,
) -> np.ndarray:
    """Ledoit-Wolf toward the graph block target, blended with EWMA, made PSD.

    ``lw_weight`` rides on the Ledoit-Wolf side, matching ``lw_blend``.
    """
    shrunk, _delta = ledoit_wolf(block, graph_block_target(block, symbols, clusters))
    ewma = ewma_cov(block, lam=config.ewma_lambda)
    return nearest_psd(blend_cov(shrunk, ewma, lw_weight=config.lw_blend))


def _horizon_covariance(
    sigma: np.ndarray, symbols: tuple[str, ...], views: tuple[View, ...]
) -> np.ndarray:
    """``sigma`` restated over each name's own holding horizon: ``D S D``,
    ``D = diag(sqrt(H_i))``.

    **This is a units fix, and it was worth 130x.** The objective
    ``mu'w - gamma*w'Sw - kappa*||w - w_cur||`` is a one-period mean-variance
    problem, and a one-period problem requires ``mu`` and ``S`` to describe the
    *same* period. They did not:

      * ``mu`` comes from ``views.drag_adjusted_edge``, which is
        ``hedged_edge - 0.5*vol^2*H`` -- a **cumulative H-session** return;
      * ``S`` comes from ``MarketPanel.trailing_returns``, which are **one-day**
        returns;
      * ``kappa`` is a one-time cost, correctly commensurate with an H-session
        return because a position is established once and unwound once.

    So the risk term alone was stated per day while its two neighbours were
    stated per horizon, dividing it by H. Measured on the ten-year replay's
    signal geometry (H=20 throughout), at the solved weights the median terms
    were ``mu'w = 4.4e-2``, ``gamma*w'Sw = 3.4e-4``, ``kappa*||dw|| = 9.4e-4``:
    risk was 130x smaller than return, so the solve degenerated into "rank by
    mu and fill to the caps". Gross sat at exactly 1.000 on the median day and
    the book was bang-bang -- and a bang-bang solution *flips* when mu moves a
    hair, which is itself a turnover engine.

    The rescaling is invariant to which period you choose to write the problem
    in. For any period ``P`` sessions: ``mu_P = mu_H*(P/H)``, ``S_P = P*S``, and
    a name held ``H`` sessions is traded ``P/H`` times, so
    ``cost_P = kappa*(P/H)*||dw||``. Multiplying through by ``H/P`` returns
    ``mu_H'w - gamma*H*w'Sw - kappa*||dw||`` whatever ``P`` was. ``gamma``
    itself is period-free: it is ``E[excess]/variance``, and both scale with the
    period, which is why the textbook 2.5 needs no adjustment.

    Per-name rather than one scalar because horizons genuinely differ across
    admitted cells: the covariance of *i*'s ``H_i``-session return with *j*'s
    ``H_j``-session return is ``sqrt(H_i*H_j)*S_ij`` for overlapping windows.
    ``D S D`` with ``D = diag(sqrt(H))`` is exactly that, and it preserves PSD
    (a congruence by a positive diagonal), so ``nearest_psd`` upstream still
    holds downstream.

    Note what does NOT read this: ``targeting_cov`` / ``vol_target_scale`` keep
    the daily estimate and annualize it by ``sqrt(252)`` themselves. Handing
    them a horizon covariance would target volatility over the wrong span.
    """
    horizons = {view.symbol: max(int(view.horizon_days), 1) for view in views}
    # A symbol with no view is not in ``invested`` and cannot reach here, but a
    # missing horizon must still be a no-op rather than a zero: scaling a
    # variance to zero would price the name as riskless.
    lengths = np.array([horizons.get(symbol, 1) for symbol in symbols], dtype=np.float64)
    # ``sqrt(outer(H, H))`` rather than ``outer(sqrt(H), sqrt(H))``: identical in
    # exact arithmetic, but the single root is exact on the diagonal and on
    # every equal-horizon pair, so a one-horizon day scales the covariance by
    # the integer itself and stays byte-comparable with the unscaled estimate.
    return np.asarray(sigma, dtype=np.float64) * np.sqrt(np.outer(lengths, lengths))


def _optimizer_covariance(sigma: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``sigma`` with an unmeasurable name's variance marked NaN.

    ``solve_max_sharpe`` pins a name with a non-finite variance to zero weight.
    That is the behaviour wanted here: a repaired zero would price the name as
    riskless and hand it the largest position precisely when least is known
    about it.
    """
    unusable = np.asarray(counts) < _MIN_COV_OBS
    if not unusable.any():
        return sigma
    out = np.array(sigma, dtype=np.float64, copy=True)
    diagonal = np.diag(out).copy()
    diagonal[unusable] = np.nan
    np.fill_diagonal(out, diagonal)
    return out


def _current_weights(
    state: AccountState,
    universe: tuple[str, ...],
    prices: Mapping[str, float],
    equity_ref: float,
) -> np.ndarray:
    """Holdings as a fraction of decision-time equity, at day t-1's closes."""
    weights = np.zeros(len(universe), dtype=np.float64)
    if equity_ref <= 0.0:
        return weights
    for column, symbol in enumerate(universe):
        position = state.positions.get(symbol)
        price = prices.get(symbol)
        if position is not None and price is not None:
            weights[column] = position.shares * price / equity_ref
    return weights


def _subset(weights: np.ndarray, universe: tuple[str, ...], symbols: tuple[str, ...]) -> np.ndarray:
    """Re-index a weight vector from ``universe`` onto ``symbols``."""
    lookup = {symbol: column for column, symbol in enumerate(universe)}
    return np.array([weights[lookup[symbol]] for symbol in symbols], dtype=np.float64)


# --------------------------------------------------------------------------- #
# execution                                                                    #
# --------------------------------------------------------------------------- #
def _planned_trades(
    panel: MarketPanel,
    row: int,
    state: AccountState,
    costs: CostModel,
    *,
    as_of: date,
    universe: tuple[str, ...],
    delisted: tuple[str, ...],
    w_target: np.ndarray,
    w_current: np.ndarray,
    equity_ref: float,
    reasons: Mapping[str, str],
) -> tuple[_Trade, ...]:
    """Sells first, then buys, each in symbol order so a replay is reproducible.

    Sale proceeds fund purchases, so the ordering is an invariant of the day and
    not a convenience: a buy attempted first would overdraw an account that is
    fully invested and raise mid-replay.
    """
    sells: list[_Trade] = []
    buys: list[_Trade] = []

    for symbol in delisted:
        position = state.positions.get(symbol)
        stale = panel.last_known_price(symbol, as_of=as_of)
        if position is None or stale is None:
            continue
        # Exit slippage still applies. A forced exit into a market that has
        # stopped printing is not cheaper than an ordinary one, and inventing a
        # frictionless fill would flatter every delisting in the replay.
        sells.append(
            _Trade(
                symbol=symbol,
                side=Side.SELL,
                shares=position.shares,
                price=fill_price(stale, is_buy=False, model=costs),
                reason="delisted",
            )
        )

    for column, symbol in enumerate(universe):
        if symbol in delisted:
            continue
        raw_open = _raw_open(panel, row, panel.column_of(symbol))
        if raw_open is None:
            continue  # no open, no execution -- the position simply rides
        position = state.positions.get(symbol)
        shares_held = position.shares if position is not None else 0.0
        if float(w_target[column]) <= _WEIGHT_DUST:
            delta = -shares_held
        else:
            delta = float(w_target[column] - w_current[column]) * equity_ref / raw_open
        if delta == 0.0:
            continue
        reason = reasons.get(symbol, "rebalance")
        if delta > 0.0:
            buys.append(
                _Trade(
                    symbol=symbol,
                    side=Side.BUY,
                    shares=delta,
                    price=fill_price(raw_open, is_buy=True, model=costs),
                    reason=reason,
                )
            )
        elif shares_held > 0.0:
            sells.append(
                _Trade(
                    symbol=symbol,
                    side=Side.SELL,
                    shares=min(-delta, shares_held),
                    price=fill_price(raw_open, is_buy=False, model=costs),
                    reason=reason,
                )
            )
    return (*sells, *buys)


def _settle(
    state: AccountState,
    trades: tuple[_Trade, ...],
    config: PortfolioConfig,
    costs: CostModel,
    *,
    as_of: date,
) -> tuple[tuple[Order, ...], tuple[Fill, ...], AccountState]:
    """Apply the day's trades in order, clamping each to what the book allows.

    An order is an intent and may go unfilled or fill short: a buy is clamped to
    the cash actually on hand at the moment it executes, a sell to the shares
    actually held. Clamping rather than raising is deliberate -- a 2,500-day
    replay cannot stop on day 1,400 because a fill was a cent too large.
    """
    fractional = bool(config.fractional_shares)
    orders: list[Order] = []
    fills: list[Fill] = []
    for trade in trades:
        key = f"{trade.symbol}:{as_of.isoformat()}:{trade.side.value}:{config.simulation_version}"
        orders.append(
            Order(
                order_id=f"po:{key}",
                symbol=trade.symbol,
                side=trade.side,
                shares=_round_near(trade.shares, fractional=fractional),
                as_of=as_of,
                reason=trade.reason,
            )
        )
        if trade.side is Side.SELL:
            position = state.positions.get(trade.symbol)
            if position is None:
                continue
            shares = min(_round_near(trade.shares, fractional=fractional), position.shares)
        else:
            affordable = state.cash / (trade.price + config.commission_per_share)
            shares = _round_down(min(trade.shares, affordable), fractional=fractional)
        if shares <= 0.0:
            continue
        fill = Fill(
            fill_id=f"pf:{key}",
            order_id=f"po:{key}",
            symbol=trade.symbol,
            side=trade.side,
            shares=shares,
            price=trade.price,
            commission=commission(shares, costs),
            as_of=as_of,
        )
        state = apply_fill(state, fill, fractional=fractional)
        fills.append(fill)
    return tuple(orders), tuple(fills), state


def _round_down(shares: float, *, fractional: bool) -> float:
    """Quantise toward zero -- used for buys, which must never overspend."""
    if not fractional:
        return float(math.floor(shares))
    return math.floor(shares * 1e4) / 1e4


def _round_near(shares: float, *, fractional: bool) -> float:
    """Quantise to the nearest tick -- used for sells, which are clamped to the
    holding afterwards, so rounding up cannot oversell but flooring *could*
    leave a dust position the governor was told to remove."""
    if not fractional:
        return float(round(shares))
    return round(shares, 4)


# --------------------------------------------------------------------------- #
# prices                                                                       #
# --------------------------------------------------------------------------- #
def _raw_open(panel: MarketPanel, row: int, column: int) -> float | None:
    """Day ``row``'s open in raw price space, or None when it cannot be priced.

    The panel stores an adjusted open and a raw close, so the row's own
    adjustment factor (``raw_close / close``) de-adjusts the open exactly. On a
    row with no corporate action the factor is 1 and this returns the open
    unchanged, bit for bit.
    """
    adjusted_open = float(panel.open[row, column])
    adjusted_close = float(panel.close[row, column])
    raw_close = float(panel.raw_close[row, column])
    if not math.isfinite(adjusted_open) or adjusted_open <= 0.0:
        return None
    if not math.isfinite(adjusted_close) or adjusted_close <= 0.0:
        return None
    if not math.isfinite(raw_close) or raw_close <= 0.0:
        return None
    return adjusted_open * raw_close / adjusted_close


def _mark_prices(
    panel: MarketPanel, row: int, symbols: tuple[str, ...]
) -> dict[str, float]:
    """Raw closes for ``symbols`` on ``row``, falling back to the last known.

    Never looks forward: the fallback is ``last_known_price``, which walks
    backwards only. A symbol with no price at all is omitted, and ``mark`` is
    called with ``stale_ok=True`` so it values that holding at its own cost --
    reporting no gain rather than an invented one.
    """
    if row < 0 or not symbols:
        return {}
    day: date = panel.calendar[row].astype(object)
    prices: dict[str, float] = {}
    for symbol in symbols:
        value = float(panel.raw_close[row, panel.column_of(symbol)])
        if not math.isfinite(value):
            fallback = panel.last_known_price(symbol, as_of=day)
            if fallback is None:
                continue
            value = fallback
        prices[symbol] = value
    return prices


def _is_delisted(panel: MarketPanel, row: int, symbol: str) -> bool:
    """True once a held symbol has missed ``_DELISTED_SESSIONS`` in a row.

    Point-in-time by construction: it reads the window ending at ``row`` and
    never past it. Nothing here can know a symbol is missing *forever*, only
    that it has been missing for a full trading week.
    """
    window = panel.raw_close[max(0, row - _DELISTED_SESSIONS + 1) : row + 1]
    column = window[:, panel.column_of(symbol)]
    if column.size < _DELISTED_SESSIONS:
        return False
    return not bool(np.isfinite(column).any())


def _equity_returns(equity_curve: np.ndarray, starting_cash: float) -> np.ndarray:
    """Day-over-day account returns, the first measured against opening capital."""
    if equity_curve.size == 0:
        return np.zeros(0, dtype=np.float64)
    previous = np.concatenate(([float(starting_cash)], equity_curve[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = equity_curve / previous - 1.0
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)


def _benchmark_returns(
    panel: MarketPanel, spy_index: int | None, low: int, high: int
) -> np.ndarray | None:
    """Benchmark returns aligned to the simulated days, or None with no benchmark."""
    if spy_index is None:
        return None
    closes = panel.close[:, spy_index]
    out = np.zeros(high - low, dtype=np.float64)
    for position, row in enumerate(range(low, high)):
        if row == 0:
            continue
        previous, current = float(closes[row - 1]), float(closes[row])
        if math.isfinite(previous) and previous > 0.0 and math.isfinite(current):
            out[position] = current / previous - 1.0
    return out


# --------------------------------------------------------------------------- #
# stress                                                                       #
# --------------------------------------------------------------------------- #
def _panel_returns(panel: MarketPanel) -> np.ndarray:
    """``[T-1, N]`` simple returns with gaps read as no move."""
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = panel.close[1:] / panel.close[:-1] - 1.0
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)


def _first_finite_close(panel: MarketPanel) -> np.ndarray:
    """Each column's first observed close, for starting a synthetic price path."""
    base = np.full(len(panel.symbols), 100.0, dtype=np.float64)
    for column in range(len(panel.symbols)):
        finite = np.flatnonzero(np.isfinite(panel.close[:, column]))
        if finite.size:
            base[column] = float(panel.close[finite[0], column])
    return base


def _rebuild_panel(
    panel: MarketPanel, returns: np.ndarray, day_rows: np.ndarray, calendar: np.ndarray
) -> MarketPanel:
    """Compound ``returns`` into a synthetic panel on ``calendar``.

    ``day_rows[i]`` names the original row whose open/close geometry output row
    *i* borrows, so an overnight gap is not silently flattened into a
    close-to-close series -- execution happens at the open, and a stress panel
    with no gaps would understate exactly the risk it exists to test.

    The synthetic panel has no corporate actions, so ``raw_close == close`` and
    the ledger and the statistics read the same numbers.
    """
    close = np.empty((calendar.size, len(panel.symbols)), dtype=np.float64)
    close[0] = _first_finite_close(panel)
    if calendar.size > 1:
        close[1:] = close[0] * np.cumprod(
            1.0 + np.clip(returns, _MIN_STRESSED_RETURN, None), axis=0
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = panel.open / panel.close
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    open_ = close * ratio[day_rows]
    return MarketPanel(
        calendar=calendar,
        symbols=panel.symbols,
        open=open_,
        high=np.maximum(open_, close),
        low=np.minimum(open_, close),
        close=close,
        raw_close=close.copy(),
    )


def block_bootstrap_panel(
    bundle: FeedBundle, *, block_size: int = 21, seed: int = 0, n_days: int | None = None
) -> FeedBundle:
    """Resample BLOCKS OF DATES, not per-symbol series.

    Each sampled block carries every symbol's returns for those dates together,
    so the cross-sectional correlation structure survives. Independent
    per-symbol resampling would produce an uncorrelated panel and a stress test
    that proves nothing.

    Distinct from ``executable_validation.moving_block_bootstrap``, which
    resamples realized net-return *observations* for outcome-level confidence
    intervals. This one synthesizes a panel; that one bounds a statistic.
    """
    panel = bundle.panel
    total = int(panel.calendar.size)
    if total < 3:
        return bundle
    target = total if n_days is None else int(np.clip(int(n_days), 2, total))
    returns = _panel_returns(panel)
    width = int(np.clip(int(block_size), 1, returns.shape[0]))
    needed = target - 1

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, returns.shape[0] - width + 1, size=math.ceil(needed / width))
    rows = np.concatenate([np.arange(start, start + width) for start in starts])[:needed]
    # Return index k is the move *into* panel row k+1, so that row supplies the
    # sampled day's open/close geometry.
    day_rows = np.concatenate(([0], rows + 1))
    calendar = panel.calendar[:target].copy()
    return replace(bundle, panel=_rebuild_panel(panel, returns[rows], day_rows, calendar))


def _slice_panel(panel: MarketPanel, low: int, high: int) -> MarketPanel:
    """A contiguous window of real prices -- no synthesis, no rebuild."""
    return MarketPanel(
        calendar=panel.calendar[low:high].copy(),
        symbols=panel.symbols,
        open=panel.open[low:high].copy(),
        high=panel.high[low:high].copy(),
        low=panel.low[low:high].copy(),
        close=panel.close[low:high].copy(),
        raw_close=panel.raw_close[low:high].copy(),
    )


def _crash_window(panel: MarketPanel) -> tuple[int, int]:
    """The 2020-02..04 rows, or the panel's own deepest stretch if it never
    reaches them.

    A panel that stops in 2019 cannot replay COVID, and fabricating those
    returns would be inventing data. Its own worst window is the closest thing
    the data actually contains; which one was used is logged rather than
    guessed at from the output.
    """
    calendar = panel.calendar
    low = int(np.searchsorted(calendar, np.datetime64(_COVID_START, "D"), "left"))
    high = int(np.searchsorted(calendar, np.datetime64(_COVID_END, "D"), "right"))
    if high - low >= 2:
        _log.info("stress_covid_window", source="historical", rows=high - low)
        return low, high

    returns = _panel_returns(panel)
    width = min(_CRASH_WINDOW_SESSIONS, returns.shape[0])
    market = returns.mean(axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(market)))
    totals = cumulative[width:] - cumulative[:-width]
    start = int(np.argmin(totals))
    _log.info("stress_covid_window", source="worst_observed", rows=width + 1)
    return start, start + width + 1


def stress_transform(bundle: FeedBundle, scenario: StressScenario, *, seed: int = 0) -> FeedBundle:
    """Apply one stress: doubled vol, correlations -> 1, or the 2020-02..04 replay."""
    if scenario == "bootstrap":
        return block_bootstrap_panel(bundle, seed=seed)
    panel = bundle.panel
    if panel.calendar.size < 3:
        return bundle
    if scenario == "covid":
        low, high = _crash_window(panel)
        return replace(bundle, panel=_slice_panel(panel, low, high))

    returns = _panel_returns(panel)
    day_rows = np.arange(panel.calendar.size)
    if scenario == "vol_x2":
        return replace(
            bundle, panel=_rebuild_panel(panel, 2.0 * returns, day_rows, panel.calendar.copy())
        )
    if scenario == "corr_one":
        # Every name becomes a scaled copy of the equal-weight market, so the
        # cross-correlation is exactly one while each name keeps its own
        # volatility -- the diversification is removed, not the risk.
        market = returns.mean(axis=1)
        market_vol = float(market.std(ddof=1))
        if market_vol <= 0.0:
            return bundle
        forced = np.outer(market, returns.std(axis=0, ddof=1) / market_vol)
        return replace(
            bundle, panel=_rebuild_panel(panel, forced, day_rows, panel.calendar.copy())
        )
    raise ValueError(f"unknown stress scenario {scenario!r}")


@dataclass(frozen=True)
class StressOutcome:
    """Risk-control verdict for one scenario. Preregistered pass criteria."""

    scenario: StressScenario
    realized_vol: float
    max_drawdown: float
    governor_fired: bool
    vol_within_target: bool
    drawdown_within_halt: bool

    @property
    def passed(self) -> bool:
        """Vol <= 1.25x target AND drawdown <= halt + 5pts AND governor observed."""
        return self.vol_within_target and self.drawdown_within_halt and self.governor_fired


def run_stress_suite(
    bundle: FeedBundle,
    config: PortfolioConfig,
    *,
    scenarios: tuple[StressScenario, ...] = ("bootstrap", "vol_x2", "corr_one", "covid"),
    seed: int = 0,
) -> tuple[StressOutcome, ...]:
    """Run each scenario and report risk-control outcomes. Never tunes anything.

    The criteria are read off ``config`` -- the same numbers the replay ran
    under -- so a scenario cannot be passed by relaxing what it was measured
    against.
    """
    outcomes: list[StressOutcome] = []
    for scenario in scenarios:
        result = replay(stress_transform(bundle, scenario, seed=seed), config)
        returns = result.daily_returns
        realized_vol = (
            float(returns.std(ddof=1) * math.sqrt(_PERIODS_PER_YEAR)) if returns.size > 1 else 0.0
        )
        drawdown = max_drawdown(result.equity_curve)
        outcomes.append(
            StressOutcome(
                scenario=scenario,
                realized_vol=realized_vol,
                max_drawdown=drawdown,
                governor_fired=any(
                    decision.governor_scale < 1.0 for decision in result.decisions
                ),
                vol_within_target=realized_vol <= _STRESS_VOL_MULTIPLE * config.target_vol,
                drawdown_within_halt=drawdown
                <= config.governor_flatten_drawdown + _STRESS_DRAWDOWN_MARGIN,
            )
        )
        _log.info(
            "stress_scenario",
            scenario=scenario,
            realized_vol=outcomes[-1].realized_vol,
            max_drawdown=outcomes[-1].max_drawdown,
            passed=outcomes[-1].passed,
        )
    return tuple(outcomes)


__all__ = [
    "DayDecision",
    "ReplayResult",
    "StressOutcome",
    "StressScenario",
    "block_bootstrap_panel",
    "execute_day",
    "replay",
    "run_stress_suite",
    "step_day",
    "stress_transform",
]
