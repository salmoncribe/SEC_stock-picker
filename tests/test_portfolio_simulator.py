"""Simulator tests: three leak canaries, a hand-computed golden, and the stress harness.

The canaries are the reason this file exists, so each is built to be *capable of
failing*:

  * the covariance canary mutates day *t* and asserts the weights do not move --
    but it also asserts the mutation reached the day (the fills changed) and
    that the *same* mutation one row earlier DOES move the weights. Without
    those two controls a bit-identical weight vector could just mean the edit
    never arrived.
  * the shuffle canary plants an edge that is only reachable if the signal is
    read on the right date, and shuffles by a **derangement** so no signal can
    keep its own date. A shuffle with fixed points is not a shuffle.
  * the t+1 canary asserts both halves: nothing fills on the availability date,
    and the fill on the next session is at that session's open.

Panels are built by hand with ``adj_close == close`` so adjusted and raw space
coincide -- the ledger prices are then exactly the numbers the test wrote --
and ``open`` is the previous session's close so a t+1 fill price is nameable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from market_intelligence.analytics.backtest import BacktestSignal
from market_intelligence.analytics.backtest_data import AdmittedCell
from market_intelligence.config import PortfolioConfig
from market_intelligence.portfolio import simulator
from market_intelligence.portfolio.account import AccountState, Position, Side, genesis
from market_intelligence.portfolio.costs import CostModel, fill_price
from market_intelligence.portfolio.feed import FeedBundle, MarketPanel, build_panel
from market_intelligence.portfolio.optimizer import apply_no_trade_band
from market_intelligence.portfolio.risk import (
    blend_cov,
    commercial_clusters,
    ewma_cov,
    graph_block_target,
    ledoit_wolf,
    nearest_psd,
    targeting_cov,
    vol_target_scale,
)
from market_intelligence.portfolio.simulator import (
    _returns_block,
    block_bootstrap_panel,
    replay,
    run_stress_suite,
    step_day,
    stress_transform,
)
from market_intelligence.portfolio.views import View
from market_intelligence.signals.trade_plan import Bar

# --------------------------------------------------------------------------- #
# fixtures, built by hand -- nothing here opens a database                      #
# --------------------------------------------------------------------------- #

_HORIZON = 3


def _cell(horizon: int = _HORIZON) -> AdmittedCell:
    return AdmittedCell(
        event_type="insider_transaction",
        event_subtype="purchase",
        horizon_days=horizon,
        direction=1,
        mean_car=0.06,
        hit_rate=0.62,
        n_clusters=40,
    )


def _sessions(count: int, start: date = date(2024, 1, 2)) -> list[date]:
    """``count`` weekday sessions from ``start``."""
    days: list[date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _panel_from_closes(symbols: tuple[str, ...], days: list[date], closes: np.ndarray):
    """Panel whose open is the previous session's close.

    ``adj_close == close`` on every bar, so ``build_panel``'s adjustment factor
    is exactly 1.0 and the raw prices the ledger records are the numbers written
    here. A non-finite close means the symbol simply has no bar that day.
    """
    bars: dict[str, list[Bar]] = {}
    for col, symbol in enumerate(symbols):
        rows: list[Bar] = []
        previous: float | None = None
        for index, day in enumerate(days):
            close = float(closes[index, col])
            if not math.isfinite(close):
                continue
            open_ = previous if previous is not None else close
            rows.append(
                Bar(
                    date=day,
                    open=open_,
                    high=max(open_, close),
                    low=min(open_, close),
                    close=close,
                    adj_close=close,
                )
            )
            previous = close
        bars[symbol] = rows
    return build_panel(bars, symbols)


def _bundle(
    symbols: tuple[str, ...],
    days: list[date],
    closes: np.ndarray,
    signals: list[BacktestSignal],
    *,
    horizon: int = _HORIZON,
    edge: float = 0.05,
) -> FeedBundle:
    by_day: dict[date, list[BacktestSignal]] = defaultdict(list)
    for signal in signals:
        by_day[signal.available_on].append(signal)
    cell = _cell(horizon)
    return FeedBundle(
        panel=_panel_from_closes(symbols, days, closes),
        signals_by_day=dict(by_day),
        cells=[cell],
        hedged_edges={cell.key: edge},
        betas={},
        edges=(),
        spy_index=None,
    )


def _config(**overrides: object) -> PortfolioConfig:
    base: dict[str, object] = {
        "rebalance_days": 1,
        "no_trade_band": 0.0,
        "starting_cash": 10_000.0,
    }
    base.update(overrides)
    return PortfolioConfig(**base)  # type: ignore[arg-type]


def _costs(config: PortfolioConfig) -> CostModel:
    return CostModel(
        slippage_bps=config.slippage_bps, commission_per_share=config.commission_per_share
    )


def _signal(symbol: str, day: date, *, horizon: int = _HORIZON) -> BacktestSignal:
    return BacktestSignal(
        symbol=symbol,
        available_on=day,
        direction=1,
        predicted_move=0.06,
        horizon_days=horizon,
        confidence=90,
    )


def _drift_closes(n_days: int, n_symbols: int, *, base: float, step: float) -> np.ndarray:
    """Deterministic compounding drift -- constant returns, so sample vol is 0."""
    closes = np.empty((n_days, n_symbols), dtype=np.float64)
    for col in range(n_symbols):
        closes[:, col] = base * (1.0 + step) ** np.arange(n_days)
    return closes


def _gross(state: AccountState, prices: dict[str, float]) -> float:
    """Market value of the holdings, in dollars."""
    return sum(position.shares * prices[symbol] for symbol, position in state.positions.items())


# --------------------------------------------------------------------------- #
# 1. t+1 execution                                                             #
# --------------------------------------------------------------------------- #
def test_a_signal_available_on_day_t_fills_at_the_next_session_open() -> None:
    """A signal knowable on day *t* may not trade until *t+1*, and at *t+1*'s open.

    Both halves are asserted. Only checking that a fill exists on *t+1* would
    pass a simulator that filled on *t* as well.
    """
    days = _sessions(8)
    symbols = ("AAA", "BBB")
    closes = np.column_stack(
        [
            100.0 * (1.01 ** np.arange(8)),
            np.full(8, 50.0) + np.arange(8) * 0.1,
        ]
    )
    bundle = _bundle(symbols, days, closes, [_signal("AAA", days[3])])
    config = _config()

    result = replay(bundle, config)

    assert not [fill for fill in result.fills if fill.as_of <= days[3]]
    entries = [fill for fill in result.fills if fill.as_of == days[4]]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.symbol == "AAA"
    assert entry.side is Side.BUY
    # day 4's open is day 3's close, by construction of the panel
    assert entry.price == pytest.approx(
        fill_price(float(closes[3, 0]), is_buy=True, model=_costs(config)), abs=1e-12
    )

    by_day = {decision.as_of: decision for decision in result.decisions}
    assert by_day[days[3]].views == ()  # not yet knowable
    assert len(by_day[days[4]].views) == 1


# --------------------------------------------------------------------------- #
# 2. covariance leak canary                                                    #
# --------------------------------------------------------------------------- #
def _scaled_row(matrix: np.ndarray, row: int, factor: float) -> np.ndarray:
    out = matrix.copy()
    out[row] = out[row] * factor
    return out


def _mutate_day(bundle: FeedBundle, row: int, factor: float) -> FeedBundle:
    """Scale every price on one calendar row. Nothing else changes."""
    panel = bundle.panel
    mutated = MarketPanel(
        calendar=panel.calendar,
        symbols=panel.symbols,
        open=_scaled_row(panel.open, row, factor),
        high=_scaled_row(panel.high, row, factor),
        low=_scaled_row(panel.low, row, factor),
        close=_scaled_row(panel.close, row, factor),
        raw_close=_scaled_row(panel.raw_close, row, factor),
    )
    return replace(bundle, panel=mutated)


def test_mutating_day_t_cannot_move_day_t_weights_but_day_t_minus_one_must() -> None:
    """The covariance window ends STRICTLY before the decision day.

    Risk aversion is raised so the optimum is interior rather than pinned to the
    position cap: a weight sitting on a cap is insensitive to the covariance by
    construction, and a canary that cannot move is not a canary. The test
    asserts the interior-ness, asserts the day-*t* edit was genuinely visible
    (it changed the fills), and asserts the identical edit one row earlier moves
    the weights -- which is what proves a one-row shift is detectable at all.
    """
    n_days = 14
    days = _sessions(n_days)
    symbols = ("AAA", "BBB", "CCC")
    rng = np.random.default_rng(20260731)
    steps = rng.normal(0.0, 0.012, size=(n_days, len(symbols)))
    closes = 100.0 * np.cumprod(1.0 + steps, axis=0)
    signals = [_signal(symbol, days[6], horizon=6) for symbol in symbols]
    bundle = _bundle(symbols, days, closes, signals, horizon=6)
    config = _config(risk_aversion=5000.0, max_position=1.0, max_cluster=1.0)
    costs = _costs(config)

    warmup = replay(bundle, config, end=days[10])
    state, history = warmup.final_state, warmup.equity_curve

    def decide(feed: FeedBundle):
        return step_day(
            state, feed, config, as_of=days[11], equity_history=history, costs=costs
        )

    base_decision, base_state = decide(bundle)
    same_day_decision, same_day_state = decide(_mutate_day(bundle, 11, 1.4))
    prior_day_decision, _ = decide(_mutate_day(bundle, 10, 1.4))

    # the canary has room to move: no weight is pinned to a bound
    assert base_decision.target_weights.size >= 2
    assert float(base_decision.target_weights.max()) < 0.99
    assert float(base_decision.target_weights.sum()) > 1e-6

    # the day-t edit really did reach the day -- it repriced the execution
    assert same_day_state.cash != base_state.cash

    # ... and yet the decision is bit-identical
    assert np.array_equal(base_decision.target_weights, same_day_decision.target_weights)
    assert base_decision.symbols == same_day_decision.symbols
    assert base_decision.views == same_day_decision.views
    assert base_decision.vol_scale == same_day_decision.vol_scale

    # the same edit one row earlier is inside the window and must be felt
    assert not np.array_equal(base_decision.target_weights, prior_day_decision.target_weights)


# --------------------------------------------------------------------------- #
# 3. shuffle canary                                                            #
# --------------------------------------------------------------------------- #
def _planted_edge_panel(
    symbols: tuple[str, ...], days: list[date], signal_rows: list[int], *, horizon: int
) -> np.ndarray:
    """Flat, zero-drift noise everywhere except a bump only a correct read reaches.

    Symbol *j* jumps 2%/day on the ``horizon`` rows following its own signal
    row. The position is bought at the open of ``row + 1`` (which is row
    ``row``'s close), so those are exactly the rows a correctly-timed trade
    collects.
    """
    n_days, n_symbols = len(days), len(symbols)
    returns = np.where(
        (np.arange(n_days)[:, None] + np.arange(n_symbols)[None, :]) % 2 == 0, 0.001, -0.001
    )
    for col, row in enumerate(signal_rows):
        returns[row + 1 : row + 1 + horizon, col] = 0.02
    closes = np.empty((n_days, n_symbols), dtype=np.float64)
    closes[0] = 100.0
    closes[1:] = 100.0 * np.cumprod(1.0 + returns[1:], axis=0)
    return closes


def _derangement(size: int, seed: int) -> np.ndarray:
    """A permutation with no fixed point. A shuffle that leaves a signal on its
    own date has not shuffled it."""
    rng = np.random.default_rng(seed)
    identity = np.arange(size)
    while True:
        candidate = rng.permutation(size)
        if not np.any(candidate == identity):
            return candidate


def test_shuffling_signal_dates_destroys_the_edge() -> None:
    """An edge that survives a date shuffle was never in the signals."""
    horizon = 4
    symbols = ("S0", "S1", "S2", "S3", "S4")
    signal_rows = [4, 12, 20, 28, 36]
    days = _sessions(44)
    closes = _planted_edge_panel(symbols, days, signal_rows, horizon=horizon)
    config = _config()

    signals = [
        _signal(symbol, days[row], horizon=horizon)
        for symbol, row in zip(symbols, signal_rows, strict=True)
    ]
    truth = replay(_bundle(symbols, days, closes, signals, horizon=horizon), config)
    true_return = float(truth.equity_curve[-1]) / config.starting_cash - 1.0
    assert true_return > 0.03, "the planted edge must be reachable before it can be destroyed"

    shuffled_returns: list[float] = []
    for seed in (1, 7):
        order = _derangement(len(signals), seed)
        shuffled = [
            _signal(symbols[index], days[signal_rows[int(target)]], horizon=horizon)
            for index, target in enumerate(order)
        ]
        run = replay(_bundle(symbols, days, closes, shuffled, horizon=horizon), config)
        shuffled_returns.append(float(run.equity_curve[-1]) / config.starting_cash - 1.0)

    assert all(abs(value) < 0.2 * true_return for value in shuffled_returns), shuffled_returns


# --------------------------------------------------------------------------- #
# 4. hand-computed golden                                                      #
# --------------------------------------------------------------------------- #
def test_golden_ledger_is_hand_computed_to_1e_9() -> None:
    """Every dollar of a 2-symbol, 7-session replay, derived by hand.

    One input comes from the run: the target weight. An interior-point solver
    lands within ~1e-9 of the position cap, not on it, so hard-coding 0.15 would
    be asserting Clarabel's tolerance rather than the simulator's arithmetic.
    The weight is therefore read back (and asserted to be at the cap), and every
    dollar after it -- share count, slipped fill price, cash, and the daily
    ``cash + shares * close`` mark -- is computed in the test.
    """
    days = _sessions(7)
    symbols = ("AAA", "BBB")
    closes = np.column_stack(
        [100.0 * (1.01 ** np.arange(7)), np.full(7, 20.0)]
    )
    bundle = _bundle(symbols, days, closes, [_signal("AAA", days[3])])
    config = _config(no_trade_band=0.02)

    result = replay(bundle, config)

    entry = result.decisions[4]
    assert entry.symbols == ("AAA",)
    weight = float(entry.target_weights[0])
    assert weight == pytest.approx(config.max_position, abs=1e-6)
    assert entry.vol_scale == 1.0

    open_price = float(closes[3, 0])  # day 4's open is day 3's close
    price = open_price * (1.0 + config.slippage_bps / 10_000.0)
    shares = math.floor(weight * config.starting_cash / open_price * 1e4) / 1e4
    cash = config.starting_cash - shares * price

    expected = [
        config.starting_cash,
        config.starting_cash,
        config.starting_cash,
        config.starting_cash,
        cash + shares * float(closes[4, 0]),
        cash + shares * float(closes[5, 0]),
        cash + shares * float(closes[6, 0]),
    ]
    assert result.equity_curve == pytest.approx(expected, abs=1e-9)

    # exactly one trade: the band suppressed the drift-sized rebalances after it
    assert len(result.fills) == 1
    assert result.final_state.cash == pytest.approx(cash, abs=1e-9)
    assert result.final_state.positions["AAA"].shares == pytest.approx(shares, abs=1e-12)


# --------------------------------------------------------------------------- #
# 5. sells settle before buys                                                  #
# --------------------------------------------------------------------------- #
def test_a_buy_funded_only_by_the_days_sale_does_not_overdraw() -> None:
    """The purchase is unaffordable until the sale settles, and it still fills."""
    days = _sessions(8)
    symbols = ("AAA", "BBB")
    closes = np.column_stack([np.full(8, 50.0) + np.arange(8) * 0.05, np.full(8, 20.0)])
    bundle = _bundle(symbols, days, closes, [_signal("BBB", days[3])])
    config = _config()
    costs = _costs(config)

    held = {"AAA": Position(symbol="AAA", shares=200.0, cost_basis=9_000.0)}
    state = AccountState(as_of=days[3], cash=50.0, positions=held, high_water_mark=10_000.0)
    history = np.full(4, 10_000.0)

    decision, after = step_day(
        state, bundle, config, as_of=days[4], equity_history=history, costs=costs
    )

    sides = [order.side for order in decision.orders]
    assert Side.SELL in sides and Side.BUY in sides
    assert sides.index(Side.SELL) < sides.index(Side.BUY), "sells must settle first"

    open_bbb = float(closes[3, 1])
    intended = config.max_position * (50.0 + 200.0 * float(closes[3, 0]))
    assert state.cash < intended, "the buy has to be funded by the sale, or this proves nothing"

    assert "AAA" not in after.positions  # no live view -> horizon exit
    assert after.cash >= 0.0
    assert after.positions["BBB"].shares * open_bbb == pytest.approx(intended, rel=5e-3)


# --------------------------------------------------------------------------- #
# 6. zero-view days ride                                                       #
# --------------------------------------------------------------------------- #
def test_days_with_no_live_view_hold_rather_than_liquidate() -> None:
    """No live view is not a sell signal; it is the absence of a decision."""
    days = _sessions(12)
    symbols = ("AAA", "BBB")
    closes = np.column_stack([np.full(12, 40.0) + np.arange(12) * 0.02, np.full(12, 15.0)])
    bundle = _bundle(symbols, days, closes, [_signal("AAA", days[3])])
    config = _config(no_trade_band=0.05)

    result = replay(bundle, config)

    assert [fill.side for fill in result.fills] == [Side.BUY]
    entry_shares = result.fills[0].shares
    assert result.final_state.positions["AAA"].shares == pytest.approx(entry_shares, abs=1e-12)

    # once the horizon has elapsed there are no views left at all, and the
    # account rides: no optimizer run, no orders.
    tail = [decision for decision in result.decisions if decision.as_of > days[8]]
    assert tail, "the panel must outlive the horizon for this to test anything"
    assert all(decision.views == () for decision in tail)
    assert all(decision.rebalanced is False for decision in tail)
    assert all(decision.orders == () for decision in tail)


# --------------------------------------------------------------------------- #
# 7. delisting                                                                 #
# --------------------------------------------------------------------------- #
def test_a_symbol_that_goes_nan_forever_is_closed_at_its_last_known_price() -> None:
    """A delisting books a real exit; an unrelated mid-panel gap does not."""
    n_days = 20
    days = _sessions(n_days)
    symbols = ("AAA", "BBB", "CCC")
    closes = np.column_stack(
        [
            np.full(n_days, 60.0) + np.arange(n_days) * 0.1,
            np.full(n_days, 25.0),
            np.full(n_days, 12.0),
        ]
    )
    closes[10:, 0] = np.nan  # AAA never trades again
    closes[6:9, 2] = np.nan  # CCC simply misses three sessions
    bundle = _bundle(symbols, days, closes, [_signal("AAA", days[3])])
    config = _config(no_trade_band=0.05)

    result = replay(bundle, config)  # must not raise on the NaN gaps

    exits = [fill for fill in result.fills if fill.side is Side.SELL and fill.symbol == "AAA"]
    assert len(exits) == 1
    last_known = float(closes[9, 0])
    assert exits[0].price == pytest.approx(
        fill_price(last_known, is_buy=False, model=_costs(config)), abs=1e-12
    )
    assert exits[0].as_of == days[14]  # five consecutive missing sessions
    assert "AAA" not in result.final_state.positions
    assert result.final_state.cash == pytest.approx(result.equity_curve[-1], abs=1e-9)


# --------------------------------------------------------------------------- #
# 8. the governor fires, and the target gross is REACHED                       #
# --------------------------------------------------------------------------- #
def _crash_bundle() -> tuple[FeedBundle, list[date], np.ndarray, tuple[str, ...]]:
    n_days, horizon = 30, 6
    days = _sessions(n_days)
    symbols = tuple(f"S{index}" for index in range(8))
    closes = _drift_closes(n_days, len(symbols), base=100.0, step=0.001)
    closes[25:] = closes[25:] * 0.75  # one 25% gap down, sustained
    signals = [
        _signal(symbol, days[row], horizon=horizon)
        for row in (2, 7, 12, 17, 22)
        for symbol in symbols
    ]
    return _bundle(symbols, days, closes, signals, horizon=horizon), days, closes, symbols


def test_governor_flattens_on_a_crash_and_the_target_gross_is_actually_reached() -> None:
    """A >15% drawdown must not merely *request* zero gross -- it must get it.

    Two independent mechanisms now guarantee it, and the control below isolates
    the second:

      * a full exit is never banded at all (``optimizer.apply_no_trade_band``),
        so a *flatten* reaches zero however wide the band is; and
      * ``governor_bypasses_band`` (Michael, 2026-07-31), which is what covers
        the governor's *partial* de-risk -- halving gross trims positions rather
        than closing them, so the exit exemption does not apply and only the
        bypass keeps a wide band from swallowing the sells.

    The control therefore exercises a halved day, not a flattened one. Asserting
    the bypass on a flatten would pass for the wrong reason now that the exit
    exemption also covers it.
    """
    bundle, days, closes, symbols = _crash_bundle()
    config = _config()

    before = replay(bundle, config, end=days[24])
    prices = {symbol: float(closes[24, index]) for index, symbol in enumerate(symbols)}
    invested = _gross(before.final_state, prices)
    assert invested / float(before.equity_curve[-1]) > 0.9, "the account must be at risk first"

    result = replay(bundle, config)
    scales = [decision.governor_scale for decision in result.decisions]
    assert 0.0 in scales, "a 25% drawdown must flatten the book"
    flatten = next(d for d in result.decisions if d.governor_scale == 0.0)
    assert float(flatten.target_weights.sum()) == pytest.approx(0.0, abs=1e-12)

    assert result.final_state.positions == {}
    assert result.final_state.cash == pytest.approx(result.equity_curve[-1], abs=1e-9)

    # A wide band cannot stop a flatten: the exit exemption covers it whether
    # or not the bypass is on. Both settings must reach zero gross.
    crashed = replay(bundle, config, end=days[25])
    assert crashed.final_state.positions, "the control needs something to sell"
    for bypass in (False, True):
        settings = _config(no_trade_band=0.5, governor_bypasses_band=bypass)
        _, flattened = step_day(
            crashed.final_state,
            bundle,
            settings,
            as_of=days[26],
            equity_history=crashed.equity_curve,
            costs=_costs(settings),
        )
        assert flattened.positions == {}, (
            f"a full exit must not be banded (bypass={bypass}): a position too small to "
            "sell is a position held forever"
        )

    # ...but a PARTIAL de-risk is a trim, not an exit, so only the bypass saves
    # it from a wide band. This is the leg Michael's decision actually governs.
    halved = np.array([0.5, 0.5])
    held = np.array([0.6, 0.6])
    assert apply_no_trade_band(halved, held, band=0.5, bypass=False) == pytest.approx(held), (
        "without the bypass a wide band swallows the governor's partial sells"
    )
    assert apply_no_trade_band(halved, held, band=0.5, bypass=True) == pytest.approx(halved), (
        "with the bypass the governor's partial de-risk executes in full"
    )


# --------------------------------------------------------------------------- #
# 8b. the governor is not an absorbing state -- and is still a governor         #
# --------------------------------------------------------------------------- #
def _crash_then_recover_bundle(
    n_days: int = 46,
) -> tuple[FeedBundle, list[date], np.ndarray, tuple[str, ...]]:
    """One sustained 25% gap down, then a steady climb back.

    Signals arrive on *every* session from row 2 on, so the moment the governor
    permits gross the optimizer has something to buy. Without that, "the account
    never re-invests" would be a statement about the signal feed rather than
    about the governor, and the test would prove nothing.
    """
    horizon = 6
    days = _sessions(n_days)
    symbols = tuple(f"S{index}" for index in range(8))
    closes = _drift_closes(n_days, len(symbols), base=100.0, step=0.001)
    closes[25:] = closes[25:] * 0.75
    closes[25:] = closes[25:] * (1.004 ** np.arange(n_days - 25))[:, None]
    signals = [
        _signal(symbol, days[row], horizon=horizon)
        for row in range(2, n_days - 1)
        for symbol in symbols
    ]
    return _bundle(symbols, days, closes, signals, horizon=horizon), days, closes, symbols


def test_a_flattened_account_wins_its_gross_back_instead_of_dying_flat() -> None:
    """The absorbing state is gone.

    Measured on the real replay before this fix: the account flattened on
    2020-03-12 at a 18.97% drawdown and then moved on 0 of the following 1,603
    sessions, because a flat account cannot make a new high, so the high-water
    mark could not move, so the drawdown stayed pinned above the flatten
    threshold forever. Recovery required being invested and the governor
    forbade being invested.

    The decaying mark closes that loop: while equity sits below the mark, the
    mark drifts toward it, so the drawdown heals even in cash. A five-session
    half-life is used so the panel stays hand-checkable -- with the shipped 63
    it is the same arithmetic over ~60 sessions.
    """
    bundle, _days, _closes, _symbols = _crash_then_recover_bundle()
    config = _config(hwm_decay_halflife_days=5)

    result = replay(bundle, config)
    scales = [decision.governor_scale for decision in result.decisions]

    assert 0.0 in scales, "the crash must actually flatten the book first"
    flattened_on = scales.index(0.0)
    # The account really is in cash, which is the state the ratchet could not
    # leave -- equity is constant from here until the governor lets it move.
    flat_state_days = [
        index
        for index, decision in enumerate(result.decisions)
        if index > flattened_on and not decision.orders and decision.governor_scale == 0.0
    ]
    assert flat_state_days, "the flatten must persist for at least a session"

    rearmed = [index for index in range(flattened_on, len(scales)) if scales[index] > 0.0]
    assert rearmed, "the governor never returned any gross at all: still absorbing"
    assert rearmed[0] - flattened_on <= 12, (
        f"took {rearmed[0] - flattened_on} sessions to re-arm off a five-day half-life"
    )
    assert max(scales[rearmed[0] :]) == 1.0, "and it must reach full gross, not just half"

    # ...and the account actually puts money back to work afterwards.
    assert result.final_state.positions, "re-armed but never re-invested"
    assert result.equity_curve[-1] > result.equity_curve[rearmed[0]], (
        "it re-entered a recovering tape and still made nothing"
    )


def test_a_steady_bleed_never_wins_full_gross_back() -> None:
    """The safety rail survives the fix. This is the more important half.

    A fix that always re-enters is exactly as broken as one that never does, so
    the panel here grinds down 4% a session -- deliberately harsher than any
    real tape, so the assertion cannot be marginal. The threshold the decay has
    to lose against is ``1 - 0.5 ** (1/63) = 1.09%`` a session: at the halved
    gross the governor permits, the account still bleeds ~2% a session, so the
    gap to the mark widens faster than the mark can forget it and the drawdown
    keeps growing. The governor therefore oscillates between halved and flat and
    never buys its way back to full gross.
    """
    n_days, horizon = 70, 6
    days = _sessions(n_days)
    symbols = tuple(f"S{index}" for index in range(6))
    closes = _drift_closes(n_days, len(symbols), base=100.0, step=0.001)
    closes = closes * (0.96 ** np.maximum(np.arange(n_days) - 10, 0))[:, None]
    signals = [
        _signal(symbol, days[row], horizon=horizon)
        for row in range(2, n_days - 1)
        for symbol in symbols
    ]
    bundle = _bundle(symbols, days, closes, signals, horizon=horizon)

    result = replay(bundle, _config())  # the shipped 63-session half-life
    scales = [decision.governor_scale for decision in result.decisions]

    assert min(scales) < 1.0, "the governor must fire at all for this to mean anything"
    first_derisk = next(index for index, scale in enumerate(scales) if scale < 1.0)
    assert max(scales[first_derisk:]) < 1.0, (
        "the mark forgot faster than the account bled: the rail is disabled"
    )
    # The damage is bounded too -- the account is not simply riding it down.
    assert float(result.equity_curve[-1]) > 0.5 * float(result.equity_curve[:first_derisk].max())


# --------------------------------------------------------------------------- #
# 9. fractional vs integer share modes                                         #
# --------------------------------------------------------------------------- #
def test_share_mode_changes_the_quantities_and_nothing_else() -> None:
    """Integer mode measures rounding error, so only the quantities may differ.

    Compared at the level of a single decision from an identical state. Over a
    full replay the two must diverge -- that divergence *is* the rounding error
    integer mode exists to measure -- so equality is asserted where the claim is
    actually true.
    """
    days = _sessions(8)
    symbols = ("AAA", "BBB")
    closes = np.column_stack(
        [47.31 * (1.004 ** np.arange(8)), 63.17 * (1.002 ** np.arange(8))]
    )
    bundle = _bundle(symbols, days, closes, [_signal("AAA", days[3])])

    fractional = _config(fractional_shares=True)
    integral = _config(fractional_shares=False)
    warmup = replay(bundle, fractional, end=days[3])

    def decide(config: PortfolioConfig):
        return step_day(
            warmup.final_state,
            bundle,
            config,
            as_of=days[4],
            equity_history=warmup.equity_curve,
            costs=_costs(config),
        )

    frac_decision, frac_state = decide(fractional)
    int_decision, int_state = decide(integral)

    assert np.array_equal(frac_decision.target_weights, int_decision.target_weights)
    assert frac_decision.symbols == int_decision.symbols
    assert frac_decision.views == int_decision.views
    assert frac_decision.optimizer_status == int_decision.optimizer_status
    assert frac_decision.governor_scale == int_decision.governor_scale
    assert frac_decision.vol_scale == int_decision.vol_scale
    assert frac_decision.rebalanced == int_decision.rebalanced
    assert [(o.symbol, o.side) for o in frac_decision.orders] == [
        (o.symbol, o.side) for o in int_decision.orders
    ]

    frac_shares = frac_state.positions["AAA"].shares
    int_shares = int_state.positions["AAA"].shares
    assert int_shares == float(int(int_shares))
    assert frac_shares != int_shares
    assert abs(frac_shares - int_shares) < 1.0


# --------------------------------------------------------------------------- #
# 10. stress transforms actually transform                                     #
# --------------------------------------------------------------------------- #
def _returns_of(panel) -> np.ndarray:
    return panel.close[1:] / panel.close[:-1] - 1.0


def _stress_bundle() -> FeedBundle:
    n_days = 40
    days = _sessions(n_days)
    symbols = ("AAA", "BBB", "CCC")
    rng = np.random.default_rng(11)
    steps = rng.normal(0.0004, 0.011, size=(n_days, len(symbols)))
    closes = 100.0 * np.cumprod(1.0 + steps, axis=0)
    signals = [_signal("AAA", days[5]), _signal("BBB", days[15]), _signal("CCC", days[25])]
    return _bundle(symbols, days, closes, signals)


def test_stress_transforms_are_verified_not_assumed() -> None:
    """Doubled vol doubles vol, forced correlation forces it, and the bootstrap
    keeps each sampled day's cross-section intact."""
    bundle = _stress_bundle()
    original = _returns_of(bundle.panel)

    doubled = _returns_of(stress_transform(bundle, "vol_x2").panel)
    assert doubled.std(axis=0, ddof=1) == pytest.approx(
        2.0 * original.std(axis=0, ddof=1), rel=1e-9
    )

    forced = _returns_of(stress_transform(bundle, "corr_one").panel)
    correlation = np.corrcoef(forced, rowvar=False)
    off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]
    assert float(off_diagonal.min()) > 0.999
    assert float(np.corrcoef(original, rowvar=False)[0, 1]) < 0.9

    # the cross-section: a panel whose symbols are exact linear images of each
    # other. Resampling per symbol independently would break these identities;
    # resampling whole dates cannot.
    n_days = 30
    days = _sessions(n_days)
    symbols = ("X", "Y", "Z")
    base = np.arange(1, n_days) * 1e-4
    returns = np.column_stack([base, -base, 2.0 * base])
    closes = np.empty((n_days, 3), dtype=np.float64)
    closes[0] = 100.0
    closes[1:] = 100.0 * np.cumprod(1.0 + returns, axis=0)
    linked = _bundle(symbols, days, closes, [])

    sampled = _returns_of(block_bootstrap_panel(linked, block_size=5, seed=3).panel)
    assert sampled.shape == (n_days - 1, 3)
    assert sampled[:, 1] == pytest.approx(-sampled[:, 0], abs=1e-12)
    assert sampled[:, 2] == pytest.approx(2.0 * sampled[:, 0], abs=1e-12)
    # and it really resampled -- the output is not just the input back
    assert not np.allclose(sampled[:, 0], returns[:, 0])

    outcomes = run_stress_suite(bundle, _config(), scenarios=("vol_x2",))
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.scenario == "vol_x2"
    assert outcome.passed == (
        outcome.vol_within_target and outcome.drawdown_within_halt and outcome.governor_fired
    )


# --------------------------------------------------------------------------- #
# 11. resume continuity                                                        #
# --------------------------------------------------------------------------- #
def test_a_split_replay_equals_a_whole_one() -> None:
    """1..N in one call == 1..K then K+1..N from the persisted state.

    This is the ``portfolio step`` contract. The per-day equity is re-derived
    here from ``cash + sum(shares * close)`` rather than read back from the
    simulator, so the equity identity is checked independently on every stepped
    day as a side effect.
    """
    n_days = 16
    days = _sessions(n_days)
    symbols = ("AAA", "BBB")
    rng = np.random.default_rng(5)
    steps = rng.normal(0.0005, 0.009, size=(n_days, len(symbols)))
    closes = 100.0 * np.cumprod(1.0 + steps, axis=0)
    signals = [_signal("AAA", days[3]), _signal("BBB", days[8])]
    bundle = _bundle(symbols, days, closes, signals)
    config = _config()
    costs = _costs(config)

    whole = replay(bundle, config)

    cut = 9
    part = replay(bundle, config, end=days[cut])
    state = part.final_state
    curve = list(part.equity_curve)
    for index in range(cut + 1, n_days):
        _, state = step_day(
            state,
            bundle,
            config,
            as_of=days[index],
            equity_history=np.asarray(curve, dtype=np.float64),
            costs=costs,
        )
        prices = {symbol: float(closes[index, col]) for col, symbol in enumerate(symbols)}
        curve.append(state.cash + _gross(state, prices))

    assert np.asarray(curve) == pytest.approx(whole.equity_curve, abs=1e-9)
    assert state.cash == pytest.approx(whole.final_state.cash, abs=1e-9)
    assert set(state.positions) == set(whole.final_state.positions)
    for symbol, position in state.positions.items():
        assert position.shares == pytest.approx(
            whole.final_state.positions[symbol].shares, abs=1e-12
        )


# --------------------------------------------------------------------------- #
# 12. volatility targeting engages -- and does not touch portfolio construction #
# --------------------------------------------------------------------------- #
_REGIME_CALM_SESSIONS = 251
_REGIME_VIOLENT_SESSIONS = 9
_REGIME_SESSIONS = _REGIME_CALM_SESSIONS + _REGIME_VIOLENT_SESSIONS
_REGIME_HORIZON = 6
#: Last calm decision day, and the first day whose window holds the whole shift.
_CALM_ROW = 248
_SHIFTED_ROW = _REGIME_SESSIONS - 1


def _regime_closes(*, calm_vol: float = 0.002, violent_vol: float = 0.015) -> np.ndarray:
    """One common factor plus a little idiosyncratic noise, volatility stepping up.

    A shared factor (correlation ~0.96) is deliberate: it makes predicted
    portfolio volatility roughly the factor's volatility *whatever* the
    optimizer allocates, so this test measures the estimator rather than
    measuring how three independent names happened to diversify. The step is
    0.2%/day (3.2% a year) to 1.5%/day (24% a year) -- a real regime change,
    and nowhere near the 50%/yr SPY printed in March 2020.
    """
    rng = np.random.default_rng(20260731)
    vols = np.array(
        [calm_vol] * _REGIME_CALM_SESSIONS + [violent_vol] * _REGIME_VIOLENT_SESSIONS
    )
    factor = rng.normal(0.0, 1.0, size=_REGIME_SESSIONS) * vols
    idiosyncratic = rng.normal(0.0, 1.0, size=(_REGIME_SESSIONS, 3)) * vols[:, None] * 0.2
    returns = factor[:, None] + idiosyncratic
    returns[0] = 0.0
    return 100.0 * np.cumprod(1.0 + returns, axis=0)


def _regime_bundle() -> tuple[FeedBundle, list[date]]:
    days = _sessions(_REGIME_SESSIONS, start=date(2023, 1, 2))
    symbols = ("AAA", "BBB", "CCC")
    signals = [
        _signal(symbol, days[row], horizon=_REGIME_HORIZON)
        for row in (_CALM_ROW - 5, _SHIFTED_ROW - 5)
        for symbol in symbols
    ]
    bundle = _bundle(symbols, days, _regime_closes(), signals, horizon=_REGIME_HORIZON)
    return bundle, days


def _regime_config(**overrides: object) -> PortfolioConfig:
    """Caps opened up so the *volatility* budget is what binds, not a weight cap.

    A book pinned to ``max_position`` would hold its weights whatever the
    covariance said, which would make a vol-targeting test unable to fail.
    """
    base: dict[str, object] = {
        "max_position": 1.0,
        "max_cluster": 1.0,
        "turnover_penalty_bps": 0.0,
        "cov_lookback": 252,
    }
    base.update(overrides)
    return _config(**base)


def _decide_regime_day(
    bundle: FeedBundle, config: PortfolioConfig, days: list[date], row: int
):
    """One decision from a flat, at-high-water account -- so the governor is idle.

    Starting from genesis on the day itself keeps the drawdown at zero, which
    isolates ``vol_scale``: a governor cut would also shrink gross and the test
    could not say which control did it.
    """
    as_of = days[row]
    return step_day(
        genesis(config.starting_cash, as_of),
        bundle,
        config,
        as_of=as_of,
        equity_history=np.array([config.starting_cash], dtype=np.float64),
        costs=_costs(config),
    )[0]


def _blend_the_optimizer_reads(
    bundle: FeedBundle, config: PortfolioConfig, symbols: tuple[str, ...], as_of: date
) -> tuple[np.ndarray, np.ndarray]:
    """``(block, blended covariance)`` for ``as_of``, spelled out from risk primitives.

    Written from the public functions rather than imported from the simulator
    so this file states the covariance it expects instead of asking the module
    under test what it thinks it produced.
    """
    block, _counts = _returns_block(
        bundle.panel, symbols, as_of=as_of, lookback=config.cov_lookback
    )
    clusters = commercial_clusters(symbols, bundle.edges, as_of=as_of)
    shrunk, _delta = ledoit_wolf(block, graph_block_target(block, symbols, clusters))
    blended = nearest_psd(
        blend_cov(
            shrunk, ewma_cov(block, lam=config.ewma_lambda), lw_weight=config.lw_blend
        )
    )
    return block, blended


def test_volatility_targeting_engages_on_a_regime_shift_instead_of_staying_at_one() -> None:
    """The defect this closes: ``vol_scale`` was 1.000 through the whole COVID crash.

    Both halves are asserted, because "it cut" alone would pass a simulator
    that cuts every day for unrelated reasons: the calm day before the shift
    must sit at exactly 1.0, and the day whose window holds the shift must cut.

    The control at the bottom is what makes this a regression test rather than
    a description. On the *same* weights and the *same* window, the blended
    covariance the optimizer reads still predicts volatility under target and
    would not have cut at all -- which is exactly the 1.5%-of-sessions
    behaviour measured on ten years of real data.
    """
    bundle, days = _regime_bundle()
    config = _regime_config()

    calm = _decide_regime_day(bundle, config, days, _CALM_ROW)
    shifted = _decide_regime_day(bundle, config, days, _SHIFTED_ROW)

    assert calm.rebalanced and shifted.rebalanced
    assert calm.optimizer_status == shifted.optimizer_status == "clarabel"
    assert calm.governor_scale == shifted.governor_scale == 1.0  # not the governor

    assert calm.vol_scale == 1.0
    assert shifted.vol_scale < 1.0
    assert float(np.abs(shifted.target_weights).sum()) < float(
        np.abs(calm.target_weights).sum()
    )

    # ... and the slow estimate, on those same weights, would have done nothing.
    raw_weights = shifted.target_weights / shifted.vol_scale
    block, blended = _blend_the_optimizer_reads(
        bundle, config, shifted.symbols, days[_SHIFTED_ROW]
    )
    slow_scale = vol_target_scale(
        raw_weights, blended, target_vol=config.target_vol, periods_per_year=252
    )
    assert slow_scale == 1.0

    fast = targeting_cov(block, lam=config.vol_estimate_lambda)
    assert float(np.sqrt(raw_weights @ fast @ raw_weights * 252)) > config.target_vol
    assert float(np.sqrt(raw_weights @ blended @ raw_weights * 252)) < config.target_vol


def test_the_scaler_changed_but_portfolio_construction_did_not(monkeypatch) -> None:
    """A "risk fix" that silently alters which stocks get bought is not a risk fix.

    Three things are pinned. The covariance handed to ``optimize`` is the
    blend, byte for byte, and is *not* the fast estimate. The weights
    ``optimize`` returns do not move when ``vol_estimate_lambda`` moves --
    proved on the shifted day, where the scaler demonstrably does something
    different under the two lambdas. And on the calm day, where neither lambda
    binds, the target weights themselves come out identical.

    0.97 is the comparator because it is the blend's own EWMA leg: if the
    scaler had been left reading the blend, this is roughly the speed it would
    have had, so "0.94 cuts strictly more than 0.97" is the change measured
    against the thing it replaced.
    """
    bundle, days = _regime_bundle()

    def run(lam: float, row: int):
        config = _regime_config(vol_estimate_lambda=lam)
        seen: list[tuple[np.ndarray, np.ndarray]] = []
        real = simulator.optimize

        def spy(mu, sigma, *args, **kwargs):
            result = real(mu, sigma, *args, **kwargs)
            seen.append(
                (
                    np.array(sigma, dtype=np.float64, copy=True),
                    np.array(result.weights, dtype=np.float64, copy=True),
                )
            )
            return result

        monkeypatch.setattr(simulator, "optimize", spy)
        decision = _decide_regime_day(bundle, config, days, row)
        monkeypatch.setattr(simulator, "optimize", real)
        assert len(seen) == 1
        return decision, seen[0]

    fast_decision, (fast_sigma, fast_weights) = run(0.94, _SHIFTED_ROW)
    slow_decision, (slow_sigma, slow_weights) = run(0.97, _SHIFTED_ROW)

    # the lambda genuinely reached the scaler
    assert fast_decision.vol_scale < slow_decision.vol_scale < 1.0

    # ...and reached nothing else
    assert np.array_equal(fast_sigma, slow_sigma)
    assert np.array_equal(fast_weights, slow_weights)

    config = _regime_config()
    block, blended = _blend_the_optimizer_reads(
        bundle, config, fast_decision.symbols, days[_SHIFTED_ROW]
    )
    # The blend, restated over the view horizon. Portfolio construction happens
    # in horizon space (``simulator._horizon_covariance``) because the views are
    # horizon-cumulative returns; every name here carries the same horizon, so
    # the scaling is the scalar ``_REGIME_HORIZON``. What this assertion is
    # for is unchanged: the optimizer reads the *blend*, never the scaler's
    # fast estimate.
    assert np.array_equal(fast_sigma, blended * _REGIME_HORIZON)
    assert not np.allclose(fast_sigma, targeting_cov(block, lam=config.vol_estimate_lambda))

    # On a calm panel nothing scales, so the shipped weights match outright.
    calm_fast, _ = run(0.94, _CALM_ROW)
    calm_slow, _ = run(0.97, _CALM_ROW)
    assert calm_fast.vol_scale == calm_slow.vol_scale == 1.0
    assert calm_fast.symbols == calm_slow.symbols
    assert np.array_equal(calm_fast.target_weights, calm_slow.target_weights)


# --------------------------------------------------------------------------- #
# 13. the objective is stated over ONE period                                  #
# --------------------------------------------------------------------------- #
def test_the_risk_term_is_priced_over_the_view_horizon_not_over_one_day() -> None:
    """``mu`` is an H-session return, so the covariance it is priced against
    must cover H sessions too.

    The defect this closes, measured on the ten-year replay's geometry: at the
    solved weights the median objective terms were ``mu'w = 4.4e-2``,
    ``gamma*w'Sw = 3.4e-4`` and ``kappa*||dw|| = 9.4e-4``. Risk was 130x
    smaller than return because ``mu`` was cumulative over the horizon while
    ``S`` was one day, so the solve degenerated into "rank by mu, fill to the
    caps".

    Asserted as an identity rather than as an outcome: the scaling is
    ``sqrt(H_i * H_j)``, which for one shared horizon is the scalar H, and it
    must be exactly that -- not "bigger", which a wrong constant would also
    satisfy.
    """
    sigma = np.array([[4.0e-4, 1.0e-4], [1.0e-4, 9.0e-4]], dtype=np.float64)
    symbols = ("AAA", "BBB")
    views = (
        View(
            symbol="AAA",
            expected_excess_return=0.05,
            confidence=0.9,
            horizon_days=4,
            expires_on=date(2024, 2, 1),
            source="insider",
        ),
        View(
            symbol="BBB",
            expected_excess_return=0.05,
            confidence=0.9,
            horizon_days=9,
            expires_on=date(2024, 2, 1),
            source="insider",
        ),
    )
    scaled = simulator._horizon_covariance(sigma, symbols, views)
    expected = sigma * np.array([[4.0, 6.0], [6.0, 9.0]])  # sqrt(H_i * H_j)
    assert np.allclose(scaled, expected, rtol=0.0, atol=0.0)

    # a congruence by a positive diagonal cannot break positive semidefiniteness
    assert float(np.linalg.eigvalsh(scaled).min()) >= 0.0

    # a name the views do not cover is left alone rather than scaled to zero,
    # which would price it as riskless
    assert np.allclose(simulator._horizon_covariance(sigma, symbols, ()), sigma)


def test_the_horizon_scaled_covariance_is_what_actually_reaches_the_optimizer(
    monkeypatch,
) -> None:
    """The identity above is worthless if ``execute_day`` hands over the daily one."""
    days = _sessions(40)
    symbols = ("AAA", "BBB", "CCC")
    rng = np.random.default_rng(7)
    closes = 50.0 * np.cumprod(1.0 + rng.normal(0.0, 0.012, size=(40, 3)), axis=0)
    horizon = 7
    bundle = _bundle(
        symbols,
        days,
        closes,
        [_signal(symbol, days[30], horizon=horizon) for symbol in symbols],
        horizon=horizon,
    )
    config = _config(cov_lookback=25)

    seen: list[np.ndarray] = []
    real = simulator.optimize

    def spy(mu, sigma, *args, **kwargs):
        seen.append(np.array(sigma, dtype=np.float64, copy=True))
        return real(mu, sigma, *args, **kwargs)

    monkeypatch.setattr(simulator, "optimize", spy)
    decision, _state = step_day(
        genesis(config.starting_cash, days[31]),
        bundle,
        config,
        as_of=days[31],
        equity_history=np.array([config.starting_cash], dtype=np.float64),
        costs=_costs(config),
    )
    assert len(seen) == 1
    assert len(decision.views) == 3

    _block, blended = _blend_the_optimizer_reads(bundle, config, decision.symbols, days[31])
    assert np.allclose(seen[0], blended * horizon)
    assert not np.allclose(seen[0], blended)


# --------------------------------------------------------------------------- #
# 14. turnover is measured, bounded, and damped on purpose                     #
# --------------------------------------------------------------------------- #
#: Sessions, names, and signal cadence of the churn fixture. Sized so the live
#: view set genuinely rotates: a signal every ``_CHURN_STRIDE`` sessions per
#: name, staggered across names, against a ``_CHURN_HORIZON``-session horizon.
_CHURN_SESSIONS = 180
_CHURN_SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
_CHURN_HORIZON = 10
_CHURN_STRIDE = 12
#: Annualized turnover the churn fixture must stay under, on the same both-sides
#: definition ``cli._turnover_series`` uses. Measured at 12.93x with the shipped
#: configuration; the bound carries ~16% headroom so ordinary solver noise
#: cannot trip it while a structural regression still does.
#:
#: The absolute level is high on purpose and is not a target: a 10-session
#: horizon held at ~70% gross costs ``2 * 0.7 * 252/10`` = 35x/yr of round
#: trips before anything damps it. That identity -- turnover is roughly
#: ``2 * gross * 252 / holding_days`` -- is why the account's own 400%/yr
#: figure is unreachable at a 20-session edge, and it is pinned here so a
#: change that quietly lengthens or shortens the holding period is visible.
_CHURN_TURNOVER_BOUND = 15.0


def _churn_bundle() -> tuple[FeedBundle, list[date]]:
    """A fixture whose weights would churn every rebalance if nothing damped them.

    Correlated-but-not-identical names with real dispersion in their returns, so
    the optimizer's ranking genuinely re-sorts between rebalances. A fixture
    whose weights never move could not tell a working damper from a broken one.
    """
    days = _sessions(_CHURN_SESSIONS, start=date(2023, 1, 2))
    rng = np.random.default_rng(20260731)
    size = len(_CHURN_SYMBOLS)
    factor = rng.normal(0.0, 0.008, size=_CHURN_SESSIONS)
    noise = rng.normal(0.0, 0.014, size=(_CHURN_SESSIONS, size))
    returns = factor[:, None] + noise
    returns[0] = 0.0
    closes = 60.0 * np.cumprod(1.0 + returns, axis=0)
    signals = [
        _signal(symbol, days[row], horizon=_CHURN_HORIZON)
        for index, symbol in enumerate(_CHURN_SYMBOLS)
        for row in range(20 + index, _CHURN_SESSIONS - 2, _CHURN_STRIDE)
    ]
    return _bundle(_CHURN_SYMBOLS, days, closes, signals, horizon=_CHURN_HORIZON), days


def _annual_turnover(result) -> float:
    """Traded notional as a fraction of that day's equity, annualized.

    Spelled out rather than imported so this file states the definition it is
    asserting on. Matches ``cli._turnover_series``: BOTH sides of a round trip
    count, which is twice the SEC N-1A convention.
    """
    traded: dict[date, float] = defaultdict(float)
    for fill in result.fills:
        traded[fill.as_of] += abs(fill.shares * fill.price)
    daily = [
        traded.get(result.calendar[row].astype(object), 0.0) / float(result.equity_curve[row])
        for row in range(result.calendar.size)
        if float(result.equity_curve[row]) > 0.0
    ]
    return float(np.mean(daily)) * 252.0


def _churn_config(**overrides: object) -> PortfolioConfig:
    base: dict[str, object] = {"rebalance_days": 5, "no_trade_band": 0.01}
    base.update(overrides)
    return _config(**base)


def test_replay_turnover_on_the_churn_fixture_stays_under_its_pinned_bound() -> None:
    """A regression that re-loosens portfolio construction has to show up here.

    Turnover is the criterion this account has actually failed against, so it
    gets an assertion rather than a report line. The bound is stated, not
    derived from the run: a test that recomputes its own expectation cannot
    fail.
    """
    bundle, _days = _churn_bundle()
    result = replay(bundle, _churn_config())

    assert result.fills, "a fixture that never trades cannot bound turnover"
    assert _annual_turnover(result) < _CHURN_TURNOVER_BOUND


def test_the_band_and_the_turnover_penalty_each_cut_trading_on_their_own() -> None:
    """Both dampers are load-bearing, and each is measured with the other off.

    Measuring them together would let one carry the other: a band that did
    nothing would still pass if the penalty were doing all the work. So the
    baseline turns *both* off, and each is then switched on alone.
    """
    bundle, _days = _churn_bundle()

    loose = replay(bundle, _churn_config(no_trade_band=0.0, turnover_penalty_bps=0.0))
    banded = replay(bundle, _churn_config(no_trade_band=0.03, turnover_penalty_bps=0.0))
    priced = replay(bundle, _churn_config(no_trade_band=0.0, turnover_penalty_bps=250.0))

    assert len(banded.fills) < len(loose.fills)
    assert len(priced.fills) < len(loose.fills)
    assert _annual_turnover(banded) < _annual_turnover(loose)
    assert _annual_turnover(priced) < _annual_turnover(loose)
