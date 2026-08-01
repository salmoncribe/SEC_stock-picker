"""Deterministic portfolio replay for research-only strategy experiments.

This module intentionally accepts already-produced signals.  Signal generation
and portfolio replay are separate so a backtest cannot accidentally use a
future feature while constructing its own labels.

Three choices here exist to keep the replay measuring the same thing the
validation gate measured, rather than something else wearing its name:

* **Exits land on the calibrated horizon, counted in trading days.** A cell's
  verdict is a mean cumulative abnormal return over *H bars*. Counting calendar
  days instead retires a 20-bar position after roughly 14 bars, and
  ``exit_style="target"`` closes it at the mean move -- which clips every winner
  at the average while leaving the full ATR stop to pay for the losers. That
  geometry needs a win rate near 90% just to break even, so it can turn a
  genuinely positive-mean cell into a losing strategy. ``exit_style="horizon"``
  holds the window out and keeps the stop purely as risk control.
* **Beta is hedged, because the label is abnormal return.**
  ``analytics.returns`` grades every event against a market model, so a cell's
  edge is alpha, not direction. Replaying it as naked exposure measures the
  market's drift over the sample far more than it measures the signal --
  decisively so here, where most admitted cells are shorts. With
  ``hedge_symbol`` set, the book carries a beta-weighted offset in that
  instrument, rebalanced daily.
* **Sizing reads the same ATR window the live path reads.**
  ``signals.trade_alerts`` passes the last ``atr_period * 4`` bars, so a replay
  smoothing over ten years of history sizes trades the production system would
  never have taken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from market_intelligence.signals.trade_plan import Bar, build_trade_plan

#: Hold to the calibrated horizon; the stop is risk control, not a profit target.
EXIT_HORIZON = "horizon"
#: Legacy geometry: take profit at the expected move, stop at the ATR multiple.
EXIT_TARGET = "target"

#: Size from the risk budget and the stop distance (``trade_plan``'s own rule).
SIZING_RISK = "risk"
#: Size to a fixed fraction of equity, independent of stop distance.
#:
#: Fixed-fractional risk sizing ties exposure to stop width, which is the right
#: rule when the stop defines the trade. For a horizon-held cell the stop is only
#: a disaster brake, so widening it to stop noise-triggered exits would silently
#: shrink every position toward zero and mute the edge along with the losses.
SIZING_EQUAL_WEIGHT = "equal_weight"


@dataclass(frozen=True)
class BacktestSignal:
    symbol: str
    available_on: date
    direction: int
    predicted_move: float
    horizon_days: int
    confidence: int = 100


@dataclass(frozen=True)
class BacktestConfig:
    starting_equity: float = 10_000.0
    risk_pct_per_trade: float = 1.0
    atr_period: int = 14
    atr_stop_multiple: float = 2.0
    max_position_pct: float = 20.0
    min_confidence: int = 0
    entry_slippage_bps: float = 5.0
    exit_slippage_bps: float = 5.0
    commission_per_share: float = 0.0
    max_positions: int = 10
    #: ``EXIT_HORIZON`` or ``EXIT_TARGET``; see the module docstring.
    exit_style: str = EXIT_HORIZON
    #: Bars of history handed to the sizing model. 0 means the full series;
    #: ``atr_period * 4`` matches the live trade-alert path.
    atr_lookback_bars: int = 56
    #: Ceiling on gross notional as a percentage of equity. 0 disables it, which
    #: lets short proceeds finance an unbounded book.
    max_gross_exposure_pct: float = 200.0
    #: Instrument used to neutralise market exposure. None replays naked
    #: direction, which does not match the abnormal-return label.
    hedge_symbol: str | None = None
    hedge_slippage_bps: float = 1.0
    #: ``SIZING_RISK`` or ``SIZING_EQUAL_WEIGHT``.
    sizing: str = SIZING_RISK
    #: Target exposure per position, as a percentage of equity. Only read under
    #: ``SIZING_EQUAL_WEIGHT``; still bounded by ``max_position_pct``.
    target_weight_pct: float = 2.0


@dataclass(frozen=True)
class BacktestTrade:
    symbol: str
    direction: int
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: int
    gross_pnl: float
    costs: float
    net_pnl: float
    reason: str


@dataclass(frozen=True)
class EquityPoint:
    day: date
    equity: float


@dataclass(frozen=True)
class BacktestResult:
    starting_equity: float
    ending_equity: float
    total_return: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    profit_factor: float
    expectancy: float
    trades: tuple[BacktestTrade, ...]
    equity_curve: tuple[EquityPoint, ...]

    @property
    def objective(self) -> float:
        """Profit objective with a drawdown penalty for parameter search."""
        return self.total_return - 0.75 * self.max_drawdown


@dataclass
class _Position:
    signal: BacktestSignal
    entry_date: date
    entry_price: float
    stop: float
    target: float
    shares: int
    #: Index into the symbol's own bar series, so the horizon counts trading
    #: days rather than calendar days.
    horizon_exit_index: int
    beta: float


@dataclass(frozen=True)
class _Series:
    """One symbol's bars plus a date index, so day lookups are not linear scans."""

    bars: tuple[Bar, ...]
    index: dict[date, int]

    def at(self, day: date) -> Bar | None:
        position = self.index.get(day)
        return None if position is None else self.bars[position]


def _fill(raw: float, direction: int, bps: float, entry: bool) -> float:
    rate = bps / 10_000.0
    if entry:
        return raw * (1.0 + rate) if direction > 0 else raw * (1.0 - rate)
    return raw * (1.0 - rate) if direction > 0 else raw * (1.0 + rate)


def _daily_volatility(curve: list[EquityPoint]) -> float:
    if len(curve) < 2:
        return 0.0
    returns = [curve[i].equity / curve[i - 1].equity - 1.0 for i in range(1, len(curve))]
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / len(returns)
    return math.sqrt(variance)


def _metrics(
    starting: float, ending: float, trades: list[BacktestTrade], curve: list[EquityPoint]
) -> BacktestResult:
    peak = starting
    max_dd = 0.0
    for point in curve:
        peak = max(peak, point.equity)
        max_dd = max(max_dd, (peak - point.equity) / peak if peak else 0.0)
    wins = [trade.net_pnl for trade in trades if trade.net_pnl > 0]
    losses = [-trade.net_pnl for trade in trades if trade.net_pnl < 0]
    gross_loss = sum(losses)
    vol = _daily_volatility(curve)
    sharpe = 0.0
    if vol > 0 and len(curve) > 1:
        daily_mean = sum(
            curve[i].equity / curve[i - 1].equity - 1.0 for i in range(1, len(curve))
        ) / (len(curve) - 1)
        sharpe = daily_mean / vol * math.sqrt(252.0)
    return BacktestResult(
        starting_equity=starting,
        ending_equity=ending,
        total_return=ending / starting - 1.0 if starting else 0.0,
        max_drawdown=max_dd,
        sharpe=sharpe,
        win_rate=len(wins) / len(trades) if trades else 0.0,
        profit_factor=sum(wins) / gross_loss if gross_loss else (math.inf if wins else 0.0),
        expectancy=sum(trade.net_pnl for trade in trades) / len(trades) if trades else 0.0,
        trades=tuple(trades),
        equity_curve=tuple(curve),
    )


def _build_series(bars_by_symbol: dict[str, list[Bar]]) -> dict[str, _Series]:
    series: dict[str, _Series] = {}
    for symbol, raw in bars_by_symbol.items():
        ordered = tuple(sorted((bar for bar in raw if bar.close > 0), key=lambda bar: bar.date))
        series[symbol] = _Series(
            bars=ordered, index={bar.date: i for i, bar in enumerate(ordered)}
        )
    return series


def run_backtest(
    signals: list[BacktestSignal],
    bars_by_symbol: dict[str, list[Bar]],
    config: BacktestConfig,
    betas: dict[str, float] | None = None,
) -> BacktestResult:
    """Replay signals chronologically with next-bar entries and daily OHLC exits.

    ``betas`` supplies the market-model beta used to size the hedge; a symbol
    missing from it is hedged at 1.0 rather than left naked, since an unhedged
    leg silently changes which strategy the result describes.
    """
    if config.starting_equity <= 0 or config.max_positions < 1:
        raise ValueError("starting_equity must be positive and max_positions must be at least 1")
    betas = betas or {}
    series = _build_series(bars_by_symbol)
    hedge = series.get(config.hedge_symbol) if config.hedge_symbol else None

    # The hedge instrument carries no alpha claim of its own; trading it as a
    # signal as well would double-count its exposure.
    signals = sorted(
        (
            signal
            for signal in signals
            if signal.confidence >= config.min_confidence
            and signal.symbol != config.hedge_symbol
            and signal.symbol in series
        ),
        key=lambda signal: (signal.available_on, signal.symbol),
    )
    all_days = sorted({bar.date for item in series.values() for bar in item.bars})

    # Entry lands on the first bar strictly after the signal was available.
    pending: dict[date, list[BacktestSignal]] = {}
    for signal in signals:
        symbol_series = series[signal.symbol]
        entry_index = next(
            (
                i
                for i, bar in enumerate(symbol_series.bars)
                if bar.date > signal.available_on
            ),
            None,
        )
        if entry_index is not None:
            pending.setdefault(symbol_series.bars[entry_index].date, []).append(signal)

    equity = config.starting_equity
    cash = equity
    positions: list[_Position] = []
    trades: list[BacktestTrade] = []
    curve: list[EquityPoint] = []
    hedge_shares = 0.0
    bankrupt = False

    def close(position: _Position, day: date, raw_exit: float, reason: str) -> None:
        nonlocal cash
        exit_price = _fill(raw_exit, position.signal.direction, config.exit_slippage_bps, False)
        gross = (
            (exit_price - position.entry_price) * position.shares * position.signal.direction
        )
        costs = config.commission_per_share * position.shares * 2
        cash += exit_price * position.shares * position.signal.direction - costs
        trades.append(BacktestTrade(
            symbol=position.signal.symbol, direction=position.signal.direction,
            entry_date=position.entry_date, exit_date=day,
            entry_price=position.entry_price, exit_price=exit_price,
            shares=position.shares, gross_pnl=gross, costs=costs,
            net_pnl=gross - costs, reason=reason,
        ))

    for day in all_days:
        # Exits are evaluated before new entries. A bar touching both levels is
        # a stop-out, the conservative assumption used by the live grader.
        for position in list(positions):
            symbol_series = series[position.signal.symbol]
            day_bar = symbol_series.at(day)
            if day_bar is None:
                continue
            reason: str | None = None
            raw_exit: float | None = None
            if position.signal.direction > 0:
                if day_bar.low <= position.stop:
                    reason, raw_exit = "stop", min(position.stop, day_bar.open)
                elif config.exit_style == EXIT_TARGET and day_bar.high >= position.target:
                    reason, raw_exit = "target", max(position.target, day_bar.open)
            else:
                if day_bar.high >= position.stop:
                    reason, raw_exit = "stop", max(position.stop, day_bar.open)
                elif config.exit_style == EXIT_TARGET and day_bar.low <= position.target:
                    reason, raw_exit = "target", min(position.target, day_bar.open)
            if reason is None and symbol_series.index[day] >= position.horizon_exit_index:
                reason, raw_exit = "horizon", day_bar.close
            if reason is None or raw_exit is None:
                continue
            close(position, day, raw_exit, reason)
            positions.remove(position)

        # Capacity is scarce -- insider filings arrive in clumps -- so it goes to
        # the largest measured edge rather than to whichever symbol sorts first.
        held = {position.signal.symbol for position in positions}
        candidates = sorted(
            pending.get(day, ()), key=lambda signal: (-signal.predicted_move, signal.symbol)
        )
        for signal in candidates:
            if bankrupt or len(positions) >= config.max_positions or signal.symbol in held:
                continue
            symbol_series = series[signal.symbol]
            entry_index = symbol_series.index[day]
            day_bar = symbol_series.bars[entry_index]
            history = list(symbol_series.bars[:entry_index])
            if config.atr_lookback_bars > 0:
                history = history[-config.atr_lookback_bars :]
            entry = _fill(day_bar.open, signal.direction, config.entry_slippage_bps, True)
            plan = build_trade_plan(
                bars=history, direction=signal.direction, predicted_move=signal.predicted_move,
                horizon_days=signal.horizon_days, as_of=signal.available_on,
                account_equity=max(equity, 0.0), risk_pct_per_trade=config.risk_pct_per_trade,
                atr_period=config.atr_period, atr_stop_multiple=config.atr_stop_multiple,
                max_position_pct=config.max_position_pct, entry_override=entry,
            )
            if plan is None:
                continue
            shares = plan.shares
            if config.sizing == SIZING_EQUAL_WEIGHT:
                weight = min(config.target_weight_pct, config.max_position_pct)
                shares = math.floor(max(equity, 0.0) * (weight / 100.0) / entry)
            if shares <= 0:
                continue
            notional = entry * shares
            if config.max_gross_exposure_pct > 0:
                gross = sum(
                    abs(item.entry_price * item.shares) for item in positions
                ) + notional
                if gross > equity * (config.max_gross_exposure_pct / 100.0):
                    continue
            if signal.direction > 0 and notional > cash:
                continue
            cash -= notional * signal.direction
            positions.append(_Position(
                signal=signal, entry_date=day, entry_price=entry,
                stop=plan.stop, target=plan.target, shares=shares,
                horizon_exit_index=entry_index + signal.horizon_days,
                beta=betas.get(signal.symbol, 1.0),
            ))
            held.add(signal.symbol)

        # Rebalance the beta hedge against the book as it now stands. The label
        # being replayed is an abnormal return, so the market leg is not part of
        # the bet.
        hedge_bar = hedge.at(day) if hedge else None
        if hedge_bar is not None:
            beta_notional = sum(
                position.beta
                * position.signal.direction
                * position.shares
                * (series[position.signal.symbol].at(day) or hedge_bar).close
                for position in positions
            )
            target_shares = -beta_notional / hedge_bar.close
            delta = target_shares - hedge_shares
            if delta:
                cash -= delta * hedge_bar.close
                cash -= abs(delta) * hedge_bar.close * (config.hedge_slippage_bps / 10_000.0)
                hedge_shares = target_shares

        mark = cash
        for position in positions:
            bar = series[position.signal.symbol].at(day)
            if bar:
                mark += bar.close * position.shares * position.signal.direction
        if hedge_bar is not None:
            mark += hedge_shares * hedge_bar.close
        equity = mark
        curve.append(EquityPoint(day=day, equity=equity))

        # A book replayed from non-positive equity is not a strategy any more.
        if equity <= 0 and not bankrupt:
            bankrupt = True
            for position in list(positions):
                bar = series[position.signal.symbol].at(day)
                if bar:
                    close(position, day, bar.close, "bankrupt")
                    positions.remove(position)

    # Force a final close at the last observed bar so the result is complete.
    if positions:
        last_day = all_days[-1]
        for position in list(positions):
            bar = series[position.signal.symbol].at(last_day)
            if bar is None:
                bar = series[position.signal.symbol].bars[-1]
            close(position, last_day, bar.close, "end_of_data")
            positions.remove(position)
        if hedge is not None and hedge_shares:
            hedge_bar = hedge.at(last_day) or hedge.bars[-1]
            cash += hedge_shares * hedge_bar.close
            hedge_shares = 0.0
        equity = cash
        if curve and curve[-1].day == last_day:
            curve[-1] = EquityPoint(last_day, equity)
    return _metrics(config.starting_equity, equity, trades, curve)


def chronological_split(
    signals: list[BacktestSignal], split_date: date
) -> tuple[list[BacktestSignal], list[BacktestSignal]]:
    """Return discovery and untouched holdout signals."""
    return (
        [signal for signal in signals if signal.available_on < split_date],
        [signal for signal in signals if signal.available_on >= split_date],
    )


def optimize(
    signals: list[BacktestSignal], bars_by_symbol: dict[str, list[Bar]],
    base: BacktestConfig, split_date: date, stop_multiples: tuple[float, ...] = (1.5, 2.0, 2.5),
) -> tuple[BacktestConfig, BacktestResult, BacktestResult]:
    """Choose parameters on discovery only, then report the frozen holdout result."""
    discovery, holdout = chronological_split(signals, split_date)
    best_config = base
    best_result = run_backtest(discovery, bars_by_symbol, base)
    for multiple in stop_multiples:
        candidate = BacktestConfig(**{**base.__dict__, "atr_stop_multiple": multiple})
        result = run_backtest(discovery, bars_by_symbol, candidate)
        if result.objective > best_result.objective:
            best_config, best_result = candidate, result
    return best_config, best_result, run_backtest(holdout, bars_by_symbol, best_config)


__all__ = [
    "BacktestConfig", "BacktestResult", "BacktestSignal", "BacktestTrade", "EquityPoint",
    "chronological_split", "optimize", "run_backtest",
]
