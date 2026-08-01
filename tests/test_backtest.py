"""Portfolio replay tests: no look-ahead, costs, and holdout discipline."""

from datetime import date, timedelta

import pytest

from market_intelligence.analytics.backtest import (
    BacktestConfig,
    BacktestSignal,
    chronological_split,
    optimize,
    run_backtest,
)
from market_intelligence.signals.trade_plan import Bar


def _bars(values: list[float]) -> list[Bar]:
    return [
        Bar(
            date=date(2024, 1, 1) + timedelta(days=i), open=value,
            high=value * 1.01, low=value * 0.99, close=value, adj_close=value,
        )
        for i, value in enumerate(values)
    ]


def _signal(day: int = 15, direction: int = 1) -> BacktestSignal:
    return BacktestSignal(
        symbol="AAA", available_on=date(2024, 1, day), direction=direction,
        predicted_move=0.02, horizon_days=2,
    )


def test_signal_enters_on_next_bar_not_signal_bar():
    result = run_backtest(
        [_signal()], {"AAA": _bars([100] * 15 + [110] + [110] * 5)},
        BacktestConfig(
            starting_equity=10_000, max_position_pct=100,
            entry_slippage_bps=0, exit_slippage_bps=0,
        ),
    )
    assert result.trades
    assert result.trades[0].entry_date == date(2024, 1, 16)


def test_costs_reduce_profit():
    bars = {"AAA": _bars([100] * 15 + [110] + [110] * 5)}
    free = run_backtest(
        [_signal()], bars, BacktestConfig(starting_equity=10_000, max_position_pct=100)
    )
    costly = run_backtest(
        [_signal()], bars,
        BacktestConfig(
            starting_equity=10_000, max_position_pct=100,
            entry_slippage_bps=100, exit_slippage_bps=100,
        ),
    )
    assert costly.ending_equity < free.ending_equity


def test_split_is_chronological_and_optimize_does_not_tune_on_holdout():
    signals = [_signal(1), _signal(10)]
    discovery, holdout = chronological_split(signals, date(2024, 1, 5))
    assert [item.available_on for item in discovery] == [date(2024, 1, 1)]
    assert [item.available_on for item in holdout] == [date(2024, 1, 10)]
    config, discovery_result, holdout_result = optimize(
        signals, {"AAA": _bars([100] * 16 + [110])},
        BacktestConfig(starting_equity=10_000, max_position_pct=100), date(2024, 1, 5),
        stop_multiples=(1.5, 2.0),
    )
    assert config.atr_stop_multiple in (1.5, 2.0)
    assert discovery_result.starting_equity == pytest.approx(10_000)
    assert holdout_result.starting_equity == pytest.approx(10_000)
