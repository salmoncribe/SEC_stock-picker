"""Replay mechanics: horizon exits, capacity ranking, exposure caps, beta hedge.

These are the properties that decide whether the replay measures the edge the
gate validated or something else wearing its name.
"""

from __future__ import annotations

from datetime import date, timedelta

from market_intelligence.analytics.backtest import (
    BacktestConfig,
    BacktestSignal,
    run_backtest,
)
from market_intelligence.signals.trade_plan import Bar

TRADING_DAYS = [
    day
    for day in (date(2024, 1, 1) + timedelta(days=i) for i in range(260))
    if day.weekday() < 5
]


def _bars(values: list[float], days: list[date] | None = None) -> list[Bar]:
    """Bars on trading days only, so calendar and trading horizons diverge."""
    days = days or TRADING_DAYS
    return [
        Bar(
            date=days[i], open=value, high=value * 1.005,
            low=value * 0.995, close=value, adj_close=value,
        )
        for i, value in enumerate(values)
    ]


def _flat_then(values: list[float], lead: int = 20, level: float = 100.0) -> list[float]:
    return [level] * lead + values


def _signal(
    day: date, *, direction: int = 1, move: float = 0.02, horizon: int = 5, symbol: str = "AAA"
) -> BacktestSignal:
    return BacktestSignal(
        symbol=symbol, available_on=day, direction=direction,
        predicted_move=move, horizon_days=horizon,
    )


def _config(**overrides) -> BacktestConfig:
    base = {
        "starting_equity": 100_000.0,
        "max_position_pct": 100.0,
        "entry_slippage_bps": 0.0,
        "exit_slippage_bps": 0.0,
    }
    return BacktestConfig(**{**base, **overrides})


# --- horizon exits ---------------------------------------------------------


def test_horizon_exit_counts_trading_days_not_calendar_days():
    """A 5-day cell measured over 5 *bars* must be held for 5 bars.

    The signal lands on TRADING_DAYS[19]; entry is the next bar (index 20).
    Five trading days later is index 25 -- seven calendar days, because the
    window spans a weekend. Counting calendar days would exit at index 23.
    """
    bars = {"AAA": _bars(_flat_then([100.0] * 40))}
    result = run_backtest(
        [_signal(TRADING_DAYS[19], horizon=5, move=9.0)],
        bars,
        _config(exit_style="horizon"),
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_date == TRADING_DAYS[25]
    assert result.trades[0].reason == "horizon"


def test_horizon_style_does_not_clip_a_winner_at_the_expected_move():
    """The measured edge is a mean over the window, not a take-profit level."""
    prices = _flat_then([100.0, 103.0, 106.0, 109.0, 112.0, 115.0])
    bars = {"AAA": _bars(prices)}
    signal = _signal(TRADING_DAYS[19], horizon=5, move=0.01)

    clipped = run_backtest([signal], bars, _config(exit_style="target"))
    held = run_backtest([signal], bars, _config(exit_style="horizon"))

    assert clipped.trades[0].reason == "target"
    assert held.trades[0].net_pnl > clipped.trades[0].net_pnl


def test_a_stop_still_protects_a_horizon_trade():
    """Dropping the profit target must not drop the risk control."""
    prices = _flat_then([100.0, 80.0, 80.0, 80.0, 80.0, 80.0])
    bars = {"AAA": _bars(prices)}
    result = run_backtest(
        [_signal(TRADING_DAYS[19], horizon=5)], bars,
        _config(exit_style="horizon", atr_stop_multiple=2.0),
    )

    assert result.trades[0].reason == "stop"


# --- capacity allocation ---------------------------------------------------


def test_scarce_capacity_goes_to_the_larger_expected_move():
    """With one slot and two candidates, the stronger cell trades -- not the
    alphabetically-first symbol."""
    prices = _flat_then([100.0] * 40)
    bars = {"AAA": _bars(prices), "ZZZ": _bars(prices)}
    signals = [
        _signal(TRADING_DAYS[19], symbol="AAA", move=0.004),
        _signal(TRADING_DAYS[19], symbol="ZZZ", move=0.018),
    ]

    result = run_backtest(
        signals, bars, _config(exit_style="horizon", max_positions=1, max_position_pct=50.0)
    )

    assert [trade.symbol for trade in result.trades] == ["ZZZ"]


# --- exposure -------------------------------------------------------------


def test_gross_exposure_cap_limits_short_book_size():
    """Short proceeds must not silently finance an unbounded book."""
    prices = _flat_then([100.0] * 40)
    bars = {name: _bars(prices) for name in ("AAA", "BBB", "CCC", "DDD")}
    signals = [
        _signal(TRADING_DAYS[19], symbol=name, direction=-1, move=0.01)
        for name in ("AAA", "BBB", "CCC", "DDD")
    ]
    config = {
        "exit_style": "horizon", "max_positions": 4, "max_position_pct": 50.0,
        "risk_pct_per_trade": 10.0,
    }

    capped = run_backtest(signals, bars, _config(**config, max_gross_exposure_pct=60.0))
    uncapped = run_backtest(signals, bars, _config(**config, max_gross_exposure_pct=0.0))

    capped_notional = sum(abs(t.entry_price * t.shares) for t in capped.trades)
    assert capped_notional <= 60_000.0 + 1e-6
    assert len(capped.trades) < len(uncapped.trades)


# --- ATR window parity with the live path ---------------------------------


def test_atr_lookback_bars_bounds_the_volatility_window():
    """Live sizing reads the last ``atr_period * 4`` bars; the replay must match.

    Ancient turbulence followed by a long calm stretch gives a wide full-history
    ATR and a narrow recent one, so the two windows produce different stops and
    therefore different share counts.
    """
    prices = [100.0 if i % 2 else 140.0 for i in range(60)] + [100.0] * 60
    bars = {"AAA": _bars(prices)}
    signal = _signal(TRADING_DAYS[110], horizon=5, move=0.01)

    bounded = run_backtest([signal], bars, _config(exit_style="horizon", atr_lookback_bars=56))
    unbounded = run_backtest([signal], bars, _config(exit_style="horizon", atr_lookback_bars=0))

    assert bounded.trades[0].shares > unbounded.trades[0].shares


# --- beta hedge -----------------------------------------------------------


def test_equal_weight_sizing_does_not_shrink_when_the_stop_widens():
    """Size must not be welded to stop distance.

    ``shares = risk_budget / stop_distance`` means widening the stop shrinks the
    position. For a horizon-held edge the stop is only a disaster brake, so
    widening it should leave the intended exposure untouched.
    """
    bars = {"AAA": _bars(_flat_then([100.0] * 40))}
    signal = _signal(TRADING_DAYS[19], horizon=5)

    narrow = run_backtest(
        [signal], bars,
        _config(sizing="equal_weight", target_weight_pct=10.0, atr_stop_multiple=2.0),
    )
    wide = run_backtest(
        [signal], bars,
        _config(sizing="equal_weight", target_weight_pct=10.0, atr_stop_multiple=25.0),
    )

    assert narrow.trades[0].shares == wide.trades[0].shares
    assert narrow.trades[0].shares == 100  # 10% of $100k equity at $100


def test_risk_sizing_still_shrinks_when_the_stop_widens():
    """The original mode must keep its meaning; equal weight is an addition."""
    bars = {"AAA": _bars(_flat_then([100.0] * 40))}
    signal = _signal(TRADING_DAYS[19], horizon=5)

    narrow = run_backtest([signal], bars, _config(sizing="risk", atr_stop_multiple=2.0))
    wide = run_backtest([signal], bars, _config(sizing="risk", atr_stop_multiple=10.0))

    assert narrow.trades[0].shares > wide.trades[0].shares


def test_equal_weight_still_respects_the_position_cap():
    bars = {"AAA": _bars(_flat_then([100.0] * 40))}
    result = run_backtest(
        [_signal(TRADING_DAYS[19], horizon=5)], bars,
        _config(sizing="equal_weight", target_weight_pct=50.0, max_position_pct=5.0),
    )

    assert result.trades[0].shares == 50  # capped at 5% of 100k, not 50%


def test_beta_hedge_strips_the_market_move_from_a_long():
    """The gate grades abnormal returns, so the replay must neutralise beta.

    The name and the index both rise 10%: pure market move, zero alpha. Hedged,
    the trade should end roughly flat; unhedged it books the whole rally.
    """
    prices = _flat_then([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
    bars = {"AAA": _bars(prices), "SPY": _bars(prices)}
    signal = _signal(TRADING_DAYS[19], horizon=5, move=0.01)

    hedged = run_backtest(
        [signal], bars,
        _config(exit_style="horizon", hedge_symbol="SPY", max_position_pct=50.0),
        betas={"AAA": 1.0},
    )
    unhedged = run_backtest(
        [signal], bars, _config(exit_style="horizon", max_position_pct=50.0)
    )

    assert unhedged.total_return > 0.0
    assert abs(hedged.total_return) < 0.2 * unhedged.total_return


def test_hedge_symbol_is_not_traded_as_a_signal():
    """The hedge instrument carries no alpha claim; it only offsets exposure."""
    prices = _flat_then([100.0] * 40)
    bars = {"AAA": _bars(prices), "SPY": _bars(prices)}
    result = run_backtest(
        [_signal(TRADING_DAYS[19], symbol="SPY", horizon=5)],
        bars,
        _config(exit_style="horizon", hedge_symbol="SPY"),
        betas={"SPY": 1.0},
    )

    assert result.trades == ()


def test_missing_beta_falls_back_to_one_rather_than_dropping_the_hedge():
    """An unhedged position is a silent change of strategy; assume beta 1."""
    prices = _flat_then([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
    bars = {"AAA": _bars(prices), "SPY": _bars(prices)}

    result = run_backtest(
        [_signal(TRADING_DAYS[19], horizon=5)], bars,
        _config(exit_style="horizon", hedge_symbol="SPY", max_position_pct=50.0),
        betas={},
    )

    assert abs(result.total_return) < 0.01
