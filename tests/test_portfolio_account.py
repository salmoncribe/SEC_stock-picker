"""Account ledger tests: hand-computed P&L and the invariants that must not bend."""

from datetime import date
from itertools import pairwise

import pytest

from market_intelligence.portfolio.account import (
    EQUITY_TOLERANCE,
    AccountError,
    AccountState,
    Fill,
    Side,
    apply_fill,
    equity_identity_holds,
    genesis,
    genesis_fill,
    mark,
    replay,
)

DAY = date(2026, 1, 5)


def _fill(
    side: Side,
    shares: float,
    price: float,
    *,
    symbol: str = "AAA",
    commission: float = 0.0,
    as_of: date = DAY,
    tag: str = "1",
) -> Fill:
    return Fill(
        fill_id=f"pf:{symbol}:{as_of.isoformat()}:{side.value}:{tag}",
        order_id=f"po:{symbol}:{as_of.isoformat()}:{side.value}:{tag}",
        symbol=symbol,
        side=side,
        shares=shares,
        price=price,
        commission=commission,
        as_of=as_of,
    )


def test_golden_buy_pnl():
    state = apply_fill(genesis(10_000.0, DAY), _fill(Side.BUY, 10.0, 100.0))
    assert state.cash == 9_000.0
    assert state.positions["AAA"].shares == 10.0
    assert state.positions["AAA"].cost_basis == 1_000.0


def test_golden_sell_with_commission_reduces_basis_proportionally():
    # Buy 10 @ 100 with $5 commission: basis capitalises the commission.
    bought = apply_fill(genesis(10_000.0, DAY), _fill(Side.BUY, 10.0, 100.0, commission=5.0))
    assert bought.cash == 8_995.0
    assert bought.positions["AAA"].cost_basis == 1_005.0

    # Sell 4 @ 125 with $5 commission: proceeds 500 - 5 = 495, basis falls 40%.
    sold = apply_fill(bought, _fill(Side.SELL, 4.0, 125.0, commission=5.0, tag="2"))
    assert sold.cash == 9_490.0
    assert sold.positions["AAA"].shares == 6.0
    assert sold.positions["AAA"].cost_basis == 603.0


def test_overdraft_raises():
    with pytest.raises(AccountError):
        apply_fill(genesis(1_000.0, DAY), _fill(Side.BUY, 11.0, 100.0))


def test_oversell_raises_because_the_account_is_long_only():
    bought = apply_fill(genesis(10_000.0, DAY), _fill(Side.BUY, 10.0, 100.0))
    with pytest.raises(AccountError):
        apply_fill(bought, _fill(Side.SELL, 10.5, 100.0, tag="2"))
    with pytest.raises(AccountError):
        apply_fill(genesis(10_000.0, DAY), _fill(Side.SELL, 1.0, 100.0, symbol="BBB"))


def test_backdated_fill_raises():
    opened = apply_fill(genesis(10_000.0, DAY), _fill(Side.BUY, 1.0, 100.0))
    with pytest.raises(AccountError):
        apply_fill(opened, _fill(Side.BUY, 1.0, 100.0, as_of=date(2026, 1, 2), tag="2"))


def test_mark_satisfies_the_equity_identity():
    state = apply_fill(
        apply_fill(genesis(10_000.0, DAY), _fill(Side.BUY, 10.0, 100.0)),
        _fill(Side.BUY, 5.0, 40.0, symbol="BBB", tag="2"),
    )
    marked = mark(state, {"AAA": 111.5, "BBB": 37.25})
    expected = state.cash + 10.0 * 111.5 + 5.0 * 37.25
    assert abs(marked.equity - expected) <= EQUITY_TOLERANCE
    assert equity_identity_holds(marked)


def test_mark_missing_price_raises_unless_stale_ok():
    state = apply_fill(genesis(10_000.0, DAY), _fill(Side.BUY, 10.0, 100.0))
    with pytest.raises(AccountError):
        mark(state, {"BBB": 10.0})
    marked = mark(state, {"BBB": 10.0}, stale_ok=True)
    assert equity_identity_holds(marked)
    assert marked.equity == pytest.approx(10_000.0, abs=EQUITY_TOLERANCE)


def test_replay_reproduces_incremental_application():
    opening = genesis_fill(10_000.0, DAY, "test")
    fills = (
        _fill(Side.BUY, 10.0, 100.0, commission=1.0),
        _fill(Side.BUY, 5.0, 40.0, symbol="BBB", commission=1.0, tag="2"),
        _fill(Side.SELL, 4.0, 125.0, commission=1.0, as_of=date(2026, 1, 6), tag="3"),
    )

    live = genesis(10_000.0, DAY)
    for fill in fills:
        live = apply_fill(live, fill)
    replayed = replay([opening, *fills])

    assert replayed.cash == pytest.approx(live.cash, abs=1e-9)
    assert replayed.high_water_mark == pytest.approx(live.high_water_mark, abs=1e-9)
    assert replayed.as_of == live.as_of
    assert set(replayed.positions) == set(live.positions)
    for symbol, position in live.positions.items():
        assert replayed.positions[symbol].shares == pytest.approx(position.shares, abs=1e-9)
        assert replayed.positions[symbol].cost_basis == pytest.approx(
            position.cost_basis, abs=1e-9
        )


def test_high_water_mark_ratchets_and_drawdown_measures_from_the_peak():
    state = apply_fill(genesis(10_000.0, DAY), _fill(Side.BUY, 10.0, 100.0))

    peak = mark(state, {"AAA": 200.0})
    assert peak.equity == pytest.approx(11_000.0)
    assert peak.state.high_water_mark == pytest.approx(11_000.0)
    assert peak.drawdown == pytest.approx(0.0)

    trough = mark(peak.state, {"AAA": 100.0})
    assert trough.equity == pytest.approx(10_000.0)
    assert trough.state.high_water_mark == pytest.approx(11_000.0)
    assert trough.drawdown == pytest.approx(1_000.0 / 11_000.0)


def test_mark_without_a_halflife_is_bit_for_bit_the_pure_ratchet():
    """The backward-compatibility pin. ``None`` must mean *exactly* today.

    Every caller that predates the decay passes nothing, and every one of them
    must keep the mark it had. Asserted as equality, not ``approx``: a ratchet
    that returns 10_999.999999999998 is a different function.
    """
    state = apply_fill(genesis(10_000.0, DAY), _fill(Side.BUY, 10.0, 100.0))
    peak = mark(state, {"AAA": 200.0})
    assert peak.state.high_water_mark == 11_000.0

    for kwargs in ({}, {"hwm_decay_halflife_days": None}):
        trough = mark(peak.state, {"AAA": 100.0}, **kwargs)  # type: ignore[arg-type]
        assert trough.state.high_water_mark == 11_000.0
        assert trough.equity == 10_000.0
        assert trough.drawdown == pytest.approx(1_000.0 / 11_000.0)

    # ...and it stays pinned however many times it is re-marked. This *is* the
    # 2020-03-12 absorbing state: 1,603 sessions of constant equity, constant
    # mark, constant drawdown.
    pinned = peak.state
    for _ in range(50):
        pinned = mark(pinned, {"AAA": 100.0}).state
    assert pinned.high_water_mark == 11_000.0


def test_decay_lets_a_flattened_account_heal_a_drawdown_it_cannot_trade_out_of():
    """An all-cash account cannot move equity, so only the mark can move.

    Positions are empty on purpose: this is the account the governor has already
    flattened, which is precisely the state the pure ratchet could never leave.
    """
    flat = AccountState(as_of=DAY, cash=8_000.0, positions={}, high_water_mark=10_000.0)
    assert mark(flat, {}).drawdown == pytest.approx(0.20)

    state = flat
    drawdowns: list[float] = []
    equities: list[float] = []
    for _ in range(20):
        marked = mark(state, {}, hwm_decay_halflife_days=5)
        drawdowns.append(marked.drawdown)
        equities.append(marked.equity)
        state = marked.state

    assert equities == [8_000.0] * 20  # equity genuinely never moved
    assert all(later < earlier for earlier, later in pairwise(drawdowns))
    # Hand-computed: 20 sessions is four half-lives, so 1/16 of the $2,000 gap
    # survives -- a mark of $8,125 and a drawdown of 125/8125.
    assert state.high_water_mark == pytest.approx(8_125.0)
    assert drawdowns[-1] == pytest.approx(125.0 / 8_125.0)
    assert drawdowns[-1] < 0.10  # below the governor's halve threshold: re-entry


def test_share_precision_rounds_and_integer_mode_rejects_fractions():
    fractional = apply_fill(genesis(10_000.0, DAY), _fill(Side.BUY, 3.14159265, 100.0))
    assert fractional.positions["AAA"].shares == pytest.approx(3.1416, abs=1e-12)
    assert fractional.cash == pytest.approx(10_000.0 - 314.16, abs=1e-9)

    # Dust below half a share-precision unit closes the position outright.
    closed = apply_fill(fractional, _fill(Side.SELL, 3.1416, 100.0, tag="2"))
    assert "AAA" not in closed.positions

    whole = apply_fill(genesis(10_000.0, DAY), _fill(Side.BUY, 3.0, 100.0), fractional=False)
    assert whole.positions["AAA"].shares == 3.0
    with pytest.raises(AccountError):
        apply_fill(genesis(10_000.0, DAY), _fill(Side.BUY, 3.5, 100.0), fractional=False)
