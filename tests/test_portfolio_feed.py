"""Feed tests: adjusted space, calendar alignment, and the no-lookahead boundary.

``trailing_returns(end_exclusive=...)`` is the only place a return from day *t*
can leak into a decision made on day *t*, so it gets the hardest tests here: a
distinctive spike is planted on one day and the boundary is walked across it in
both directions.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import duckdb
import numpy as np
import pytest

from market_intelligence.portfolio.feed import build_panel, load_feed
from market_intelligence.signals.trade_plan import Bar


def _bar(day: date, close: float, *, adj_close: float | None = None) -> Bar:
    """A flat-ish bar whose OHLC is a fixed spread around ``close``."""
    return Bar(
        date=day,
        open=close,
        high=close * 1.1,
        low=close * 0.9,
        close=close,
        adj_close=close if adj_close is None else adj_close,
    )


def _closes(symbol: str, pairs: list[tuple[date, float]]) -> dict[str, list[Bar]]:
    return {symbol: [_bar(day, close) for day, close in pairs]}


D = [date(2024, 1, day) for day in (2, 3, 4, 5, 8, 9)]


# --------------------------------------------------------------------------- #
# 1. adjusted space                                                            #
# --------------------------------------------------------------------------- #
def test_adjusted_space_is_hand_computed_and_raw_close_is_untouched():
    """A 2-for-1 split halves OHLC in adjusted space; the ledger price does not move.

    Hand-computed: factor = adj_close / close = 50 / 100 = 0.5, so the bar's
    100/110/90/100 becomes 50/55/45/50 while ``raw_close`` stays 100 -- a broker
    charged 100 a share that day and the ledger has to say so.
    """
    bars = {"AAA": [Bar(date=D[0], open=100.0, high=110.0, low=90.0, close=100.0, adj_close=50.0)]}

    panel = build_panel(bars, ("AAA",))

    assert panel.open[0, 0] == pytest.approx(50.0)
    assert panel.high[0, 0] == pytest.approx(55.0)
    assert panel.low[0, 0] == pytest.approx(45.0)
    assert panel.close[0, 0] == pytest.approx(50.0)
    assert panel.raw_close[0, 0] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# 2. calendar                                                                  #
# --------------------------------------------------------------------------- #
def test_calendar_is_strictly_ascending_and_index_of_refuses_absent_days():
    """Two symbols, unsorted input, overlapping dates -- one clean ascending axis."""
    bars = {
        "AAA": [_bar(D[2], 12.0), _bar(D[0], 10.0)],
        "BBB": [_bar(D[1], 20.0), _bar(D[2], 21.0)],
    }

    panel = build_panel(bars, ("AAA", "BBB"))

    assert panel.calendar.tolist() == [D[0], D[1], D[2]]
    assert np.all(np.diff(panel.calendar) > np.timedelta64(0, "D"))
    assert len(set(panel.calendar.tolist())) == panel.calendar.size
    assert panel.index_of(D[1]) == 1
    assert panel.index_of(D[2]) == 2
    with pytest.raises(KeyError):
        panel.index_of(date(2024, 1, 6))  # a Saturday: absent, never nearest-matched


# --------------------------------------------------------------------------- #
# 3. the leak canary                                                           #
# --------------------------------------------------------------------------- #
def test_trailing_returns_excludes_the_decision_day_itself():
    """Day D[3]'s return is +400%; a decision made on D[3] must not be able to see it.

    Closes 10, 11, 12, 60 make the return on row 3 exactly 4.0 and nothing else
    close to it, so a single off-by-one at the boundary is visible by inspection
    rather than by arithmetic.
    """
    panel = build_panel(
        _closes("AAA", [(D[0], 10.0), (D[1], 11.0), (D[2], 12.0), (D[3], 60.0), (D[4], 61.0)]),
        ("AAA",),
    )
    spike = 60.0 / 12.0 - 1.0

    before = panel.trailing_returns(end_exclusive=D[3], lookback=2)
    after = panel.trailing_returns(end_exclusive=D[4], lookback=2)

    assert before.shape == (2, 1)
    assert not np.any(np.isclose(before, spike))
    assert before[:, 0] == pytest.approx([11.0 / 10.0 - 1.0, 12.0 / 11.0 - 1.0])
    assert after.shape == (2, 1)
    assert np.isclose(after[-1, 0], spike)
    assert after[:, 0] == pytest.approx([12.0 / 11.0 - 1.0, spike])


# --------------------------------------------------------------------------- #
# 4. column order                                                              #
# --------------------------------------------------------------------------- #
def test_trailing_returns_column_order_follows_the_requested_symbols():
    """Requested order wins over the panel's internal order.

    Getting this backwards transposes what a covariance matrix means without
    changing its shape, so nothing downstream would ever raise.
    """
    bars = {
        "AAA": [_bar(D[0], 10.0), _bar(D[1], 11.0), _bar(D[2], 12.0)],
        "BBB": [_bar(D[0], 100.0), _bar(D[1], 200.0), _bar(D[2], 300.0)],
        "CCC": [_bar(D[0], 50.0), _bar(D[1], 55.0), _bar(D[2], 60.5)],
    }
    panel = build_panel(bars, ("AAA", "BBB", "CCC"))
    assert panel.symbols == ("AAA", "BBB", "CCC")

    requested = panel.trailing_returns(end_exclusive=D[2], lookback=1, symbols=("CCC", "AAA"))
    natural = panel.trailing_returns(end_exclusive=D[2], lookback=1)

    assert requested.shape == (1, 2)
    assert requested[0, 0] == pytest.approx(55.0 / 50.0 - 1.0)  # CCC first
    assert requested[0, 1] == pytest.approx(11.0 / 10.0 - 1.0)  # AAA second
    assert natural[0].tolist() == pytest.approx([requested[0, 1], 1.0, requested[0, 0]])


# --------------------------------------------------------------------------- #
# 5. last known price                                                          #
# --------------------------------------------------------------------------- #
def test_last_known_price_reaches_backwards_across_a_gap_and_never_forwards():
    """A NaN hole must resolve to the last real print, not the next one."""
    bars = {
        "AAA": [_bar(D[0], 10.0), _bar(D[2], 12.0), _bar(D[4], 99.0)],
        "BBB": [_bar(D[1], 20.0), _bar(D[2], 21.0), _bar(D[3], 22.0), _bar(D[4], 23.0)],
    }
    panel = build_panel(bars, ("AAA", "BBB"))
    assert np.isnan(panel.close[panel.index_of(D[3]), panel.column_of("AAA")])

    assert panel.last_known_price("AAA", as_of=D[3]) == pytest.approx(12.0)
    assert panel.last_known_price("AAA", as_of=D[3]) != pytest.approx(99.0)
    assert panel.last_known_price("AAA", as_of=D[4]) == pytest.approx(99.0)
    assert panel.last_known_price("BBB", as_of=D[0]) is None  # nothing precedes it


# --------------------------------------------------------------------------- #
# 6. alignment                                                                 #
# --------------------------------------------------------------------------- #
def test_build_panel_aligns_uneven_coverage_into_one_calendar():
    """Disjoint coverage aligns on dates, with NaN where a symbol has no bar."""
    bars = {
        "AAA": [_bar(D[0], 10.0), _bar(D[1], 11.0)],
        "BBB": [_bar(D[1], 20.0), _bar(D[2], 21.0)],
    }

    panel = build_panel(bars, ("AAA", "BBB"))

    assert panel.calendar.size == 3
    assert panel.close.shape == (3, 2)
    aaa, bbb = panel.column_of("AAA"), panel.column_of("BBB")
    assert panel.close[panel.index_of(D[1]), bbb] == pytest.approx(20.0)
    assert panel.close[panel.index_of(D[0]), aaa] == pytest.approx(10.0)
    assert np.isnan(panel.close[panel.index_of(D[2]), aaa])
    assert np.isnan(panel.close[panel.index_of(D[0]), bbb])
    assert np.isnan(panel.raw_close[panel.index_of(D[2]), aaa])


# --------------------------------------------------------------------------- #
# 7. the bundle                                                                #
# --------------------------------------------------------------------------- #
def _seed_feed(con: duckdb.DuckDBPyConnection) -> None:
    """One admitted, beta-hedge-surviving cell on AAA, plus SPY bars and edges."""
    con.execute(
        """
        INSERT INTO signal_status (
            signal_id, event_type, event_subtype, edge_type, horizon_days, status,
            confirm_streak, fail_streak, last_verdict, mean_car, hit_rate, n_clusters,
            direction, first_seen_time, schema_version
        ) VALUES ('sig-a', 'insider_transaction', 'P', 'self', 20, 'candidate',
                  1, 0, 'admitted', 0.01, 0.55, 500, 1, ?, '1.0.0')
        """,
        [datetime(2026, 1, 1, tzinfo=UTC)],
    )
    con.execute(
        """
        INSERT INTO event_samples (
            sample_id, event_id, edge_id, event_type, event_subtype, source_ticker,
            target_ticker, horizon_days, available_on, t0, window_end,
            forward_abnormal_return, split, schema_version
        ) VALUES ('s1', 'e1', 'self', 'insider_transaction', 'P', 'AAA', 'AAA', 20,
                  ?, ?, ?, 0.01, 'discovery', '1.0.0')
        """,
        [date(2020, 5, 1), date(2020, 5, 1), date(2020, 5, 4)],
    )
    for day, beta in ((date(2020, 5, 1), 1.0), (date(2020, 5, 4), 1.4)):
        con.execute(
            """
            INSERT INTO daily_returns (
                return_id, symbol, price_date, total_return, market_return,
                abnormal_return, beta, method, schema_version
            ) VALUES (?, 'AAA', ?, 0.01, 0.002, 0.008, ?, 'market_model', '1.0.0')
            """,
            [f"r-{day}", day, beta],
        )
    for symbol, close in (("AAA", 10.0), ("SPY", 400.0)):
        for offset, day in enumerate((date(2020, 5, 1), date(2020, 5, 4), date(2020, 5, 5))):
            price = close + offset
            con.execute(
                """
                INSERT INTO daily_prices (
                    price_id, symbol, price_date, open, high, low, close, adj_close,
                    volume, provider, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1000, 'test', '1.0.0')
                """,
                [f"p-{symbol}-{day}", symbol, day, price, price, price, price, price],
            )
    for edge_type, target in (("customer", "BBB"), ("shared_board_member", "CCC")):
        con.execute(
            """
            INSERT INTO company_edges (
                edge_id, edge_key, source_cik, source_ticker, target, target_name,
                target_ticker, edge_type, resolution_status, report_date, schema_version
            ) VALUES (?, ?, '0000001', 'AAA', ?, ?, ?, ?, 'resolved', ?, '1.0.0')
            """,
            [
                f"edge-{edge_type}", f"key-{edge_type}", target, target, target, edge_type,
                date(2020, 1, 1),
            ],
        )


def test_load_feed_returns_one_consistent_symbol_universe(memory_db):
    """Panel, signals and betas must all speak about the same names.

    ``spy_index`` resolves only when SPY actually has price history: a benchmark
    column of NaN would silently turn every benchmark-relative metric into NaN
    instead of saying the benchmark is missing.
    """
    _seed_feed(memory_db)

    bundle = load_feed(memory_db, as_of=date(2024, 1, 1))

    assert bundle.panel.symbols == ("AAA", "SPY")
    assert bundle.spy_index == bundle.panel.column_of("SPY")
    assert bundle.panel.calendar.size == 3
    assert [cell.event_subtype for cell in bundle.cells] == ["P"]
    assert set(bundle.signals_by_day) == {date(2020, 5, 1)}
    fired = [signal.symbol for group in bundle.signals_by_day.values() for signal in group]
    assert fired == ["AAA"]
    assert set(fired) <= set(bundle.panel.symbols)
    assert set(bundle.betas) <= set(bundle.panel.symbols)
    assert bundle.betas["AAA"] == pytest.approx(1.2)
    # (0.01 - 1.0*0.002) + (0.01 - 1.4*0.002), one sample, direction +1
    assert bundle.hedged_edges[("insider_transaction", "P", 20)] == pytest.approx(0.0152)
    assert bundle.edges == (("AAA", "BBB", "customer", date(2020, 1, 1)),)

    memory_db.execute("DELETE FROM daily_prices WHERE symbol = 'SPY'")
    without_spy = load_feed(memory_db, as_of=date(2024, 1, 1))

    assert without_spy.spy_index is None
    assert without_spy.panel.symbols == ("AAA",)
