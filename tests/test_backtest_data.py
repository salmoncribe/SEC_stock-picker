"""Loader tests: only admitted cells become signals, and one signal per symbol-day."""

from __future__ import annotations

from datetime import UTC, date, datetime

import duckdb
import pytest

from market_intelligence.analytics.backtest_data import (
    AdmittedCell,
    filter_tradeable,
    load_admitted_cells,
    load_bars,
    load_betas,
    load_replay_inputs,
    load_signals,
    signals_from_rows,
)


def _status_row(
    con: duckdb.DuckDBPyConnection,
    *,
    signal_id: str,
    subtype: str | None,
    horizon: int,
    direction: int,
    mean_car: float,
    status: str = "candidate",
    verdict: str = "admitted",
) -> None:
    con.execute(
        """
        INSERT INTO signal_status (
            signal_id, event_type, event_subtype, edge_type, horizon_days, status,
            confirm_streak, fail_streak, last_verdict, mean_car, hit_rate, n_clusters,
            direction, first_seen_time, schema_version
        ) VALUES (?, 'insider_transaction', ?, 'self', ?, ?, 1, 0, ?, ?, 0.55, 500, ?, ?, '1.0.0')
        """,
        [
            signal_id, subtype, horizon, status, verdict, mean_car, direction,
            datetime(2026, 1, 1, tzinfo=UTC),
        ],
    )


def _sample_row(
    con: duckdb.DuckDBPyConnection,
    *,
    sample_id: str,
    subtype: str | None,
    horizon: int,
    ticker: str,
    available_on: date,
    split: str = "discovery",
    edge_id: str = "self",
) -> None:
    con.execute(
        """
        INSERT INTO event_samples (
            sample_id, event_id, edge_id, event_type, event_subtype, source_ticker,
            target_ticker, horizon_days, available_on, t0, window_end,
            forward_abnormal_return, split, schema_version
        ) VALUES (?, ?, ?, 'insider_transaction', ?, ?, ?, ?, ?, ?, ?, 0.01, ?, '1.0.0')
        """,
        [
            sample_id, f"event-{sample_id}", edge_id, subtype, ticker, ticker, horizon,
            available_on, available_on, available_on, split,
        ],
    )


def test_retired_and_unadmitted_cells_are_not_tradeable(memory_db):
    _status_row(memory_db, signal_id="a", subtype="P", horizon=20, direction=1, mean_car=0.01)
    _status_row(
        memory_db, signal_id="b", subtype="J", horizon=20, direction=-1, mean_car=-0.014,
        status="retired", verdict="demoted",
    )
    _status_row(
        memory_db, signal_id="c", subtype="X", horizon=20, direction=-1, mean_car=-0.01,
        verdict="insufficient_evidence",
    )

    cells = load_admitted_cells(memory_db)

    assert [cell.event_subtype for cell in cells] == ["P"]


def test_cell_carries_direction_and_absolute_expected_move():
    cell = AdmittedCell(
        event_type="insider_transaction", event_subtype="S", horizon_days=20,
        direction=-1, mean_car=-0.0087, hit_rate=0.45, n_clusters=41445,
    )
    assert cell.predicted_move == pytest.approx(0.0087)


def test_one_signal_per_symbol_day_keeps_the_strongest_cell():
    """Two admitted cells fire on the same name the same day; only the bigger edge trades."""
    weak = AdmittedCell("insider_transaction", "M", 20, -1, -0.0070, 0.46, 26048)
    strong = AdmittedCell("insider_transaction", "S", 20, -1, -0.0087, 0.45, 41445)
    rows = [
        ("insider_transaction", "M", 20, "AAA", date(2024, 3, 1)),
        ("insider_transaction", "S", 20, "AAA", date(2024, 3, 1)),
    ]

    signals = signals_from_rows(rows, [weak, strong])

    assert len(signals) == 1
    assert signals[0].predicted_move == pytest.approx(0.0087)
    assert signals[0].direction == -1


def test_conflicting_directions_on_one_symbol_day_are_dropped():
    """A buy cell and a sell cell on the same name the same day cancel; no coin flip."""
    long_cell = AdmittedCell("insider_transaction", "P", 20, 1, 0.0100, 0.55, 4752)
    short_cell = AdmittedCell("insider_transaction", "S", 20, -1, -0.0087, 0.45, 41445)
    rows = [
        ("insider_transaction", "P", 20, "AAA", date(2024, 3, 1)),
        ("insider_transaction", "S", 20, "AAA", date(2024, 3, 1)),
    ]

    assert signals_from_rows(rows, [long_cell, short_cell]) == []


def test_rows_without_a_matching_admitted_cell_are_ignored():
    cell = AdmittedCell("insider_transaction", "P", 20, 1, 0.01, 0.55, 4752)
    rows = [("insider_transaction", "P", 5, "AAA", date(2024, 3, 1))]  # horizon not admitted

    assert signals_from_rows(rows, [cell]) == []


def test_load_signals_reads_only_self_edges_of_the_requested_split(memory_db):
    _status_row(memory_db, signal_id="a", subtype="P", horizon=20, direction=1, mean_car=0.01)
    _sample_row(
        memory_db, sample_id="s1", subtype="P", horizon=20, ticker="AAA",
        available_on=date(2020, 5, 1),
    )
    _sample_row(
        memory_db, sample_id="s2", subtype="P", horizon=20, ticker="BBB",
        available_on=date(2024, 5, 1), split="holdout",
    )
    _sample_row(
        memory_db, sample_id="s3", subtype="P", horizon=20, ticker="CCC",
        available_on=date(2020, 6, 1), edge_id="customer",
    )

    cells = load_admitted_cells(memory_db)
    discovery = load_signals(memory_db, cells, split="discovery")

    assert [signal.symbol for signal in discovery] == ["AAA"]


def test_a_cell_whose_edge_is_only_the_alpha_term_is_not_tradeable():
    """The gate grades alpha-and-beta-adjusted returns; a trade can only hedge beta.

    An insider-sale cell can show a large positive label edge made entirely of
    subtracted trailing alpha -- insiders sell into strength. Beta-hedged, the
    same cell loses money, so it must not reach the replay.
    """
    survives = AdmittedCell("insider_transaction", "P", 20, 1, 0.0100, 0.55, 4752)
    artifact = AdmittedCell("insider_transaction", "S", 20, -1, -0.0087, 0.45, 41445)
    marginal = AdmittedCell("insider_transaction", "G", 20, -1, -0.0054, 0.47, 7846)
    hedged_edges = {
        survives.key: 0.01604,
        artifact.key: -0.00293,
        marginal.key: 0.0005,  # positive but under the cost of trading it
    }

    kept = filter_tradeable(
        [survives, artifact, marginal], hedged_edges, min_hedged_edge=0.0015
    )

    assert [cell.event_subtype for cell in kept] == ["P"]


def test_a_cell_with_no_measured_hedged_edge_is_not_assumed_tradeable():
    cell = AdmittedCell("insider_transaction", "P", 20, 1, 0.01, 0.55, 4752)

    assert filter_tradeable([cell], {}, min_hedged_edge=0.0) == []


def test_betas_are_estimated_before_a_cutoff_so_the_hedge_is_knowable(memory_db):
    """A hedge ratio fitted on the era it hedges is a hindsight hedge."""
    for day, beta in (
        (date(2020, 1, 2), 1.0),
        (date(2020, 1, 3), 1.2),
        (date(2024, 6, 3), 9.0),  # after the cutoff: must not influence the ratio
    ):
        memory_db.execute(
            """
            INSERT INTO daily_returns (
                return_id, symbol, price_date, total_return, market_return,
                abnormal_return, beta, method, schema_version
            ) VALUES (?, 'AAA', ?, 0.01, 0.005, 0.005, ?, 'market_model', '1.0.0')
            """,
            [f"r-{day}", day, beta],
        )

    betas = load_betas(memory_db, ["AAA"], before=date(2023, 1, 1))

    assert betas["AAA"] == pytest.approx(1.1)


def test_symbols_without_a_fitted_beta_are_absent_rather_than_zero(memory_db):
    assert load_betas(memory_db, ["NOPE"], before=date(2023, 1, 1)) == {}


def test_replay_inputs_include_the_hedge_series_even_with_no_signal_on_it(memory_db):
    """The hedge instrument needs bars, and it never appears as a signal."""
    _status_row(memory_db, signal_id="a", subtype="P", horizon=20, direction=1, mean_car=0.01)
    _sample_row(
        memory_db, sample_id="s1", subtype="P", horizon=20, ticker="AAA",
        available_on=date(2020, 5, 1),
    )
    for symbol in ("AAA", "SPY"):
        memory_db.execute(
            """
            INSERT INTO daily_prices (
                price_id, symbol, price_date, open, high, low, close, adj_close, volume,
                provider, schema_version
            ) VALUES (?, ?, ?, 10, 11, 9, 10, 10, 1000, 'test', '1.0.0')
            """,
            [f"p-{symbol}", symbol, date(2020, 5, 4)],
        )

    inputs = load_replay_inputs(
        memory_db, hedge_symbol="SPY", split_date=date(2023, 1, 1), tradeable_only=False
    )

    assert "SPY" in inputs.bars
    assert [signal.symbol for signal in inputs.signals["discovery"]] == ["AAA"]


def test_load_bars_returns_chronological_bars_per_symbol(memory_db):
    for day, close in ((date(2024, 1, 3), 11.0), (date(2024, 1, 2), 10.0)):
        memory_db.execute(
            """
            INSERT INTO daily_prices (
                price_id, symbol, price_date, open, high, low, close, adj_close, volume,
                provider, schema_version
            ) VALUES (?, 'AAA', ?, ?, ?, ?, ?, ?, 1000, 'test', '1.0.0')
            """,
            [f"p-{day}", day, close, close * 1.01, close * 0.99, close, close],
        )

    bars = load_bars(memory_db, ["AAA"])

    assert [bar.date for bar in bars["AAA"]] == [date(2024, 1, 2), date(2024, 1, 3)]
