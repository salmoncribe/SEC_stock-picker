"""Offline fail-closed tests for real-time market and read-only broker contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from market_intelligence.clients.realtime_market import (
    ReadOnlyBrokerClient,
    RealtimeMarketClient,
    UnavailableReadOnlyBrokerClient,
    UnavailableRealtimeMarketClient,
)
from market_intelligence.collectors.market_snapshots import (
    capture_market_snapshot,
    evaluate_market_data_health,
)
from market_intelligence.schemas.market_snapshot import (
    AccountState,
    BorrowStatus,
    MarketSession,
    MarketSnapshotPolicy,
    MarketStatus,
    RealtimeQuote,
    RealtimeTrade,
    TradeSide,
)

NOW = datetime(2026, 7, 24, 15, 0, tzinfo=UTC)


def quote(**changes: object) -> RealtimeQuote:
    defaults: dict[str, object] = {
        "symbol": "AAA",
        "bid": 100.0,
        "ask": 100.05,
        "bid_size": 500,
        "ask_size": 600,
        "venue": "XNYS",
        "exchange_time": NOW - timedelta(milliseconds=200),
        "received_at": NOW - timedelta(milliseconds=100),
        "sequence": 101,
    }
    return RealtimeQuote(**{**defaults, **changes})  # type: ignore[arg-type]


def status(**changes: object) -> MarketStatus:
    defaults: dict[str, object] = {
        "symbol": "AAA",
        "as_of": NOW - timedelta(seconds=1),
        "session": MarketSession.REGULAR,
        "is_trading": True,
        "session_open_at": NOW - timedelta(minutes=20),
        "session_close_at": NOW + timedelta(minutes=30),
    }
    return MarketStatus(**{**defaults, **changes})  # type: ignore[arg-type]


def account(**changes: object) -> AccountState:
    defaults: dict[str, object] = {
        "as_of": NOW - timedelta(seconds=1),
        "broker_acknowledged_at": NOW - timedelta(seconds=1),
    }
    return AccountState(**{**defaults, **changes})  # type: ignore[arg-type]


def borrow(**changes: object) -> BorrowStatus:
    defaults: dict[str, object] = {
        "symbol": "AAA",
        "as_of": NOW - timedelta(seconds=1),
        "broker_acknowledged_at": NOW - timedelta(seconds=1),
        "locate_id": "loc-123",
        "approved_quantity": 100,
        "expires_at": NOW + timedelta(minutes=10),
        "is_borrowable": True,
    }
    return BorrowStatus(**{**defaults, **changes})  # type: ignore[arg-type]


def health(**changes: object):
    values = {
        "symbol": "AAA",
        "side": TradeSide.LONG,
        "quantity": 25,
        "observed_at": NOW,
        "quote": quote(),
        "market_status": status(),
        "account": account(),
        "borrow": None,
        "policy": MarketSnapshotPolicy(),
    }
    return evaluate_market_data_health(**{**values, **changes})


def test_healthy_regular_long_is_eligible() -> None:
    assert health().healthy


@pytest.mark.parametrize(
    ("bad_quote", "reason"),
    [
        (None, "no_quote"),
        (quote(bid=100.05, ask=100.0), "quote_crossed"),
        (quote(bid=100.0, ask=100.0), "quote_locked"),
        (quote(received_at=NOW - timedelta(seconds=6)), "quote_stale"),
    ],
)
def test_quote_problems_fail_closed(bad_quote: RealtimeQuote | None, reason: str) -> None:
    result = health(quote=bad_quote)
    assert not result.healthy
    assert reason in result.suppression_reasons


def test_feed_gap_and_clock_drift_fail_closed() -> None:
    result = health(
        quote=quote(sequence=103, exchange_time=NOW - timedelta(seconds=4)),
        previous_feed_sequence=101,
    )
    assert {"feed_sequence_gap", "quote_clock_drift"} <= set(result.suppression_reasons)


@pytest.mark.parametrize(
    ("bad_status", "reason"),
    [
        (status(is_trading=False), "market_halt"),
        (status(luld_active=True), "luld_active"),
        (status(ssr_active=True), "ssr_active"),
        (status(session=MarketSession.PREMARKET), "outside_regular_session"),
    ],
)
def test_status_blocks_fail_closed(bad_status: MarketStatus, reason: str) -> None:
    result = health(market_status=bad_status)
    assert not result.healthy
    assert reason in result.suppression_reasons


def test_short_requires_current_sufficient_unrecalled_locate() -> None:
    result = health(
        side=TradeSide.SHORT,
        quantity=101,
        borrow=borrow(approved_quantity=100, is_recalled=True),
    )
    assert {"borrow_recalled", "locate_quantity_exceeded"} <= set(result.suppression_reasons)


def test_delayed_broker_acknowledgement_blocks_both_directions() -> None:
    result = health(account=account(broker_acknowledged_at=NOW - timedelta(seconds=11)))
    assert not result.healthy
    assert "broker_acknowledgment_delayed" in result.suppression_reasons


class PaperMarket:
    provider_name = "paper-feed"

    def get_quote(self, symbol: str) -> RealtimeQuote | None:
        return quote(symbol=symbol)

    def get_latest_trade(self, symbol: str) -> RealtimeTrade | None:
        return RealtimeTrade(
            symbol=symbol,
            price=100.03,
            size=100,
            venue="XNYS",
            exchange_time=NOW - timedelta(milliseconds=200),
            received_at=NOW - timedelta(milliseconds=100),
            sequence=101,
        )

    def get_market_status(self, symbol: str) -> MarketStatus | None:
        return status(symbol=symbol)

    def get_latest_one_minute_bar(self, symbol: str, *, as_of: datetime):
        return None


class PaperBroker:
    provider_name = "paper-broker"

    def get_account_state(self) -> AccountState | None:
        return account()

    def get_borrow_status(self, symbol: str) -> BorrowStatus | None:
        return borrow(symbol=symbol)


def test_protocols_are_read_only_and_snapshot_is_reproducible() -> None:
    market = PaperMarket()
    broker = PaperBroker()
    assert isinstance(market, RealtimeMarketClient)
    assert isinstance(broker, ReadOnlyBrokerClient)

    first = capture_market_snapshot(
        market, broker, symbol="aaa", side="short", quantity=25, observed_at=NOW
    )
    second = capture_market_snapshot(
        market, broker, symbol="AAA", side=TradeSide.SHORT, quantity=25, observed_at=NOW
    )
    assert first.eligible
    assert first.snapshot_id == second.snapshot_id
    with pytest.raises(FrozenInstanceError):
        first.quantity = 30  # type: ignore[misc]


def test_unconfigured_adapters_are_research_only() -> None:
    snapshot = capture_market_snapshot(
        UnavailableRealtimeMarketClient(),
        UnavailableReadOnlyBrokerClient(),
        symbol="AAA",
        side=TradeSide.LONG,
        quantity=25,
        observed_at=NOW,
    )
    assert not snapshot.eligible
    assert "market_data_unavailable" in snapshot.suppression_reasons
    assert "broker_data_unavailable" in snapshot.suppression_reasons
