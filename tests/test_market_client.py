"""Offline tests for the market-data provider interface.

No network, no clock: every assertion uses fixed dates and the deterministic
``MockMarketDataProvider``.
"""

from __future__ import annotations

from datetime import date

import pytest

from market_intelligence.clients.market import (
    CorporateAction,
    DailyPrice,
    LatestPrice,
    MarketDataProvider,
    MockMarketDataProvider,
    UnimplementedMarketDataProvider,
)


def test_daily_prices_count_and_invariants() -> None:
    provider = MockMarketDataProvider()
    prices = provider.get_daily_prices("AAPL", date(2026, 1, 1), date(2026, 1, 5))

    assert len(prices) == 5
    assert [p.date for p in prices] == [
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 3),
        date(2026, 1, 4),
        date(2026, 1, 5),
    ]
    for p in prices:
        assert isinstance(p, DailyPrice)
        assert p.low > 0
        assert p.high >= p.low
        assert p.high >= max(p.open, p.close)
        assert min(p.open, p.close) >= p.low
        assert p.volume > 0


def test_daily_prices_are_deterministic() -> None:
    provider = MockMarketDataProvider()
    first = provider.get_daily_prices("AAPL", date(2026, 1, 1), date(2026, 1, 5))
    second = provider.get_daily_prices("AAPL", date(2026, 1, 1), date(2026, 1, 5))

    assert first == second


def test_latest_price_uses_injected_as_of() -> None:
    provider = MockMarketDataProvider()
    latest = provider.get_latest_price("AAPL", as_of=date(2026, 1, 5))

    assert isinstance(latest, LatestPrice)
    assert latest.symbol == "AAPL"
    assert latest.as_of == date(2026, 1, 5)
    assert latest.price > 0


def test_corporate_actions_returns_list() -> None:
    provider = MockMarketDataProvider()
    actions = provider.get_corporate_actions("AAPL", date(2026, 1, 1), date(2026, 12, 31))

    assert isinstance(actions, list)
    for action in actions:
        assert isinstance(action, CorporateAction)


def test_mock_is_a_market_data_provider() -> None:
    assert isinstance(MockMarketDataProvider(), MarketDataProvider)


def test_unimplemented_provider_raises() -> None:
    provider = UnimplementedMarketDataProvider()

    with pytest.raises(NotImplementedError):
        provider.get_daily_prices("AAPL", date(2026, 1, 1), date(2026, 1, 5))
    with pytest.raises(NotImplementedError):
        provider.get_latest_price("AAPL")
    with pytest.raises(NotImplementedError):
        provider.get_corporate_actions("AAPL", date(2026, 1, 1), date(2026, 1, 5))
