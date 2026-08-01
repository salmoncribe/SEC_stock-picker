"""Read-only protocols for licensed real-time market and broker adapters.

No concrete network client is registered here.  Until a vendor is selected,
the unavailable adapters make collection ineligible rather than substituting
daily data or mock prices.  Protocols intentionally omit every order-routing,
order-modification, and account-mutation method.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from market_intelligence.schemas.market_snapshot import (
    AccountState,
    BorrowStatus,
    MarketStatus,
    OneMinuteBar,
    RealtimeQuote,
    RealtimeTrade,
)


class RealtimeDataUnavailable(RuntimeError):
    """A required read-only real-time observation could not be obtained."""


@runtime_checkable
class RealtimeMarketClient(Protocol):
    """Vendor-neutral, read-only real-time market-data contract."""

    provider_name: str

    def get_quote(self, symbol: str) -> RealtimeQuote | None:
        """Return current NBBO/top-of-book data, or ``None`` when unavailable."""

    def get_latest_trade(self, symbol: str) -> RealtimeTrade | None:
        """Return the latest consolidated trade and its conditions."""

    def get_market_status(self, symbol: str) -> MarketStatus | None:
        """Return current session, halt, LULD, and SSR state."""

    def get_latest_one_minute_bar(self, symbol: str, *, as_of: datetime) -> OneMinuteBar | None:
        """Return the latest completed one-minute bar at ``as_of``."""


@runtime_checkable
class ReadOnlyBrokerClient(Protocol):
    """Broker contract restricted to state, restrictions, and borrow evidence."""

    provider_name: str

    def get_account_state(self) -> AccountState | None:
        """Return read-only account restriction state; never mutate it."""

    def get_borrow_status(self, symbol: str) -> BorrowStatus | None:
        """Return read-only locate/borrow state for ``symbol``."""


class UnavailableRealtimeMarketClient:
    """Default adapter: live data is intentionally unconfigured and unusable."""

    provider_name = "unconfigured-realtime-market"
    _MESSAGE = "No licensed real-time market-data adapter is configured; research is suppressed."

    def get_quote(self, symbol: str) -> RealtimeQuote | None:
        raise RealtimeDataUnavailable(self._MESSAGE)

    def get_latest_trade(self, symbol: str) -> RealtimeTrade | None:
        raise RealtimeDataUnavailable(self._MESSAGE)

    def get_market_status(self, symbol: str) -> MarketStatus | None:
        raise RealtimeDataUnavailable(self._MESSAGE)

    def get_latest_one_minute_bar(self, symbol: str, *, as_of: datetime) -> OneMinuteBar | None:
        raise RealtimeDataUnavailable(self._MESSAGE)


class UnavailableReadOnlyBrokerClient:
    """Default adapter: no broker state is assumed or invented."""

    provider_name = "unconfigured-readonly-broker"
    _MESSAGE = "No read-only broker adapter is configured; research is suppressed."

    def get_account_state(self) -> AccountState | None:
        raise RealtimeDataUnavailable(self._MESSAGE)

    def get_borrow_status(self, symbol: str) -> BorrowStatus | None:
        raise RealtimeDataUnavailable(self._MESSAGE)


__all__ = [
    "ReadOnlyBrokerClient",
    "RealtimeDataUnavailable",
    "RealtimeMarketClient",
    "UnavailableReadOnlyBrokerClient",
    "UnavailableRealtimeMarketClient",
]
