"""Immutable, vendor-neutral inputs used to assess market tradeability.

This module deliberately models observations rather than execution.  A quote,
market-status update, and broker acknowledgement can be supplied by any
licensed vendor, but the resulting :class:`MarketSnapshot` never routes an
order.  Missing or ambiguous information is represented explicitly so the
collector can fail closed and preserve its reasons for suppression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class TradeSide(StrEnum):
    """Candidate direction; this is a research direction, never an order."""

    LONG = "long"
    SHORT = "short"


class MarketSession(StrEnum):
    PREMARKET = "premarket"
    REGULAR = "regular"
    POSTMARKET = "postmarket"
    OPENING_AUCTION = "opening_auction"
    CLOSING_AUCTION = "closing_auction"
    CLOSED = "closed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RealtimeQuote:
    """A top-of-book observation, including both exchange and receipt clocks."""

    symbol: str
    bid: float | None
    ask: float | None
    bid_size: int | None
    ask_size: int | None
    venue: str | None
    exchange_time: datetime | None
    received_at: datetime | None
    sequence: int | None
    is_firm: bool = True
    conditions: tuple[str, ...] = ()

    @property
    def midpoint(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class RealtimeTrade:
    """Most recent consolidated trade, retained independently from the quote."""

    symbol: str
    price: float | None
    size: int | None
    venue: str | None
    exchange_time: datetime | None
    received_at: datetime | None
    sequence: int | None
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketStatus:
    """Trading-status observation from an exchange or consolidated feed."""

    symbol: str
    as_of: datetime | None
    session: MarketSession = MarketSession.UNKNOWN
    is_trading: bool = False
    is_halted: bool = False
    luld_active: bool = False
    ssr_active: bool = False
    session_open_at: datetime | None = None
    session_close_at: datetime | None = None
    reason: str | None = None


@dataclass(frozen=True)
class OneMinuteBar:
    """A completed one-minute volume/price bar; useful for liquidity checks."""

    symbol: str
    start_at: datetime
    end_at: datetime
    volume: int | None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    trade_conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class BorrowStatus:
    """Read-only broker borrow/locate state for a single symbol.

    ``approved_quantity`` and ``locate_id`` are evidence for a short locate;
    they are intentionally not an instruction to borrow or sell short.
    """

    symbol: str
    as_of: datetime | None
    broker_acknowledged_at: datetime | None
    locate_id: str | None = None
    approved_quantity: int | None = None
    expires_at: datetime | None = None
    is_borrowable: bool = False
    is_hard_to_borrow: bool = False
    is_recalled: bool = False
    account_restricted: bool = False
    restrictions: tuple[str, ...] = ()


@dataclass(frozen=True)
class AccountState:
    """Minimal read-only account state needed to detect a broker-wide block."""

    as_of: datetime | None
    broker_acknowledged_at: datetime | None
    account_restricted: bool = False
    restrictions: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketSnapshotPolicy:
    """Fail-closed freshness and session controls for one snapshot evaluation."""

    version: str = "market-snapshot-v1"
    max_quote_age_seconds: float = 5.0
    max_status_age_seconds: float = 5.0
    max_broker_age_seconds: float = 10.0
    max_clock_drift_seconds: float = 2.0
    allow_extended_hours: bool = False
    allow_opening_auction: bool = False
    allow_closing_auction: bool = False
    allow_first_minutes: bool = False
    first_minutes: int = 5
    allow_last_minutes: bool = False
    last_minutes: int = 15
    allow_ssr: bool = False


@dataclass(frozen=True)
class MarketDataHealth:
    """Health result retained with the snapshot instead of recomputed later."""

    healthy: bool
    suppression_reasons: tuple[str, ...] = ()
    quote_age_seconds: float | None = None
    status_age_seconds: float | None = None
    broker_age_seconds: float | None = None
    clock_drift_seconds: float | None = None


@dataclass(frozen=True)
class MarketSnapshot:
    """Frozen decision-time state.  It is immutable and contains no credentials."""

    snapshot_id: str
    symbol: str
    side: TradeSide
    quantity: int
    observed_at: datetime
    policy_version: str
    quote: RealtimeQuote | None
    latest_trade: RealtimeTrade | None
    market_status: MarketStatus | None
    one_minute_bar: OneMinuteBar | None
    borrow: BorrowStatus | None
    account: AccountState | None
    health: MarketDataHealth
    market_provider: str | None = None
    broker_provider: str | None = None

    @property
    def eligible(self) -> bool:
        """Whether base market and broker facts permit further research evaluation."""
        return self.health.healthy

    @property
    def suppression_reasons(self) -> tuple[str, ...]:
        return self.health.suppression_reasons

    def canonical_payload(self) -> dict[str, Any]:
        """Stable, serialization-ready content for audit/replay callers."""
        return asdict(self)


__all__ = [
    "AccountState",
    "BorrowStatus",
    "MarketDataHealth",
    "MarketSession",
    "MarketSnapshot",
    "MarketSnapshotPolicy",
    "MarketStatus",
    "OneMinuteBar",
    "RealtimeQuote",
    "RealtimeTrade",
    "TradeSide",
]
