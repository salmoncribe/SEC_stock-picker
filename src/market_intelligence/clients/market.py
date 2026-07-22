"""Provider interface for future market-price data (extension point).

This module defines the contract a market-price data source must implement so
the rest of the platform can consume daily OHLCV bars, latest prices, and
corporate actions without caring where they come from. It deliberately does
**not** wire up a live data source.

What ships in this baseline:

* ``MarketDataProvider`` -- the abstract contract (protocol/ABC) a real source
  must satisfy.
* ``MockMarketDataProvider`` -- a fully deterministic, offline implementation
  used by tests and local development. Its values are computed from a stable
  hash of ``(symbol, date)``. They are SYNTHETIC and are **not real market
  prices**; they must never be treated as such, stored as authoritative, or
  shown to a user as real quotes.
* ``UnimplementedMarketDataProvider`` -- a production placeholder whose methods
  all raise ``NotImplementedError``. It documents, in code, that no live price
  source is configured in this phase.

To add a real source, implement ``MarketDataProvider`` against a licensed
end-of-day price API (one with documented, reliable, terms-compliant access)
and register it in place of the placeholder. This baseline intentionally ships
no live price source, and depends on no undocumented or unreliable endpoint.
"""

from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass
from datetime import date, timedelta

# Synthetic price band for the mock. Plausible-looking, but arbitrary: these
# bounds shape fake numbers and carry no market meaning.
_PRICE_MIN = 5.0
_PRICE_SPAN = 495.0


@dataclass(frozen=True)
class DailyPrice:
    """A single daily OHLCV bar for one symbol."""

    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int


@dataclass(frozen=True)
class LatestPrice:
    """The most recent known price for one symbol, as of a given date."""

    symbol: str
    price: float
    as_of: date


@dataclass(frozen=True)
class CorporateAction:
    """A corporate action (e.g. a dividend or a split) for one symbol."""

    symbol: str
    date: date
    action_type: str
    value: float
    details: str | None = None


class MarketDataProvider(abc.ABC):
    """Abstract contract for a market-price data source.

    Implementations return normalized price data for a symbol. A real provider
    should back these methods with a licensed end-of-day price API.
    """

    @abc.abstractmethod
    def get_daily_prices(self, symbol: str, start: date, end: date) -> list[DailyPrice]:
        """Return one OHLCV bar per calendar day in ``[start, end]`` inclusive."""

    @abc.abstractmethod
    def get_latest_price(self, symbol: str) -> LatestPrice:
        """Return the most recent known price for ``symbol``."""

    @abc.abstractmethod
    def get_corporate_actions(self, symbol: str, start: date, end: date) -> list[CorporateAction]:
        """Return corporate actions for ``symbol`` within ``[start, end]`` inclusive."""


def _seed(symbol: str, d: date) -> int:
    """Stable 256-bit seed derived from ``(symbol, iso_date)``.

    Deterministic and offline: the same inputs always yield the same seed.
    """
    digest = hashlib.sha256(f"{symbol}:{d.isoformat()}".encode()).hexdigest()
    return int(digest, 16)


def _unit(seed: int, slot: int) -> float:
    """Extract a deterministic float in ``[0.0, 1.0)`` from one 16-bit slot."""
    return ((seed >> (slot * 16)) & 0xFFFF) / 65536.0


class MockMarketDataProvider(MarketDataProvider):
    """SYNTHETIC test data -- deterministic, NOT real market prices.

    Every value is a pure function of ``(symbol, date)`` via a SHA-256 seed, so
    identical inputs always produce identical outputs and nothing touches the
    network or a clock (except the optional ``today()`` fallback in
    ``get_latest_price`` when no ``as_of`` is supplied). These numbers are fake
    and must never be used as, or mistaken for, real quotes.
    """

    def _daily_price(self, symbol: str, d: date) -> DailyPrice:
        """Build one deterministic, invariant-satisfying bar for ``(symbol, d)``."""
        seed = _seed(symbol, d)

        base = _PRICE_MIN + _unit(seed, 0) * _PRICE_SPAN
        close = base * (1.0 + (_unit(seed, 1) - 0.5) * 0.10)  # +/- 5% vs. base
        high_extra = _unit(seed, 2) * 0.03  # up to +3% above the body
        low_extra = _unit(seed, 3) * 0.03  # up to -3% below the body

        open_px = round(base, 2)
        close_px = round(close, 2)
        body_top = max(open_px, close_px)
        body_bottom = min(open_px, close_px)

        # Round first, then re-assert the ordering invariants so rounding can
        # never flip high/low relative to the body.
        high_px = max(round(body_top * (1.0 + high_extra), 2), body_top)
        low_px = min(round(body_bottom * (1.0 - low_extra), 2), body_bottom)
        low_px = max(low_px, 0.01)  # keep strictly positive

        adj_close = max(round(close_px * (1.0 - _unit(seed, 4) * 0.02), 2), 0.01)
        volume = 1000 + seed % 9_000_000  # always > 0

        return DailyPrice(
            symbol=symbol,
            date=d,
            open=open_px,
            high=high_px,
            low=low_px,
            close=close_px,
            adj_close=adj_close,
            volume=volume,
        )

    def get_daily_prices(self, symbol: str, start: date, end: date) -> list[DailyPrice]:
        """Return one synthetic bar per calendar day in ``[start, end]`` inclusive."""
        prices: list[DailyPrice] = []
        day = start
        while day <= end:
            prices.append(self._daily_price(symbol, day))
            day += timedelta(days=1)
        return prices

    def get_latest_price(self, symbol: str, *, as_of: date | None = None) -> LatestPrice:
        """Return the synthetic price for ``as_of`` (defaults to today if omitted).

        Tests always pass ``as_of`` to stay deterministic; the ``today()``
        fallback exists only for convenience in ad-hoc local use.
        """
        when = as_of if as_of is not None else date.today()
        return LatestPrice(symbol=symbol, price=self._daily_price(symbol, when).close, as_of=when)

    def get_corporate_actions(self, symbol: str, start: date, end: date) -> list[CorporateAction]:
        """Return a small deterministic list of synthetic actions (may be empty).

        The count (0-2) and contents are a pure function of the symbol and the
        window, so results are stable across runs.
        """
        span = (end - start).days
        if span < 0:
            return []

        symbol_seed = int(hashlib.sha256(symbol.encode()).hexdigest(), 16)
        count = symbol_seed % 3  # 0, 1, or 2 actions

        actions: list[CorporateAction] = []
        for i in range(count):
            slot = symbol_seed >> (i * 24)
            when = start + timedelta(days=(slot % (span + 1)))
            if i % 2 == 0:
                action_type = "dividend"
                value = round(0.10 + (slot % 200) / 100.0, 2)
                details = "Synthetic quarterly dividend (mock data)."
            else:
                action_type = "split"
                value = 2.0
                details = "Synthetic 2-for-1 split (mock data)."
            actions.append(
                CorporateAction(
                    symbol=symbol,
                    date=when,
                    action_type=action_type,
                    value=value,
                    details=details,
                )
            )
        return actions


class UnimplementedMarketDataProvider(MarketDataProvider):
    """Production placeholder: no live market-data source is configured.

    Every method raises ``NotImplementedError``. Replace this with a real
    ``MarketDataProvider`` implementation once a licensed price source is wired
    up.
    """

    _MESSAGE = (
        "No production market-data provider is configured. Implement "
        "MarketDataProvider against a licensed EOD price API and register it "
        "here. This baseline intentionally ships no live price source."
    )

    def get_daily_prices(self, symbol: str, start: date, end: date) -> list[DailyPrice]:
        raise NotImplementedError(self._MESSAGE)

    def get_latest_price(self, symbol: str) -> LatestPrice:
        raise NotImplementedError(self._MESSAGE)

    def get_corporate_actions(self, symbol: str, start: date, end: date) -> list[CorporateAction]:
        raise NotImplementedError(self._MESSAGE)


__all__ = [
    "CorporateAction",
    "DailyPrice",
    "LatestPrice",
    "MarketDataProvider",
    "MockMarketDataProvider",
    "UnimplementedMarketDataProvider",
]
