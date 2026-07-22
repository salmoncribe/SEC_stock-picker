"""Yahoo Finance implementation of ``MarketDataProvider``.

This is the first live price source in the platform. Two things about it are
worth stating plainly, because they differ from the SEC and FRED clients.

**It is an unofficial endpoint.** Yahoo publishes no terms-covered API for this
data; ``yfinance`` reverse-engineers a public one that can change shape without
notice. That is a deliberate trade for zero cost, and it is contained: this
class is the only place that knows about Yahoo. Swapping in a licensed EOD feed
means implementing ``MarketDataProvider`` once and changing a registration --
nothing downstream of the collector is aware of where prices came from.

**Raw preservation is a snapshot, not a wire capture.** Every other client in
this platform hands the raw HTTP response to the raw store byte-for-byte.
``yfinance`` owns its own HTTP layer and returns a DataFrame, so what gets
preserved here is that frame serialized to CSV: the provider's *output*, not
the bytes on the wire. It is still content-hashed, deterministic, and
re-derivable, but it is one transformation removed from the source and is
labelled as such rather than passed off as a verbatim response.

Two ``yfinance`` defaults are overridden, both of which would otherwise corrupt
the data quietly:

* ``auto_adjust=True`` (default) folds split/dividend adjustment into OHLC and
  drops ``Adj Close``, so the raw close is lost. We want both -- the unadjusted
  close for reporting and the adjusted close for returns.
* ``raise_errors=False`` (default) makes a failed fetch return an *empty frame*
  rather than raise. That is precisely the failure this platform must not
  swallow: a delisted or mistyped symbol would read as "nothing happened"
  instead of "the request failed". It is turned on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from market_intelligence.clients.market import (
    CorporateAction,
    DailyPrice,
    LatestPrice,
    MarketDataProvider,
)
from market_intelligence.logging_config import get_logger

_log = get_logger("clients.market_yfinance")

PROVIDER_NAME = "yfinance"

# Columns requested from yfinance with auto_adjust=False, actions=True.
_COLUMN_OPEN = "Open"
_COLUMN_HIGH = "High"
_COLUMN_LOW = "Low"
_COLUMN_CLOSE = "Close"
_COLUMN_ADJ_CLOSE = "Adj Close"
_COLUMN_VOLUME = "Volume"
_COLUMN_DIVIDEND = "Dividends"
_COLUMN_SPLIT = "Stock Splits"


class MarketDataUnavailable(Exception):
    """A price fetch failed or returned nothing usable for a symbol.

    Raised rather than returning an empty list so a dead symbol is a loud,
    countable failure instead of an absence the collector might read as "this
    company simply did not trade".
    """


@dataclass(frozen=True)
class PriceFetch:
    """One symbol's fetched bars plus the artifact to preserve in the raw store."""

    symbol: str
    bars: list[DailyPrice]
    actions: list[CorporateAction]
    raw_csv: bytes
    source_url: str


def _to_date(value: Any) -> date:
    """Coerce a pandas timestamp (tz-aware or not) to a plain calendar date."""
    if isinstance(value, date) and not hasattr(value, "date"):
        return value
    return value.date()


def _to_float(value: Any) -> float | None:
    """Coerce a cell to float, mapping NaN/None to ``None``."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # NaN is the only float that is not equal to itself.
    return None if result != result else result


def _to_int(value: Any) -> int | None:
    result = _to_float(value)
    return None if result is None else int(result)


class YFinanceMarketDataProvider(MarketDataProvider):
    """Live end-of-day prices from Yahoo Finance via ``yfinance``.

    ``fetcher`` exists for testability: it is called as
    ``fetcher(symbol, start, end)`` and must return a pandas DataFrame shaped
    like ``yfinance``'s output. Tests inject a stub so the suite never touches
    the network; production leaves it ``None`` and the real library is used.
    """

    def __init__(
        self,
        *,
        fetcher: Callable[[str, date, date], Any] | None = None,
        timeout: int = 30,
        logger: Any = None,
    ) -> None:
        self._fetcher = fetcher
        self._timeout = timeout
        self._log = logger or _log

    # -- fetching ----------------------------------------------------------- #
    def _default_fetcher(self, symbol: str, start: date, end: date) -> Any:
        import yfinance

        ticker = yfinance.Ticker(symbol)
        return ticker.history(
            start=start.isoformat(),
            # yfinance treats `end` as exclusive; add a day so the caller's
            # inclusive [start, end] contract holds.
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=True,
            raise_errors=True,
            timeout=self._timeout,
        )

    def fetch(self, symbol: str, start: date, end: date) -> PriceFetch:
        """Fetch bars and corporate actions for ``symbol`` over ``[start, end]``.

        Raises ``MarketDataUnavailable`` when the provider errors or returns an
        empty frame. An empty frame is treated as a failure, not as a symbol
        that happened not to trade: distinguishing "no data" from "no trading"
        is impossible here, and silently accepting it is how a delisting turns
        into a flat line in the return series.
        """
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        fetcher = self._fetcher or self._default_fetcher
        try:
            frame = fetcher(symbol, start, end)
        except Exception as exc:  # provider-specific failures are opaque
            raise MarketDataUnavailable(f"{symbol}: fetch failed: {exc}") from exc

        if frame is None or len(frame) == 0:
            raise MarketDataUnavailable(f"{symbol}: provider returned no rows for {start}..{end}")

        bars, actions = self._parse_frame(symbol, frame)
        if not bars:
            raise MarketDataUnavailable(f"{symbol}: no usable bars in provider response")

        return PriceFetch(
            symbol=symbol,
            bars=bars,
            actions=actions,
            raw_csv=frame.to_csv().encode("utf-8"),
            source_url=f"https://finance.yahoo.com/quote/{symbol}/history",
        )

    def _parse_frame(
        self, symbol: str, frame: Any
    ) -> tuple[list[DailyPrice], list[CorporateAction]]:
        """Translate a yfinance DataFrame into normalized records."""
        columns = set(frame.columns)
        bars: list[DailyPrice] = []
        actions: list[CorporateAction] = []

        for index, row in frame.iterrows():
            day = _to_date(index)

            close = _to_float(row.get(_COLUMN_CLOSE))
            # Older/adjusted responses may omit Adj Close; fall back to close so
            # the bar is still usable, and let validation flag the difference.
            adj_close = (
                _to_float(row.get(_COLUMN_ADJ_CLOSE)) if _COLUMN_ADJ_CLOSE in columns else close
            )

            # A missing numeric becomes 0.0 rather than None, because DailyPrice
            # declares plain floats. That is not a silent default: validation
            # rejects any bar with a non-positive close, so a missing price
            # surfaces as a counted rejection instead of a plausible-looking
            # zero. See validators.market.validate_price.
            bars.append(
                DailyPrice(
                    symbol=symbol,
                    date=day,
                    open=_to_float(row.get(_COLUMN_OPEN)) or 0.0,
                    high=_to_float(row.get(_COLUMN_HIGH)) or 0.0,
                    low=_to_float(row.get(_COLUMN_LOW)) or 0.0,
                    close=close or 0.0,
                    adj_close=adj_close or 0.0,
                    volume=_to_int(row.get(_COLUMN_VOLUME)) or 0,
                )
            )

            dividend = _to_float(row.get(_COLUMN_DIVIDEND)) if _COLUMN_DIVIDEND in columns else None
            if dividend:
                actions.append(
                    CorporateAction(
                        symbol=symbol,
                        date=day,
                        action_type="dividend",
                        value=dividend,
                        details=f"{PROVIDER_NAME} reported cash dividend",
                    )
                )

            split = _to_float(row.get(_COLUMN_SPLIT)) if _COLUMN_SPLIT in columns else None
            # yfinance reports 0.0 on non-split days and the ratio on split days.
            if split:
                actions.append(
                    CorporateAction(
                        symbol=symbol,
                        date=day,
                        action_type="split",
                        value=split,
                        details=f"{PROVIDER_NAME} reported {split}-for-1 split",
                    )
                )

        return bars, actions

    # -- MarketDataProvider ------------------------------------------------- #
    def get_daily_prices(self, symbol: str, start: date, end: date) -> list[DailyPrice]:
        return self.fetch(symbol, start, end).bars

    def get_latest_price(self, symbol: str, *, as_of: date | None = None) -> LatestPrice:
        """Return the most recent bar's close within a short trailing window."""
        end = as_of or date.today()
        # 10 calendar days comfortably spans a long weekend plus holidays.
        bars = self.fetch(symbol, end - timedelta(days=10), end).bars
        if not bars:
            raise MarketDataUnavailable(f"{symbol}: no recent bars on or before {end}")
        latest = max(bars, key=lambda bar: bar.date)
        return LatestPrice(symbol=symbol, price=latest.close, as_of=latest.date)

    def get_corporate_actions(self, symbol: str, start: date, end: date) -> list[CorporateAction]:
        return self.fetch(symbol, start, end).actions


__all__ = [
    "PROVIDER_NAME",
    "MarketDataUnavailable",
    "PriceFetch",
    "YFinanceMarketDataProvider",
]
