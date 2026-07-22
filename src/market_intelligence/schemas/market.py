"""Market-side normalized records: index membership, prices, and returns.

Three record types mirror the ``index_constituents`` / ``daily_prices`` /
``daily_returns`` DuckDB tables.

The distinction between the three matters for how much they can be trusted:

* ``ConstituentRecord`` is *reference* data -- who was in an index, and when.
* ``DailyPriceRecord`` is *ingested* data -- what a provider reported.
* ``DailyReturnRecord`` is *derived* data -- what this platform computed from
  the prices. It carries the estimation window that produced its ``beta`` so a
  stored row can be re-derived and checked for lookahead after the fact.
"""

from __future__ import annotations

from datetime import date

from market_intelligence.schemas.common import ProvenanceModel, Source


class ConstituentRecord(ProvenanceModel):
    """One index-membership window (a row of ``index_constituents``).

    A company that leaves and later rejoins an index produces two records, not
    one mutated record, so the membership history stays exact.
    ``removed_date is None`` means the membership is still open.
    """

    constituent_id: str
    index_id: str
    ticker: str
    company_id: str | None = None
    cik: str | None = None
    company_name: str | None = None
    added_date: date
    removed_date: date | None = None
    source: Source = Source.REFERENCE

    def was_member_on(self, day: date) -> bool:
        """Whether this membership window covers ``day`` (inclusive start)."""
        if day < self.added_date:
            return False
        return self.removed_date is None or day < self.removed_date


class DailyPriceRecord(ProvenanceModel):
    """One daily OHLCV bar as reported by a provider (``daily_prices``).

    ``is_delisted_gap`` marks a bar the collector believes is missing because
    the symbol stopped trading, rather than because nothing happened. Free price
    APIs return an empty series for a dead ticker, which reads as "no data"
    instead of "-100%"; flagging it keeps that ambiguity explicit rather than
    letting a delisting silently look like a flat day.
    """

    price_id: str
    symbol: str
    price_date: date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    adj_close: float | None = None
    volume: int | None = None
    provider: str | None = None
    is_delisted_gap: bool = False
    source: Source = Source.MARKET


class DailyReturnRecord(ProvenanceModel):
    """One derived daily return row (``daily_returns``).

    ``abnormal_return`` is the label the signal layer is graded against: the
    part of a day's move not explained by the market (and, where available, the
    sector). ``estimation_window_start`` records the trailing window used to fit
    ``alpha``/``beta``; that window always ends strictly before ``price_date``.
    """

    return_id: str
    symbol: str
    price_date: date
    total_return: float | None = None
    market_return: float | None = None
    sector_return: float | None = None
    abnormal_return: float | None = None
    beta: float | None = None
    alpha: float | None = None
    method: str | None = None
    estimation_window_start: date | None = None
    source: Source = Source.DERIVED


__all__ = [
    "ConstituentRecord",
    "DailyPriceRecord",
    "DailyReturnRecord",
]
