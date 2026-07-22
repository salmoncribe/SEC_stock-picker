"""Validation rules for market-side records.

Follows the platform contract: a problem is recorded on the record itself and
classified ``warning`` (stored, flagged) or ``rejected`` (kept for audit, not
persisted to the analytical tables). Nothing is ever silently discarded.

The price rules exist because the price source is an unofficial endpoint that
can change shape without notice (see ``clients.market_yfinance``). Rather than
trusting it, every bar is checked against invariants that are true of any real
OHLCV bar, so a provider regression shows up as a spike in rejections rather
than as quietly wrong returns.
"""

from __future__ import annotations

from datetime import date

from market_intelligence.schemas.market import (
    ConstituentRecord,
    DailyPriceRecord,
    DailyReturnRecord,
)

# A daily move beyond this is flagged as a likely unadjusted corporate action.
#
# The value is set by the splits it must catch, not picked as a round number.
# The most common split ratios produce these apparent one-day moves:
#
#     2-for-1  -> -50.0%      3-for-2 -> -33.3%      4-for-1 -> -75.0%
#
# A threshold above 33% would miss 3-for-2 entirely and, worse, a threshold
# above 50% would miss 2-for-1 -- by far the most frequent split there is. Real
# one-day moves of this size do happen (an earnings collapse, a failed trial),
# but they are rare enough that flagging them costs little: this raises a
# warning, never a rejection, so a true move is recorded and merely annotated.
EXTREME_DAILY_MOVE = 0.30

# Adjusted close should not exceed raw close by a wide margin: adjustment
# divides out dividends and splits, so adj_close <= close for any normal
# history. A large excess suggests the two columns were mixed up.
ADJ_CLOSE_TOLERANCE = 1.05


def validate_price(record: DailyPriceRecord, *, today: date | None = None) -> DailyPriceRecord:
    """Validate one OHLCV bar in isolation.

    Rejections are reserved for bars that cannot be true of a real security:
    a non-positive close (which also catches a missing price coerced to 0.0),
    an impossible high/low ordering, or negative volume. Everything else is a
    warning.
    """
    if record.price_date is None:
        record.add_error("price_date is required", reject=True)
        return record

    horizon = today or date.today()
    if record.price_date > horizon:
        record.add_error(f"price_date {record.price_date} is in the future", reject=True)

    if record.close is None or record.close <= 0:
        record.add_error(f"close must be positive, got {record.close}", reject=True)

    if record.adj_close is None or record.adj_close <= 0:
        record.add_error(f"adj_close must be positive, got {record.adj_close}", reject=True)

    if record.volume is not None and record.volume < 0:
        record.add_error(f"volume must be non-negative, got {record.volume}", reject=True)

    high, low = record.high, record.low
    if high is not None and low is not None and high < low:
        record.add_error(f"high {high} is below low {low}", reject=True)

    # The body of the candle must sit inside the high/low range.
    body = [value for value in (record.open, record.close) if value is not None and value > 0]
    if body:
        if high is not None and high > 0 and high < max(body):
            record.add_error(f"high {high} is below the open/close body {max(body)}", reject=True)
        if low is not None and low > 0 and low > min(body):
            record.add_error(f"low {low} is above the open/close body {min(body)}", reject=True)

    if record.close and record.adj_close and record.adj_close > record.close * ADJ_CLOSE_TOLERANCE:
        record.add_error(
            f"adj_close {record.adj_close} exceeds close {record.close}; columns may be swapped"
        )

    if record.volume == 0:
        record.add_error("zero volume: the bar may be a placeholder rather than a trading day")

    return record


def validate_price_series(records: list[DailyPriceRecord]) -> list[DailyPriceRecord]:
    """Validate a symbol's bars against each other.

    Catches problems invisible to a single bar: duplicate dates, and moves so
    large they usually indicate an unadjusted corporate action rather than a
    real price change.
    """
    ordered = sorted(
        (r for r in records if r.price_date is not None),
        key=lambda r: r.price_date,
    )

    seen: dict[date, DailyPriceRecord] = {}
    for record in ordered:
        if record.price_date in seen:
            record.add_error(f"duplicate bar for {record.price_date}", reject=True)
        else:
            seen[record.price_date] = record

    previous: DailyPriceRecord | None = None
    for record in ordered:
        if record.is_rejected:
            continue
        if (
            previous is not None
            and previous.adj_close
            and record.adj_close
            and abs(record.adj_close / previous.adj_close - 1.0) > EXTREME_DAILY_MOVE
        ):
            record.add_error(
                f"move of {record.adj_close / previous.adj_close - 1.0:.1%} from "
                f"{previous.price_date}: possible unadjusted corporate action"
            )
        previous = record

    return records


def validate_constituent(record: ConstituentRecord) -> ConstituentRecord:
    """Validate one index-membership window.

    An inverted or empty window would corrupt every point-in-time universe
    query built on it, so both are rejections.
    """
    if not record.ticker or not record.ticker.strip():
        record.add_error("ticker is required", reject=True)

    if not record.index_id or not record.index_id.strip():
        record.add_error("index_id is required", reject=True)

    if record.added_date is None:
        record.add_error("added_date is required", reject=True)
        return record

    if record.removed_date is not None and record.removed_date <= record.added_date:
        record.add_error(
            f"removed_date {record.removed_date} is not after added_date {record.added_date}",
            reject=True,
        )

    if record.cik is None and record.company_id is None:
        record.add_error("membership has no company_id or cik; it cannot be joined to filings")

    return record


def validate_return(record: DailyReturnRecord) -> DailyReturnRecord:
    """Validate one derived return row.

    The lookahead check is the important one: a beta fitted on a window that
    does not end strictly before the day it prices is a leak, and a leak is
    worth more than a warning because every downstream statistic inherits it.
    """
    if record.price_date is None:
        record.add_error("price_date is required", reject=True)
        return record

    if (
        record.estimation_window_start is not None
        and record.estimation_window_start >= record.price_date
    ):
        record.add_error(
            f"estimation window starts {record.estimation_window_start}, on or after the "
            f"day it prices ({record.price_date}): lookahead",
            reject=True,
        )

    if record.total_return is not None and abs(record.total_return) > EXTREME_DAILY_MOVE:
        record.add_error(f"extreme total_return {record.total_return:.1%}")

    if (
        record.abnormal_return is not None
        and record.beta is None
        and record.method == "market_model"
    ):
        record.add_error("market_model abnormal return recorded without a fitted beta")

    return record


__all__ = [
    "ADJ_CLOSE_TOLERANCE",
    "EXTREME_DAILY_MOVE",
    "validate_constituent",
    "validate_price",
    "validate_price_series",
    "validate_return",
]
