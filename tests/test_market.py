"""Tests for the market price provider and market-side validation.

The provider is exercised through an injected ``fetcher``, so the suite never
touches Yahoo. The frames built here mimic the shape yfinance returns with
``auto_adjust=False, actions=True``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from market_intelligence.clients.market_yfinance import (
    MarketDataUnavailable,
    YFinanceMarketDataProvider,
)
from market_intelligence.schemas.market import (
    ConstituentRecord,
    DailyPriceRecord,
    DailyReturnRecord,
)
from market_intelligence.validators.market import (
    validate_constituent,
    validate_price,
    validate_price_series,
    validate_return,
)

TODAY = date(2024, 3, 1)


def frame(rows: list[dict], *, index: list[str], columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows, index=pd.to_datetime(index))
    return df[columns] if columns else df


def standard_frame() -> pd.DataFrame:
    return frame(
        [
            {
                "Open": 100.0,
                "High": 105.0,
                "Low": 99.0,
                "Close": 104.0,
                "Adj Close": 103.0,
                "Volume": 1_000_000,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            },
            {
                "Open": 104.0,
                "High": 108.0,
                "Low": 103.0,
                "Close": 107.0,
                "Adj Close": 106.0,
                "Volume": 1_200_000,
                "Dividends": 0.5,
                "Stock Splits": 0.0,
            },
        ],
        index=["2024-01-02", "2024-01-03"],
    )


def provider_returning(df: pd.DataFrame | None) -> YFinanceMarketDataProvider:
    return YFinanceMarketDataProvider(fetcher=lambda symbol, start, end: df)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def test_fetch_parses_bars():
    result = provider_returning(standard_frame()).fetch("AAA", date(2024, 1, 1), date(2024, 1, 5))

    assert [bar.date for bar in result.bars] == [date(2024, 1, 2), date(2024, 1, 3)]
    assert result.bars[0].close == pytest.approx(104.0)
    assert result.bars[0].adj_close == pytest.approx(103.0)
    assert result.bars[0].volume == 1_000_000


def test_fetch_extracts_dividends_and_splits():
    df = standard_frame()
    df.loc[pd.Timestamp("2024-01-03"), "Stock Splits"] = 2.0

    result = provider_returning(df).fetch("AAA", date(2024, 1, 1), date(2024, 1, 5))

    kinds = {(a.action_type, a.date) for a in result.actions}
    assert ("dividend", date(2024, 1, 3)) in kinds
    assert ("split", date(2024, 1, 3)) in kinds
    # Non-event days must not produce zero-valued actions.
    assert all(a.date != date(2024, 1, 2) for a in result.actions)


def test_empty_frame_is_a_failure_not_an_absence():
    """A delisted symbol must be loud, not read as 'did not trade'."""
    with pytest.raises(MarketDataUnavailable):
        provider_returning(pd.DataFrame()).fetch("DEAD", date(2024, 1, 1), date(2024, 1, 5))


def test_none_frame_is_a_failure():
    with pytest.raises(MarketDataUnavailable):
        provider_returning(None).fetch("DEAD", date(2024, 1, 1), date(2024, 1, 5))


def test_provider_exception_is_wrapped():
    def explode(symbol, start, end):
        raise RuntimeError("upstream changed shape")

    with pytest.raises(MarketDataUnavailable, match="fetch failed"):
        YFinanceMarketDataProvider(fetcher=explode).fetch("AAA", date(2024, 1, 1), date(2024, 1, 5))


def test_inverted_window_is_rejected():
    with pytest.raises(ValueError, match="after end"):
        provider_returning(standard_frame()).fetch("AAA", date(2024, 5, 1), date(2024, 1, 1))


def test_missing_adj_close_falls_back_to_close():
    df = standard_frame().drop(columns=["Adj Close"])

    result = provider_returning(df).fetch("AAA", date(2024, 1, 1), date(2024, 1, 5))

    assert result.bars[0].adj_close == pytest.approx(result.bars[0].close)


def test_nan_price_becomes_zero_and_is_later_rejected():
    df = standard_frame()
    df.loc[pd.Timestamp("2024-01-02"), "Close"] = float("nan")

    result = provider_returning(df).fetch("AAA", date(2024, 1, 1), date(2024, 1, 5))
    record = DailyPriceRecord(
        price_id="x", symbol="AAA", price_date=result.bars[0].date, close=result.bars[0].close
    )

    assert result.bars[0].close == 0.0
    assert validate_price(record, today=TODAY).is_rejected


def test_raw_csv_is_deterministic():
    first = provider_returning(standard_frame()).fetch("AAA", date(2024, 1, 1), date(2024, 1, 5))
    second = provider_returning(standard_frame()).fetch("AAA", date(2024, 1, 1), date(2024, 1, 5))

    assert first.raw_csv == second.raw_csv


def test_get_daily_prices_satisfies_the_abc():
    bars = provider_returning(standard_frame()).get_daily_prices(
        "AAA", date(2024, 1, 1), date(2024, 1, 5)
    )

    assert len(bars) == 2


# ---------------------------------------------------------------------------
# Price validation
# ---------------------------------------------------------------------------


def price(**overrides) -> DailyPriceRecord:
    defaults = {
        "price_id": "p1",
        "symbol": "AAA",
        "price_date": date(2024, 1, 2),
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 104.0,
        "adj_close": 103.0,
        "volume": 1_000_000,
    }
    return DailyPriceRecord(**{**defaults, **overrides})


def test_valid_bar_passes():
    assert validate_price(price(), today=TODAY).validation_status == "valid"


def test_non_positive_close_is_rejected():
    assert validate_price(price(close=0.0), today=TODAY).is_rejected
    assert validate_price(price(close=-1.0), today=TODAY).is_rejected


def test_non_positive_adj_close_is_rejected():
    assert validate_price(price(adj_close=0.0), today=TODAY).is_rejected


def test_future_bar_is_rejected():
    assert validate_price(price(price_date=date(2030, 1, 1)), today=TODAY).is_rejected


def test_high_below_low_is_rejected():
    assert validate_price(price(high=90.0, low=95.0), today=TODAY).is_rejected


def test_high_below_the_candle_body_is_rejected():
    assert validate_price(price(high=101.0, close=104.0), today=TODAY).is_rejected


def test_low_above_the_candle_body_is_rejected():
    assert validate_price(price(low=101.0, open=100.0), today=TODAY).is_rejected


def test_negative_volume_is_rejected():
    assert validate_price(price(volume=-5), today=TODAY).is_rejected


def test_zero_volume_is_only_a_warning():
    result = validate_price(price(volume=0), today=TODAY)

    assert result.validation_status == "warning"
    assert not result.is_rejected


def test_adj_close_far_above_close_warns():
    result = validate_price(price(close=100.0, adj_close=140.0), today=TODAY)

    assert result.validation_status == "warning"


def test_duplicate_bars_are_rejected():
    records = [price(price_id="a"), price(price_id="b")]

    validate_price_series(records)

    assert not records[0].is_rejected
    assert records[1].is_rejected


@pytest.mark.parametrize(
    ("ratio", "label"),
    [(0.5, "2-for-1"), (1 / 1.5, "3-for-2"), (0.25, "4-for-1")],
)
def test_common_split_ratios_are_flagged(ratio: float, label: str):
    """The threshold must sit below every common split, especially 2-for-1."""
    first = price(price_date=date(2024, 1, 2), adj_close=100.0)
    second = price(price_id="p2", price_date=date(2024, 1, 3), adj_close=100.0 * ratio)

    validate_price_series([first, second])

    assert second.validation_status == "warning", f"{label} split went unflagged"
    assert "unadjusted" in second.validation_errors[0]


def test_ordinary_move_is_not_flagged():
    first = price(price_date=date(2024, 1, 2), adj_close=100.0)
    second = price(price_id="p2", price_date=date(2024, 1, 3), adj_close=94.0)

    validate_price_series([first, second])

    assert second.validation_status == "valid"


# ---------------------------------------------------------------------------
# Constituent validation
# ---------------------------------------------------------------------------


def constituent(**overrides) -> ConstituentRecord:
    defaults = {
        "constituent_id": "c1",
        "index_id": "SP500",
        "ticker": "AAA",
        "cik": "0000000001",
        "added_date": date(2020, 1, 1),
    }
    return ConstituentRecord(**{**defaults, **overrides})


def test_valid_membership_passes():
    assert validate_constituent(constituent()).validation_status == "valid"


def test_removal_before_addition_is_rejected():
    assert validate_constituent(constituent(removed_date=date(2019, 1, 1))).is_rejected


def test_same_day_removal_is_rejected():
    assert validate_constituent(constituent(removed_date=date(2020, 1, 1))).is_rejected


def test_membership_without_a_company_link_warns():
    result = validate_constituent(constituent(cik=None, company_id=None))

    assert result.validation_status == "warning"


def test_missing_ticker_is_rejected():
    assert validate_constituent(constituent(ticker="  ")).is_rejected


def test_membership_window_covers_the_expected_days():
    member = constituent(added_date=date(2020, 1, 1), removed_date=date(2022, 1, 1))

    assert not member.was_member_on(date(2019, 12, 31))
    assert member.was_member_on(date(2020, 1, 1))
    assert member.was_member_on(date(2021, 6, 1))
    assert not member.was_member_on(date(2022, 1, 1))


def test_open_membership_covers_any_later_day():
    assert constituent(added_date=date(2020, 1, 1)).was_member_on(date(2099, 1, 1))


# ---------------------------------------------------------------------------
# Return validation
# ---------------------------------------------------------------------------


def test_lookahead_estimation_window_is_rejected():
    record = DailyReturnRecord(
        return_id="r1",
        symbol="AAA",
        price_date=date(2024, 1, 10),
        estimation_window_start=date(2024, 1, 10),
    )

    result = validate_return(record)

    assert result.is_rejected
    assert "lookahead" in result.validation_errors[0]


def test_trailing_estimation_window_passes():
    record = DailyReturnRecord(
        return_id="r1",
        symbol="AAA",
        price_date=date(2024, 1, 10),
        estimation_window_start=date(2023, 1, 10),
    )

    assert validate_return(record).validation_status == "valid"


def test_market_model_without_a_beta_warns():
    record = DailyReturnRecord(
        return_id="r1",
        symbol="AAA",
        price_date=date(2024, 1, 10),
        abnormal_return=0.01,
        method="market_model",
    )

    assert validate_return(record).validation_status == "warning"
