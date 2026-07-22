"""Offline tests for the price collector.

Every test injects a stub ``fetcher`` into ``YFinanceMarketDataProvider`` (see
``clients.market_yfinance``); nothing here touches the network or yfinance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import pytest

from market_intelligence import database
from market_intelligence.clients.market_yfinance import (
    MarketDataUnavailable,
    YFinanceMarketDataProvider,
)
from market_intelligence.collectors import prices as price_collector
from market_intelligence.config import Config
from market_intelligence.storage import duckdb as duckdb_store

END = date(2024, 3, 1)


def bars_frame(dates: list[date], *, base: float = 100.0) -> pd.DataFrame:
    """A minimal frame that satisfies every ``validate_price`` invariant."""
    rows = []
    for i in range(len(dates)):
        level = base + i
        rows.append(
            {
                "Open": level,
                "High": level + 2.0,
                "Low": level - 2.0,
                "Close": level + 1.0,
                "Adj Close": level + 1.0,
                "Volume": 1_000_000,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            }
        )
    return pd.DataFrame(rows, index=pd.to_datetime([d.isoformat() for d in dates]))


def invalid_bars_frame(dates: list[date]) -> pd.DataFrame:
    """Like ``bars_frame``, but the first bar has a non-positive close."""
    df = bars_frame(dates)
    df.iloc[0, df.columns.get_loc("Close")] = -5.0
    df.iloc[0, df.columns.get_loc("Adj Close")] = -5.0
    return df


@dataclass
class StubFetcher:
    """Records every call and answers per-symbol from ``frames``.

    A symbol mapped to an ``Exception`` instance raises it instead of
    returning a frame -- how a ``MarketDataUnavailable`` failure is simulated
    offline. Symbols not in ``frames`` fall back to ``default``.
    """

    frames: dict[str, pd.DataFrame | Exception] = field(default_factory=dict)
    default: pd.DataFrame | None = None
    calls: list[tuple[str, date, date]] = field(default_factory=list)

    def __call__(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        result = self.frames.get(symbol, self.default)
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise AssertionError(f"unexpected fetch call for {symbol!r}")
        return result


def _provider(fetcher: StubFetcher) -> YFinanceMarketDataProvider:
    return YFinanceMarketDataProvider(fetcher=fetcher)


def _prices(config: Config) -> list[tuple]:
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        return con.execute(
            "SELECT symbol, price_date, close FROM daily_prices ORDER BY symbol, price_date"
        ).fetchall()


def _seed_price(config: Config, symbol: str, price_date: date, *, close: float = 100.0) -> None:
    """Insert one bar directly, bypassing the collector, to control DB state."""
    with database.connection(config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_daily_prices(
            con,
            [
                {
                    "price_id": f"seed-{symbol}-{price_date.isoformat()}",
                    "symbol": symbol,
                    "price_date": price_date,
                    "close": close,
                    "adj_close": close,
                    "open": close,
                    "high": close,
                    "low": close,
                    "volume": 1000,
                    "provider": "yfinance",
                    "validation_status": "valid",
                    "source": "market",
                }
            ],
        )


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_sync_persists_bars_and_preserves_raw(tmp_config: Config) -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    fetcher = StubFetcher(default=bars_frame(dates))

    summary = price_collector.sync(
        tmp_config, symbols=["AAA", "BBB"], end=END, provider=_provider(fetcher)
    )

    assert summary.status == "success"
    assert summary.inserted == 6  # 2 bars x (AAA, BBB, SPY)
    assert summary.rejected == 0

    rows = _prices(tmp_config)
    assert {r[0] for r in rows} == {"AAA", "BBB", "SPY"}  # benchmark always included

    raw_dir = tmp_config.paths.raw_dir / "market" / "daily_prices" / "AAA"
    assert any(raw_dir.rglob("*.csv"))


# --------------------------------------------------------------------------- #
# idempotency
# --------------------------------------------------------------------------- #
def test_sync_is_idempotent_across_reruns(tmp_config: Config) -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    # Deliberately ignores whatever window it is asked for, so this test
    # isolates the upsert-dedup path from the resumability optimization
    # (covered separately below).
    fetcher = StubFetcher(default=bars_frame(dates))

    first = price_collector.sync(tmp_config, symbols=["AAA"], end=END, provider=_provider(fetcher))
    second = price_collector.sync(tmp_config, symbols=["AAA"], end=END, provider=_provider(fetcher))

    assert first.inserted == 4 and first.updated == 0  # AAA + SPY, 2 bars each
    assert second.inserted == 0
    assert second.updated == 4

    rows = _prices(tmp_config)
    assert len(rows) == 4  # row count unchanged


# --------------------------------------------------------------------------- #
# resumability
# --------------------------------------------------------------------------- #
def test_resumes_from_the_day_after_the_last_stored_bar(tmp_config: Config) -> None:
    _seed_price(tmp_config, "AAA", date(2024, 1, 5))
    fetcher = StubFetcher(default=bars_frame([date(2024, 1, 10)]))

    price_collector.sync(tmp_config, symbols=["AAA"], end=END, provider=_provider(fetcher))

    aaa_calls = [c for c in fetcher.calls if c[0] == "AAA"]
    assert aaa_calls == [("AAA", date(2024, 1, 6), END)]


def test_backfills_when_asked_for_history_older_than_is_stored(tmp_config: Config) -> None:
    """An explicit earlier start must extend history backwards, not no-op.

    Resuming from the newest bar here would leave the requested earlier window
    permanently unreachable while still reporting success -- a silent hole in
    the history every backtest reads from.
    """
    _seed_price(tmp_config, "AAA", date(2024, 1, 5))
    fetcher = StubFetcher(default=bars_frame([date(2020, 1, 2)]))

    price_collector.sync(
        tmp_config,
        symbols=["AAA"],
        start=date(2020, 1, 1),
        end=END,
        provider=_provider(fetcher),
    )

    aaa_calls = [c for c in fetcher.calls if c[0] == "AAA"]
    assert aaa_calls == [("AAA", date(2020, 1, 1), END)]


def test_start_no_older_than_stored_history_still_resumes(tmp_config: Config) -> None:
    """A start inside the stored range must not trigger a redundant refetch."""
    _seed_price(tmp_config, "AAA", date(2024, 1, 5))
    fetcher = StubFetcher(default=bars_frame([date(2024, 1, 10)]))

    price_collector.sync(
        tmp_config,
        symbols=["AAA"],
        start=date(2024, 1, 5),
        end=END,
        provider=_provider(fetcher),
    )

    aaa_calls = [c for c in fetcher.calls if c[0] == "AAA"]
    assert aaa_calls == [("AAA", date(2024, 1, 6), END)]


def test_earlier_symbols_survive_a_later_symbol_crashing(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work already done must be durable when the run dies partway through.

    This is what makes resumption real. _fetch_start resumes from
    max(price_date), so if a run buffered every symbol and wrote once at the
    end, a crash near the finish would leave nothing stored and the next run
    would restart from zero. The failure injected here is a bare RuntimeError
    -- deliberately not a MarketDataUnavailable, which the collector catches --
    so it tears the run down the way an OOM or a killed process would.
    """

    class Boom(RuntimeError):
        pass

    fetcher = StubFetcher(default=bars_frame([date(2024, 1, 2), date(2024, 1, 3)]))

    # Fail on the write for ZZZ, not inside the fetch: the provider converts
    # every fetch-path exception into MarketDataUnavailable, which the
    # collector is supposed to absorb per symbol. The crash has to land after
    # AAA has been handed to storage for this to test durability at all.
    real_upsert = duckdb_store.upsert_daily_prices

    def exploding_upsert(con, rows):
        materialized = list(rows)
        if any(row.get("symbol") == "ZZZ" for row in materialized):
            raise Boom("killed mid-run")
        return real_upsert(con, materialized)

    monkeypatch.setattr(price_collector.duckdb_store, "upsert_daily_prices", exploding_upsert)

    with pytest.raises(Boom):
        price_collector.sync(
            tmp_config, symbols=["AAA", "ZZZ"], end=END, provider=_provider(fetcher)
        )

    with database.connection(tmp_config.paths.database_path) as con:
        stored = con.execute("SELECT count(*) FROM daily_prices WHERE symbol = 'AAA'").fetchone()[0]

    assert stored == 2, "bars collected before the crash were lost"


def test_no_network_work_when_already_current(tmp_config: Config) -> None:
    _seed_price(tmp_config, "AAA", END)
    _seed_price(tmp_config, "SPY", END)
    fetcher = StubFetcher()

    summary = price_collector.sync(
        tmp_config, symbols=["AAA"], end=END, provider=_provider(fetcher)
    )

    assert fetcher.calls == []
    assert summary.inserted == 0
    assert summary.updated == 0


# --------------------------------------------------------------------------- #
# one bad symbol must not abort the run
# --------------------------------------------------------------------------- #
def test_one_symbol_failure_does_not_abort_the_run(tmp_config: Config) -> None:
    dates = [date(2024, 1, 2)]
    fetcher = StubFetcher(
        frames={"DEAD": MarketDataUnavailable("DEAD: delisted")},
        default=bars_frame(dates),
    )

    summary = price_collector.sync(
        tmp_config, symbols=["AAA", "DEAD", "BBB"], end=END, provider=_provider(fetcher)
    )

    assert summary.status == "success"
    rows = _prices(tmp_config)
    assert {r[0] for r in rows} == {"AAA", "BBB", "SPY"}  # DEAD absent, others present
    assert any("DEAD" in note for note in summary.notes)
    assert summary.stage["price_fetch_failed"] == 1


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_invalid_bar_is_rejected_and_excluded(tmp_config: Config) -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    fetcher = StubFetcher(default=invalid_bars_frame(dates))

    summary = price_collector.sync(
        tmp_config, symbols=["AAA"], end=END, provider=_provider(fetcher)
    )

    assert summary.collected == 4  # AAA + SPY, 2 bars each
    assert summary.rejected == 2  # the bad first bar, once per symbol
    assert summary.inserted == 2

    rows = _prices(tmp_config)
    assert dates[0] not in {r[1] for r in rows}
    assert dates[1] in {r[1] for r in rows}


# --------------------------------------------------------------------------- #
# default universe
# --------------------------------------------------------------------------- #
def test_default_universe_falls_back_to_companies_and_always_includes_benchmark(
    tmp_config: Config,
) -> None:
    fetcher = StubFetcher(default=bars_frame([date(2024, 1, 2)]))

    price_collector.sync(tmp_config, end=END, provider=_provider(fetcher))

    requested = {c[0] for c in fetcher.calls}
    configured = {c.ticker.upper() for c in tmp_config.companies.companies}
    assert configured <= requested
    assert "SPY" in requested


def test_default_universe_prefers_populated_index_constituents(tmp_config: Config) -> None:
    with database.connection(tmp_config.paths.database_path) as con:
        database.init_db(con)
        duckdb_store.upsert_constituents(
            con,
            [
                {
                    "constituent_id": "c1",
                    "index_id": "SP500",
                    "ticker": "ZZZ",
                    "added_date": date(2020, 1, 1),
                    "validation_status": "valid",
                    "source": "reference",
                }
            ],
        )

    fetcher = StubFetcher(default=bars_frame([date(2024, 1, 2)]))
    price_collector.sync(tmp_config, end=END, provider=_provider(fetcher))

    requested = {c[0] for c in fetcher.calls}
    assert requested == {"ZZZ", "SPY"}
