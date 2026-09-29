from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.core.calendar import sessions, sessions_back
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import scan
from portfolio_lab.data.ingest import prices
from portfolio_lab.data.ingest.prices import (
    OVERLAP_SESSIONS,
    build_price_rows,
    plan_fetches,
    update_prices,
    verify_prices,
)
from portfolio_lab.data.sources.alpaca import BAR_SCHEMA

DAYS = sessions(date(2024, 1, 2), date(2024, 3, 28))
SPLIT_DAY = date(2024, 2, 26)  # 2:1 split: raw price halves from this day
DIVIDEND_DAY = date(2024, 3, 11)  # $1 dividend, ex-date


class FakeProvider:
    """Mimics Alpaca: raw bars, and adjusted bars rebased for events known as of ``asof``."""

    def __init__(self, symbols, seed=0):
        rng = np.random.default_rng(seed)
        self.asof = DAYS[-1]
        self.raw = {}
        for symbol in symbols:
            close = 100 * np.cumprod(1 + rng.normal(0, 0.02, len(DAYS)))
            close[DAYS.index(SPLIT_DAY) :] /= 2
            opens = np.r_[close[0], close[:-1] * (1 + rng.normal(0, 0.005, len(DAYS) - 1))]
            self.raw[symbol] = (opens, close)

    def factor(self, symbol, i):
        """Backward adjustment factor for day index ``i`` given events known as of ``asof``."""
        _, close = self.raw[symbol]
        f = 1.0
        for day, event in ((SPLIT_DAY, "split"), (DIVIDEND_DAY, "div")):
            j = DAYS.index(day)
            if day <= self.asof and i < j:
                f *= 0.5 if event == "split" else 1 - 1.0 / close[j - 1]
        return f

    def fetch_bars(self, client, symbols, start, end, adjustment="raw", timeframe="1Day"):
        rows = []
        for symbol in symbols:
            if symbol not in self.raw:
                continue
            opens, close = self.raw[symbol]
            for i, day in enumerate(DAYS):
                if start <= day <= min(end, self.asof):
                    f = self.factor(symbol, i) if adjustment == "all" else 1.0
                    o, c = opens[i] * f, close[i] * f
                    rows.append(
                        (symbol, day, o, max(o, c) * 1.01, min(o, c) * 0.99, c, 1e6, c, 100)
                    )
        return pl.DataFrame(rows, schema=BAR_SCHEMA, orient="row")


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider(["AAA", "BBB", "NEW"])
    monkeypatch.setattr(prices, "fetch_bars", fake.fetch_bars)
    return fake


def _stored(settings):
    return (
        scan(DataPaths(settings.data_dir).prices_daily, "year=*/data.parquet")
        .collect()
        .sort("symbol", "date")
    )


def _expected(provider, symbols):
    raw = provider.fetch_bars(None, symbols, DAYS[0], DAYS[-1], "raw")
    adjusted = provider.fetch_bars(None, symbols, DAYS[0], DAYS[-1], "all")
    return build_price_rows(raw, adjusted)


@pytest.mark.parametrize("chunk", [500, 1])
def test_incremental_across_split_and_dividend_matches_full_fetch(
    settings, provider, monkeypatch, chunk
):
    monkeypatch.setattr(prices, "CHUNK_SYMBOLS", chunk)
    dataset = DataPaths(settings.data_dir).prices_daily

    provider.asof = date(2024, 2, 15)  # before either corporate action
    update_prices(settings, None, ["AAA", "BBB"], dataset, "prices", end=provider.asof)

    provider.asof = DAYS[-1]  # both events now known; adjusted history rebased
    summary = update_prices(
        settings, None, ["AAA", "BBB", "NEW"], dataset, "prices", end=provider.asof
    )
    assert summary["symbols_full_history"] == 1  # NEW
    assert summary["symbols_incremental"] == 2

    stored = _stored(settings)
    expected = _expected(provider, ["AAA", "BBB", "NEW"])
    assert stored.select("symbol", "date").equals(expected.select("symbol", "date"))
    for col in ("ret_cc", "ret_co", "close"):
        np.testing.assert_allclose(
            stored[col].fill_null(0).to_numpy(), expected[col].fill_null(0).to_numpy(), rtol=1e-12
        )
    # Only each symbol's very first bar lacks a return.
    assert stored["ret_cc"].null_count() == 3


def test_rerun_is_idempotent(settings, provider):
    dataset = DataPaths(settings.data_dir).prices_daily
    update_prices(settings, None, ["AAA"], dataset, "prices", end=DAYS[-1])
    first = _stored(settings)
    update_prices(settings, None, ["AAA"], dataset, "prices", end=DAYS[-1])
    assert _stored(settings).equals(first)


def test_plan_fetches():
    extent = pl.DataFrame(
        {
            "symbol": ["OLD", "RECENT", "LAGGING"],
            "first": [date(2024, 1, 2)] * 3,
            "last": [date(2024, 1, 5), date(2024, 3, 28), date(2024, 3, 20)],
        }
    )
    fresh, recent, start = plan_fetches(["OLD", "RECENT", "LAGGING", "NEW"], extent, full=False)
    assert fresh == ["OLD", "NEW"]  # stale and never-stored symbols get full history
    assert recent == ["RECENT", "LAGGING"]
    assert start == sessions_back(date(2024, 3, 20), OVERLAP_SESSIONS)

    fresh, recent, start = plan_fetches(["OLD", "RECENT"], extent, full=True)
    assert (fresh, recent, start) == (["OLD", "RECENT"], [], None)


def test_verify_repairs_corrupted_returns(settings, provider):
    paths = DataPaths(settings.data_dir)
    update_prices(settings, None, ["AAA", "BBB"], paths.prices_daily, "prices", end=DAYS[-1])
    part = DataPaths.year_partition(paths.prices_daily, 2024)
    corrupted = pl.read_parquet(part).with_columns(
        pl.when((pl.col("symbol") == "AAA") & (pl.col("date") == date(2024, 3, 1)))
        .then(0.5)
        .otherwise(pl.col("ret_cc"))
        .alias("ret_cc")
    )
    corrupted.write_parquet(part)

    summary = verify_prices(settings, None, paths.prices_daily, sample=10, seed=1)
    assert summary["repaired"] == ["AAA"]
    repaired = _stored(settings).filter(
        (pl.col("symbol") == "AAA") & (pl.col("date") == date(2024, 3, 1))
    )
    assert repaired["ret_cc"][0] != 0.5
