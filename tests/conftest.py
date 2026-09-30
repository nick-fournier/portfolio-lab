from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from portfolio_lab.core.calendar import rebalance_dates, sessions
from portfolio_lab.core.config import Settings
from portfolio_lab.research.panel import EligibilityRules, Panel

PRICE_COLUMNS = ["symbol", "date", "close", "volume", "ret_cc", "ret_co"]


@pytest.fixture
def settings(tmp_path):
    """Settings pointed at a temporary data dir, ignoring the real .env (and its keys)."""
    return Settings(_env_file=None, PORTFOLIO_DATA_DIR=tmp_path / "data")


def make_long_prices(symbols, start=date(2020, 1, 2), days=300, seed=0, price=20.0):
    """Synthetic long price rows (random walks) over real NYSE sessions."""
    dates = sessions(start, start + timedelta(days=days * 2))[:days]
    rng = np.random.default_rng(seed)
    rows = []
    for symbol in symbols:
        close = price * np.cumprod(1 + rng.normal(0.0005, 0.02, len(dates)))
        prev = np.r_[np.nan, close[:-1]]
        opens = np.where(np.isnan(prev), close, prev * (1 + rng.normal(0, 0.005, len(dates))))
        for i, (d, c, o, p) in enumerate(zip(dates, close, opens, prev, strict=True)):
            first = i == 0
            rows.append(
                (symbol, d, c, 1e6, None if first else c / p - 1, None if first else o / p - 1)
            )
    return pl.DataFrame(rows, schema=PRICE_COLUMNS, orient="row")


@pytest.fixture
def make_panel():
    """Factory for small synthetic panels with permissive eligibility rules."""

    def factory(symbols=("AAA", "BBB", "CCC"), benchmarks=("SPY",), days=300, seed=0):
        prices = make_long_prices([*symbols, *benchmarks], days=days, seed=seed)
        dates = prices["date"].unique().sort()
        rates = pl.DataFrame({"date": dates, "rate": 0.05})
        rules = EligibilityRules(min_price=1, min_dollar_volume=0, adv_window=5, min_history=5)
        panel = Panel.from_long(prices, symbols, rates, rules)
        panel.fundamentals = make_fundamentals(symbols, dates)
        panel.features = make_features(symbols, dates)
        panel.predictions = make_predictions(symbols, dates)
        return panel

    return factory


def make_features(symbols, dates):
    """Synthetic monthly feature rows: earnings_yield grows with the symbol's position."""
    month_ends = rebalance_dates(list(dates), "M")
    rows = [(d, s, 0.01 * (k + 1)) for d in month_ends for k, s in enumerate(symbols)]
    return pl.DataFrame(rows, schema=["date", "symbol", "earnings_yield"], orient="row")


def make_predictions(symbols, dates):
    """Synthetic forecast scores at month ends: later symbols rated higher."""
    month_ends = rebalance_dates(list(dates), "M")
    rows = [(d, s, k / len(symbols)) for d in month_ends for k, s in enumerate(symbols)]
    return pl.DataFrame(rows, schema=["date", "symbol", "score"], orient="row")


def make_fundamentals(symbols, dates):
    """Synthetic F-scores: a filing every ~250 sessions; every other symbol scores 9."""
    rows = [
        (symbol, dates[i], dates[max(i - 40, 0)], 9 if k % 2 == 0 else 5, 9)
        for k, symbol in enumerate(symbols)
        for i in range(20, len(dates), 250)
    ]
    schema = ["symbol", "filed", "fiscal_end", "fscore", "n_signals"]
    return pl.DataFrame(rows, schema=schema, orient="row").sort("filed")
