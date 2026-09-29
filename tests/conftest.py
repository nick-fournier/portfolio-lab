from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from portfolio_lab.core.calendar import sessions
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
        rates = pl.DataFrame({"date": prices["date"].unique().sort(), "rate": 0.05})
        rules = EligibilityRules(min_price=1, min_dollar_volume=0, adv_window=5, min_history=5)
        return Panel.from_long(prices, symbols, rates, rules)

    return factory
