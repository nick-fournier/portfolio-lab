"""Production's expected return, for comparison: each stock's trailing one-year return.

The same model mean-variance uses (``strategies.meanvar.forecast``, ``historical_mean``): at
each month end, the stock's growth over the last :data:`LOOKBACK` daily prices, as a pace
per :data:`HORIZON` sessions. Stocks with bars on fewer than :data:`MIN_COVERAGE` of the
window are left out; missing days count as no change (as the production code carries prices
forward).
"""

import numpy as np
import polars as pl

from portfolio_lab.research.panel import Panel

LOOKBACK = 252
HORIZON = 21
MIN_COVERAGE = 0.95


def trailing(panel: Panel, month_ends: list) -> pl.DataFrame:
    """date, symbol and ``production`` (next month's return at last year's pace)."""
    rets = panel.field("ret_cc")
    frames = []
    for day in month_ends:
        i = panel.date_index[day]
        if i < LOOKBACK:
            continue
        window = rets[i - LOOKBACK + 2 : i + 1]  # the returns between LOOKBACK prices
        ok = np.isfinite(window).mean(axis=0) >= MIN_COVERAGE
        pace = np.log1p(np.nan_to_num(window[:, ok])).sum(axis=0) * HORIZON / len(window)
        symbols = np.asarray(panel.symbols)[ok]
        frames.append(pl.DataFrame({"date": [day] * len(symbols), "symbol": symbols,
                                    "production": np.expm1(pace)}))  # fmt: skip
    return pl.concat(frames)
