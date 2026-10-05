"""Production's forecast, for comparison: AR(1) on each stock's trailing daily returns.

The same model mean-variance uses (``strategies.meanvar.forecast``, ``ar1_logret``): at each
month end, an AR(1) fitted by least squares to the last :data:`LOOKBACK` daily log returns,
summed over the next :data:`HORIZON` sessions. Stocks with bars on fewer than
:data:`MIN_COVERAGE` of the window are left out; missing days count as no change (as the
production code carries prices forward). With daily returns barely autocorrelated, this is
close to the trailing mean.
"""

import numpy as np
import polars as pl

from portfolio_lab.research.panel import Panel

LOOKBACK = 252
HORIZON = 21
MIN_COVERAGE = 0.95


def ar1(panel: Panel, month_ends: list) -> pl.DataFrame:
    """date, symbol and ``production`` (next month's return as AR(1) forecasts it)."""
    rets = panel.field("ret_cc")
    frames = []
    for day in month_ends:
        i = panel.date_index[day]
        if i < LOOKBACK:
            continue
        window = rets[i - LOOKBACK + 1 : i + 1]
        ok = np.isfinite(window).mean(axis=0) >= MIN_COVERAGE
        r = np.log1p(np.nan_to_num(window[:, ok]))
        x, y = r[:-1], r[1:]
        xm, ym = x.mean(axis=0), y.mean(axis=0)
        var = ((x - xm) ** 2).sum(axis=0)
        phi = np.where(var > 0, ((x - xm) * (y - ym)).sum(axis=0) / np.where(var > 0, var, 1), 0)
        phi = np.clip(phi, -0.99, 0.99)
        mu = (ym - phi * xm) / (1 - phi)
        total = HORIZON * mu + (r[-1] - mu) * phi * (1 - phi**HORIZON) / (1 - phi)
        symbols = np.asarray(panel.symbols)[ok]
        frames.append(pl.DataFrame({"date": [day] * len(symbols), "symbol": symbols,
                                    "production": np.expm1(total)}))  # fmt: skip
    return pl.concat(frames)
