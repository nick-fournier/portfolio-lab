"""Market-state signals at each month end: fragile, bear, rebound.

``market_state`` is production's bear/rebound switch; :func:`signals` builds every
indicator per month end (the forecaster uses them as inputs).

Three different situations, each with its own candidate signals, read at every month end
from data known then:

1. **Fragile** (the market still near its high, but shaky underneath):
   ``absorption_rising`` (stocks moving more as one than over the past year, Kritzman et
   al.), ``credit_divergence`` (credit spreads up over 0.25 points in three months while
   SPY is within 5% of its high), ``breadth_divergence`` (the share of stocks above their
   200-day average down 10 points in three months while SPY is within 5% of its high),
   ``curve_inverted`` (10-year yield below 3-month), ``complacent`` (SPY's 3-month
   volatility in the calmest fifth of its history so far).
2. **Bear** (``bear``): SPY below its 200-day average for three straight month ends and at
   least 15% below its high.
3. **Rebound** (``rebound``): SPY at least 20% below its high and the VIX at least 20%
   below its peak of the past three months: panic easing.

Outcomes after each month end: SPY's worst drawdown over the next six months, SPY's return
over the next one, three and six months, and how the strongest and weakest tenths of the
500 most liquid stocks by 12-month return did over the next three months.
"""

import numpy as np
import pandas as pd
import polars as pl

from portfolio_lab.core.calendar import rebalance_dates
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.stress import absorption

YEAR = 252
NEAR_HIGH = 0.05
BEAR_DRAWDOWN = 0.15
REBOUND_DRAWDOWN = 0.20
VIX_EASING = 0.20
POOL = 500


def market_state(
    market: "pd.Series",
    vix: list[float],
    bear_drawdown: float = BEAR_DRAWDOWN,
    vix_easing: float = VIX_EASING,
) -> str:
    """``bear``, ``rebound`` or ``normal`` at the last date of ``market`` (daily SPY returns).

    Bear: below the 200-day average now and one and two months ago, and at least
    ``bear_drawdown`` below the two-year high. Rebound (takes precedence): at least
    :data:`REBOUND_DRAWDOWN` below the high and the latest month-end VIX at least
    ``vix_easing`` below its peak of the last three month ends (``vix``, oldest first).
    """
    r = market.dropna().to_numpy()
    if len(r) < YEAR + 42:
        return "normal"
    level = np.cumprod(1 + r)
    drawdown = 1 - level[-1] / level[-2 * YEAR :].max()
    easing = len(vix) >= 3 and vix[-1] <= (1 - vix_easing) * max(vix[-3:])
    if drawdown >= REBOUND_DRAWDOWN and easing:
        return "rebound"
    below = all(level[-1 - k] < level[-200 - k : len(level) - k].mean() for k in (0, 21, 42))
    return "bear" if below and drawdown >= bear_drawdown else "normal"


def _expanding_pct(values: list[float | None]) -> list[float | None]:
    """Each value's percentile among the values up to and including it (no look-ahead)."""
    seen, out = [], []
    for v in values:
        if v is None or not np.isfinite(v):
            out.append(None)
            continue
        seen.append(v)
        out.append(float(np.mean(np.asarray(seen) <= v)))
    return out


def _month_end(panel: Panel, level: np.ndarray, stocks: np.ndarray, i: int) -> dict:
    """Market state at session ``i`` and what followed."""
    spy_ret = panel.field("ret_cc")[:, panel.symbol_index["SPY"]]
    live = np.flatnonzero(panel.eligible[i])
    liquid = live[np.argsort(-np.nan_to_num(panel.field("adv")[i, live], nan=-np.inf))[:POOL]]
    window = pd.DataFrame(panel.field("ret_cc")[i - YEAR + 1 : i + 1][:, liquid])
    row = {
        "date": panel.dates[i],
        "drawdown": float(1 - level[i] / level[max(0, i - 2 * YEAR) : i + 1].max()),
        "below_ma200": bool(level[i] < level[i - 199 : i + 1].mean()),
        "vol3m": float(np.nanstd(spy_ret[i - 62 : i + 1]) * np.sqrt(YEAR)),
        "breadth": float(np.mean(stocks[i, live] > stocks[i - 199 : i + 1, live].mean(axis=0))),
        "absorption": absorption(window),
    }  # fmt: skip
    for h, name in ((21, "fwd_1m"), (63, "fwd_3m"), (126, "fwd_6m")):
        row[name] = float(level[i + h] / level[i] - 1) if i + h < len(level) else None
    row["fwd_max_dd_6m"] = None
    if i + 126 < len(level):
        path = level[i : i + 127] / level[i]
        row["fwd_max_dd_6m"] = float((path / np.maximum.accumulate(path) - 1).min())
    row["winners_3m"] = row["losers_3m"] = None
    if i + 63 < len(level):
        mom = stocks[i - 21, liquid] / stocks[i - YEAR, liquid] - 1
        fwd = stocks[i + 63, liquid] / stocks[i, liquid] - 1
        ok = np.isfinite(mom) & np.isfinite(fwd)
        order = np.argsort(mom[ok])
        tenth = max(1, ok.sum() // 10)
        row["winners_3m"] = float(fwd[ok][order[-tenth:]].mean())
        row["losers_3m"] = float(fwd[ok][order[:tenth]].mean())
    return row


def signals(panel: Panel, env: pl.DataFrame) -> pl.DataFrame:
    """Every signal and outcome at each month end (see module docs)."""
    level = np.cumprod(1 + np.nan_to_num(panel.field("ret_cc")[:, panel.symbol_index["SPY"]]))
    stocks = np.cumprod(1 + np.nan_to_num(panel.field("ret_cc")), axis=0)
    ends = [panel.date_index[d] for d in rebalance_dates(list(panel.dates), "M")]
    frame = pl.DataFrame([_month_end(panel, level, stocks, i) for i in ends if i >= YEAR])
    frame = frame.join(env.select("date", "vix", "baa_spread", "curve_10y_3m"), on="date",
                       how="left").sort("date")  # fmt: skip
    near_high = pl.col("drawdown") <= NEAR_HIGH
    past = pl.col("absorption").shift(1)
    return frame.with_columns(
        pl.Series("vol_pct", _expanding_pct(frame["vol3m"].to_list()), dtype=pl.Float64),
    ).with_columns(
        ((pl.col("absorption") - past.rolling_mean(12)) / past.rolling_std(12) > 1)
        .alias("absorption_rising"),
        (near_high & (pl.col("baa_spread") - pl.col("baa_spread").shift(3) > 0.25))
        .alias("credit_divergence"),
        (near_high & (pl.col("breadth") - pl.col("breadth").shift(3) < -0.10))
        .alias("breadth_divergence"),
        (pl.col("curve_10y_3m") < 0).alias("curve_inverted"),
        (pl.col("vol_pct") <= 0.2).alias("complacent"),
        ((pl.col("below_ma200").cast(pl.Int8).rolling_sum(3) == 3)
         & (pl.col("drawdown") >= BEAR_DRAWDOWN)).alias("bear"),
        ((pl.col("drawdown") >= REBOUND_DRAWDOWN)
         & (pl.col("vix") <= (1 - VIX_EASING) * pl.col("vix").rolling_max(3)))
        .alias("rebound"),
    )  # fmt: skip
