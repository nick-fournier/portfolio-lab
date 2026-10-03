"""The forecaster's data: one row per stock-month, and one row per month for the market.

:func:`stocks`: every stock at every month end, its :data:`INPUTS` as within-month
percentiles centered on 0 (missing stays missing; the linear part reads it as 0, the
median), its size, and the target. The target is the return from this month end to the
next (delisting as in the backtest) minus that month's average across stocks: ``actual``
as is, ``y`` capped at the month's :data:`CAP` quantiles for fitting. The latest month has
no target yet; it is the one being forecast.

:func:`market`: per month end, the environment (``env_*``), the market state (``mkt_*``,
``research.regimes``) and the two conditions the linear part interacts with, each a
percentile of its own history so far, centered on 0: ``vix`` (the VIX's percentile, from
the environment) and ``dispersion`` (the spread of last month's stock returns).
"""

from itertools import pairwise

import numpy as np
import polars as pl

from portfolio_lab.research import regimes
from portfolio_lab.research.characteristics.build import COLUMNS as EXTRA
from portfolio_lab.research.dataset import rank_features
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.scoreboard import forward_returns

#: Inputs from the monthly feature panel (``research.features``).
ORIGINAL = (
    "earnings_yield", "book_to_market", "cf_yield", "fcf_yield", "sales_yield",
    "dividend_yield", "roa", "cfo_to_assets", "accruals", "gross_profitability",
    "operating_margin", "leverage", "current_ratio", "d_roa", "d_lt_debt", "d_current_ratio",
    "d_gross_margin", "d_asset_turnover", "share_issuance", "asset_growth", "sales_growth",
    "mom_12_1", "ret_1m", "volatility", "beta", "fscore", "earnings_yield_ind",
    "book_to_market_ind", "cf_yield_ind", "sales_yield_ind", "roa_ind",
    "gross_profitability_ind", "asset_growth_ind", "mom_12_1_ind",
)  # fmt: skip
#: Every stock input: the feature panel's plus ``research.characteristics``'.
INPUTS = (*ORIGINAL, *EXTRA)
#: Market-state indicators from ``research.regimes`` (as known at the month end).
MARKET_STATE = ("drawdown", "below_ma200", "vol3m", "breadth", "absorption", "vol_pct",
                "absorption_rising", "credit_divergence", "breadth_divergence",
                "curve_inverted", "complacent", "bear", "rebound")  # fmt: skip
#: The target is capped at these quantiles of each month for fitting.
CAP = (0.001, 0.999)


def stocks(panel: Panel, features: pl.DataFrame, extra: pl.DataFrame) -> pl.DataFrame:
    """Stock-months with inputs and target (module docs).

    Args:
        panel: Prices, for the returns between month ends.
        features: The monthly feature panel (``DataPaths.features``).
        extra: The extra inputs (``DataPaths.characteristics``).

    Returns:
        date, symbol, actual, y, size (market value's percentile, centered on 0) and
        :data:`INPUTS`, sorted by date and symbol.
    """
    data = features.select("date", "symbol", "market_value", *ORIGINAL).join(
        extra, on=["date", "symbol"], how="left"
    )
    data = rank_features(data, list(INPUTS)).with_columns(pl.col(c) - 0.5 for c in INPUTS)
    ends = sorted(data["date"].unique().to_list())
    frames = []
    for start, end in pairwise(ends):
        i, j = panel.date_index[start], panel.date_index[end]
        frames.append(pl.DataFrame({"date": [start] * len(panel.symbols),
                                    "symbol": panel.symbols,
                                    "R": forward_returns(panel, i, j - i)}))  # fmt: skip
    target = pl.concat(frames).with_columns(pl.col("R").fill_nan(None))
    labeled = data.join(target, on=["date", "symbol"], how="inner").drop_nulls("R")
    latest = data.filter(pl.col("date") == ends[-1]).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("R")
    )
    data = pl.concat([labeled, latest.select(labeled.columns)])
    r = pl.col("R") - pl.col("R").mean().over("date")
    lo, hi = (r.quantile(q).over("date") for q in CAP)
    size = pl.col("market_value").rank().over("date") / pl.len().over("date") - 0.5
    return data.select(
        "date", "symbol", r.alias("actual"), r.clip(lo, hi).alias("y"), size.alias("size"),
        *INPUTS,
    ).sort("date", "symbol")  # fmt: skip


def _expanding_pct(values: list[float]) -> list[float]:
    """Each value's percentile among the values up to and including it."""
    seen: list[float] = []
    out = []
    for v in values:
        seen.append(v)
        out.append(float(np.mean(np.array(seen) <= v)))
    return out


def market(panel: Panel, env: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """Per month end: ``env_*``, ``mkt_*``, ``vix`` and ``dispersion`` (module docs).

    Args:
        panel: Prices, for the market state.
        env: The monthly environment (``DataPaths.environment``).
        features: The monthly feature panel (for last month's returns).
    """
    numeric = [c for c, t in env.schema.items() if c != "date" and t.is_numeric()]
    state = regimes.signals(panel, env).select(
        "date", *[pl.col(c).cast(pl.Float64).alias(f"mkt_{c}") for c in MARKET_STATE]
    )
    spread = features.group_by("date").agg(pl.col("ret_1m").std().alias("disp")).sort("date")
    spread = spread.with_columns(
        pl.Series("dispersion", _expanding_pct(spread["disp"].to_list())) - 0.5
    )
    months = features.select("date").unique().sort("date")
    return (
        months.join(
            env.select("date", *[pl.col(c).alias(f"env_{c}") for c in numeric]),
            on="date",
            how="left",
        )
        .join(state, on="date", how="left")
        .join(spread.select("date", "dispersion"), on="date", how="left")
        .with_columns((pl.col("env_vix_pct") - 0.5).alias("vix"))
    )
