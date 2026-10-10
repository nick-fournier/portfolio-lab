"""The forecaster: least squares on nine terms, refit every month on all earlier months.

Each month end, every stock in the feature panel gets a forecast of next month's return
over the T-bill. The nine terms:

- seven stock inputs: one-year momentum (:data:`MOMENTUM_SESSIONS` daily returns, compounded),
  six-month momentum, operating cash flow over assets, and the free-cash-flow, sales,
  earnings and R&D-to-market yields;
- one-year and six-month momentum each times SPY's one-year return, so momentum's payoff
  can change with the market's trend.

Each stock input is put on one scale per month: divided by the month's median absolute value,
passed through ``asinh`` (which tames the long tails while keeping the sign), then
standardized across that month's stocks. R&D-to-market, whose tail survives ``asinh``, is
also clipped at the month's 1st and 99th percentiles and standardized again. A missing
input counts as the month's average (0).

The target is the stock's return from one month end to the next minus that month's T-bill,
uncapped. The fit is ordinary least squares with an intercept on every earlier month's
stocks that have a one-year momentum, from per-month sums; the first forecast comes once
:data:`MIN_MONTHS` months are available.
"""

import numpy as np
import pandas as pd
import polars as pl

from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.scoreboard import forward_returns

#: Where :func:`walk`'s forecasts (with each stock's terms) are kept, in ``DataPaths.forecaster``.
FILE = "nine.parquet"
#: Where :func:`slopes` are kept.
SLOPES = "nine_slopes.parquet"
#: Months of history before the first forecast.
MIN_MONTHS = 24
#: Daily returns in one-year momentum (and SPY's one-year return).
MOMENTUM_SESSIONS = 251
#: One-year momentum needs bars on this share of its window.
MIN_COVERAGE = 0.95
#: Stock inputs: name -> (source column, table it comes from).
INPUTS = {
    "mom12": ("mom12_raw", "panel"),
    "mom6": ("mom6m", "monthly"),
    "cfo_assets": ("cfo_to_assets", "monthly"),
    "fcf_yield": ("fcf_yield", "monthly"),
    "sales_yield": ("sales_yield", "monthly"),
    "earnings_yield": ("earnings_yield", "monthly"),
    "rd_mve": ("rd_mve", "monthly"),
}
#: Inputs clipped at the month's 1st/99th percentiles after the transform.
CLIPPED = ("rd_mve",)
#: The fitted terms, in design order (after the intercept).
TERMS = (*INPUTS, "mom12_x_spy", "mom6_x_spy")


def _standardize(e: pl.Expr) -> pl.Expr:
    return (e - e.mean().over("date")) / e.std().over("date")


def _scaled(col: str, clip: bool) -> pl.Expr:
    """``asinh(x / median |x|)`` standardized within the month (module docs)."""
    x = pl.when(pl.col(col).cast(pl.Float64).is_finite()).then(pl.col(col).cast(pl.Float64))
    scale = x.abs().median().over("date")
    scale = pl.when(scale > 0).then(scale).otherwise(x.abs().mean().over("date"))
    e = _standardize((x / scale).arcsinh())
    if clip:
        e = _standardize(e.clip(e.quantile(0.01).over("date"), e.quantile(0.99).over("date")))
    return e


def _momentum(panel: Panel, i: int) -> np.ndarray:
    """Every symbol's compounded return over the :data:`MOMENTUM_SESSIONS` before ``i``."""
    window = panel.field("ret_cc")[i - MOMENTUM_SESSIONS + 1 : i + 1]
    ok = np.isfinite(window).mean(axis=0) >= MIN_COVERAGE
    growth = np.expm1(np.log1p(np.nan_to_num(window)).sum(axis=0))
    return np.where(ok, growth, np.nan)


def table(panel: Panel, monthly: pl.DataFrame) -> pl.DataFrame:
    """Stock-months with the nine terms and the target (module docs).

    Args:
        panel: Prices, the T-bill and the market.
        monthly: The monthly stock inputs (``DataPaths.features``); its rows are the stocks.

    Returns:
        date, symbol, ``actual`` (next month's return minus the T-bill; null in the last
        month), ``has_momentum``, ``spy12`` (SPY's one-year return) and :data:`TERMS`,
        sorted by date and symbol.
    """
    monthly_cols = [src for src, where in INPUTS.values() if where == "monthly"]
    data = monthly.select("date", "symbol", *monthly_cols)
    ends = sorted(d for d in data["date"].unique().to_list() if d in panel.date_index)
    spy = panel.market
    frames = []
    for k, day in enumerate(ends):
        i = panel.date_index[day]
        if i < MOMENTUM_SESSIONS:
            continue
        nxt = None
        if k + 1 < len(ends):
            nxt = forward_returns(panel, i, panel.date_index[ends[k + 1]] - i)
        rf = DataView(panel, i).risk_free() / 12
        frames.append(pl.DataFrame({
            "date": [day] * len(panel.symbols), "symbol": panel.symbols,
            "mom12_raw": _momentum(panel, i).astype(np.float64),
            "actual": (nxt - rf if nxt is not None else np.full(len(panel.symbols), np.nan))
            .astype(np.float64),
            "spy12": float(_momentum(panel, i)[spy]),
        }))  # fmt: skip
    priced = pl.concat(frames).with_columns(pl.col("mom12_raw", "actual").fill_nan(None))
    data = data.join(priced, on=["date", "symbol"], how="inner")
    last = data["date"].max()
    data = data.filter(pl.col("actual").is_not_null() | (pl.col("date") == last))
    inputs = [_scaled(src, name in CLIPPED).fill_nan(None).alias(name)
              for name, (src, _) in INPUTS.items()]  # fmt: skip
    out = data.select("date", "symbol", "actual", "spy12",
                      pl.col("mom12_raw").is_not_null().alias("has_momentum"), *inputs)  # fmt: skip
    out = out.with_columns(pl.col(n).fill_null(0.0) for n in INPUTS)
    return out.select(
        "date", "symbol", "actual", "has_momentum", "spy12", *INPUTS,
        (pl.col("mom12") * pl.col("spy12").fill_nan(0.0)).alias("mom12_x_spy"),
        (pl.col("mom6") * pl.col("spy12").fill_nan(0.0)).alias("mom6_x_spy"),
    ).sort("date", "symbol")  # fmt: skip


def walk(data: pl.DataFrame) -> pl.DataFrame:
    """Forecasts for every month with :data:`MIN_MONTHS` earlier months (module docs).

    Args:
        data: From :func:`table`.

    Returns:
        date, symbol, actual, ``forecast`` (next month's return over the T-bill), ``spy12``
        and the :data:`TERMS` (the stock's exposures, for the factor covariance).
    """
    data = data.sort("date", "symbol")
    x = np.column_stack([np.ones(data.height), data.select(TERMS).to_numpy()])
    y = data["actual"].to_numpy().astype(float)
    fit = np.isfinite(y) & data["has_momentum"].to_numpy()
    dates = data["date"].to_numpy()
    months = sorted(set(data["date"].to_list()))
    bounds = np.searchsorted(dates, np.array(months, dtype=dates.dtype))
    xtx = np.zeros((x.shape[1], x.shape[1]))
    xty = np.zeros(x.shape[1])
    fitted, out = 0, []
    for a, b in zip(bounds, [*bounds[1:], len(dates)], strict=True):
        if fitted >= MIN_MONTHS:
            coef = np.linalg.lstsq(xtx, xty, rcond=None)[0]
            out.append(data[a:b].select("date", "symbol", "actual", "spy12", *TERMS).with_columns(
                pl.Series("forecast", x[a:b] @ coef)))  # fmt: skip
        rows = slice(a, b)
        use = fit[rows]
        if use.any():
            xs = x[rows][use]
            xtx += xs.T @ xs
            xty += xs.T @ y[rows][use]
            fitted += 1
    return pl.concat(out) if out else pl.DataFrame()


#: Fewest stocks with a target for a month's slopes.
MIN_STOCKS = 200


def spy12_center(data: pl.DataFrame) -> float:
    """SPY's one-year return averaged over every month in ``data``.

    As in research (to be revisited): the covariance's interaction terms are centered on
    this full-sample mean, which uses months after each decision date. The forecasts are
    unaffected (the main terms absorb any centering).
    """
    months = data.group_by("date").agg(pl.col("spy12").first())["spy12"]
    return float(months.filter(months.is_finite()).mean())


def centered(data: pl.DataFrame, center: float) -> pl.DataFrame:
    """``data`` with the interaction terms built on ``spy12 - center`` (0 where unknown)."""
    s12 = pl.when(pl.col("spy12").is_finite()).then(pl.col("spy12") - center).otherwise(0.0)
    return data.with_columns((pl.col("mom12") * s12).alias("mom12_x_spy"),
                             (pl.col("mom6") * s12).alias("mom6_x_spy"))  # fmt: skip


def slopes(data: pl.DataFrame) -> pl.DataFrame:
    """Each month's cross-sectional slopes of the target on the :data:`TERMS` (Fama-MacBeth).

    The covariance of these slopes over earlier months is how the terms' payoffs move
    together, the factor part of the strategy's covariance (``strategies.meanvar.nine``).
    The interactions are centered (:func:`centered`) on :func:`spy12_center`.

    Args:
        data: From :func:`table`.

    Returns:
        date, one column per term and ``center`` (the same in every row); months with
        fewer than :data:`MIN_STOCKS` are left out.
    """
    center = spy12_center(data)
    data = centered(data, center)
    rows = []
    for (day,), month in data.filter(
        pl.col("has_momentum") & pl.col("actual").is_not_null()
    ).group_by("date"):
        if month.height < MIN_STOCKS:
            continue
        x = np.column_stack([np.ones(month.height), month.select(TERMS).to_numpy()])
        coef = np.linalg.lstsq(x, month["actual"].to_numpy(), rcond=None)[0][1:]
        rows.append({"date": day, **dict(zip(TERMS, coef.tolist(), strict=True))})
    return pl.DataFrame(rows).sort("date").with_columns(pl.lit(center).alias("center"))


#: Trading sessions in a year (Grinold's volatility is annualized with it).
TRADING_DAYS = 252


def trailing_annual(prices: pd.DataFrame) -> pd.Series:
    """Each column's trailing annual return, as production's mean-variance computes it.

    Grinold's scale is set to the spread of these across the candidates, so the forecasts
    come out at the size of production's expected returns. Columns production cannot
    forecast (too short, non-positive prices) are left out.
    """
    from portfolio_lab.strategies.meanvar.forecast import (  # noqa: PLC0415 - avoids a cycle
        ForecastSpec,
        forecast_one,
    )

    spec = ForecastSpec("historical_mean")
    out = {s: forecast_one(prices[s].to_numpy(), spec) for s in prices.columns}
    return pd.Series({s: v for s, v in out.items() if np.isfinite(v)}, dtype=float)
