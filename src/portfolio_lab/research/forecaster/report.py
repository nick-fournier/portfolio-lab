"""Summary of the forecaster for the Forecasts page, rebuilt with the forecasts nightly.

Graded on unseen months from :data:`FIRST_GRADED`, on the stock-months both forecasts
cover: the nine-term forecaster (``nine``) and production's expected return
(``baseline``: the trailing one-year return). Outcomes are each stock's return relative to
the month's average stock, so the two are graded on the same footing. Headline grades of
each; IC by year and 12-month trailing IC; forecast against outcome by forecast group; the
best and worst forecast tenths. Also graded where the strategies pick, among the month's
:data:`LIQUID` most liquid stocks: the IC, the compounded return of the :data:`TOP`
best-forecast stocks held in equal weights, and forecast against outcome after Grinold's
rule (the forecasts as class 2 hands them to the optimizer, ``strategies.meanvar.nine``).
"""

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from portfolio_lab.research.forecaster import baseline, grade, nine
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.scoreboard import forward_returns

SUMMARY = "summary.json"
#: First year graded (the forecaster needs earlier months to learn from).
FIRST_GRADED = 2009
#: The two forecasts, as graded on the page.
PIECES = {
    "forecast": "Class 2: least squares on nine terms",
    "production": "Class 1 (production): trailing one-year return",
}
TRAILING = 12
#: The strategies' candidates: this many most liquid stocks each month.
LIQUID = 400
#: Stocks held in the equal-weight grade.
TOP = 30
#: Sessions of prices behind Grinold's volatility and scale (the strategies' window).
WINDOW = 252
MIN_COVERAGE = 0.95
GRADES = ("ic", "ic_t", "months_right", "years_right", "years", "slope", "r2", "tenth_yr",
          "ic_by_size")  # fmt: skip


def build(combined: pl.DataFrame, liquid: pl.DataFrame, grinold: pl.DataFrame) -> dict[str, Any]:
    """The page's summary.

    Args:
        combined: date, symbol, ``actual`` (next month relative to the average stock),
            ``size`` (market value's percentile, centered on 0), ``forecast`` and
            ``production``; from :data:`FIRST_GRADED`.
        liquid: date, symbol, ``liquidity`` (higher = more liquid) and ``r`` (next month's
            return), for grading among the most liquid stocks.
        grinold: date, symbol, ``forecast`` (monthly, over the T-bill, in Grinold's form)
            and ``actual`` (next month over the T-bill), for the most liquid stocks.
    """
    pieces = {}
    for col, label in PIECES.items():
        r = grade.report(combined.with_columns(pl.col(col).alias("forecast")))
        pieces[col] = {"label": label, **{k: r[k] for k in GRADES},
                       **_liquid_grades(combined, liquid, col)}  # fmt: skip
    head = grade.report(combined)
    old = {r["year"]: r["ic"] for r in grade.report(
        combined.with_columns(pl.col("production").alias("forecast")))["yearly"]}  # fmt: skip
    yearly = [{**r, "old": old.get(r["year"])} for r in head["yearly"]]
    trailing = _trailing(combined, "forecast", "ic").join(
        _trailing(combined, "production", "old"), on="date"
    )
    tenths = _tenths(combined, "forecast").join(
        _tenths(combined, "production").rename({"top": "p_top", "bottom": "p_bottom"}), on="date"
    )
    cols = ("top", "bottom", "p_top", "p_bottom")
    growth = tenths.sort("date").with_columns((1 + pl.col(c)).cum_prod().alias(c) for c in cols)
    after = grade.fit_bins(grinold)
    x, a = after["forecast"].to_numpy(), after["actual"].to_numpy()
    return {
        "start": str(head["first"]), "end": str(head["last"]), "months": head["months"],
        # how much a year's IC moves by chance
        "yearly_noise": grade.grade_months(combined)["ic"].std() / 12**0.5,
        "pieces": pieces, "yearly": yearly, "bins": head["bins"],
        "grinold_bins": after.to_dicts(), "grinold_slope": float(np.polyfit(x, a, 1)[0]),
        "trailing": trailing.drop_nulls().sort("date").to_dicts(),
        "tenths": growth.with_columns(pl.col("date").cast(pl.String)).to_dicts(),
    }  # fmt: skip


def _liquid_grades(combined: pl.DataFrame, liquid: pl.DataFrame, col: str) -> dict[str, float]:
    """IC among the :data:`LIQUID` most liquid, and their :data:`TOP` stocks' return per year."""
    rank = pl.col("liquidity").rank("ordinal", descending=True).over("date")
    pool = (
        combined.join(liquid, on=["date", "symbol"]).filter(rank <= LIQUID).drop_nulls([col, "r"])
    )
    ic = grade.grade_months(pool.with_columns(pl.col(col).alias("forecast")))["ic"].mean()
    top = (
        pool.sort(col, descending=True)
        .group_by("date", maintain_order=True)
        .agg(pl.col("r").head(TOP).mean())
        .sort("date")
    )["r"].to_numpy()
    return {"liquid_ic": float(ic),
            "top_yr": float(np.prod(1 + top) ** (12 / len(top)) - 1)}  # fmt: skip


def _trailing(combined: pl.DataFrame, col: str, name: str) -> pl.DataFrame:
    """12-month trailing IC of ``col``, as ``name``."""
    months = grade.grade_months(combined.with_columns(pl.col(col).alias("forecast"))).sort("date")
    return months.select("date", pl.col("ic").rolling_mean(TRAILING).alias(name))


def _tenths(combined: pl.DataFrame, col: str) -> pl.DataFrame:
    """Per month: actual return of the top and bottom tenth by ``col``, vs the average stock."""
    rank = pl.col(col).rank("ordinal").over("date") / pl.len().over("date")
    return (
        combined.drop_nulls([col, "actual"])
        .with_columns(rank.alias("_q"))
        .group_by("date")
        .agg(
            pl.col("actual").filter(pl.col("_q") > 0.9).mean().alias("top"),
            pl.col("actual").filter(pl.col("_q") <= 0.1).mean().alias("bottom"),
        )
    )


def liquid_returns(panel: Panel, month_ends: list) -> pl.DataFrame:
    """date, symbol, ``liquidity`` (trailing dollar volume) and ``r`` (next month's return)."""
    frames = []
    for day, nxt in pairwise(month_ends):
        i, j = panel.date_index[day], panel.date_index[nxt]
        frames.append(pl.DataFrame({
            "date": [day] * len(panel.symbols), "symbol": panel.symbols,
            "liquidity": panel.field("adv")[i].astype(np.float64),
            "r": forward_returns(panel, i, j - i).astype(np.float64),
        }))  # fmt: skip
    return pl.concat(frames).with_columns(pl.col("liquidity", "r").fill_nan(None))


def grinold_form(panel: Panel, forecasts: pl.DataFrame, liquid: pl.DataFrame) -> pl.DataFrame:
    """The :data:`LIQUID` most liquid stocks' forecasts in Grinold's form, each month.

    As class 2 builds them for its candidates: ``k x volatility x z``, with ``z`` the
    forecast standardized across the stocks, volatility over the last :data:`WINDOW`
    sessions and ``k`` matching the spread of their trailing annual returns
    (``nine.trailing_annual``, production's expected returns); monthly, over the T-bill.
    """
    rank = pl.col("liquidity").rank("ordinal", descending=True).over("date")
    pool = forecasts.join(liquid.select("date", "symbol", "liquidity"), on=["date", "symbol"])
    pool = pool.filter(rank <= LIQUID).drop_nulls(["forecast", "actual"])
    rets = panel.field("ret_cc")
    frames = []
    for (day,), month in pool.group_by("date"):
        i = panel.date_index[day]
        if i < WINDOW:
            continue
        cols = [panel.symbol_index[s] for s in month["symbol"]]
        window = rets[i - WINDOW + 1 : i + 1][:, cols].astype(np.float64)
        ok = np.isfinite(window).mean(axis=0) >= MIN_COVERAGE
        prices = pd.DataFrame(np.cumprod(1 + np.nan_to_num(window[:, ok]), axis=0),
                              columns=np.asarray(month["symbol"])[ok])  # fmt: skip
        reference = nine.trailing_annual(prices)
        f = pd.Series(month["forecast"].to_numpy(), index=month["symbol"]).reindex(reference.index)
        if len(f) < 2 or f.std() == 0:
            continue
        vol = prices[reference.index].pct_change().std() * nine.TRADING_DAYS**0.5
        raw = (f - f.mean()) / f.std() * vol
        k = float(reference.std() / raw.std()) if raw.std() > 0 else 1.0
        actual = dict(zip(month["symbol"], month["actual"], strict=True))
        frames.append(pl.DataFrame({
            "date": [day] * len(raw), "symbol": list(raw.index),
            "forecast": (k * raw / 12).to_numpy(), "actual": [actual[s] for s in raw.index],
        }))  # fmt: skip
    return pl.concat(frames)


def publish(folder: Path, panel: Panel, monthly: pl.DataFrame) -> Path:
    """Write :func:`build`'s summary of ``folder``'s forecasts (``nine.FILE``) into it.

    Args:
        folder: The forecaster's results (``DataPaths.forecaster``).
        panel: Prices, for production's forecast, liquidity and outcomes.
        monthly: The monthly stock inputs, for each stock's size.
    """
    forecasts = pl.read_parquet(folder / nine.FILE).filter(pl.col("date").dt.year() >= FIRST_GRADED)
    months = sorted(forecasts["date"].unique().to_list())
    size = monthly.select(
        "date", "symbol",
        (pl.col("market_value").rank().over("date") / pl.len().over("date") - 0.5).alias("size"),
    )  # fmt: skip
    combined = (
        forecasts.select("date", "symbol", "forecast",
                         (pl.col("actual") - pl.col("actual").mean().over("date")).alias("actual"))
        .join(baseline.trailing(panel, months), on=["date", "symbol"])
        .join(size, on=["date", "symbol"], how="left")
    )  # fmt: skip
    liquid = liquid_returns(panel, months)
    summary = build(combined, liquid, grinold_form(panel, forecasts, liquid))
    path = folder / SUMMARY
    path.write_text(json.dumps(summary, indent=1, default=str))
    return path
