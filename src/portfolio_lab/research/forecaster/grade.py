"""Grading forecasts on unseen months.

Per month (:func:`grade_months`): the rank correlation of forecast and outcome (IC), and
the gap between the best- and worst-forecast tenths' outcomes. Over all months
(:func:`report`): average IC, its t-statistic, years with a positive IC, the slope of
outcome on forecast across :data:`BINS` forecast groups (1 = right-sized forecasts) and
separately for the below- and above-average halves, R² against forecasting the average
for every stock, the best-minus-worst tenth per year, and IC by size third.
"""

from typing import Any

import numpy as np
import polars as pl

#: Forecast groups per month for the slope.
BINS = 20
#: Size thirds by market value's percentile (centered on 0), smallest first.
SIZE_GROUPS = {"small": (-0.5, -1 / 6), "mid": (-1 / 6, 1 / 6), "large": (1 / 6, 0.5)}
Z90 = 1.645


def grade_months(forecasts: pl.DataFrame, min_stocks: int = 50) -> pl.DataFrame:
    """Per month: stocks, IC (0 when the forecast is constant) and best-minus-worst tenth."""
    rows = []
    for (day,), month in forecasts.drop_nulls(["forecast", "actual"]).group_by("date"):
        f, a = month["forecast"].to_numpy(), month["actual"].to_numpy()
        if len(f) < min_stocks:
            continue
        ic = 0.0
        if np.std(f) > 0:
            ic = float(np.corrcoef(month["forecast"].rank().to_numpy(),
                                   month["actual"].rank().to_numpy())[0, 1])  # fmt: skip
        tenth = max(len(f) // 10, 1)
        order = np.argsort(f)
        spread = float(a[order[-tenth:]].mean() - a[order[:tenth]].mean())
        rows.append({"date": day, "stocks": len(f), "ic": ic, "spread": spread})
    return pl.DataFrame(rows).sort("date")


def fit_bins(forecasts: pl.DataFrame) -> pl.DataFrame:
    """Average forecast and outcome per forecast group (:data:`BINS` per month).

    With a 90% margin on the outcome's average, from how much it varied between months,
    and the middle half of single stocks' outcomes (``q25``, ``q75``).
    """
    f = forecasts.drop_nulls(["forecast", "actual"]).with_columns(
        (pl.col("forecast").rank("ordinal").over("date") * BINS
         // (pl.len().over("date") + 1)).alias("bin"))  # fmt: skip
    monthly = f.group_by("bin", "date").agg(pl.col("actual").mean().alias("m"))
    margin = monthly.group_by("bin").agg(
        (Z90 * pl.col("m").std() / pl.len().sqrt()).alias("margin")
    )
    return (
        f.group_by("bin")
        .agg(
            pl.col("forecast").mean(),
            pl.col("actual").mean(),
            pl.col("actual").quantile(0.25).alias("q25"),
            pl.col("actual").quantile(0.75).alias("q75"),
        )
        .join(margin, on="bin")
        .sort("bin")
    )


def report(forecasts: pl.DataFrame) -> dict[str, Any]:
    """Headline grades of ``forecasts`` (date, symbol, forecast, actual, size; module docs)."""
    months = grade_months(forecasts)
    ic = months["ic"]
    yearly = (
        months.group_by(pl.col("date").dt.year().alias("year"))
        .agg(pl.col("ic").mean(), (Z90 * pl.col("ic").std() / pl.len().sqrt()).alias("margin"))
        .sort("year")
    )
    centered = forecasts.drop_nulls(["forecast", "actual"]).with_columns(
        pl.col("forecast") - pl.col("forecast").mean().over("date")
    )
    r2 = 1 - float(((centered["actual"] - centered["forecast"]) ** 2).mean()) / float(
        (centered["actual"] ** 2).mean()
    )
    # forecasts relative to the month's average, so the loser/winner split is at 0
    bins = fit_bins(
        forecasts.with_columns(pl.col("forecast") - pl.col("forecast").mean().over("date"))
    )
    x, a = bins["forecast"].to_numpy(), bins["actual"].to_numpy()
    sizes = {}
    for name, (lo, hi) in SIZE_GROUPS.items():
        group = forecasts.filter(pl.col("size").is_between(lo, hi, "left"))
        sizes[name] = float(grade_months(group)["ic"].mean())
    return {
        "first": months["date"].min(), "last": months["date"].max(), "months": months.height,
        "ic": float(ic.mean()), "ic_t": float(ic.mean() / ic.std() * months.height**0.5),
        "years_right": int((yearly["ic"] > 0).sum()), "years": yearly.height,
        "months_right": int((ic > 0).sum()),
        "slope": float(np.polyfit(x, a, 1)[0]),
        "slope_losers": float(np.polyfit(x[x < 0], a[x < 0], 1)[0]),
        "slope_winners": float(np.polyfit(x[x >= 0], a[x >= 0], 1)[0]),
        "r2": r2,
        "tenth_yr": float((1 + months["spread"]).product() ** (12 / months.height) - 1),
        "ic_by_size": sizes,
        "yearly": yearly.to_dicts(),
        "bins": bins.to_dicts(),
    }  # fmt: skip
