"""Summary of the forecaster for the Forecasts page (``plab forecast publish``).

Graded on unseen months from ``walk.FIRST_GRADED``, on the stock-months production's own
forecast covers (``baseline``: stocks with about a year of prices): headline grades of the
forecast, of its pieces and of production's forecast; IC by year and 12-month trailing IC,
each against production's; forecast against outcome by forecast group.
"""

import json
from pathlib import Path
from typing import Any

import polars as pl

from portfolio_lab.research.forecaster import baseline, grade, walk
from portfolio_lab.research.panel import Panel

SUMMARY = "summary.json"
#: The forecast and its pieces, as graded on the page.
PIECES = {
    "forecast": "Forecast: linear + trees, averaged with the nets",
    "linear_trees": "Linear + trees alone (half of the forecast)",
    "nets": "Nets alone (the other half)",
    "linear": "Linear part alone",
    "production": "Production today: trailing one-year return",
}
TRAILING = 12


def build(forecasts: pl.DataFrame, production: pl.DataFrame) -> dict[str, Any]:
    """The page's summary.

    Args:
        forecasts: The saved forecasts (``walk.run``'s output).
        production: Production's expected return (``baseline.trailing``).
    """
    combined = (
        walk.combine(forecasts)
        .filter(pl.col("date").dt.year() >= walk.FIRST_GRADED)
        .join(production, on=["date", "symbol"])
    )
    keys = (
        "ic",
        "ic_t",
        "months_right",
        "years_right",
        "years",
        "slope",
        "r2",
        "tenth_yr",
        "ic_by_size",
    )
    pieces = {}
    for col, label in PIECES.items():
        r = grade.report(combined.with_columns(pl.col(col).alias("forecast")))
        pieces[col] = {"label": label, **{k: r[k] for k in keys}}
    head = grade.report(combined)
    old = {r["year"]: r["ic"] for r in pieces_yearly(combined, "production")}
    yearly = [{**r, "old": old.get(r["year"])} for r in head["yearly"]]
    months = grade.grade_months(combined).sort("date")
    old_months = grade.grade_months(combined.with_columns(pl.col("production").alias("forecast")))
    trailing = months.select("date", pl.col("ic").rolling_mean(TRAILING).alias("ic")).join(
        old_months.select("date", pl.col("ic").rolling_mean(TRAILING).alias("old")),
        on="date",
    )  # fmt: skip
    tenths = (
        _tenths(combined, "forecast")
        .join(
            _tenths(combined, "production").rename({"top": "p_top", "bottom": "p_bottom"}),
            on="date",
        )
        .sort("date")
    )
    growth = tenths.with_columns((1 + pl.col(c)).cum_prod().alias(c)
                                 for c in ("top", "bottom", "p_top", "p_bottom"))  # fmt: skip
    corr = combined.group_by("date").agg(pl.corr("linear_trees", "nets").alias("r"))["r"].median()
    return {
        "start": str(head["first"]), "end": str(head["last"]), "months": head["months"],
        # how alike the two models' forecasts are, and how much a year's IC moves by chance
        "corr": corr, "yearly_noise": months["ic"].std() / 12**0.5,
        "pieces": pieces, "yearly": yearly, "bins": head["bins"],
        "trailing": trailing.drop_nulls().to_dicts(),
        "tenths": growth.with_columns(pl.col("date").cast(pl.String)).to_dicts(),
    }  # fmt: skip


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


def pieces_yearly(combined: pl.DataFrame, col: str) -> list[dict]:
    """IC by year of one piece."""
    return grade.report(combined.with_columns(pl.col(col).alias("forecast")))["yearly"]


def publish(source: Path, dest: Path, panel: Panel) -> Path:
    """Write :func:`build`'s summary of ``source``'s forecasts into the ``dest`` folder."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / SUMMARY
    forecasts = pl.read_parquet(source / "forecasts.parquet")
    production = baseline.trailing(panel, sorted(forecasts["date"].unique().to_list()))
    summary = build(forecasts, production)
    path.write_text(json.dumps(summary, indent=1, default=str))
    return path
