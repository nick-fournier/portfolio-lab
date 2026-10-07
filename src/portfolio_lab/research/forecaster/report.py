"""Summary of the forecaster for the Forecasts page (``plab forecast publish``).

Graded on unseen months from ``walk.FIRST_GRADED``, on the stock-months all three forecasts
cover: the forecaster (``nine``), the previous forecaster (``walk``: linear + trees averaged
with the nets) and production's (``baseline``: trailing one-year return). Headline grades of
each; IC by year and 12-month trailing IC; forecast against outcome by forecast group; the
best and worst forecast tenths. Outcomes are each stock's return relative to the month's
average stock, so the three are graded on the same footing. Also graded where the strategy
picks: among the month's :data:`LIQUID` most liquid stocks, the IC and the compounded return
of the :data:`TOP` best-forecast stocks held in equal weights.
"""

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from portfolio_lab.research.forecaster import baseline, grade, nine, walk
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.scoreboard import forward_returns

SUMMARY = "summary.json"
#: The forecast and its pieces, as graded on the page.
PIECES = {
    "forecast": "Forecast: least squares on nine terms",
    "previous": "Previous forecaster: linear + trees, averaged with the nets",
    "production": "Production's forecast: trailing one-year return",
}
TRAILING = 12
#: The strategy's candidates: this many most liquid stocks each month.
LIQUID = 400
#: Stocks held in the equal-weight grade.
TOP = 30


def build(
    forecasts: pl.DataFrame,
    previous: pl.DataFrame,
    production: pl.DataFrame,
    liquid: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """The page's summary.

    Args:
        forecasts: The forecaster's forecasts (``nine.walk``'s output).
        previous: The previous forecaster's saved forecasts (``walk.run``'s output).
        production: Production's expected return (``baseline.trailing``).
        liquid: date, symbol, ``liquidity`` (higher = more liquid) and ``r`` (next month's
            return) for grading among the most liquid stocks; ``None`` leaves that out.
    """
    combined = (
        walk.combine(previous)
        .select("date", "symbol", "actual", "size", pl.col("forecast").alias("previous"))
        .filter(pl.col("date").dt.year() >= walk.FIRST_GRADED)
        .join(forecasts.select("date", "symbol", "forecast"), on=["date", "symbol"])
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
        if liquid is not None:
            pieces[col].update(_liquid_grades(combined, liquid, col))
    head = grade.report(combined)
    refs = {"old": "production", "prev": "previous"}
    by_year = {key: {r["year"]: r["ic"] for r in pieces_yearly(combined, col)}
               for key, col in refs.items()}  # fmt: skip
    yearly = [{**r, **{key: by_year[key].get(r["year"]) for key in refs}} for r in head["yearly"]]
    trailing = _trailing(combined, "forecast", "ic")
    for key, col in refs.items():
        trailing = trailing.join(_trailing(combined, col, key), on="date")
    tenths = _tenths(combined, "forecast")
    for prefix, col in (("p_", "production"), ("q_", "previous")):
        tenths = tenths.join(
            _tenths(combined, col).rename({"top": f"{prefix}top", "bottom": f"{prefix}bottom"}),
            on="date",
        )
    cols = ("top", "bottom", "p_top", "p_bottom", "q_top", "q_bottom")
    growth = tenths.sort("date").with_columns((1 + pl.col(c)).cum_prod().alias(c) for c in cols)
    months = grade.grade_months(combined)
    return {
        "start": str(head["first"]), "end": str(head["last"]), "months": head["months"],
        # how much a year's IC moves by chance
        "yearly_noise": months["ic"].std() / 12**0.5,
        "pieces": pieces, "yearly": yearly, "bins": head["bins"],
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


def pieces_yearly(combined: pl.DataFrame, col: str) -> list[dict]:
    """IC by year of one piece."""
    return grade.report(combined.with_columns(pl.col(col).alias("forecast")))["yearly"]


def liquid_returns(panel: Panel, month_ends: list) -> pl.DataFrame:
    """date, symbol, ``liquidity`` (trailing dollar volume) and ``r`` (next month's return)."""
    frames = []
    for day, nxt in pairwise(month_ends):
        i, j = panel.date_index[day], panel.date_index[nxt]
        frames.append(pl.DataFrame({
            "date": [day] * len(panel.symbols), "symbol": panel.symbols,
            "liquidity": panel.field("adv")[i], "r": forward_returns(panel, i, j - i),
        }))  # fmt: skip
    return pl.concat(frames).with_columns(pl.col("liquidity", "r").fill_nan(None))


def publish(source: Path, dest: Path, panel: Panel) -> Path:
    """Write :func:`build`'s summary of ``source``'s forecasts into the ``dest`` folder."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / SUMMARY
    forecasts = pl.read_parquet(source / nine.FILE)
    previous = pl.read_parquet(source / "forecasts.parquet")
    months = sorted(forecasts["date"].unique().to_list())
    production = baseline.trailing(panel, months)
    summary = build(forecasts, previous, production, liquid_returns(panel, months))
    path.write_text(json.dumps(summary, indent=1, default=str))
    return path
