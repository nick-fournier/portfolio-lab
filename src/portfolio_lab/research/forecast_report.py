"""The Forecasts page's summary of the forecasting study (``research.forecasting``).

The chosen forecaster is :data:`LOCKED`: the slim regime-aware linear model plus gradient-
boosted trees trained on its residuals, added at half strength (``linear_trees`` without the
size and macro-sensitivity inputs). :func:`build` reads the study's saved results (run on
the Sharadar history) and condenses them into a small JSON file for the page: headline
numbers, ranking by year against the linear model alone, the models tried, input groups'
importance, and the ranking by stock size and credit conditions. Only derived statistics
are written, no prices or tickers.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm

from portfolio_lab.research.forecasting import blend, by_size, grade_months, summarize

#: The chosen forecaster: study run, and the strength of the trees' correction.
LOCKED = ("linear_trees-no_size-no_sensitivities", 0.5)
#: Models tried, shown on the page: (label, study run, correction strength or None).
TRIED = (
    ("Momentum alone", "momentum", None),
    ("Linear, all inputs", "linear", None),
    ("Trees, all inputs + market", "trees", None),
    ("Linear, slim + regime", "linear_regime-no_size-no_sensitivities", None),
    ("50/50 blend of linear and trees", "blend", None),
    ("Trees on linear, full strength", LOCKED[0], 1.0),
    ("Trees on linear, half strength (chosen)", LOCKED[0], LOCKED[1]),
)
#: Input groups' importance, from the linear model with every group (labels for the page).
GROUP_LABELS = {
    "health": "Health & profitability", "price": "Price & momentum",
    "changes": "Year-on-year changes", "valuation": "Valuation",
    "industry": "Industry-relative", "regime_terms": "Momentum in bears and rebounds",
    "sensitivities": "Macro sensitivities", "size": "Size & liquidity",
}  # fmt: skip
DROPPED = ("sensitivities", "size")
#: Groups in the fit plot: each month's stocks split into this many by forecast.
FIT_BINS = 20
#: Margins are 90% (1.645 standard errors either side).
Z90 = 1.645
SUMMARY = "summary.json"


def _strength(forecasts: pl.DataFrame, k: float) -> pl.DataFrame:
    """A stacked run's forecasts with the trees' correction at strength ``k``."""
    base = pl.col("forecast") - pl.col("correction")
    return forecasts.with_columns((base + k * pl.col("correction")).alias("forecast"))


#: The two runs the "blend" entry of :data:`TRIED` averages (as within-month ranks).
BLENDED = ("linear_regime-no_size-no_sensitivities", "trees_regime-no_size-no_sensitivities")


def _months(folder: Path, run: str, k: float | None) -> pl.DataFrame:
    if run == "blend":
        a, b = (pl.read_parquet(folder / f"{r}.forecasts.parquet") for r in BLENDED)
        return grade_months(blend(a, b, 0.5))
    if k is None:
        return pl.read_parquet(folder / f"{run}.parquet")
    return grade_months(_strength(pl.read_parquet(folder / f"{run}.forecasts.parquet"), k))


def _headline(months: pl.DataFrame) -> dict[str, Any]:
    s = summarize(months)
    years = months.group_by(pl.col("date").dt.year()).agg(pl.col("ic").mean())
    return {
        "ic": s["all"]["ic"], "t": s["all"]["ic_t"], "first_half": s["first_half"]["ic"],
        "second_half": s["second_half"]["ic"], "spread": s["all"]["spread_yr"],
        "years": years.height, "years_right": int((years["ic"] > 0).sum()),
        "start": months["date"].min().year, "end": months["date"].max().year,
    }  # fmt: skip


def _conditions(forecasts: pl.DataFrame, env: pl.DataFrame) -> list[dict[str, Any]]:
    """Ranking with and without the trees' correction by credit-spread third."""
    k = LOCKED[1]
    both = (
        grade_months(_strength(forecasts, 0.0)).select("date", pl.col("ic").alias("linear"))
        .join(grade_months(_strength(forecasts, k)).select("date", pl.col("ic").alias("model")),
              on="date")
        .join(env.select("date", "baa_spread_pct"), on="date", how="left")
        .drop_nulls("baa_spread_pct")
        .with_columns(pl.col("baa_spread_pct").qcut(3, labels=["calm", "middle", "stressed"])
                      .alias("credit"))
    )  # fmt: skip
    table = both.group_by("credit").agg(pl.col("linear").mean(), pl.col("model").mean())
    order = {"calm": 0, "middle": 1, "stressed": 2}
    return sorted(table.to_dicts(), key=lambda r: order[r["credit"]])


#: Years of earlier results needed before Grinold's rule has an IC to use.
GRINOLD_MIN_YEARS = 3
TRADING_DAYS_PER_MONTH = 21


def grinold(forecasts: pl.DataFrame, volatility: pl.DataFrame) -> pl.DataFrame:
    """Forecasts rebuilt as expected returns with Grinold's rule: alpha = IC x vol x score.

    Score: the forecast's percentile within the month as a standard normal value. Vol: the
    stock's daily volatility over the past year (``volatility``: date, symbol, volatility)
    scaled to a month. IC: the average monthly IC of all earlier test years, so no year uses
    its own result; the first :data:`GRINOLD_MIN_YEARS` years are dropped. Returns the
    forecasts with ``forecast`` replaced by alpha.
    """
    f = forecasts.drop_nulls(["forecast", "actual"])
    ic = grade_months(f).with_columns(pl.col("date").dt.year().alias("year"))
    years = sorted(ic["year"].unique().to_list())
    past = {y: float(ic.filter(pl.col("year") < y)["ic"].mean())
            for k, y in enumerate(years) if k >= GRINOLD_MIN_YEARS}  # fmt: skip
    f = (
        f.join(volatility, on=["date", "symbol"], how="inner")
        .with_columns(pl.col("date").dt.year().alias("year"))
        .filter(pl.col("year").is_in(list(past)))
        .with_columns(((pl.col("forecast").rank("average").over("date") - 0.5)
                       / pl.len().over("date")).alias("_pct"))
    )  # fmt: skip
    score = pl.Series("_z", norm.ppf(f["_pct"].to_numpy()))
    month_vol = pl.col("volatility") * np.sqrt(TRADING_DAYS_PER_MONTH)
    alpha = pl.col("year").replace_strict(past) * month_vol * pl.col("_z")
    return (
        f.with_columns(score)
        .with_columns(alpha.alias("forecast"))
        .drop("_pct", "_z", "year", "volatility")
        .drop_nulls("forecast")
    )


def build(
    folder: Path, env: pl.DataFrame | None, volatility: pl.DataFrame | None = None
) -> dict[str, Any]:
    """The page's summary from the study results in ``folder`` (see module docs).

    ``volatility`` (date, symbol, volatility from the feature panel) adds Grinold's rule.
    """
    run, k = LOCKED
    forecasts = pl.read_parquet(folder / f"{run}.forecasts.parquet")
    chosen = grade_months(_strength(forecasts, k))
    linear = grade_months(_strength(forecasts, 0.0))
    yearly = (
        chosen.group_by(pl.col("date").dt.year().alias("year")).agg(
            pl.col("ic").mean().alias("model"),
            (Z90 * pl.col("ic").std() / pl.len().sqrt()).alias("margin"))
        .join(linear.group_by(pl.col("date").dt.year().alias("year"))
              .agg(pl.col("ic").mean().alias("linear")), on="year")
        .sort("year")
    )  # fmt: skip
    importance = pl.read_parquet(folder / "linear_regime.importance.parquet")
    groups = importance.group_by("group").agg(pl.col("drop").mean()).sort("drop", descending=True)
    sizes = {
        "model": {g: s["all"]["ic"] for g, s in by_size(_strength(forecasts, k)).items()},
        "linear": {g: s["all"]["ic"] for g, s in by_size(_strength(forecasts, 0.0)).items()},
    }
    return {
        "headline": _headline(chosen),
        "yearly": yearly.to_dicts(),
        "tried": [{"label": label, **_headline(_months(folder, r, s))} for label, r, s in TRIED],
        "groups": [
            {"label": GROUP_LABELS.get(g, g), "drop": d, "dropped": g in DROPPED}
            for g, d in groups.iter_rows()
        ],
        "sizes": sizes,
        "conditions": _conditions(forecasts, env) if env is not None else [],
        "strength": k,
        "fit": fit_bins(_strength(forecasts, k)),
        "grinold": fit_bins(grinold(_strength(forecasts, k), volatility))
        if volatility is not None
        else [],
    }


def fit_bins(forecasts: pl.DataFrame) -> list[dict[str, Any]]:
    """Forecast against outcome by forecast group (see the Forecasts page's fit plot).

    Each month the stocks are split into :data:`FIT_BINS` equal groups by forecast. Per
    group, over all months: the average forecast and outcome (next-month return relative
    to the month's average), a 90% margin on the outcome's average (from how much it varied
    between months), and the middle half of individual stocks' outcomes.
    """
    f = forecasts.drop_nulls(["forecast", "actual"]).with_columns(
        (pl.col("forecast").rank("ordinal").over("date") * FIT_BINS
         // (pl.len().over("date") + 1)).alias("bin"))  # fmt: skip
    monthly = f.group_by("bin", "date").agg(pl.col("actual").mean().alias("m"))
    margin = monthly.group_by("bin").agg(
        (Z90 * pl.col("m").std() / pl.len().sqrt()).alias("margin")
    )
    return (
        f.group_by("bin")
        .agg(pl.col("forecast").mean(), pl.col("actual").mean(),
             pl.col("actual").quantile(0.25).alias("q25"),
             pl.col("actual").quantile(0.75).alias("q75"))
        .join(margin, on="bin").sort("bin").to_dicts()
    )  # fmt: skip


def publish(
    source: Path, dest: Path, env: pl.DataFrame | None, volatility: pl.DataFrame | None = None
) -> Path:
    """Write :func:`build`'s summary to ``dest`` (a forecast_study results folder)."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / SUMMARY
    path.write_text(json.dumps(build(source, env, volatility), indent=1, default=str))
    return path
