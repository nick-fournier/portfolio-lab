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

import polars as pl

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


def build(folder: Path, env: pl.DataFrame | None) -> dict[str, Any]:
    """The page's summary from the study results in ``folder`` (see module docs)."""
    run, k = LOCKED
    forecasts = pl.read_parquet(folder / f"{run}.forecasts.parquet")
    chosen = grade_months(_strength(forecasts, k))
    linear = grade_months(_strength(forecasts, 0.0))
    yearly = (
        chosen.group_by(pl.col("date").dt.year().alias("year")).agg(pl.col("ic").alias("model"))
        .join(linear.group_by(pl.col("date").dt.year().alias("year"))
              .agg(pl.col("ic").alias("linear")), on="year")
        .with_columns(pl.col("model").list.mean(), pl.col("linear").list.mean())
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
    }


def publish(source: Path, dest: Path, env: pl.DataFrame | None) -> Path:
    """Write :func:`build`'s summary to ``dest`` (a forecast_study results folder)."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / SUMMARY
    path.write_text(json.dumps(build(source, env), indent=1, default=str))
    return path
