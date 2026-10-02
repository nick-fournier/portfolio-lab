"""Can we forecast next month's stock returns? A forecasting study, independent of trading.

Every stock-month in the data directory (all stocks, dead ones included) with every input
we have: the stock's features (valuation, profitability, health and their changes, price
history, size, liquidity, industry-relative versions, macro sensitivities and tailwinds)
and the month's market environment (``env_*``: rates, spreads, VIX, inflation, ...).

The target is each stock's next-month return **relative to that month's average**: the
part that differs between stocks. (Whether the whole market rises is a separate question.)
For training it is clipped at each month's 1st and 99th percentiles, so a few extreme
stocks don't dominate.

Models, walk-forward: each test year is predicted by a model fit only on months whose
next month ended before the year began.

- ``momentum``: the 12-month return excluding the last month, as one plain rule.
- ``linear``: ridge regression on every stock input (as monthly percentiles).
- ``trees``: gradient-boosted trees on every stock input plus the environment, so it can
  learn interactions (e.g. an input that works only when VIX is high).

Graded per month on all stocks: rank correlation of forecast and outcome (IC), the slope
of outcome on forecast (1 = realistic size), out-of-sample R² (share of the variation
between stocks explained), and the gap between the best- and worst-forecast tenths.
"""

import logging
from datetime import date

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from portfolio_lab.research.dataset import STOCK_COLUMNS, build_dataset, rank_features
from portfolio_lab.research.panel import Panel

log = logging.getLogger(__name__)

H = 21
TARGET = "target"
#: First test year: earlier years are training only (five years of history).
FIRST_TEST_YEAR = 2004
MODELS = ("momentum", "linear", "trees")
#: Clip the training target at these monthly quantiles.
CLIP = (0.01, 0.99)
TREE_PARAMS = {
    "learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31,
    "min_samples_leaf": 2000, "l2_regularization": 1.0, "early_stopping": True,
    "validation_fraction": 0.1, "n_iter_no_change": 20, "random_state": 0,
}  # fmt: skip


def load(panel: Panel, features: pl.DataFrame, env: pl.DataFrame | None) -> pl.DataFrame:
    """Stock-months with ranked stock inputs, environment and the relative target."""
    data = build_dataset(panel, features, env)
    stock = [c for c in STOCK_COLUMNS if c in data.columns]
    data = rank_features(data, stock).with_columns(pl.col(c) - 0.5 for c in stock)
    fwd = pl.col(f"fwd_{H}")
    rel = fwd - fwd.mean().over("date")
    lo, hi = (rel.quantile(q).over("date") for q in CLIP)
    return data.with_columns(rel.alias("actual"), rel.clip(lo, hi).alias(TARGET))


def _inputs(data: pl.DataFrame, model: str) -> list[str]:
    stock = [c for c in STOCK_COLUMNS if c in data.columns]
    env = [c for c in data.columns if c.startswith("env_")]
    return stock + env if model == "trees" else stock


def walk_forward(data: pl.DataFrame, model: str) -> pl.DataFrame:
    """Out-of-sample forecasts for every test year (see module docs).

    Returns:
        date, symbol, forecast, actual (next-month return relative to the month's mean).
    """
    out = []
    labeled = data.filter(pl.col(TARGET).is_not_null())
    years = sorted({d.year for d in labeled["date"].to_list() if d.year >= FIRST_TEST_YEAR})
    for year in years:
        test = labeled.filter(pl.col("date").dt.year() == year)
        if model == "momentum":
            forecast = test["mom_12_1"].fill_null(0.0).to_numpy()
        else:
            first = test["session"].min()
            train = labeled.filter(pl.col(f"label_end_{H}") < first)
            # Inputs with no values yet (a series that starts later) break the tree binning.
            columns = [c for c in _inputs(data, model) if train[c].drop_nans().drop_nulls().len()]
            x_train = train.select(columns).to_numpy()
            x_test = test.select(columns).to_numpy()
            if model == "linear":
                fit = Ridge(alpha=10.0).fit(np.nan_to_num(x_train), train[TARGET].to_numpy())
                forecast = fit.predict(np.nan_to_num(x_test))
            else:
                fit = HistGradientBoostingRegressor(**TREE_PARAMS)
                fit.fit(x_train, train[TARGET].to_numpy())
                forecast = fit.predict(x_test)
        log.info("forecast %s %d: %d stock-months", model, year, test.height)
        out.append(test.select("date", "symbol", "actual").with_columns(
            pl.Series("forecast", forecast, dtype=pl.Float64)))  # fmt: skip
    return pl.concat(out)


def grade_months(forecasts: pl.DataFrame) -> pl.DataFrame:
    """Per month: IC, slope, R², best-minus-worst tenth and stocks graded."""
    rows = []
    for (day,), month in forecasts.drop_nulls(["forecast", "actual"]).group_by("date"):
        f, a = month["forecast"].to_numpy(), month["actual"].to_numpy()
        if len(f) < 50 or np.std(f) == 0:
            continue
        lo, hi = np.quantile(a, CLIP)
        a_clip = np.clip(a, lo, hi)
        fc = f - f.mean()
        slope = float((fc * (a_clip - a_clip.mean())).sum() / (fc**2).sum())
        rank_f = month["forecast"].rank().to_numpy()
        ic = float(np.corrcoef(rank_f, month["actual"].rank().to_numpy())[0, 1])
        r2 = 1 - float(
            ((a_clip - a_clip.mean() - fc) ** 2).sum() / ((a_clip - a_clip.mean()) ** 2).sum()
        )
        tenth = max(len(f) // 10, 1)
        order = np.argsort(f)
        spread = float(a[order[-tenth:]].mean() - a[order[:tenth]].mean())
        rows.append({"date": day, "stocks": len(f), "ic": ic, "slope": slope, "r2": r2,
                     "spread": spread})  # fmt: skip
    return pl.DataFrame(rows).sort("date")


def summarize(months: pl.DataFrame, split: date = date(2015, 1, 1)) -> dict:
    """Headline numbers overall and per half (before and after ``split``)."""

    def stats(m: pl.DataFrame) -> dict:
        ic = m["ic"]
        return {
            "months": m.height,
            "ic": ic.mean(),
            "ic_t": ic.mean() / ic.std() * m.height**0.5,
            "ic_positive": (ic > 0).mean(),
            "slope": m["slope"].median(),
            "r2": m["r2"].mean(),
            "spread_yr": (1 + m["spread"]).product() ** (12 / m.height) - 1,
        }

    return {
        "all": stats(months),
        "first_half": stats(months.filter(pl.col("date") < split)),
        "second_half": stats(months.filter(pl.col("date") >= split)),
        "years_ic_positive": months.group_by(pl.col("date").dt.year())
        .agg(pl.col("ic").mean())
        .select((pl.col("ic") > 0).mean())
        .item(),
    }
