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
- ``linear_regime``: ``linear`` plus momentum and last month's return interacted with the
  market state (bear, rebound, drawdown), since momentum behaves differently then.
- ``trees_regime``: ``trees`` plus the market-state indicators (``mkt_*``, from
  ``research.regimes``: drawdown, trend, volatility, breadth, absorption, warning flags,
  bear and rebound), all as known at the month end.
- ``pls``: partial least squares on every stock input: inputs compressed to the few
  combinations that best predict returns (a check on dropping inputs instead).

:func:`importance` measures how much each input group matters: the drop in a fitted
model's out-of-sample IC when that group's values are shuffled between stocks.

Graded per month on all stocks: rank correlation of forecast and outcome (IC), the slope
of outcome on forecast (1 = realistic size), out-of-sample R² (share of the variation
between stocks explained), and the gap between the best- and worst-forecast tenths.
"""

import logging
from datetime import date

import numpy as np
import polars as pl
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from portfolio_lab.research import regimes
from portfolio_lab.research.dataset import STOCK_COLUMNS, build_dataset, rank_features
from portfolio_lab.research.panel import Panel

log = logging.getLogger(__name__)

H = 21
TARGET = "target"
#: First test year: earlier years are training only (five years of history).
FIRST_TEST_YEAR = 2004
MODELS = ("momentum", "linear", "trees", "linear_regime", "trees_regime", "pls")
#: Input groups, for importance and for dropping (``mkt`` and ``env`` are market-wide).
GROUPS: dict[str, tuple[str, ...]] = {
    "valuation": ("earnings_yield", "book_to_market", "cf_yield", "fcf_yield", "sales_yield",
                  "dividend_yield"),
    "health": ("roa", "cfo_to_assets", "accruals", "gross_profitability", "operating_margin",
               "leverage", "current_ratio", "fscore"),
    "changes": ("d_roa", "d_lt_debt", "d_current_ratio", "d_gross_margin", "d_asset_turnover",
                "share_issuance", "asset_growth", "sales_growth"),
    "price": ("mom_12_1", "ret_1m", "volatility", "beta"),
    "size": ("log_size", "log_adv"),
    "industry": tuple(c for c in STOCK_COLUMNS if c.endswith("_ind")),
    "sensitivities": tuple(c for c in STOCK_COLUMNS if c.endswith(("_beta", "_tailwind"))
                           and c != "beta"),
}  # fmt: skip
#: Market-state indicators from ``research.regimes`` (as known at the month end).
MARKET = ("drawdown", "below_ma200", "vol3m", "breadth", "absorption", "vol_pct",
          "absorption_rising", "credit_divergence", "breadth_divergence", "curve_inverted",
          "complacent", "bear", "rebound")  # fmt: skip
#: Stock inputs interacted with the market state in ``linear_regime``.
INTERACTED = ("mom_12_1", "ret_1m")
STATES = ("bear", "rebound", "drawdown")
#: Clip the training target at these monthly quantiles.
CLIP = (0.01, 0.99)
TREE_PARAMS = {
    "learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31,
    "min_samples_leaf": 2000, "l2_regularization": 1.0, "early_stopping": True,
    "validation_fraction": 0.1, "n_iter_no_change": 20, "random_state": 0,
}  # fmt: skip


def load(panel: Panel, features: pl.DataFrame, env: pl.DataFrame | None) -> pl.DataFrame:
    """Stock-months with ranked stock inputs, environment, market state and the target."""
    data = build_dataset(panel, features, env)
    stock = [c for c in STOCK_COLUMNS if c in data.columns]
    data = rank_features(data, stock).with_columns(pl.col(c) - 0.5 for c in stock)
    if env is not None:
        market = regimes.signals(panel, env).select(
            "date", *[pl.col(c).cast(pl.Float64).alias(f"mkt_{c}") for c in MARKET]
        )
        data = data.join(market, on="date", how="left")
        data = data.with_columns(
            (pl.col(c) * pl.col(f"mkt_{m}").fill_null(0.0)).alias(f"{c}_x_{m}")
            for c in INTERACTED
            for m in STATES
        )
    fwd = pl.col(f"fwd_{H}")
    rel = fwd - fwd.mean().over("date")
    lo, hi = (rel.quantile(q).over("date") for q in CLIP)
    return data.with_columns(rel.alias("actual"), rel.clip(lo, hi).alias(TARGET))


def _inputs(data: pl.DataFrame, model: str, drop: tuple[str, ...] = ()) -> list[str]:
    """The model's inputs, less the ``drop`` groups (keys of :data:`GROUPS`, ``env``, ``mkt``)."""
    dropped = {c for g in drop for c in GROUPS.get(g, ())}
    stock = [c for c in STOCK_COLUMNS if c in data.columns and c not in dropped]
    env = [c for c in data.columns if c.startswith("env_") and "env" not in drop]
    market = [c for c in data.columns if c.startswith("mkt_") and "mkt" not in drop]
    crossed = [f"{c}_x_{m}" for c in INTERACTED for m in STATES
               if f"{c}_x_{m}" in data.columns and c not in dropped]  # fmt: skip
    return {
        "trees": stock + env,
        "trees_regime": stock + env + market,
        "linear_regime": stock + crossed,
    }.get(model, stock)


def _fit(model: str, x: np.ndarray, y: np.ndarray):
    """A fitted estimator for ``model`` (anything with ``predict``)."""
    if model.startswith("linear"):
        return Ridge(alpha=10.0).fit(np.nan_to_num(x), y)
    if model == "pls":
        return PLSRegression(n_components=5, scale=False).fit(np.nan_to_num(x), y)
    return HistGradientBoostingRegressor(**TREE_PARAMS).fit(x, y)


def _predict(fit, model: str, x: np.ndarray) -> np.ndarray:
    x = x if model.startswith("trees") else np.nan_to_num(x)
    return np.ravel(fit.predict(x))


def _monthly_ic(dates: np.ndarray, forecast: np.ndarray, actual: np.ndarray) -> float:
    """Average over months of the rank correlation of forecast and actual."""
    frame = pl.DataFrame({"d": dates, "f": forecast, "a": actual}).drop_nulls().drop_nans()
    ic = frame.group_by("d").agg(pl.corr(pl.col("f").rank(), pl.col("a").rank()).alias("ic"))
    return float(ic["ic"].mean())


def walk_forward(
    data: pl.DataFrame, model: str, drop: tuple[str, ...] = (), importance: list | None = None
) -> pl.DataFrame:
    """Out-of-sample forecasts for every test year (see module docs).

    Args:
        data: From :func:`load`.
        model: One of :data:`MODELS`.
        drop: Input groups left out.
        importance: If a list, each test year's IC drop per shuffled input group is
            appended to it (year, group, drop).

    Returns:
        date, symbol, forecast, actual (next-month return relative to the month's mean).
    """
    out = []
    rng = np.random.default_rng(0)
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
            columns = [
                c for c in _inputs(data, model, drop) if train[c].drop_nans().drop_nulls().len()
            ]
            x_test = test.select(columns).to_numpy()
            fit = _fit(model, train.select(columns).to_numpy(), train[TARGET].to_numpy())
            forecast = _predict(fit, model, x_test)
            if importance is not None:
                importance += _importance(fit, model, test, columns, x_test, forecast, rng)
        log.info("forecast %s %d: %d stock-months", model, year, test.height)
        out.append(test.select("date", "symbol", "actual").with_columns(
            pl.Series("forecast", forecast, dtype=pl.Float64)))  # fmt: skip
    return pl.concat(out)


def _importance(fit, model, test, columns, x_test, forecast, rng) -> list[dict]:
    """IC drop per input group when its columns are shuffled between the test rows."""
    dates, actual = test["date"].to_numpy(), test["actual"].to_numpy()
    base = _monthly_ic(dates, forecast, actual)
    groups = {**GROUPS, "env": tuple(c for c in columns if c.startswith("env_")),
              "mkt": tuple(c for c in columns if c.startswith("mkt_")),
              "regime_terms": tuple(c for c in columns if "_x_" in c)}  # fmt: skip
    year = int(test["date"][0].year)
    rows = []
    for group, members in groups.items():
        idx = [columns.index(c) for c in members if c in columns]
        if not idx:
            continue
        shuffled = x_test.copy()
        order = rng.permutation(len(shuffled))
        shuffled[:, idx] = shuffled[order][:, idx]
        ic = _monthly_ic(dates, _predict(fit, model, shuffled), actual)
        rows.append({"year": year, "group": group, "drop": base - ic})
    return rows


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
