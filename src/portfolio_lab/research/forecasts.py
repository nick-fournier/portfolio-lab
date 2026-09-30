"""Forecasts with direction and magnitude, across horizons (measurement only).

Every stock-month is scored from what was known then, walk-forward (trained each January on
earlier months whose labels ended before the year, as in ``research.models``), with
gradient-boosted trees on all eligible stocks (traits as monthly percentiles plus the
market environment). Per horizon (1, 3, 6 and 12 months):

- ``direction``: probability of beating the median stock.
- ``single``: the stock's predicted return rank that month (one model for "how good").
- ``size``: the predicted size of its move relative to the median (sign ignored).
- ``hurdle``: expected edge = (2 x direction - 1) x size, direction then magnitude.
- Baselines: ``trailing_12m`` (meanvar's forecast: the past year's return) and
  ``momentum`` (the past year excluding the last month).

:func:`evaluate` measures each score: rank IC, the hit rates and average excess return of
the top and bottom tenth, and for ``size`` how well it matches the realized size.
"""

from datetime import date
from itertools import pairwise

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from portfolio_lab.research.dataset import HORIZONS, STOCK_COLUMNS, month_weights
from portfolio_lab.research.features import FUNDAMENTAL
from portfolio_lab.research.models import MIN_TRAIN_MONTHS, training_rows

_TREE = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 1000,
         "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}  # fmt: skip
#: Model name -> (target column prefix, is classifier).
TARGETS = {"direction": ("y", True), "single": ("rank", False), "size": ("absexcess", False)}
POOLS = {"all": pl.lit(True), "top500": pl.col("top500")}


def inputs(data: pl.DataFrame) -> list[str]:
    """Model inputs: stock traits (as percentiles) and the environment."""
    stock = [c for c in STOCK_COLUMNS if c in data.columns]
    return [*stock, *[c for c in data.columns if c.startswith("env_")]]


def _fit(classify: bool, x: np.ndarray, y: np.ndarray, w: np.ndarray):
    tree = (
        HistGradientBoostingClassifier(**_TREE)
        if classify
        else HistGradientBoostingRegressor(**_TREE)
    )
    return tree.fit(x, y, sample_weight=w)


def walk_forward(data: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Out-of-sample direction, single and size predictions for one horizon.

    Returns:
        date, symbol, top500, horizon, direction, single, size, excess (realized return
        minus the median), fwd (realized return).
    """
    data = data.with_columns(pl.col(f"excess_{horizon}").abs().alias(f"absexcess_{horizon}"))
    labeled = data.filter(pl.col(f"y_{horizon}").is_not_null())
    columns = inputs(data)
    months = sorted(labeled["date"].unique())
    if len(months) <= MIN_TRAIN_MONTHS:
        return pl.DataFrame()
    out = []
    for year in range(months[MIN_TRAIN_MONTHS].year + 1, max(data["date"]).year + 1):
        test = data.filter(pl.col("date").dt.year() == year)
        train = (
            training_rows(labeled, year, horizon, test["session"].min()) if test.height else test
        )
        if test.is_empty() or train["date"].n_unique() < MIN_TRAIN_MONTHS:
            continue
        x_train = train.select(columns).to_numpy().astype(float)
        x_test = test.select(columns).to_numpy().astype(float)
        weights = month_weights(train["date"])
        weights = weights / weights.mean()
        scores = {}
        for name, (prefix, classify) in TARGETS.items():
            model = _fit(classify, x_train, train[f"{prefix}_{horizon}"].to_numpy(), weights)
            scores[name] = model.predict_proba(x_test)[:, 1] if classify else model.predict(x_test)
        out.append(
            test.select(
                "date", "symbol", "top500",
                pl.col(f"excess_{horizon}").alias("excess"), pl.col(f"fwd_{horizon}").alias("fwd"),
            ).with_columns(pl.lit(horizon).alias("horizon"),
                           *[pl.Series(k, v) for k, v in scores.items()])
        )  # fmt: skip
    return pl.concat(out) if out else pl.DataFrame()


def add_scores(predictions: pl.DataFrame, data: pl.DataFrame) -> pl.DataFrame:
    """Add the hurdle score and the two price baselines (from the raw features)."""
    base = data.select(
        "date", "symbol",
        ((1 + pl.col("mom_12_1")) * (1 + pl.col("ret_1m")) - 1).alias("trailing_12m"),
        pl.col("mom_12_1").alias("momentum"),
    )  # fmt: skip
    return predictions.join(base, on=["date", "symbol"], how="left").with_columns(
        ((2 * pl.col("direction") - 1) * pl.col("size")).alias("hurdle")
    )


SCORES = ("direction", "single", "hurdle", "trailing_12m", "momentum")


def _per_date(rows: pl.DataFrame, score: str) -> pl.DataFrame:
    """Per month: rank IC, top/bottom-tenth hit rates and mean excess returns."""
    ranked = rows.drop_nulls(score).with_columns(
        (pl.col(score).rank("ordinal").over("date") / pl.len().over("date")).alias("q")
    )
    top, bottom = pl.col("q") > 0.9, pl.col("q") <= 0.1
    return (
        ranked.group_by("date")
        .agg(
            pl.corr(pl.col(score).rank(), pl.col("fwd").rank()).alias("ic"),
            (pl.col("excess").filter(top) > 0).mean().alias("top_hit"),
            (pl.col("excess").filter(bottom) < 0).mean().alias("bottom_hit"),
            pl.col("excess").filter(top).mean().alias("top_excess"),
            pl.col("excess").filter(bottom).mean().alias("bottom_excess"),
        )
        .sort("date")
    )


def evaluate(predictions: pl.DataFrame) -> pl.DataFrame:
    """Summary per score, horizon and pool (see module docs).

    t-stats use non-overlapping periods only (every ``horizon / 21``-th month), since
    longer labels overlap from one month to the next. Excess returns are per period, not
    annualized; ``*_excess_yr`` annualizes them.
    """
    labeled = predictions.filter(pl.col("fwd").is_not_null())
    rows = []
    for horizon in sorted(labeled["horizon"].unique()):
        h = labeled.filter(pl.col("horizon") == horizon)
        step = max(1, horizon // 21)
        per_year = 252 / horizon
        for pool, condition in POOLS.items():
            sub = h.filter(condition)
            for score in SCORES:
                d = _per_date(sub, score)
                if d.height < 6:
                    continue
                spaced = d.gather_every(step)["ic"].drop_nulls()
                ic_t = spaced.mean() / spaced.std() * np.sqrt(spaced.len())
                rows.append({
                    "score": score, "horizon": horizon, "pool": pool, "months": d.height,
                    "ic": d["ic"].mean(), "ic_t": ic_t,
                    "top_hit": d["top_hit"].mean(), "bottom_hit": d["bottom_hit"].mean(),
                    "top_excess_yr": d["top_excess"].mean() * per_year,
                    "bottom_excess_yr": d["bottom_excess"].mean() * per_year,
                })  # fmt: skip
            size = (
                sub.drop_nulls("size")
                .group_by("date")
                .agg(pl.corr(pl.col("size").rank(), pl.col("excess").abs().rank()).alias("r"))["r"]
            )
            rows.append({"score": "size (vs realized size)", "horizon": horizon, "pool": pool,
                         "months": size.len(), "ic": size.mean(), "ic_t": None})  # fmt: skip
    return pl.DataFrame(rows).sort("horizon", "pool", "ic", descending=[False, False, True])


def run(data: pl.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> dict[str, pl.DataFrame]:
    """Walk-forward every horizon; return ``predictions`` and ``summary``."""
    frames = [walk_forward(data, h) for h in horizons]
    predictions = add_scores(pl.concat([f for f in frames if f.height]), data)
    return {"predictions": predictions, "summary": evaluate(predictions)}


# --- Model bake-off: which learner predicts the 1-month return rank best? ---------------

BAKEOFF_HORIZON = 21


def _mlp_matrix(frame: pl.DataFrame, stock: list[str], env: list[str], stats: dict) -> np.ndarray:
    """MLP inputs: trait percentiles (missing at 0.5), missing flags, standardized env."""
    traits = frame.select(stock).to_numpy().astype(float)
    flags = np.isnan(traits).astype(float)
    traits = np.where(np.isnan(traits), 0.5, traits)
    e = frame.select(env).to_numpy().astype(float) if env else np.empty((frame.height, 0))
    e = np.nan_to_num((e - stats["mean"]) / stats["std"])
    return np.hstack([traits, flags, e])


def _fit_mlp(train: pl.DataFrame, stock: list[str], env: list[str], y: np.ndarray, w: np.ndarray):
    """A small, L2-regularized MLP with early stopping; returns (model, env scaling)."""
    from sklearn.neural_network import MLPRegressor  # noqa: PLC0415 - only the bake-off needs it

    e = train.select(env).to_numpy().astype(float) if env else np.empty((train.height, 0))
    stats = {"mean": np.nanmean(e, axis=0), "std": np.nanstd(e, axis=0) + 1e-9}
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3, max_iter=200,
        early_stopping=True, random_state=0,
    )  # fmt: skip
    model.fit(_mlp_matrix(train, stock, env, stats), y, sample_weight=w)
    return model, stats


def _test_periods(data: pl.DataFrame, labeled: pl.DataFrame, retrain: str):
    """(test rows, training cutoff date) per retraining period, from the first testable year.

    ``yearly`` retrains each January; ``monthly`` retrains every month on what is known then.
    """
    months = sorted(labeled["date"].unique())
    first = months[MIN_TRAIN_MONTHS].year + 1
    if retrain == "yearly":
        for year in range(first, max(data["date"]).year + 1):
            yield data.filter(pl.col("date").dt.year() == year), date(year, 1, 1)
    else:
        for day in sorted(d for d in data["date"].unique() if d.year >= first):
            yield data.filter(pl.col("date") == day), day


def _target(train: pl.DataFrame, target: str, horizon: int) -> np.ndarray:
    """Training labels: the return rank, or the size-relative return clipped each month.

    Clipping at the month's 1st and 99th percentiles keeps a few extreme small-cap moves
    from dominating a squared-error fit while keeping the right tail's order.
    """
    if target == "rank":
        return train[f"rank_{horizon}"].to_numpy()
    col = pl.col(f"size_excess_{horizon}")
    return (
        train.select(col.clip(col.quantile(0.01).over("date"), col.quantile(0.99).over("date")))
        .to_series()
        .to_numpy()
    )


def bakeoff(
    data: pl.DataFrame,
    horizon: int = BAKEOFF_HORIZON,
    mlp: bool = True,
    retrain: str = "yearly",
    target: str = "rank",
    env: bool = True,
    fundamentals_only: bool = False,
) -> pl.DataFrame:
    """Walk-forward predictions of the return rank by trees, an MLP and their average.

    With ``mlp=False`` only the trees are trained (what the portfolio uses). ``retrain`` is
    ``yearly`` (each January) or ``monthly``; either way a model learns only from months
    before its test period whose labels ended before the period starts. ``target`` is
    ``rank`` (the return's percentile among all stocks) or ``size_excess`` (the return
    relative to the stock's size tenth, see :func:`_target`); ``env=False`` drops the market
    environment inputs; ``fundamentals_only`` keeps only business inputs (no price-based
    features or market sensitivities).

    Returns:
        date, symbol, top500, excess, fwd, and one score column per learner (``trees``,
        ``mlp``, ``ensemble``: the average of the two within-month percentiles).
    """
    labeled = data.filter(pl.col(f"rank_{horizon}").is_not_null())
    allowed = FUNDAMENTAL if fundamentals_only else STOCK_COLUMNS
    stock = [c for c in allowed if c in data.columns]
    env = [c for c in data.columns if c.startswith("env_")] if env else []
    out = []
    for test, cutoff in _test_periods(data, labeled, retrain):
        if test.is_empty():
            continue
        train = labeled.filter(
            (pl.col("date") < cutoff) & (pl.col(f"label_end_{horizon}") < test["session"].min())
        )
        if train["date"].n_unique() < MIN_TRAIN_MONTHS:
            continue
        y = _target(train, target, horizon)
        w = month_weights(train["date"])
        w = w / w.mean()
        x_train = train.select([*stock, *env]).to_numpy().astype(float)
        seen = np.isfinite(x_train).any(axis=0)  # inputs with no history yet are left out
        trees = _fit(False, x_train[:, seen], y, w)
        net, stats = _fit_mlp(train, stock, env, y, w) if mlp else (None, None)
        x_test = test.select([*stock, *env]).to_numpy().astype(float)[:, seen]
        out.append(
            test.select(
                "date", "symbol", "top500",
                pl.col(f"excess_{horizon}").alias("excess"), pl.col(f"fwd_{horizon}").alias("fwd"),
            ).with_columns(
                pl.Series("trees", trees.predict(x_test)),
                *([pl.Series("mlp", net.predict(_mlp_matrix(test, stock, env, stats)))]
                  if mlp else []),
            )
        )  # fmt: skip
    frame = pl.concat(out)
    if not mlp:
        return frame
    pct = {c: pl.col(c).rank().over("date") / pl.len().over("date") for c in ("trees", "mlp")}
    return frame.with_columns(((pct["trees"] + pct["mlp"]) / 2).alias("ensemble"))


def rank_learners(predictions: pl.DataFrame, scores=("trees", "mlp", "ensemble")) -> pl.DataFrame:
    """Rank learners by the Sharpe ratio of the monthly top-minus-bottom-tenth spread.

    Also: mean spread per year, worst calendar year's spread, rank IC, hit rates, and top
    tenth stability (share of each month's top tenth still there the next month).
    """
    rows = []
    labeled = predictions.filter(pl.col("fwd").is_not_null())
    for pool, condition in POOLS.items():
        sub = labeled.filter(condition)
        for score in scores:
            d = _per_date(sub, score).with_columns(
                (pl.col("top_excess") - pl.col("bottom_excess")).alias("spread")
            )
            yearly = d.group_by(pl.col("date").dt.year()).agg(pl.col("spread").sum())["spread"]
            ranked = sub.with_columns(
                (pl.col(score).rank("ordinal").over("date") / pl.len().over("date")).alias("q")
            )
            tops = ranked.filter(pl.col("q") > 0.9).group_by("date").agg(pl.col("symbol"))
            tops = tops.sort("date")["symbol"].to_list()
            stay = [len(set(a) & set(b)) / max(len(a), 1) for a, b in pairwise(tops)]
            rows.append({
                "learner": score, "pool": pool, "months": d.height,
                "spread_sharpe": d["spread"].mean() / d["spread"].std() * np.sqrt(12),
                "spread_yr": d["spread"].mean() * 12, "worst_year": yearly.min(),
                "ic": d["ic"].mean(), "top_hit": d["top_hit"].mean(),
                "bottom_hit": d["bottom_hit"].mean(), "top_stays": float(np.mean(stay)),
            })  # fmt: skip
    return pl.DataFrame(rows).sort("pool", "spread_sharpe", descending=[False, True])
