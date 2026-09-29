"""Judging models on directional reliability: calibration, confident calls, stability.

Everything here uses out-of-sample predictions (``research.models.walk_forward``): a row per
stock-month with the predicted probability ``p`` of beating the median and the outcome ``y``.

- :func:`calibrate`: re-maps each year's probabilities with a logistic fit of past years'
  outcomes on past years' predictions (Platt scaling), so a stated 60% is right about 60%
  of the time. Uses only earlier out-of-sample years, so it stays point-in-time.
- :func:`summary`: per model, horizon and pool, ranking accuracy (AUC, monthly rank IC),
  calibration error, the hit rates of confident calls, and of each month's top and bottom
  tenth by probability (what a portfolio would hold or avoid).
- :func:`calibration_bins`: predicted versus realized rates by confidence decile.
- :func:`importance`: how much each input matters (the drop in AUC when it is shuffled).
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

#: A call is "confident" when p is at least this far from 0.5.
CONFIDENT = 0.05
POOLS = {"all": pl.lit(True), "top500": pl.col("top500")}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate(predictions: pl.DataFrame) -> pl.DataFrame:
    """Calibrated copies of each model's predictions (model name suffixed ``+cal``).

    Year Y is re-mapped with a logistic regression of outcomes on the log-odds of all
    labeled predictions from years before Y; the first year has nothing to learn from and
    is dropped.
    """
    out = []
    for (model, _), group in predictions.group_by("model", "horizon"):
        rows = group.with_columns(pl.col("date").dt.year().alias("_year"))
        for year in sorted(rows["_year"].unique())[1:]:
            past = rows.filter((pl.col("_year") < year) & pl.col("y").is_not_null())
            if past["y"].n_unique() < 2:
                continue
            fit = LogisticRegression().fit(_logit(past["p"].to_numpy()), past["y"].to_numpy())
            now = rows.filter(pl.col("_year") == year)
            p = fit.predict_proba(_logit(now["p"].to_numpy()))[:, 1]
            out.append(now.with_columns(pl.Series("p", p), pl.lit(f"{model}+cal").alias("model")))
    return pl.concat(out).drop("_year") if out else pl.DataFrame()


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: average gap between stated and realized rates."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    gaps = [
        abs(p[index == b].mean() - y[index == b].mean()) * (index == b).mean()
        for b in range(bins)
        if (index == b).any()
    ]
    return float(sum(gaps))


def _monthly_ic(rows: pl.DataFrame) -> np.ndarray:
    """Rank correlation of p with the realized forward return, per month."""
    per = rows.group_by("date").agg(pl.corr(pl.col("p").rank(), pl.col("fwd").rank()).alias("ic"))
    return per["ic"].drop_nulls().drop_nans().to_numpy()


def _share_hit(y: np.ndarray, mask: np.ndarray, invert: bool = False) -> float | None:
    """How often calls in ``mask`` were right (``invert``: right means it lagged)."""
    if not mask.any():
        return None
    rate = float(y[mask].mean())
    return 1 - rate if invert else rate


def _pool_summary(sub: pl.DataFrame) -> dict:
    """Reliability statistics for one model, horizon and pool (see :func:`summary`)."""
    p, y = sub["p"].to_numpy(), sub["y"].to_numpy()
    ic = _monthly_ic(sub)
    up, down = p >= 0.5 + CONFIDENT, p <= 0.5 - CONFIDENT
    monthly_up = (
        sub.filter(pl.col("p") >= 0.5 + CONFIDENT)
        .group_by("date")
        .agg(pl.col("y").mean().alias("hit"))["hit"]
    )
    tenth = sub.with_columns(
        (pl.col("p").rank("ordinal").over("date") / pl.len().over("date")).alias("q")
    )
    top, bottom = tenth.filter(pl.col("q") > 0.9), tenth.filter(pl.col("q") <= 0.1)
    top_monthly = top.group_by("date").agg(pl.col("y").mean().alias("hit"))["hit"]
    years = [
        roc_auc_score(g["y"], g["p"])
        for _, g in sub.group_by(pl.col("date").dt.year())
        if g["y"].n_unique() == 2
    ]
    return {
        "months": sub["date"].n_unique(),
        "auc": roc_auc_score(y, p),
        "ic": float(ic.mean()),
        "ic_t": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))),
        "brier": brier_score_loss(y, p),
        "ece": _ece(p, y),
        "up_share": float(up.mean()),
        "up_hit": _share_hit(y, up),
        "down_share": float(down.mean()),
        "down_hit": _share_hit(y, down, invert=True),
        "up_months_ok": float((monthly_up > 0.5).mean()) if len(monthly_up) else None,
        "top10_hit": float(top["y"].mean()),
        "bottom10_hit": float(1 - bottom["y"].mean()),
        "top10_months_ok": float((top_monthly > 0.5).mean()),
        "worst_year_auc": min(years) if years else None,
    }


def summary(predictions: pl.DataFrame) -> pl.DataFrame:
    """Reliability of every model, horizon and pool.

    Columns: model, horizon, pool, months, auc, ic, ic_t, brier, ece; ``up_share`` and
    ``up_hit`` (share of calls with p >= 0.5 + CONFIDENT, and how often they beat the
    median); ``down_share`` and ``down_hit`` (the same for p <= 0.5 - CONFIDENT, right when
    the stock lagged); ``up_months_ok`` (months where confident up-calls hit more than half
    the time); independent of calibration, ``top10_hit`` and ``bottom10_hit`` (how often
    each month's top tenth by p beat the median, and the bottom tenth lagged it) with
    ``top10_months_ok``; and ``worst_year_auc``.
    """
    rows = []
    labeled = predictions.filter(pl.col("y").is_not_null())
    for (model, horizon), group in labeled.group_by("model", "horizon"):
        for pool, condition in POOLS.items():
            sub = group.filter(condition)
            if sub.height >= 100 and sub["y"].n_unique() == 2:
                rows.append({"model": model, "horizon": horizon, "pool": pool,
                             **_pool_summary(sub)})  # fmt: skip
    return pl.DataFrame(rows).sort("horizon", "pool", "auc", descending=[False, False, True])


def calibration_bins(predictions: pl.DataFrame, bins: int = 10) -> pl.DataFrame:
    """Mean predicted versus realized rate per confidence decile (all stocks)."""
    labeled = predictions.filter(pl.col("y").is_not_null())
    group = ("model", "horizon")
    rank = pl.col("p").rank("ordinal").over(group) - 1
    decile = (rank * bins // pl.len().over(group)).cast(pl.Int32).alias("bin")
    return (
        labeled.with_columns(decile)
        .group_by("model", "horizon", "bin")
        .agg(
            pl.col("p").mean().alias("predicted"),
            pl.col("y").mean().alias("realized"),
            pl.len().alias("n"),
        )
        .sort("model", "horizon", "bin")
    )


def importance(
    fitted: dict[int, object],
    data: pl.DataFrame,
    columns: list[str],
    horizon: int,
    fill_missing: bool,
    groups: dict[str, list[str]] | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """Drop in out-of-sample AUC when each input (or group of inputs) is shuffled.

    Args:
        fitted: Model per test year (from ``walk_forward``).
        data: The prepared dataset.
        columns: The model's inputs, in training order.
        horizon: Label horizon.
        fill_missing: Whether the model was trained with missing values at 0.5.
        groups: Name -> columns shuffled together (default: each column alone).
        seed: Random seed for the shuffles.

    Returns:
        feature, auc_drop (averaged over test years), sorted largest first.
    """
    rng = np.random.default_rng(seed)
    groups = groups or {c: [c] for c in columns}
    index = {c: k for k, c in enumerate(columns)}
    drops: dict[str, list[float]] = {g: [] for g in groups}
    for year, model in fitted.items():
        in_year = pl.col("date").dt.year() == year
        test = data.filter(in_year & pl.col(f"y_{horizon}").is_not_null())
        if test.height < 100:
            continue
        x = test.select(columns).to_numpy().astype(float)
        if fill_missing:
            x = np.where(np.isnan(x), 0.5, x)
        y = test[f"y_{horizon}"].to_numpy()
        base = roc_auc_score(y, model.predict_proba(x)[:, 1])
        for name, cols in groups.items():
            shuffled = x.copy()
            order = rng.permutation(len(x))
            for c in cols:
                shuffled[:, index[c]] = x[order, index[c]]
            drops[name].append(base - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
    mean_drop = [float(np.mean(v)) if v else 0.0 for v in drops.values()]
    return pl.DataFrame({"feature": list(drops), "auc_drop": mean_drop}).sort(
        "auc_drop", descending=True
    )


def scoreboard_rows(predictions: pl.DataFrame) -> pl.DataFrame:
    """Models' predictions in the scoreboard's format (``research.scoreboard.SCHEMA``).

    Per month and pool: rank IC of p with the forward return, and the average forward return
    of the top and bottom fifth by p. Calibrated copies rank the same way and are skipped.
    """
    rows = predictions.filter(pl.col("fwd").is_not_null() & ~pl.col("model").str.ends_with("+cal"))
    out = []
    for pool, condition in POOLS.items():
        sub = rows.filter(condition).with_columns(
            (pl.col("p").rank("ordinal").over("model", "horizon", "date")
             / pl.len().over("model", "horizon", "date")).alias("q")
        )  # fmt: skip
        out.append(
            sub.group_by("model", "horizon", "date")
            .agg(
                pl.len().alias("n"),
                pl.corr(pl.col("p").rank(), pl.col("fwd").rank()).alias("ic"),
                pl.col("fwd").filter(pl.col("q") > 0.8).mean().alias("top"),
                pl.col("fwd").filter(pl.col("q") <= 0.2).mean().alias("bottom"),
            )
            .with_columns(("model: " + pl.col("model")).alias("signal"), pl.lit(pool).alias("pool"))
        )
    return pl.concat(out).select(
        "signal", "pool", pl.col("horizon").cast(pl.Int64), "date",
        pl.col("n").cast(pl.Int64), "ic", "top", "bottom",
    ).sort("signal", "pool", "date")  # fmt: skip
