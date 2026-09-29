"""Relaxed models: probability that a stock beats the median, learned walk-forward.

The ladder, each judged against the ones before it:

- ``fscore`` and ``cfo_to_assets``: one trait turned into a probability (a one-input
  logistic regression), the baselines to beat.
- ``logistic``: logistic regression on every stock trait as a monthly percentile, with
  missing values at the middle plus flags for missing groups. Additive: each trait pushes
  the probability up or down on its own.
- ``gbm``: gradient-boosted trees (shallow, strongly regularized) on the same percentiles,
  missing values left missing, plus the market environment, so it can learn thresholds and
  combinations such as "profitability matters more when the yield curve is steep".

Walk-forward: every January a model is trained on all earlier months whose labels ended
before the year starts (so no training label overlaps the test period) and predicts every
month of that year. Every month gets equal weight in training.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from portfolio_lab.research.dataset import (
    STOCK_COLUMNS,
    missing_flags,
    month_weights,
    rank_features,
)
from portfolio_lab.research.evaluation import calibrate, calibration_bins, importance, summary

#: At least this many months of training before a year is predicted.
MIN_TRAIN_MONTHS = 36
FLAGS = ("no_fundamentals", "no_fscore", "no_context")


@dataclass(frozen=True)
class ModelSpec:
    """A model on the ladder: its inputs and how to fit it."""

    name: str
    inputs: Callable[[pl.DataFrame], list[str]]
    fit: Callable[[np.ndarray, np.ndarray, np.ndarray], object]
    fill_missing: bool


def _stock(data: pl.DataFrame) -> list[str]:
    return [c for c in STOCK_COLUMNS if c in data.columns]


def _fit_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=0.05, max_iter=1000).fit(x, y, sample_weight=w)


def _fit_gbm(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=1000,
        l2_regularization=1.0, early_stopping=False, random_state=0,
    )  # fmt: skip
    return model.fit(x, y, sample_weight=w)


MODELS: dict[str, ModelSpec] = {
    "fscore": ModelSpec("fscore", lambda d: ["fscore"], _fit_logistic, True),
    "cfo_to_assets": ModelSpec("cfo_to_assets", lambda d: ["cfo_to_assets"], _fit_logistic, True),
    "logistic": ModelSpec(
        "logistic", lambda d: [*_stock(d), *[f for f in FLAGS if f in d.columns]],
        _fit_logistic, True,
    ),
    "gbm": ModelSpec(
        "gbm", lambda d: [*_stock(d), *[c for c in d.columns if c.startswith("env_")]],
        _fit_gbm, False,
    ),
}  # fmt: skip


def prepare(data: pl.DataFrame) -> pl.DataFrame:
    """Stock traits as monthly percentiles plus missing-group flags (environment untouched)."""
    return rank_features(missing_flags(data), _stock(data))


def _matrix(frame: pl.DataFrame, columns: list[str], fill: bool) -> np.ndarray:
    x = frame.select(columns).to_numpy().astype(float)
    return np.where(np.isnan(x), 0.5, x) if fill else x


def training_rows(labeled: pl.DataFrame, year: int, horizon: int, test_start: int) -> pl.DataFrame:
    """Rows a model predicting ``year`` may learn from.

    Only months before the year whose labels ended before its first session (index
    ``test_start``), so no training label overlaps the period being predicted.
    """
    return labeled.filter(
        (pl.col("date") < date(year, 1, 1)) & (pl.col(f"label_end_{horizon}") < test_start)
    )


def walk_forward(
    data: pl.DataFrame, spec: ModelSpec, horizon: int, first_year: int | None = None
) -> tuple[pl.DataFrame, dict[int, object]]:
    """Out-of-sample predictions for every test year.

    Args:
        data: Output of :func:`prepare` (with labels from ``dataset.build_dataset``).
        spec: The model.
        horizon: Label horizon in sessions (21 or 63).
        first_year: First year to predict (default: the first with enough history).

    Returns:
        Predictions (date, symbol, top500, model, horizon, p, y, fwd) and the fitted
        model per test year.
    """
    y_col, fwd_col = f"y_{horizon}", f"fwd_{horizon}"
    labeled = data.filter(pl.col(y_col).is_not_null())
    columns = spec.inputs(data)
    months = sorted(labeled["date"].unique())
    first = first_year or (months[MIN_TRAIN_MONTHS].year + 1)
    last_year = max(data["date"]).year
    predictions, fitted = [], {}
    for year in range(first, last_year + 1):
        test = data.filter(pl.col("date").dt.year() == year)
        if test.is_empty():
            continue
        train = training_rows(labeled, year, horizon, test["session"].min())
        if train["date"].n_unique() < MIN_TRAIN_MONTHS:
            continue
        weights = month_weights(train["date"])
        model = spec.fit(
            _matrix(train, columns, spec.fill_missing),
            train[y_col].to_numpy(),
            weights / weights.mean(),  # mean 1, so regularization strength is comparable
        )
        p = model.predict_proba(_matrix(test, columns, spec.fill_missing))[:, 1]
        fitted[year] = model
        labels = test.select(
            "date", "symbol", "top500", pl.col(y_col).alias("y"), pl.col(fwd_col).alias("fwd")
        )
        predictions.append(
            labels.with_columns(
                pl.lit(spec.name).alias("model"),
                pl.lit(horizon).alias("horizon"),
                pl.Series("p", p),
            )
        )
    return (pl.concat(predictions) if predictions else pl.DataFrame()), fitted


def _importance_groups(columns: list[str]) -> dict[str, list[str]]:
    """Each stock input alone; environment inputs grouped by series (level, changes, pct)."""
    groups: dict[str, list[str]] = {}
    for c in columns:
        if c.startswith("env_"):
            base = c.removeprefix("env_")
            for suffix in ("_chg3m", "_chg12m", "_pct"):
                base = base.removesuffix(suffix)
            groups.setdefault(f"environment: {base}", []).append(c)
        else:
            groups[c] = [c]
    return groups


def run_all(data: pl.DataFrame, horizons: tuple[int, ...] = (21, 63)) -> dict[str, pl.DataFrame]:
    """Walk-forward every model at every horizon and evaluate it.

    Args:
        data: Output of :func:`prepare`.
        horizons: Label horizons in sessions.

    Returns:
        ``predictions`` (raw and calibrated), ``summary``, ``calibration`` and
        ``importance`` (tree and logistic models) tables.
    """
    raw, imps = [], []
    for horizon in horizons:
        for spec in MODELS.values():
            predictions, fitted = walk_forward(data, spec, horizon)
            if predictions.is_empty():
                continue
            raw.append(predictions)
            if spec.name in ("gbm", "logistic"):
                columns = spec.inputs(data)
                table = importance(fitted, data, columns, horizon, spec.fill_missing,
                                   _importance_groups(columns))  # fmt: skip
                imps.append(table.with_columns(pl.lit(spec.name).alias("model"),
                                               pl.lit(horizon).alias("horizon")))  # fmt: skip
    predictions = pl.concat(raw)
    everything = pl.concat([predictions, calibrate(predictions)], how="diagonal_relaxed")
    return {
        "predictions": everything,
        "summary": summary(everything),
        "calibration": calibration_bins(everything),
        "importance": pl.concat(imps) if imps else pl.DataFrame(),
    }
