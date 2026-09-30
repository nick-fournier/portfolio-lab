"""Less prescribed F-scores: the Piotroski idea with smooth or learned thresholds.

The F-score is a sum of nine step functions (each metric above a fixed threshold scores 1,
all weighted equally). Two relaxations, both used to **eliminate** weak companies before
meanvar, never to pick winners:

- :func:`continuous`: each step replaced by the stock's percentile on that metric that
  month, averaged with Piotroski's directions. Nothing is learned.
- :func:`walk_forward`: boosted trees with one split per tree (``depth=1``), which makes
  the score a sum of learned step functions, one curve per metric, i.e. the F-score's form
  with learned thresholds and weights; ``depth=2`` allows pairwise interactions. Each
  metric's effect is constrained to Piotroski's direction where one is known. Two labels:

  - ``growth``: the business grows: revenue and operating cash flow both higher in the
    filing for the same fiscal period a year later (known on that filing's date).
  - ``beat_spy``: the stock's return over the next 12 months beats SPY's.

Trained each January on earlier months whose labels were known, like ``research.forecasts``.
"""

from datetime import date

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import partial_dependence

from portfolio_lab.research.dataset import month_weights
from portfolio_lab.research.piotroski import MIN_METRICS, PIOTROSKI

#: Further business inputs for the learned score (0 = no direction imposed).
EXTRA = {
    "gross_profitability": 1, "operating_margin": 1, "leverage": -1, "current_ratio": 1,
    "sales_growth": 1, "asset_growth": 0, "earnings_yield": 1, "cf_yield": 1,
    "inst_breadth_1q": 1, "log_size": 0,
}  # fmt: skip
MIN_TRAIN_MONTHS = 36
_TREE = {"learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 1000,
         "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}  # fmt: skip


#: Further health metrics with strong outside evidence, for the extended scores.
HEALTH_EXTRA = {"gross_profitability": 1, "asset_growth": -1, "leverage": -1,
                "inst_breadth_1q": 1}  # fmt: skip


def growth_labels(states: pl.DataFrame, tickers: pl.DataFrame) -> pl.DataFrame:
    """Per filing: did revenue and operating cash flow grow by the same period a year later?

    Returns:
        symbol, filed, grows (1/0), known (the later filing's date).
    """
    base = states.select("cik", "filed", "period_end", "revenue", "cfo")
    later = base.select(
        "cik", (pl.col("period_end") - pl.duration(days=365)).alias("_match"),
        pl.col("filed").alias("known"), pl.col("revenue").alias("revenue_next"),
        pl.col("cfo").alias("cfo_next"),
    ).sort("_match")  # fmt: skip
    joined = base.sort("period_end").join_asof(
        later, left_on="period_end", right_on="_match", by="cik", strategy="nearest",
        tolerance="20d", check_sortedness=False,
    )  # fmt: skip
    grows = (pl.col("revenue_next") > pl.col("revenue")) & (pl.col("cfo_next") > pl.col("cfo"))
    return (
        joined.filter(pl.col("known").is_not_null() & (pl.col("known") > pl.col("filed")))
        .join(tickers, on="cik")
        .select("symbol", "filed", grows.cast(pl.Int8).alias("grows"), "known")
        .sort("symbol", "filed")
    )


def add_labels(data: pl.DataFrame, growth: pl.DataFrame, spy_fwd: pl.DataFrame) -> pl.DataFrame:
    """Attach ``grows``/``grows_known`` (latest filing before each date) and ``beats_spy``.

    Args:
        data: Modeling rows with date, symbol, ``fwd_252`` and ``label_end_252``.
        growth: From :func:`growth_labels`.
        spy_fwd: date, spy_252 (SPY's return over the next 252 sessions).
    """
    right = growth.with_columns((pl.col("filed") + pl.duration(days=1)).alias("_visible"))
    joined = data.sort("date").join_asof(
        right.sort("_visible").drop("filed"), left_on="date", right_on="_visible", by="symbol",
        strategy="backward", check_sortedness=False,
    ).drop("_visible").rename({"known": "grows_known"})  # fmt: skip
    beats = (pl.col("fwd_252") > pl.col("spy_252")).cast(pl.Int8)
    return joined.join(spy_fwd, on="date", how="left").with_columns(
        pl.when(pl.col("fwd_252").is_not_null()).then(beats).alias("beats_spy")
    )


def _known_before(label: str, test_start: date, test_session: int) -> pl.Expr:
    if label == "grows":
        return pl.col("grows").is_not_null() & (pl.col("grows_known") < test_start)
    return pl.col("beats_spy").is_not_null() & (pl.col("label_end_252") < test_session)


def walk_forward(
    data: pl.DataFrame, label: str, depth: int = 1
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Learned F-score: out-of-sample probability of the good outcome, and learned curves.

    Args:
        data: Prepared data with labels (:func:`add_labels`).
        label: ``grows`` or ``beats_spy``.
        depth: 1 = additive (one learned curve per metric); 2 = pairwise interactions.

    Returns:
        (date, symbol, score) and the last model's curves (metric, percentile, effect):
        how the score moves as one metric goes from worst to best, others held as they are.
    """
    inputs = {c: d for c, d in (PIOTROSKI | EXTRA).items() if c in data.columns}
    columns = list(inputs)
    first = min(data["date"]).year + MIN_TRAIN_MONTHS // 12 + 1
    out, model, sample = [], None, None
    for year in range(first, max(data["date"]).year + 1):
        test = data.filter(pl.col("date").dt.year() == year)
        start = date(year, 1, 1)
        train = data.filter(
            (pl.col("date") < start) & _known_before(label, start, test["session"].min())
        )
        if train["date"].n_unique() < MIN_TRAIN_MONTHS:
            continue
        x = train.select(columns).to_numpy().astype(float)
        seen = np.isfinite(x).any(axis=0)  # inputs with no history yet are left out
        used = [c for c, keep in zip(columns, seen, strict=True) if keep]
        w = month_weights(train["date"])
        model = HistGradientBoostingClassifier(
            max_depth=depth, monotonic_cst=[inputs[c] for c in used], **_TREE
        )
        model.fit(x[:, seen], train[label].to_numpy(), sample_weight=w / w.mean())
        prob = model.predict_proba(test.select(used).to_numpy().astype(float))[:, 1]
        out.append(test.select("date", "symbol").with_columns(pl.Series("score", prob)))
        rows = np.random.default_rng(0).choice(len(x), min(len(x), 5000), replace=False)
        sample = x[rows][:, seen]
    curves = []
    if model is not None:
        grid = np.linspace(0.02, 0.98, 25)
        for k, name in enumerate(used):
            pd = partial_dependence(
                model, sample, [k], kind="average", method="brute", custom_values={k: grid}
            )
            curves += [{"metric": name, "percentile": float(g), "effect": float(v)}
                       for g, v in zip(grid, pd["average"][0], strict=True)]  # fmt: skip
    return pl.concat(out), pl.DataFrame(curves)


#: Years of past months used to estimate the principal-component weights.
PCA_YEARS = 5


def pca(
    data: pl.DataFrame, metrics: dict[str, int] = PIOTROSKI
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Label-free health score: the first principal component of the metrics.

    Each January the weights are estimated from the previous :data:`PCA_YEARS` years of
    stock-months: the metrics (default: Piotroski's nine) as percentiles, flipped where
    lower is better, missing ones at the median, then standardized. The first component
    (the mix of metrics along which firms differ most) gives the weights, signed so that
    higher ROA scores higher.
    No returns are used.

    Returns:
        (date, symbol, score) and the weights per year (year, component, metric, weight)
        for the first two components.
    """
    directions = {c: d for c, d in metrics.items() if c in data.columns}
    oriented = data.select(
        "date", "symbol",
        pl.sum_horizontal(pl.col(c).is_not_null() for c in directions).alias("_n"),
        *[(pl.col(c) if d > 0 else 1 - pl.col(c)).fill_null(0.5).alias(c)
          for c, d in directions.items()],
    ).filter(pl.col("_n") >= MIN_METRICS)  # fmt: skip
    metrics = list(directions)
    out, weights = [], []
    first = min(data["date"]).year + PCA_YEARS
    for year in range(first, max(data["date"]).year + 1):
        past = oriented.filter(pl.col("date").dt.year().is_between(year - PCA_YEARS, year - 1))
        test = oriented.filter(pl.col("date").dt.year() == year)
        if past.is_empty() or test.is_empty():
            continue
        x = past.select(metrics).to_numpy()
        mean, sd = x.mean(axis=0), x.std(axis=0)
        varies = sd > 0  # a metric with no data yet (all at the median) is left out
        eigval, eigvec = np.linalg.eigh(np.corrcoef(x[:, varies], rowvar=False))
        order = np.argsort(eigval)[::-1]
        roa = [m for m, v in zip(metrics, varies, strict=True) if v].index("roa")
        for rank, k in enumerate(order[:2], start=1):
            w = np.zeros(len(metrics))
            w[varies] = eigvec[:, k] * (1 if eigvec[roa, k] >= 0 else -1)
            weights += [{"year": year, "component": rank, "metric": m, "weight": float(v),
                         "explained": float(eigval[k] / eigval.sum())}
                        for m, v in zip(metrics, w, strict=True)]  # fmt: skip
            if rank == 1:
                z = (test.select(metrics).to_numpy() - mean) / np.where(varies, sd, 1.0)
                out.append(test.select("date", "symbol").with_columns(pl.Series("score", z @ w)))
    return pl.concat(out), pl.DataFrame(weights)
