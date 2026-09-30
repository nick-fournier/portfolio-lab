"""Research probes: measurement jobs behind M10-M12 that did not become strategies.

Return forecasts and the learner bake-off, the monthly pick test, the model portfolio,
the trash-risk model, and the learned and extended F-scores. Kept runnable so results can
be reproduced; production jobs live in ``jobs.tasks``.
"""

import logging
from datetime import date
from typing import Any

import polars as pl

from portfolio_lab.backtest.results import list_runs, load_run
from portfolio_lab.core.config import Settings
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.jobs.tasks import backtest_task
from portfolio_lab.research import fscore_model, health, pick_test
from portfolio_lab.research.dataset import build_dataset
from portfolio_lab.research.forecasts import bakeoff, rank_learners
from portfolio_lab.research.forecasts import run as run_forecasts
from portfolio_lab.research.models import prepare
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.scoreboard import forward_returns

log = logging.getLogger(__name__)


def forecasts_task(settings: Settings) -> dict:
    """Walk-forward direction, magnitude and hurdle forecasts at 1-12 months (measurement)."""
    paths = DataPaths(settings.data_dir)
    env = pl.read_parquet(paths.environment) if paths.environment.exists() else None
    data = prepare(
        build_dataset(Panel.load(settings.data_dir), pl.read_parquet(paths.features), env)
    )
    results = run_forecasts(data)
    for name, table in results.items():
        write_parquet_atomic(table, paths.models.parent / "forecasts" / f"{name}.parquet")
    return {"predictions": results["predictions"].height}


def bakeoff_task(settings: Settings) -> dict:
    """Compare learners (trees, MLP, ensemble) on the 1-month return rank (measurement)."""
    paths = DataPaths(settings.data_dir)
    env = pl.read_parquet(paths.environment) if paths.environment.exists() else None
    features = pl.read_parquet(paths.features)
    data = prepare(build_dataset(Panel.load(settings.data_dir), features, env))
    predictions = bakeoff(data)
    table = rank_learners(predictions)
    folder = paths.models.parent / "forecasts"
    write_parquet_atomic(predictions, folder / "bakeoff_predictions.parquet")
    write_parquet_atomic(table, folder / "bakeoff.parquet")
    return {"learners": table.to_dicts()}


#: Model portfolios start the first session after the first out-of-sample forecast (the
#: forecasts start in 2021), fully invested, as do their controls.
MODEL_START = date(2021, 2, 1)
MODEL_BACKTESTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("model_portfolio", {}),  # the 100 best-rated, optimized with a turnover penalty
    ("model_portfolio", {"weighting": "equal"}),  # the same 100, equal weights
    ("equal_weight", {}),  # control: every eligible stock, equal weights
)


def forecast_scores_task(settings: Settings) -> dict:
    """Walk-forward 1-month forecast scores (trees) for the model portfolio."""
    paths = DataPaths(settings.data_dir)
    env = pl.read_parquet(paths.environment) if paths.environment.exists() else None
    features = pl.read_parquet(paths.features)
    data = prepare(build_dataset(Panel.load(settings.data_dir), features, env))
    scores = bakeoff(data, mlp=False).select("date", "symbol", "trees")
    write_parquet_atomic(scores, paths.models.parent / "forecasts" / "scores.parquet")
    return {"scores": scores.height, "from": scores["date"].min(), "to": scores["date"].max()}


def _meanvar_daily(settings: Settings) -> pl.DataFrame | None:
    """Daily returns of the longest default meanvar run (the pick test's benchmark), if any."""
    default = {"model": "ar1_logret", "top_n": 100, "max_weight": 0.1}
    runs = [
        r["meta"] for r in list_runs(settings.data_dir)
        if r["meta"]["strategy"] == "meanvar"
        and all(r["meta"]["params"].get(k) == v for k, v in default.items())
        and r["meta"]["params"].get("selector", "liquidity") == "liquidity"
        and r["meta"]["params"].get("weighting", "optimized") == "optimized"
    ]  # fmt: skip
    if not runs:
        return None
    run_id = min(runs, key=lambda m: str(m["start"]))["run_id"]
    return load_run(settings.data_dir, run_id).daily


#: Model variants compared in the pick test (``forecasts.bakeoff`` options), retrained
#: yearly. Model 1, the fundamentals picker: business inputs only, the return relative to
#: similar-size firms over 3, 6 and 12 months; and the 6-month one with the environment.
PICK_VARIANTS: dict[str, dict[str, Any]] = {
    **{
        f"fund_{h // 21}m": {"target": "size_excess", "horizon": h, "env": False,
                             "fundamentals_only": True}
        for h in (63, 126, 252)
    },
    "fund_6m_env": {"target": "size_excess", "horizon": 126, "fundamentals_only": True},
}  # fmt: skip


def pick_test_task(settings: Settings) -> dict:
    """Each variant's top 20 and top 100 vs the 100 most liquid, SPY and meanvar monthly.

    ``scores_old.parquet`` in the forecasts folder, if present, is included as ``old``
    (scores saved from an earlier model, for before/after comparisons).
    """
    paths = DataPaths(settings.data_dir)
    env = pl.read_parquet(paths.environment) if paths.environment.exists() else None
    panel = Panel.load(settings.data_dir)
    raw = build_dataset(panel, pl.read_parquet(paths.features), env)
    data = prepare(raw)
    folder = paths.models.parent / "forecasts"
    old = folder / "scores_old.parquet"
    scores = {"old": pl.read_parquet(old)} if old.exists() else {}
    for name, options in PICK_VARIANTS.items():
        frame = bakeoff(data, mlp=False, **options).select("date", "symbol", "trees")
        write_parquet_atomic(frame, folder / f"scores_{name}.parquet")
        scores[name] = frame
    result = pick_test.run(scores, raw, panel, _meanvar_daily(settings))
    for name, frame in result.items():
        write_parquet_atomic(frame, folder / f"pick_test_{name}.parquet")
    return {"summary": result["summary"].to_dicts()}


def health_task(settings: Settings) -> dict:
    """Walk-forward trash-risk scores (with and without volatility), judged against F >= 7."""
    paths = DataPaths(settings.data_dir)
    env = pl.read_parquet(paths.environment) if paths.environment.exists() else None
    panel = Panel.load(settings.data_dir)
    raw = build_dataset(panel, pl.read_parquet(paths.features), env)
    data = prepare(raw)
    folder = paths.models.parent / "health"
    scores = {}
    for name, volatility in (("risk", False), ("risk_vol", True)):
        scores[name] = health.walk_forward(data, volatility=volatility)
        write_parquet_atomic(scores[name], folder / f"{name}.parquet")
    result = {}
    for pool in (None, 500, 100):
        table = health.compare(raw, scores, pool)
        write_parquet_atomic(table, folder / f"compare_{pool or 'all'}.parquet")
        result[pool or "all"] = table.to_dicts()
    return result


def fscore_models_task(settings: Settings) -> dict:
    """Continuous and learned F-scores (``research.fscore_model``), saved for backtests."""
    paths = DataPaths(settings.data_dir)
    panel = Panel.load(settings.data_dir)
    raw = build_dataset(panel, pl.read_parquet(paths.features), None)
    spy = panel.symbol_index["SPY"]
    dates = raw["date"].unique().sort().to_list()
    spy_fwd = pl.DataFrame(
        {"date": dates,
         "spy_252": [float(forward_returns(panel, panel.date_index[d], 252)[spy])
                     if panel.date_index[d] + 252 < len(panel.dates) else None for d in dates]}
    )  # fmt: skip
    growth = fscore_model.growth_labels(
        pl.read_parquet(paths.fundamentals_states), pl.read_parquet(paths.fundamentals_tickers)
    )
    data = fscore_model.add_labels(prepare(raw), growth, spy_fwd)
    folder = paths.models.parent / "fscore_models"
    scores = {"continuous": fscore_model.continuous(data)}
    scores["pca"], pca_weights = fscore_model.pca(data)
    write_parquet_atomic(pca_weights, folder / "pca_weights.parquet")
    for name, label, depth in (
        ("learned_growth", "grows", 1), ("learned_spy", "beats_spy", 1),
        ("pairwise_growth", "grows", 2), ("pairwise_spy", "beats_spy", 2),
    ):  # fmt: skip
        scores[name], curves = fscore_model.walk_forward(data, label, depth)
        write_parquet_atomic(curves, folder / f"{name}_curves.parquet")
    summary = {}
    for name, frame in scores.items():
        write_parquet_atomic(frame, folder / f"{name}.parquet")
        summary[name] = {"rows": frame.height, "from": frame["date"].min()}
    return summary


#: Share of stocks each filter keeps in the filter backtests: light, medium, and about the
#: share F-score >= 7 keeps.
FILTER_CUTS = (0.75, 0.5, 0.27)
FILTER_SCORES = (
    "continuous", "pca", "learned_growth", "learned_spy", "pairwise_growth", "pairwise_spy",
)  # fmt: skip


def filter_backtests_task(settings: Settings, only: tuple[str, ...] = ()) -> dict:
    """Meanvar behind each F-score model at each cut, vs meanvar, meanvar + F >= 7 and SPY.

    Every run starts on the first session all scores exist. Each score is installed in turn
    as the panel's predictions (``forecasts/scores.parquet``) for its runs. With ``only``,
    just those scores are run (no reference runs).
    """
    paths = DataPaths(settings.data_dir)
    folder = paths.models.parent / "fscore_models"
    scores = {n: pl.read_parquet(folder / f"{n}.parquet") for n in FILTER_SCORES}
    first = max(f.drop_nulls("score")["date"].min() for f in scores.values())
    start = next(d for d in Panel.load(settings.data_dir).dates if d > first)
    runs = {}
    if not only:
        runs = {
            "spy": backtest_task(settings, "buy_hold", start)[0],
            "meanvar": backtest_task(settings, "meanvar", start)[0],
            "meanvar_f7": backtest_task(settings, "meanvar", start, params={"min_fscore": 7})[0],
        }
    scores = {n: f for n, f in scores.items() if not only or n in only}
    installed = paths.models.parent / "forecasts" / "scores.parquet"
    for name, frame in scores.items():
        write_parquet_atomic(
            frame.select("date", "symbol", pl.col("score").alias("trees")), installed
        )
        for cut in FILTER_CUTS:
            runs[f"{name}_{cut}"] = backtest_task(
                settings, "meanvar", start, params={"healthy_share": cut}
            )[0]
    return {"start": start, "runs": runs}


def extended_scores_task(settings: Settings, cut: float = 0.27) -> dict:
    """Continuous F-score plus each extra health metric, all of them, and PCA on all.

    Each score is computed from the prepared data, then backtested behind meanvar keeping
    the ``cut`` healthiest share, from the same start as ``filter_backtests_task``.
    """
    paths = DataPaths(settings.data_dir)
    extra = fscore_model.HEALTH_EXTRA
    data = prepare(build_dataset(Panel.load(settings.data_dir), pl.read_parquet(paths.features)))
    scores = {
        "continuous": fscore_model.continuous(data),
        **{f"plus_{m}": fscore_model.continuous(data, fscore_model.PIOTROSKI | {m: d})
           for m, d in extra.items()},
        "extended": fscore_model.continuous(data, fscore_model.PIOTROSKI | extra),
    }  # fmt: skip
    scores["pca_extended"], weights = fscore_model.pca(data, fscore_model.PIOTROSKI | extra)
    del data
    folder = paths.models.parent / "fscore_models"
    write_parquet_atomic(weights, folder / "pca_extended_weights.parquet")
    start = date(2003, 2, 3)
    installed = paths.models.parent / "forecasts" / "scores.parquet"
    runs = {}
    for name, frame in scores.items():
        write_parquet_atomic(frame, folder / f"{name}.parquet")
        write_parquet_atomic(
            frame.select("date", "symbol", pl.col("score").alias("trees")), installed
        )
        runs[name] = backtest_task(settings, "meanvar", start, params={"healthy_share": cut})[0]
    return {"start": start, "runs": runs}


def model_backtests_task(settings: Settings) -> dict:
    """Backtest the model portfolios and their controls from :data:`MODEL_START`."""
    return {
        f"{name} {params}": backtest_task(settings, name, MODEL_START, params=params)[0]
        for name, params in MODEL_BACKTESTS
    }
