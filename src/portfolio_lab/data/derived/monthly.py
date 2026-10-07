"""The monthly table (every stock input at each month end) and the market environment.

``research.features`` builds the inputs from the hive panel and the fundamentals table;
``research.context`` builds the environment from the FRED series and adds each stock's
factor sensitivities. Both are rebuilt whole for now (the features take a few minutes on
the full history).
"""

import logging
from pathlib import Path

import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.data import reader
from portfolio_lab.data.derived import daily
from portfolio_lab.research.context import environment, sensitivities, tailwinds
from portfolio_lab.research.features import build_features
from portfolio_lab.research.panel import EligibilityRules

log = logging.getLogger(__name__)


def build(root: Path, rules: EligibilityRules | None = None) -> dict:
    """Rebuild the monthly and environment tables (module docs)."""
    panel = daily.panel(root, rules=rules)
    paths = DataPaths(root)
    states = pl.read_parquet(paths.fundamentals).with_columns(
        pl.col("sid").cast(pl.String).alias("symbol")
    )
    scores = states.filter(pl.col("fscore").is_not_null()).select(
        "symbol", "filed", "fscore", "n_signals"
    )
    industry = panel.ids.securities.select(pl.col("sid").cast(pl.String).alias("symbol"), "sic")
    features = build_features(panel, states.drop("fscore", "n_signals"), industry, scores)
    log.info("monthly: %d stock-months", features.height)
    observations = reader.read(root, "series").filter(pl.col("series") != "DTB3")
    dates = features["date"].unique().sort().to_list()
    env = environment(observations, dates, features)
    write_parquet_atomic(env, paths.environment)
    context = tailwinds(sensitivities(panel, observations, dates), env)
    features = features.join(context, on=["date", "symbol"], how="left")
    write_parquet_atomic(features, paths.features)
    return {"stock_months": features.height, "months": len(dates)}
