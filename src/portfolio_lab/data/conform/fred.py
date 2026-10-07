"""FRED's series (the T-bill rate and the market-context series) as the ``series`` table.

The T-bill rate (``DTB3``) counts as known on its own date; the context series carry the
date each observation was published (``sources.fred``).
"""

from pathlib import Path

import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data import reader

SOURCE = "fred"
RATE = "DTB3"


def build(root: Path, store: Path) -> dict:
    """Write the series from an old-layout store's ``rates`` and ``macro`` tables."""
    old = DataPaths(store)
    frames = []
    if old.rates.exists():
        frames.append(pl.read_parquet(old.rates).select(
            pl.lit(RATE).alias("series"), "date", pl.col("date").alias("available"),
            pl.col("rate").alias("value")))  # fmt: skip
    if old.macro.exists():
        frames.append(pl.read_parquet(old.macro).select("series", "date", "available", "value"))
    rows = pl.concat(frames) if frames else pl.DataFrame()
    return {"series": reader.write(root, SOURCE, "series", rows)}
