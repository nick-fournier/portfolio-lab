"""Tiingo's record of dead stocks as actions.

When each stopped trading (``delisted``), and whether it had already fallen off its
exchange (``otc``); see ``sources.tiingo``.
"""

from pathlib import Path

import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data import ids as ids_
from portfolio_lab.data import reader

SOURCE = "tiingo"


def build(root: Path, store: Path) -> dict:
    """Write actions from an old-layout store's ``universe/delisted.parquet``."""
    paths = DataPaths(root)
    dead = (
        pl.read_parquet(DataPaths(store).universe_delisted)
        .filter("included")
        .select(
            pl.col("symbol").str.replace("-", ".").alias("ticker"), "start", "end", "fell_to_otc"
        )
    )
    ids = ids_.Ids.load(paths.ids)
    # by the name on the last trade, else (an OTC tail the ids don't record) on the first
    at_end = ids_.lookup(ids, dead.with_columns(pl.col("end").alias("date")))
    at_start = ids_.lookup(ids, dead.with_columns(pl.col("start").alias("date")))
    found = at_end.with_columns(pl.coalesce("sid", at_start["sid"]).alias("sid")).drop_nulls("sid")
    rows = pl.concat([
        found.select("sid", "date", pl.lit("delisted").alias("action")),
        found.filter("fell_to_otc").select("sid", "date", pl.lit("otc").alias("action")),
    ])  # fmt: skip
    return {"actions": reader.write(root, SOURCE, "actions", rows)}
