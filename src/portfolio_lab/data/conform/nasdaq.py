"""The NASDAQ Trader symbol directory as listings, and the source of new ids.

The directory lists what trades today; the master table (``data.ingest.universe``)
remembers when each symbol was first and last seen. A symbol no name covers is a new
listing and gets a new id, so the ids keep growing after Sharadar's history ends.
"""

import logging
from pathlib import Path

import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data import ids as ids_
from portfolio_lab.data import reader
from portfolio_lab.data.conform.alpaca import EXCHANGES

log = logging.getLogger(__name__)

SOURCE = "nasdaq"
#: The directory's exclusion reasons -> our categories (anything else is ``other``).
CATEGORIES = {"common": "common", "etf": "etf", "fund": "cef", "preferred": "preferred",
              "unit": "unit", "warrant": "warrant", "right": "right"}  # fmt: skip


def build(root: Path, store: Path) -> dict:
    """Write listings from an old-layout store's ``universe/symbols.parquet``."""
    paths = DataPaths(root)
    ids = ids_.Ids.load(paths.ids)
    symbols = pl.read_parquet(DataPaths(store).universe_symbols).select(
        pl.col("symbol").str.replace("-", ".").alias("ticker"), "name",
        pl.col("exchange").replace(EXCHANGES), "first_seen", "last_seen",
        pl.col("exclude_reason").cast(pl.String).fill_null("common")
        .replace_strict(CATEGORIES, default="other").alias("category"),
    )  # fmt: skip
    latest = symbols["last_seen"].max()
    found = ids_.lookup(ids, symbols.with_columns(pl.col("last_seen").alias("date")))
    new = found.filter(pl.col("sid").is_null() & pl.col("category").is_in(["common", "etf"]))
    if new.height:
        ids = ids.extend(new.select("ticker", "name", "exchange", "category",
                                    pl.col("first_seen").alias("first")))  # fmt: skip
        ids.save(paths.ids)
        found = ids_.lookup(ids, symbols.with_columns(pl.col("last_seen").alias("date")))
        log.info("nasdaq: %d new ids", new.height)
    # A company's warrant or unit can resolve to the company itself: keep the common stock.
    rows = (
        found.drop_nulls("sid")
        .sort(pl.col("category") != "common")
        .unique(["sid", "first_seen"], keep="first", maintain_order=True)
        .select(
            "sid", pl.col("first_seen").alias("from"),
            pl.when(pl.col("last_seen") < latest).then(pl.col("last_seen")).alias("to"),
            "exchange", "category",
        )
    )  # fmt: skip
    return {"listings": reader.write(root, SOURCE, "listings", rows), "new_ids": new.height}
