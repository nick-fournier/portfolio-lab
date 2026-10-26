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
    seen = symbols.with_columns(pl.col("last_seen").alias("date"))
    found = ids_.covering(ids, ids_.lookup(ids, seen))  # names: holders on the last day seen
    # A symbol is a new listing only if no security held its ticker on any day it was seen:
    # the directory lists a stock for a few days past its delisting. One that two securities
    # hold on a day is a listing numbered twice, for ids.merge_duplicates and the guard.
    span = symbols.select("ticker", pl.col("first_seen").alias("date"), "last_seen")
    held = ids_.covering(ids, span, until="last_seen").select(
        "ticker", pl.col("names").alias("held")
    )
    found = found.join(held, on="ticker", how="left")
    unclear = found.filter(pl.col("sid").is_null() & (pl.col("names") > 1))
    if unclear.height:
        log.warning("nasdaq: %d symbols two securities hold, left out: %s",
                    unclear.height, unclear["ticker"].head(20).to_list())  # fmt: skip
    new = found.filter((pl.col("held") == 0) & pl.col("category").is_in(["common", "etf"]))
    if new.height:
        gone = pl.when(pl.col("last_seen") < latest).then(pl.col("last_seen"))
        ids = ids.extend(new.select("ticker", "name", "exchange", "category",
                                    pl.col("first_seen").alias("first"), gone.alias("to"),
                                    gone.alias("last")))  # fmt: skip
        ids.save(paths.ids)
        found = ids_.lookup(ids, seen)
        log.info("nasdaq: %d new ids", new.height)
    # A company's warrant or unit can resolve to the company itself: keep the common stock.
    # The directory has no flag for ADRs or closed-end funds (they read as common), so a
    # security the ids already know keeps the kind recorded there (Sharadar's, for its
    # history); the directory's own kind is used only for securities it introduced.
    kinds = ids.securities.select("sid", pl.col("category").alias("_kind"))
    rows = (
        found.drop_nulls("sid")
        .sort(pl.col("category") != "common")
        .unique(["sid", "first_seen"], keep="first", maintain_order=True)
        .join(kinds, on="sid", how="left")
        .with_columns(pl.coalesce("_kind", "category").alias("category"))
        .select(
            "sid", pl.col("first_seen").alias("from"),
            pl.when(pl.col("last_seen") < latest).then(pl.col("last_seen")).alias("to"),
            "exchange", "category",
        )
    )  # fmt: skip
    return {"listings": reader.write(root, SOURCE, "listings", rows), "new_ids": new.height}
