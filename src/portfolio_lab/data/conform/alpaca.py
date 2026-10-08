"""Alpaca's daily bars (``data.ingest.prices``' store) as conformed prices.

Alpaca files a security's whole history under its current ticker (DMC carries Del Monte's
bars from when it was FDP), and a dead stock's under its final one, so each symbol is
resolved to a ``sid`` by the name in use on its last bar (``ids.lookup``), with fallbacks
(:func:`_resolve`). Symbols no name
covers, first seen after the ids' history ends, are new listings and get new ids; the
rest are logged and left out.
"""

import logging
from pathlib import Path

import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data import ids as ids_
from portfolio_lab.data import reader

log = logging.getLogger(__name__)

SOURCE = "alpaca"
#: NASDAQ Trader exchange codes -> Sharadar's names, so one convention holds in the ids.
EXCHANGES = {"Q": "NASDAQ", "N": "NYSE", "A": "NYSEMKT", "P": "NYSEARCA", "Z": "BATS"}


def _resolve(ids: ids_.Ids, bars: pl.DataFrame) -> pl.DataFrame:
    """``bars`` with ``sid``: each symbol's name on its last bar, else first, else middle.

    The fallbacks cover dead stocks whose OTC tail outlasts the ids' record and names
    whose recorded range starts a day or two after Alpaca's first bar.
    """
    span = bars.group_by("ticker").agg(pl.col("date").min().alias("first"),
                                       pl.col("date").max().alias("last"))  # fmt: skip
    middle = pl.col("first") + (pl.col("last") - pl.col("first")) / 2
    chosen = span.select("ticker")
    for k, probe in enumerate((pl.col("last"), pl.col("first"), middle)):
        at = ids_.lookup(ids, span.select("ticker", probe.cast(pl.Date).alias("date")))
        chosen = chosen.join(at.select("ticker", pl.col("sid").alias(f"_{k}")), on="ticker")
    chosen = chosen.select("ticker", pl.coalesce("_0", "_1", "_2").alias("sid"))
    return bars.join(chosen, on="ticker", how="left")


def _one_per_day(ids: ids_.Ids, rows: pl.DataFrame) -> pl.DataFrame:
    """One row per security and day, where Alpaca carries a security under two tickers.

    Alpaca keeps a renamed security's whole history under both its old and its new ticker
    (IPOA and SPCE, DHCA and BNAI). On a day both have a bar, the row kept is the one whose
    ticker was the security's name that day (``ids.lookup``), else the ticker trading most
    recently.
    """
    named = ids_.lookup(ids, rows.select("ticker", "date").unique()).rename({"sid": "_named"})
    latest = rows.group_by("ticker").agg(pl.col("date").max().alias("_latest"))
    return (
        rows.join(named, on=["ticker", "date"], how="left")
        .join(latest, on="ticker")
        .with_columns((pl.col("_named") == pl.col("sid")).fill_null(False).alias("_was_named"))
        .sort(["sid", "date", "_was_named", "_latest"], descending=[False, False, True, True])
        .unique(["sid", "date"], keep="first", maintain_order=True)
        .drop("_named", "_latest", "_was_named")
    )


def build(root: Path, store: Path) -> dict:
    """Write Alpaca's conformed prices from the stored bars under ``store``.

    Args:
        root: The data directory (ids are read from and, with new listings, written to it).
        store: A data directory in the old layout (``prices/daily`` and
            ``prices/benchmarks`` year partitions, ``universe/symbols.parquet``).
    """
    paths, old = DataPaths(root), DataPaths(store)
    ids = ids_.Ids.load(paths.ids)
    bars = pl.concat(
        pl.scan_parquet(folder / "year=*" / "data.parquet")
        for folder in (old.prices_daily, old.prices_benchmarks)
        if any(folder.glob("year=*/data.parquet"))
    ).collect()
    bars = bars.with_columns(pl.col("symbol").str.replace("-", ".").alias("ticker"))
    found = _resolve(ids, bars)
    # A symbol no name covers, first seen after the ids' history ends, is a new listing.
    known_until = ids.securities["last"].max()  # the newest delisting: about the data end
    never = found.group_by("ticker").agg(pl.col("sid").null_count() == pl.len(),
                                         pl.col("date").min().alias("first"))  # fmt: skip
    new = never.filter(pl.col("sid") & (pl.col("first") > known_until)).select("ticker", "first")
    if new.height:
        symbols = pl.read_parquet(old.universe_symbols).select(
            pl.col("symbol").str.replace("-", ".").alias("ticker"), "name",
            pl.col("exchange").replace(EXCHANGES),
            pl.when("etf").then(pl.lit("etf")).otherwise(pl.lit("common")).alias("category"),
        )  # fmt: skip
        ids = ids.extend(new.join(symbols, on="ticker", how="left"))
        ids.save(paths.ids)
        found = _resolve(ids, bars)
    dropped = found.filter(pl.col("sid").is_null())
    if dropped.height:
        names = dropped["ticker"].unique()
        log.warning("alpaca: %d rows of %d symbols left out (no id): %s", dropped.height,
                    names.len(), names.head(20).to_list())  # fmt: skip
    rows = _one_per_day(ids, found.drop_nulls("sid"))
    return {"prices": reader.write(root, SOURCE, "prices", rows), "new_ids": new.height,
            "dropped_rows": dropped.height}  # fmt: skip
