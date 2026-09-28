"""Universe ingest: snapshot today's symbol directory and maintain the master symbol table.

Daily snapshots accumulate a point-in-time record of what was listed going forward. The
master table keeps every symbol ever seen, with ``first_seen``/``last_seen`` dates, so
delisted symbols remain identifiable after they drop out of the directory.
"""

import logging
from datetime import date

import polars as pl

from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic, write_status
from portfolio_lab.data.sources.universe import fetch_directory

log = logging.getLogger(__name__)


def merge_symbols(
    existing: pl.DataFrame | None, directory: pl.DataFrame, asof: date
) -> pl.DataFrame:
    """Fold a classified directory snapshot into the master symbol table.

    Symbols present today take today's name, exchange and classification and get
    ``last_seen = asof``. New symbols get ``first_seen = asof``. Symbols missing today keep
    their previous row unchanged.

    Args:
        existing: The current master table, or ``None`` on first run.
        directory: Today's classified directory (from ``fetch_directory``).
        asof: The snapshot date.
    """
    today = directory.with_columns(
        pl.lit(asof).alias("first_seen"), pl.lit(asof).alias("last_seen")
    )
    if existing is None or existing.is_empty():
        return today.sort("symbol")
    first_seen = existing.select("symbol", pl.col("first_seen").alias("_first"))
    updated = (
        today.join(first_seen, on="symbol", how="left")
        .with_columns(pl.coalesce("_first", "first_seen").alias("first_seen"))
        .drop("_first")
    )
    gone = existing.join(today.select("symbol"), on="symbol", how="anti")
    return pl.concat([updated, gone], how="diagonal_relaxed").sort("symbol")


def current_symbols(paths: DataPaths) -> list[str]:
    """Return the included symbols listed in the latest snapshot."""
    if not paths.universe_symbols.exists():
        return []
    symbols = pl.read_parquet(paths.universe_symbols)
    latest = symbols["last_seen"].max()
    return symbols.filter(pl.col("included") & (pl.col("last_seen") == latest))["symbol"].to_list()


def ingest_universe(settings: Settings, client: RateLimitedClient, asof: date) -> dict:
    """Fetch the symbol directory, write today's snapshot and update the master table.

    Args:
        settings: Application settings (for the data directory).
        client: HTTP client for NASDAQ Trader.
        asof: Date to stamp the snapshot with (normally today).

    Returns:
        A summary, also written to ``_status/universe.json``.
    """
    paths = DataPaths(settings.data_dir)
    directory = fetch_directory(client)
    snapshot = paths.universe_snapshots / f"date={asof.isoformat()}" / "data.parquet"
    write_parquet_atomic(directory, snapshot)

    existing = pl.read_parquet(paths.universe_symbols) if paths.universe_symbols.exists() else None
    master = merge_symbols(existing, directory, asof)
    write_parquet_atomic(master, paths.universe_symbols)

    summary = {
        "asof": asof,
        "listed": directory.height,
        "included": int(directory["included"].sum()),
        "excluded_by_reason": dict(
            directory.drop_nulls("exclude_reason").group_by("exclude_reason").len().iter_rows()
        ),
        "symbols_ever_seen": master.height,
    }
    write_status(settings.data_dir, "universe", summary)
    log.info("universe: %d listed, %d included", summary["listed"], summary["included"])
    return summary
