"""Delisted-stock ingest: add companies that stopped trading since 2016 (survivorship fix).

Today's symbol directory only lists survivors. This job takes Tiingo's list of every stock
that stopped trading since the history start, keeps the common stocks, and backfills their
exchange-traded prices from Alpaca under the same ticker. Dead stocks never trade again,
so each is fetched once; later runs only add newly dead ones. Companies Alpaca has no
exchange prices for (mostly never-listed OTC names) simply stay out.
"""

import logging
from datetime import date

import polars as pl

from portfolio_lab.core.calendar import last_complete_session, sessions_back
from portfolio_lab.core.config import HISTORY_START, Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic, write_status
from portfolio_lab.data.ingest.prices import stored_extent, update_prices
from portfolio_lab.data.sources.tiingo import dead_stocks, fetch_supported_tickers

log = logging.getLogger(__name__)

#: A ticker still trading within this many sessions is treated as alive.
ALIVE_SESSIONS = 10


def listed_today(paths: DataPaths) -> set[str]:
    """Every symbol in the latest directory snapshot (included or not)."""
    if not paths.universe_symbols.exists():
        return set()
    symbols = pl.read_parquet(paths.universe_symbols)
    latest = symbols["last_seen"].max()
    return set(symbols.filter(pl.col("last_seen") == latest)["symbol"])


def ingest_delisted(
    settings: Settings,
    public: RateLimitedClient,
    alpaca: RateLimitedClient,
    today: date | None = None,
) -> dict:
    """Refresh the delisted-stock table and backfill prices for newly found dead stocks.

    Args:
        settings: Application settings.
        public: Client for Tiingo's public ticker list.
        alpaca: Alpaca data client.
        today: Reference date (default: the last complete session).

    Returns:
        A summary, also written to ``_status/delisted.json``.
    """
    paths = DataPaths(settings.data_dir)
    today = today or last_complete_session()
    alive_after = sessions_back(today, ALIVE_SESSIONS)
    dead = dead_stocks(
        fetch_supported_tickers(public), listed_today(paths), HISTORY_START, alive_after
    )

    stored = set(stored_extent(paths.prices_daily)["symbol"])
    todo = [s for s in dead.filter("included")["symbol"] if s not in stored]
    fetched = {}
    if todo:
        log.info("delisted: backfilling %d dead stocks", len(todo))
        fetched = update_prices(
            settings, alpaca, todo, paths.prices_daily, "delisted_prices", end=today
        )

    with_prices = set(stored_extent(paths.prices_daily)["symbol"])
    table = dead.with_columns(pl.col("symbol").is_in(list(with_prices)).alias("has_prices"))
    write_parquet_atomic(table, paths.universe_delisted)

    usable = table.filter(pl.col("included") & pl.col("has_prices"))
    summary = {
        "dead_candidates": table.height,
        "included": int(table["included"].sum()),
        "fetched_now": len(todo),
        "rows_written": fetched.get("rows_written", 0),
        "with_prices": usable.height,
        "fell_to_otc": int(usable["fell_to_otc"].sum()),
    }
    write_status(settings.data_dir, "delisted", summary)
    log.info(
        "delisted: %d dead stocks with prices (%d fell to OTC)",
        usable.height,
        summary["fell_to_otc"],
    )
    return summary
