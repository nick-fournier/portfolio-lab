"""Fund prices for make-vs-buy: exchange-traded funds from Alpaca, mutual funds from Tiingo.

Small (about 20 funds), so the whole history is refreshed on each run.
"""

import logging
from datetime import date

import polars as pl

from portfolio_lab.core.config import HISTORY_START, Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic, write_status
from portfolio_lab.data.sources.alpaca import fetch_bars
from portfolio_lab.data.sources.tiingo import fetch_fund_history
from portfolio_lab.research.funds import FUNDS

log = logging.getLogger(__name__)


def ingest_funds(
    settings: Settings, alpaca: RateLimitedClient, tiingo: RateLimitedClient, today: date
) -> dict:
    """Download every fund's adjusted price history and store daily returns.

    Mutual funds are skipped (with a warning) when ``TIINGO_API_KEY`` is not set.

    Returns:
        A summary, also written to ``_status/funds.json``.
    """
    traded = [f.symbol for f in FUNDS if f.source == "alpaca"]
    bars = fetch_bars(alpaca, traded, HISTORY_START, today, adjustment="all")
    frames = [bars.select("symbol", "date", pl.col("close").alias("adj_close"))]
    mutual = [f.symbol for f in FUNDS if f.source == "tiingo"]
    if settings.tiingo_api_key is None:
        log.warning("funds: TIINGO_API_KEY not set; skipping %d mutual funds", len(mutual))
    else:
        token = settings.tiingo_api_key.get_secret_value()
        frames += [fetch_fund_history(tiingo, s, token, HISTORY_START) for s in mutual]
    previous = pl.col("adj_close").shift(1).over("symbol")
    prices = (
        pl.concat(frames)
        .sort("symbol", "date")
        .with_columns((pl.col("adj_close") / previous - 1).alias("ret"))
    )
    write_parquet_atomic(prices, DataPaths(settings.data_dir).fund_prices)
    summary = {"funds": prices["symbol"].n_unique(), "rows": prices.height,
               "latest": prices["date"].max()}  # fmt: skip
    write_status(settings.data_dir, "funds", summary)
    return summary
