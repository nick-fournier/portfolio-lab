"""Market-context ingest: every FRED series in ``fred.CONTEXT_SERIES`` (small; rewritten)."""

import logging

import polars as pl

from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic, write_status
from portfolio_lab.data.sources.fred import CONTEXT_SERIES, fetch_series

log = logging.getLogger(__name__)


def ingest_macro(settings: Settings, client: RateLimitedClient) -> dict:
    """Download every context series and replace the stored observations.

    Raises:
        RuntimeError: If ``FRED_API_KEY`` is not set.

    Returns:
        A summary, also written to ``_status/macro.json``.
    """
    if settings.fred_api_key is None:
        raise RuntimeError("FRED_API_KEY must be set for market-context series")
    key = settings.fred_api_key.get_secret_value()
    frames = [fetch_series(client, series_id, key) for series_id in CONTEXT_SERIES]
    observations = pl.concat(frames).sort("series", "date")
    write_parquet_atomic(observations, DataPaths(settings.data_dir).macro)
    latest = observations.group_by("series").agg(pl.col("available").max())
    summary = {
        "series": observations["series"].n_unique(),
        "rows": observations.height,
        "stalest_series": latest.sort("available")["series"][0],
        "stalest_available": latest["available"].min(),
    }
    write_status(settings.data_dir, "macro", summary)
    log.info("macro: %d rows for %d series", summary["rows"], summary["series"])
    return summary
