"""Risk-free rate ingest: refresh the full FRED DTB3 history (small; rewritten each run)."""

import logging

from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic, write_status
from portfolio_lab.data.sources.fred import fetch_dtb3

log = logging.getLogger(__name__)


def ingest_rates(settings: Settings, client: RateLimitedClient) -> dict:
    """Download DTB3 and replace the stored rate history.

    Returns:
        A summary, also written to ``_status/rates.json``.
    """
    rates = fetch_dtb3(client)
    write_parquet_atomic(rates, DataPaths(settings.data_dir).rates)
    summary = {
        "rows": rates.height,
        "max_date": rates["date"].max(),
        "latest_rate": rates["rate"][-1],
    }
    write_status(settings.data_dir, "rates", summary)
    log.info("rates: %d rows through %s", summary["rows"], summary["max_date"])
    return summary
