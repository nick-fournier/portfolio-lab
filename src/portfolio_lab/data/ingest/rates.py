"""Risk-free rate ingest: refresh the full FRED DTB3 history (small; rewritten each run).

When FRED is down, the days since the stored history's end come from Treasury's daily bill
rates instead; the next successful FRED run rewrites them with DTB3.
"""

import logging
from datetime import date

import httpx
import polars as pl

from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic, write_status
from portfolio_lab.data.sources.fred import FredUnavailableError, fetch_dtb3
from portfolio_lab.data.sources.treasury import fetch_bill_rates

log = logging.getLogger(__name__)


def extend_from_treasury(
    stored: pl.DataFrame, client: RateLimitedClient, today: date
) -> pl.DataFrame:
    """``stored`` plus Treasury's bill rates for the days after its last date."""
    last = stored["date"].max()
    years = range(last.year, today.year + 1)
    fresh = pl.concat([fetch_bill_rates(client, y) for y in years]).filter(pl.col("date") > last)
    return pl.concat([stored, fresh.cast(stored.schema)]).sort("date")


def ingest_rates(settings: Settings, client: RateLimitedClient, today: date | None = None) -> dict:
    """Download DTB3 and replace the stored rate history, or extend it from Treasury.

    Args:
        settings: Settings; the history lives under ``data_dir``.
        client: HTTP client for FRED and Treasury.
        today: Today's date, for the Treasury years to fetch; defaults to today.

    Returns:
        A summary, also written to ``_status/rates.json``; ``source`` says which answered.

    Raises:
        FredUnavailableError: If FRED is down and there is no stored history to extend.
        httpx.HTTPError: If FRED and Treasury both fail.
    """
    path = DataPaths(settings.data_dir).rates
    try:
        rates, source = fetch_dtb3(client), "fred"
    except (FredUnavailableError, httpx.HTTPError) as exc:
        if not path.exists():
            raise
        log.warning("rates: %s; extending the stored history from Treasury", exc)
        stored = pl.read_parquet(path)
        rates, source = extend_from_treasury(stored, client, today or date.today()), "treasury"
    write_parquet_atomic(rates, path)
    summary = {
        "rows": rates.height,
        "max_date": rates["date"].max(),
        "latest_rate": rates["rate"][-1],
        "source": source,
    }
    write_status(settings.data_dir, "rates", summary)
    log.info("rates: %d rows through %s (%s)", summary["rows"], summary["max_date"], source)
    return summary
