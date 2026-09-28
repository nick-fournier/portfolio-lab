"""Alpaca Market Data v2: historical stock bars.

Daily bars come back with UTC timestamps at midnight New York time (``04:00Z`` or ``05:00Z``
depending on daylight saving), so they are converted to ``America/New_York`` dates, never
truncated in UTC.
"""

import logging
from collections.abc import Iterator, Sequence
from datetime import date
from typing import Literal

import polars as pl

from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient

log = logging.getLogger(__name__)

BARS_PATH = "/v2/stocks/bars"
MAX_SYMBOLS_PER_REQUEST = 200
PAGE_LIMIT = 10_000
NEW_YORK = "America/New_York"

Adjustment = Literal["raw", "all"]

BAR_SCHEMA = {
    "symbol": pl.String,
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "vwap": pl.Float64,
    "trade_count": pl.Int64,
}

# Alpaca's single-letter bar fields -> our column names.
_FIELDS = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vw": "vwap",
    "n": "trade_count",
}


def make_client(settings: Settings) -> RateLimitedClient:
    """Return an authenticated, rate-limited client for the Alpaca data API."""
    return RateLimitedClient(
        base_url=settings.alpaca_data_url,
        headers=settings.alpaca_headers(),
        max_per_minute=settings.alpaca_requests_per_minute,
    )


def _batches(items: Sequence[str], size: int) -> Iterator[list[str]]:
    """Yield consecutive chunks of ``items`` of at most ``size`` elements."""
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def _to_frame(bars: dict[str, list[dict]]) -> pl.DataFrame:
    """Convert Alpaca's ``{symbol: [bar, ...]}`` payload into a typed frame."""
    rows = [
        {"symbol": symbol, "t": bar["t"], **{name: bar.get(key) for key, name in _FIELDS.items()}}
        for symbol, symbol_bars in bars.items()
        for bar in symbol_bars
    ]
    if not rows:
        return pl.DataFrame(schema=BAR_SCHEMA)
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("t")
            .str.to_datetime(time_zone="UTC")
            .dt.convert_time_zone(NEW_YORK)
            .dt.date()
            .alias("date")
        )
        .select([pl.col(name).cast(dtype) for name, dtype in BAR_SCHEMA.items()])
    )


def fetch_bars(
    client: RateLimitedClient,
    symbols: Sequence[str],
    start: date,
    end: date,
    adjustment: Adjustment = "raw",
    timeframe: str = "1Day",
) -> pl.DataFrame:
    """Fetch bars for many symbols, following pagination and batching symbols.

    Args:
        client: Client from :func:`make_client`.
        symbols: Symbols to fetch; batched in groups of 200 per request.
        start: First date (inclusive).
        end: Last date (inclusive).
        adjustment: ``"raw"`` for traded prices, ``"all"`` for split- and
            dividend-adjusted prices.
        timeframe: Alpaca timeframe, e.g. ``"1Day"``.

    Returns:
        One row per (symbol, date), sorted, with columns per :data:`BAR_SCHEMA`.
        Symbols without data in the range are simply absent.
    """
    frames = []
    for batch in _batches(sorted(set(symbols)), MAX_SYMBOLS_PER_REQUEST):
        params = {
            "symbols": ",".join(batch),
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": adjustment,
            "feed": "sip",
            "limit": PAGE_LIMIT,
            "sort": "asc",
        }
        page_token = None
        while True:
            payload = client.get_json(
                BARS_PATH, {**params, **({"page_token": page_token} if page_token else {})}
            )
            frames.append(_to_frame(payload.get("bars") or {}))
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        log.debug("fetched %s bars for %d symbols", adjustment, len(batch))
    if not frames:
        return pl.DataFrame(schema=BAR_SCHEMA)
    return pl.concat(frames).unique(subset=["symbol", "date"], keep="last").sort("symbol", "date")
