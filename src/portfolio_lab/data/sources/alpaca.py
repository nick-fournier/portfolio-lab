"""Alpaca Market Data v2: historical stock bars.

Daily bars come back with UTC timestamps at midnight New York time (``04:00Z`` or ``05:00Z``
depending on daylight saving), so they are converted to ``America/New_York`` dates, never
truncated in UTC. Minute bars keep their New York timestamps (the start of each minute).
"""

import logging
import re
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
import polars as pl

from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient

log = logging.getLogger(__name__)

BARS_PATH = "/v2/stocks/bars"
MAX_SYMBOLS_PER_REQUEST = 200
PAGE_LIMIT = 10_000
NEW_YORK = "America/New_York"
#: The free plan can't query the most recent 15 minutes of SIP data; keep a margin.
SIP_DELAY = timedelta(minutes=16)

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


MINUTE_SCHEMA = {"symbol": pl.String, "ts": pl.Datetime("us", NEW_YORK)} | {
    name: dtype for name, dtype in BAR_SCHEMA.items() if name not in ("symbol", "date")
}

# Raw payload rows: symbol, Alpaca's timestamp string, then bar fields as final types.
_ROW_SCHEMA = {"symbol": pl.String, "t": pl.String} | {
    name: BAR_SCHEMA[name] for name in _FIELDS.values()
}


def make_client(settings: Settings) -> RateLimitedClient:
    """Return an authenticated, rate-limited client for the Alpaca data API."""
    return RateLimitedClient(
        base_url=settings.alpaca_data_url,
        headers=settings.alpaca_headers(),
        max_per_minute=settings.alpaca_requests_per_minute,
    )


def end_param(end: date, now: datetime | None = None) -> str:
    """Return the ``end`` query value: end of ``end`` in New York, capped at now minus the delay.

    A bare date is read by Alpaca as reaching the end of that day, which for today falls
    inside the free plan's delayed window and is rejected with HTTP 403.
    """
    end_of_day = datetime.combine(end, time(23, 59, 59), ZoneInfo(NEW_YORK))
    latest = (now or datetime.now(UTC)) - SIP_DELAY
    return (
        min(end_of_day, latest).astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _batches(items: Sequence[str], size: int) -> Iterator[list[str]]:
    """Yield consecutive chunks of ``items`` of at most ``size`` elements."""
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def _to_frame(bars: dict[str, list[dict]], intraday: bool = False) -> pl.DataFrame:
    """Convert Alpaca's ``{symbol: [bar, ...]}`` payload into a typed frame.

    Daily frames follow :data:`BAR_SCHEMA`; intraday ones :data:`MINUTE_SCHEMA`.
    """
    schema = MINUTE_SCHEMA if intraday else BAR_SCHEMA
    rows = [
        {"symbol": symbol, "t": bar["t"], **{name: bar.get(key) for key, name in _FIELDS.items()}}
        for symbol, symbol_bars in bars.items()
        for bar in symbol_bars
    ]
    if not rows:
        return pl.DataFrame(schema=schema)
    # Explicit schema: inferring from the first rows would type a page whose early prices
    # are whole numbers as integers and silently truncate later ones (0.99 -> 0).
    local = pl.col("t").str.to_datetime(time_zone="UTC").dt.convert_time_zone(NEW_YORK)
    frame = pl.DataFrame(rows, schema=_ROW_SCHEMA).with_columns(
        (local if intraday else local.dt.date()).alias("ts" if intraday else "date")
    )
    return frame.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])


_INVALID_SYMBOL = re.compile(r"invalid symbol: ([A-Za-z0-9.$/^=+-]+)")


def _fetch_batch(
    client: RateLimitedClient, batch: list[str], params: dict, intraday: bool = False
) -> list[pl.DataFrame]:
    """Fetch every page for one batch of symbols.

    Alpaca rejects a whole request with HTTP 400 if any symbol is malformed. The offending
    symbol, when the error names it, is dropped and the rest retried; otherwise the batch is
    split in half until the bad symbol is isolated and skipped.
    """
    batch = list(batch)
    while batch:
        frames, page_token = [], None
        try:
            while True:
                query = {**params, "symbols": ",".join(batch)}
                if page_token:
                    query["page_token"] = page_token
                payload = client.get_json(BARS_PATH, query)
                frames.append(_to_frame(payload.get("bars") or {}, intraday))
                page_token = payload.get("next_page_token")
                if not page_token:
                    return frames
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400:
                raise
            match = _INVALID_SYMBOL.search(exc.response.text)
            if match and match.group(1) in batch:
                log.warning("alpaca rejected symbol %s; skipping it", match.group(1))
                batch.remove(match.group(1))
            elif len(batch) == 1:
                log.warning("alpaca rejected symbol %s; skipping it", batch[0])
                return []
            else:
                half = len(batch) // 2
                return _fetch_batch(client, batch[:half], params, intraday) + _fetch_batch(
                    client, batch[half:], params, intraday
                )
    return []


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
    params = {
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end_param(end),
        "adjustment": adjustment,
        "feed": "sip",
        "limit": PAGE_LIMIT,
        "sort": "asc",
    }
    frames = []
    for batch in _batches(sorted(set(symbols)), MAX_SYMBOLS_PER_REQUEST):
        frames += _fetch_batch(client, batch, params)
        log.debug("fetched %s bars for %d symbols", adjustment, len(batch))
    if not frames:
        return pl.DataFrame(schema=BAR_SCHEMA)
    return pl.concat(frames).unique(subset=["symbol", "date"], keep="last").sort("symbol", "date")


def fetch_minute_bars(client: RateLimitedClient, symbols: Sequence[str], day: date) -> pl.DataFrame:
    """Raw one-minute bars for one session's regular hours (9:30 to 16:00 New York).

    Args:
        client: Client from :func:`make_client`.
        symbols: Symbols to fetch; batched in groups of 200 per request.
        day: The session.

    Returns:
        Rows per :data:`MINUTE_SCHEMA`, sorted by symbol and time. A minute without trades
        has no bar.
    """
    zone = ZoneInfo(NEW_YORK)
    params = {
        "timeframe": "1Min",
        "start": datetime.combine(day, time(9, 30), zone).isoformat(),
        "end": datetime.combine(day, time(15, 59), zone).isoformat(),
        "adjustment": "raw",
        "feed": "sip",
        "limit": PAGE_LIMIT,
        "sort": "asc",
    }
    frames = []
    for batch in _batches(sorted(set(symbols)), MAX_SYMBOLS_PER_REQUEST):
        frames += _fetch_batch(client, batch, params, intraday=True)
    if not frames:
        return pl.DataFrame(schema=MINUTE_SCHEMA)
    return pl.concat(frames).unique(subset=["symbol", "ts"], keep="last").sort("symbol", "ts")
