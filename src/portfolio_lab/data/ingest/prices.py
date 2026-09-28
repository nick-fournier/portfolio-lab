"""Daily price ingest: raw bars plus split- and dividend-adjusted returns.

Stored per (symbol, date): the **raw** OHLCV that actually traded (used for price and
liquidity filters) and **adjusted returns** ``ret_cc`` (close to close) and ``ret_co``
(previous close to open), computed from ``adjustment=all`` bars. Adjusted price *levels*
are never stored: a later split rescales all earlier adjusted prices, but ratios between
two prices from the same fetch are unaffected, so stored returns stay valid and ingest
can be append-only.

Incremental runs re-fetch a few overlapping sessions so each new row's return is computed
from two prices in the same response. Symbols never stored before, or silent for longer
than :data:`STALE_SESSIONS`, get their full history instead.
"""

import logging
import random
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import polars as pl

from portfolio_lab.core.calendar import last_complete_session, sessions_back
from portfolio_lab.core.config import HISTORY_START, Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import scan, upsert_parquet, write_status
from portfolio_lab.data.sources.alpaca import fetch_bars
from portfolio_lab.data.validate import validate_price_rows

log = logging.getLogger(__name__)

KEYS = ("symbol", "date")
#: Sessions of overlap re-fetched on incremental runs.
OVERLAP_SESSIONS = 5
#: Symbols whose last stored bar is this many sessions behind the newest get a full refetch.
STALE_SESSIONS = 20
#: Tolerance when verifying stored returns against a fresh fetch.
VERIFY_TOLERANCE = 1e-6
#: Symbols fetched, validated and written per step, bounding memory on a full backfill.
CHUNK_SYMBOLS = 500


def _chunks(symbols: Sequence[str]) -> list[list[str]]:
    """Split ``symbols`` into consecutive lists of at most :data:`CHUNK_SYMBOLS`."""
    return [list(symbols[i : i + CHUNK_SYMBOLS]) for i in range(0, len(symbols), CHUNK_SYMBOLS)]


def adjusted_returns(adjusted: pl.DataFrame) -> pl.DataFrame:
    """Compute close-to-close and close-to-open returns from adjusted bars.

    Args:
        adjusted: Adjusted bars with symbol, date, open and close.

    Returns:
        symbol, date, ``ret_cc`` and ``ret_co``; returns are null on each symbol's first row.
    """
    prev_close = pl.col("close").shift(1).over("symbol")
    return adjusted.sort("symbol", "date").select(
        "symbol",
        "date",
        (pl.col("close") / prev_close - 1).alias("ret_cc"),
        (pl.col("open") / prev_close - 1).alias("ret_co"),
    )


def build_price_rows(raw: pl.DataFrame, adjusted: pl.DataFrame) -> pl.DataFrame:
    """Join raw bars with returns computed from the matching adjusted bars."""
    return raw.join(adjusted_returns(adjusted), on=list(KEYS), how="left").sort(*KEYS)


def fetch_price_rows(
    client: RateLimitedClient, symbols: Sequence[str], start: date, end: date
) -> pl.DataFrame:
    """Fetch raw and adjusted bars for ``symbols`` and combine them into price rows."""
    raw = fetch_bars(client, symbols, start, end, adjustment="raw")
    adjusted = fetch_bars(client, symbols, start, end, adjustment="all")
    return build_price_rows(raw, adjusted)


def stored_extent(dataset: Path) -> pl.DataFrame:
    """Return each stored symbol's first and last date (empty frame if nothing stored)."""
    lf = scan(dataset, "year=*/data.parquet")
    if lf is None:
        return pl.DataFrame(schema={"symbol": pl.String, "first": pl.Date, "last": pl.Date})
    return (
        lf.group_by("symbol")
        .agg(pl.col("date").min().alias("first"), pl.col("date").max().alias("last"))
        .collect()
    )


def write_price_rows(dataset: Path, rows: pl.DataFrame) -> list[int]:
    """Upsert rows into their year partitions; return the years touched."""
    years = []
    for (year,), part in rows.group_by(pl.col("date").dt.year(), maintain_order=True):
        upsert_parquet(part, DataPaths.year_partition(dataset, year), KEYS)
        years.append(year)
    return sorted(years)


def plan_fetches(
    symbols: Sequence[str], extent: pl.DataFrame, full: bool
) -> tuple[list[str], list[str], date | None]:
    """Split symbols into full-history and incremental fetches.

    Returns:
        ``(full_symbols, incremental_symbols, incremental_start)``; the start is ``None``
        when there is nothing incremental to fetch.
    """
    last = dict(zip(extent["symbol"], extent["last"], strict=True))
    known = [s for s in symbols if s in last]
    if full or not known:
        return list(symbols), [], None
    cutoff = sessions_back(max(last[s] for s in known), STALE_SESSIONS)
    recent = [s for s in known if last[s] >= cutoff]
    fresh = [s for s in symbols if s not in last or last[s] < cutoff]
    start = sessions_back(min(last[s] for s in recent), OVERLAP_SESSIONS) if recent else None
    return fresh, recent, start


def update_prices(
    settings: Settings,
    client: RateLimitedClient,
    symbols: Sequence[str],
    dataset: Path,
    job: str,
    full: bool = False,
    end: date | None = None,
) -> dict:
    """Bring a price dataset up to date for ``symbols``.

    Args:
        settings: Application settings.
        client: Alpaca data client.
        symbols: Symbols to maintain.
        dataset: Year-partitioned dataset directory to write.
        job: Status file name (e.g. ``"prices"``).
        full: Re-fetch every symbol's full history instead of incrementally.
        end: Last date to fetch; defaults to the last complete session.

    Returns:
        A summary, also written to ``_status/<job>.json``.
    """
    end = end or last_complete_session()
    extent = stored_extent(dataset)
    fresh, recent, start = plan_fetches(symbols, extent, full)
    last = extent.select("symbol", pl.col("last").alias("_last"))

    chunks = [(c, HISTORY_START) for c in _chunks(fresh)]
    if start is not None:
        chunks += [(c, start) for c in _chunks(recent)]
    rows_written, with_data, max_date, years, issues = 0, set(), None, set(), []
    for n, (chunk, chunk_start) in enumerate(chunks, 1):
        rows = fetch_price_rows(client, chunk, chunk_start, end)
        # On incremental fetches each symbol's first row in the overlap window has no
        # previous price in the response; that row is already stored with a valid return,
        # so it must not be overwritten. Never-stored symbols (null ``_last``) keep all rows.
        already_stored = pl.col("_last").is_not_null() & (pl.col("date") <= pl.col("_last"))
        rows = (
            rows.join(last, on="symbol", how="left")
            .filter(~(pl.col("ret_cc").is_null() & already_stored))
            .drop("_last")
        )
        rows, chunk_issues = validate_price_rows(rows)
        if rows.height:
            years.update(write_price_rows(dataset, rows))
            rows_written += rows.height
            with_data.update(rows["symbol"].unique().to_list())
            max_date = max(filter(None, [max_date, rows["date"].max()]))
        issues += chunk_issues
        for issue in chunk_issues:
            log.warning("%s: %s", job, issue)
        log.info("%s: chunk %d/%d, %d rows", job, n, len(chunks), rows.height)

    summary = {
        "end": end,
        "symbols_requested": len(symbols),
        "symbols_full_history": len(fresh),
        "symbols_incremental": len(recent),
        "rows_written": rows_written,
        "symbols_with_data": len(with_data),
        "max_date": max_date,
        "years_written": sorted(years),
        "issues": issues,
    }
    write_status(settings.data_dir, job, summary)
    log.info("%s: wrote %d rows through %s", job, summary["rows_written"], summary["max_date"])
    return summary


def verify_prices(
    settings: Settings,
    client: RateLimitedClient,
    dataset: Path,
    sample: int = 50,
    seed: int | None = None,
) -> dict:
    """Re-fetch full history for a random sample of symbols and compare stored returns.

    Symbols whose stored returns differ from a fresh fetch (e.g. after a provider
    correction) are repaired by re-writing their full history.

    Args:
        settings: Application settings.
        client: Alpaca data client.
        dataset: Price dataset to check.
        sample: Number of symbols to check.
        seed: Random seed, for reproducible samples.

    Returns:
        A summary, also written to ``_status/verify_prices.json``.
    """
    extent = stored_extent(dataset)
    symbols = random.Random(seed).sample(sorted(extent["symbol"]), min(sample, extent.height))
    end = extent["last"].max()
    fresh = fetch_price_rows(client, symbols, HISTORY_START, end)
    stored = (
        scan(dataset, "year=*/data.parquet")
        .filter(pl.col("symbol").is_in(symbols))
        .select(*KEYS, "ret_cc", "ret_co")
        .collect()
    )
    compared = stored.join(fresh.select(*KEYS, "ret_cc", "ret_co"), on=list(KEYS), suffix="_new")
    diff = pl.max_horizontal(
        (pl.col("ret_cc") - pl.col("ret_cc_new")).abs(),
        (pl.col("ret_co") - pl.col("ret_co_new")).abs(),
    )
    mismatched = sorted(compared.filter(diff > VERIFY_TOLERANCE)["symbol"].unique())
    if mismatched:
        repaired = fresh.filter(pl.col("symbol").is_in(mismatched))
        write_price_rows(dataset, validate_price_rows(repaired)[0])
        log.warning("verify: repaired %d symbols: %s", len(mismatched), mismatched[:10])

    summary = {"checked": len(symbols), "rows_compared": compared.height, "repaired": mismatched}
    write_status(settings.data_dir, "verify_prices", summary)
    return summary
