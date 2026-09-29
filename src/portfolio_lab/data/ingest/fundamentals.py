"""Fundamentals ingest: SEC company facts for the universe, from 10-K and 10-Q filings.

Downloads the bulk ``companyfacts.zip`` only when its ETag changes, maps the universe's
tickers to SEC CIK numbers, and extracts their facts in parallel.
Scores are computed from these facts by ``research.piotroski`` (run by ``jobs.tasks``,
keeping this layer free of research code).
"""

import logging
import multiprocessing
import zipfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import read_status, write_parquet_atomic, write_status
from portfolio_lab.data.sources.edgar import (
    FACT_SCHEMA,
    PROFILE_SCHEMA,
    download_bulk,
    extract_members,
    fetch_profiles,
    fetch_ticker_map,
    member_name,
)

log = logging.getLogger(__name__)

#: Worker processes for parsing (orange's four fast cores) and companies per task.
PARSE_WORKERS = 4
BATCH = 50


def ingest_fundamentals(
    settings: Settings, client: RateLimitedClient, symbols: Sequence[str], force: bool = False
) -> dict:
    """Refresh the facts for the companies behind ``symbols``.

    Args:
        settings: Application settings.
        client: HTTP client with the SEC User-Agent (see ``edgar``).
        symbols: Universe symbols whose companies to extract.
        force: Re-parse even if the bulk file has not changed.

    Returns:
        A summary, also written to ``_status/fundamentals.json``.
    """
    paths = DataPaths(settings.data_dir)
    previous = read_status(settings.data_dir, "fundamentals") or {}
    new_etag = download_bulk(client, paths.edgar_bulk, previous.get("etag"))
    tickers = fetch_ticker_map(client)
    write_parquet_atomic(tickers, paths.fundamentals_tickers)

    ciks = sorted(set(tickers.filter(pl.col("symbol").is_in(list(symbols)))["cik"]))
    profiles = update_profiles(client, paths, ciks)

    if new_etag is None and paths.fundamentals_facts.exists() and not force:
        summary = {**previous, "skipped": "bulk file unchanged", "profiles": profiles}
        write_status(settings.data_dir, "fundamentals", summary)
        return summary

    with zipfile.ZipFile(paths.edgar_bulk) as archive:
        present = set(archive.namelist())
    members = [member_name(c) for c in ciks if member_name(c) in present]
    batches = [members[i : i + BATCH] for i in range(0, len(members), BATCH)]
    log.info("fundamentals: extracting %d companies in %d batches", len(members), len(batches))

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(PARSE_WORKERS, mp_context=context) as pool:
        chunks = pool.map(extract_members, [paths.edgar_bulk] * len(batches), batches)
        rows = [row for chunk in chunks for row in chunk]
    facts = pl.DataFrame(rows, schema=FACT_SCHEMA, orient="row")
    write_parquet_atomic(facts, paths.fundamentals_facts)

    summary = {
        "etag": new_etag or previous.get("etag"),
        "universe_symbols": len(symbols),
        "companies_mapped": len(ciks),
        "companies_with_facts": facts["cik"].n_unique(),
        "facts": facts.height,
        "latest_filing": facts["filed"].max(),
        "profiles": profiles,
    }
    write_status(settings.data_dir, "fundamentals", summary)
    log.info(
        "fundamentals: %d facts for %d companies", facts.height, summary["companies_with_facts"]
    )
    return summary


def update_profiles(client: RateLimitedClient, paths: DataPaths, ciks: Sequence[int]) -> int:
    """Fetch SEC profiles (industry codes) for companies not seen before; return the total.

    Profiles rarely change, so each company is fetched once.
    """
    path = paths.fundamentals_companies
    known = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=PROFILE_SCHEMA)
    missing = sorted(set(ciks) - set(known["cik"]))
    if missing:
        log.info("fundamentals: fetching %d company profiles", len(missing))
        known = pl.concat([known, fetch_profiles(client, missing)])
        write_parquet_atomic(known, path)
    return known.height
