"""SEC EDGAR's company facts as conformed filings.

From the bulk ``companyfacts.zip`` (``<root>/edgar/raw``), every company with a CIK in the
ids: the facts for the tags in ``sources.edgar.TAGS`` are extracted once into
``raw/facts/`` (replaced when the zip changes), then ``research.fundamentals``
assembles what each filing reported, trailing twelve months (``period = "ttm"``), and the
company maps to its primary security: the common stock (or ADR) listed longest.
"""

import logging
import multiprocessing
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.data import ids as ids_
from portfolio_lab.data import reader, schemas
from portfolio_lab.data.sources.edgar import FACT_SCHEMA, extract_members, member_name
from portfolio_lab.research.fundamentals import filing_states

log = logging.getLogger(__name__)

SOURCE = "edgar"
#: Worker processes for parsing (orange's four fast cores) and companies per task.
WORKERS, BATCH = 4, 50
#: Share classes a CIK's filings are attached to, in order of preference.
PRIMARY = ("common", "adr")


def primary_sids(securities: pl.DataFrame) -> pl.DataFrame:
    """Cik -> sid of the company's primary security (module docs)."""
    rank = pl.col("category").replace_strict(
        {c: i for i, c in enumerate(PRIMARY)}, default=len(PRIMARY)
    )
    span = pl.col("last").fill_null(pl.date(2999, 12, 31))
    return (
        securities.drop_nulls("cik")
        .with_columns(rank.alias("_rank"), (span - pl.col("first")).alias("_span"))
        .sort("cik", "_rank", "_span", descending=[False, False, True])
        .unique("cik", keep="first", maintain_order=True)
        .select("cik", "sid")
    )


def extract(zip_path: Path, ciks: list[int], folder: Path, workers: int = WORKERS) -> None:
    """Parse every kept fact for ``ciks`` (``sources.edgar.FACT_SCHEMA``) into ``folder``.

    Batches are parsed in parallel and each written as its own parquet part as soon as it
    is done, so memory holds one batch at a time (the whole set is tens of millions of
    facts).
    """
    with zipfile.ZipFile(zip_path) as archive:
        present = set(archive.namelist())
    members = [member_name(c) for c in ciks if member_name(c) in present]
    batches = [members[i : i + BATCH] for i in range(0, len(members), BATCH)]
    log.info("edgar: extracting %d companies in %d batches", len(members), len(batches))
    if folder.exists():
        shutil.rmtree(folder)
    if workers > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            chunks = pool.map(extract_members, [zip_path] * len(batches), batches)
            for k, rows in enumerate(chunks):
                _part(folder, k, rows)
    else:
        for k, batch in enumerate(batches):
            _part(folder, k, extract_members(zip_path, batch))


def _part(folder: Path, k: int, rows: list[tuple]) -> None:
    frame = pl.DataFrame(rows, schema=FACT_SCHEMA, orient="row")
    write_parquet_atomic(frame, folder / f"part-{k:04d}.parquet")
    if k % 20 == 0:
        log.info("edgar: batch %d written", k)


def filings(facts: pl.DataFrame, sids: pl.DataFrame, workers: int = WORKERS) -> pl.DataFrame:
    """Conformed filings (trailing twelve months) from the facts, keyed by primary sid."""
    states = filing_states(facts, workers)
    keep = [c for c in states.columns if c in schemas.FILINGS and c != "sid"]
    reported = [pl.col(c).is_not_null() for c in keep if c in schemas.CONCEPTS]
    return (
        states.join(sids, on="cik")
        .select("sid", *keep)
        .with_columns(pl.lit("ttm").alias("period"))
        .filter(pl.col("period_end").is_not_null())
        # Two filings the same day for the same period (a report and its amendment): keep
        # the more complete, as the reader does between sources.
        .with_columns(pl.sum_horizontal(reported).alias("_complete"))
        .sort("_complete", descending=True)
        .unique(list(schemas.KEYS["filings"]), keep="first", maintain_order=True)
        .drop("_complete")
    )  # fmt: skip


def build(root: Path, zip_path: Path | None = None, workers: int = WORKERS) -> dict:
    """Extract the facts (unless done for this zip) and write the conformed filings.

    Args:
        root: The data directory.
        zip_path: ``companyfacts.zip`` (default: ``<root>/edgar/raw/companyfacts.zip``).
        workers: Parser processes.
    """
    paths = DataPaths(root)
    zip_path = zip_path or paths.raw(SOURCE) / "companyfacts.zip"
    sids = primary_sids(ids_.Ids.load(paths.ids).securities)
    cache = paths.raw(SOURCE) / "facts"
    stamp = f"{zip_path.stat().st_size}-{int(zip_path.stat().st_mtime)}"
    stamp_file = cache.with_suffix(".stamp")
    if not (cache.exists() and stamp_file.exists() and stamp_file.read_text() == stamp):
        extract(zip_path, sids["cik"].to_list(), cache, workers)
        stamp_file.write_text(stamp)
    facts = pl.read_parquet(cache / "*.parquet")
    rows = filings(facts, sids, workers)
    return {"companies": facts["cik"].n_unique(), "facts": facts.height,
            "filings": reader.write(root, SOURCE, "filings", rows)}  # fmt: skip
