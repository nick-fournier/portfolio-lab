"""The reader: every source's copy of a conformed table, as one continuous table.

:func:`read` unions the copies under ``<root>/<source>/conformed/<table>`` and keeps one
row per key (``schemas.KEYS``): the most complete row (most of ``schemas.REQUIRED``
filled), and among equally complete rows the source earliest in :data:`PRIORITY`. The
result is the only view of the data that anything downstream sees.

:func:`write` is how a source stores its copy: whole for small tables, by year partition
for the ones in ``schemas.PARTITIONED``.
"""

import logging
import shutil
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.data import schemas

log = logging.getLogger(__name__)

#: Sources in the order they win ties; a source not listed loses to every listed one.
PRIORITY: tuple[str, ...] = ("sharadar", "alpaca", "edgar", "nasdaq", "tiingo", "fred")


def sources(root: Path, table: str) -> list[str]:
    """Sources under ``root`` that have a copy of ``table``."""
    return sorted(
        p.parent.parent.name
        for p in root.glob(f"*/conformed/{table}")
        if any(p.glob("*.parquet")) or any(p.glob("year=*/*.parquet"))
    )


def _scan(folder: Path) -> pl.LazyFrame:
    files = sorted(folder.glob("year=*/*.parquet")) or sorted(folder.glob("*.parquet"))
    return pl.scan_parquet(files)


def read(
    root: Path,
    table: str,
    columns: Sequence[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    sids: Sequence[int] | None = None,
    priority: Sequence[str] = PRIORITY,
) -> pl.DataFrame:
    """One continuous ``table`` from every source (module docs).

    Args:
        root: The data directory.
        table: One of ``schemas.TABLES``.
        columns: Columns to return (default: all); the key is always included.
        start: Earliest date (on the table's date column) to include.
        end: Latest date to include.
        sids: Securities to include (default: all).
        priority: Source order for ties.

    Returns:
        The rows, sorted by key, in the table's schema.
    """
    paths = DataPaths(root)
    key = list(schemas.KEYS[table])
    rank = {s: i for i, s in enumerate(priority)}
    frames = []
    for source in sources(root, table):
        lf = _scan(paths.conformed(source, table))
        frames.append(lf.with_columns(pl.lit(rank.get(source, len(rank))).alias("_rank")))
    if not frames:
        return pl.DataFrame(schema=schemas.TABLES[table])
    lf = pl.concat(frames, how="diagonal_relaxed")
    when = schemas.PARTITIONED.get(table, "date" if "date" in schemas.TABLES[table] else None)
    if when and start:
        lf = lf.filter(pl.col(when) >= start)
    if when and end:
        lf = lf.filter(pl.col(when) <= end)
    if sids is not None:
        lf = lf.filter(pl.col("sid").is_in(list(sids)))
    required = [pl.col(c).is_not_null() for c in schemas.REQUIRED[table]]
    complete = pl.sum_horizontal(required) if required else pl.lit(0)
    lf = (
        lf.with_columns(complete.alias("_complete"))
        .sort([*key, "_complete", "_rank"], descending=[False] * len(key) + [True, False])
        .unique(subset=key, keep="first", maintain_order=True)
    )
    wanted = list(schemas.TABLES[table])
    if columns is not None:
        wanted = [*key, *(c for c in columns if c not in key)]
    return lf.select(wanted).collect()


def clear(root: Path, source: str, table: str) -> None:
    """Remove ``source``'s copy of ``table``."""
    folder = DataPaths(root).conformed(source, table)
    if folder.exists():
        shutil.rmtree(folder)


def write(root: Path, source: str, table: str, frame: pl.DataFrame, part: str = "data") -> int:
    """Store ``source``'s copy of ``table`` in the table's schema.

    A partitioned table (``schemas.PARTITIONED``) is written as ``year=YYYY/<part>.parquet``,
    so a source too large for memory can be written in several parts; :func:`clear` first
    to replace it. Other tables are one file, replaced.

    Returns:
        Rows written.
    """
    folder = DataPaths(root).conformed(source, table)
    frame = schemas.conform(frame, table).sort(list(schemas.KEYS[table]))
    if table in schemas.PARTITIONED:
        by = schemas.PARTITIONED[table]
        for (year,), rows in frame.group_by(pl.col(by).dt.year(), maintain_order=True):
            write_parquet_atomic(rows, folder / f"year={int(year)}" / f"{part}.parquet")
    else:
        write_parquet_atomic(frame, folder / "data.parquet")
    log.info("%s/%s: %d rows", source, table, frame.height)
    return frame.height
