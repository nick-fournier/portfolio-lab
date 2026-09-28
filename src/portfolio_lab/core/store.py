"""Parquet storage primitives: atomic writes, idempotent upserts, scans, and job status.

Every write goes to a temporary file in the target directory and is then renamed into
place, so readers (the web app) never see a partially written file. Temporary files are
named ``*.tmp-<pid>`` and never match ``*.parquet`` globs.
"""

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


def _atomic_replace(path: Path, write: Any) -> None:
    """Write via ``write(tmp_path)`` then atomically move the result to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        write(tmp)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` as parquet, replacing any existing file atomically."""
    _atomic_replace(path, lambda tmp: df.write_parquet(tmp, compression="zstd"))


def upsert_parquet(df: pl.DataFrame, path: Path, keys: Sequence[str]) -> int:
    """Merge ``df`` into the parquet file at ``path``, new rows winning on key conflicts.

    The result is sorted by ``keys``, so writing the same data twice produces the same
    file content (idempotent).

    Args:
        df: Rows to insert or replace.
        path: Parquet file to update; created if missing.
        keys: Columns that uniquely identify a row.

    Returns:
        The number of rows in the file after the merge.
    """
    keys = list(keys)
    if path.exists():
        existing = pl.read_parquet(path)
        df = pl.concat([existing, df], how="diagonal_relaxed")
    merged = df.unique(subset=keys, keep="last", maintain_order=True).sort(keys)
    write_parquet_atomic(merged, path)
    return merged.height


def scan(root: Path, pattern: str = "**/*.parquet") -> pl.LazyFrame | None:
    """Lazily scan the parquet files under ``root`` matching ``pattern``.

    Args:
        root: Dataset directory, e.g. ``data_dir / "prices" / "daily"``.
        pattern: Glob relative to ``root``.

    Returns:
        A lazy frame over all matching files, or ``None`` if there are none.
    """
    files = sorted(root.glob(pattern))
    return pl.scan_parquet(files) if files else None


def write_status(data_dir: Path, job: str, status: dict[str, Any]) -> None:
    """Record the outcome of a job run in ``_status/<job>.json``.

    A ``finished_at`` UTC timestamp is added. Dates and other non-JSON values are
    stringified.
    """
    payload = {**status, "finished_at": datetime.now(UTC).isoformat(timespec="seconds")}
    path = data_dir / "_status" / f"{job}.json"
    text = json.dumps(payload, indent=2, default=str)
    _atomic_replace(path, lambda tmp: tmp.write_text(text))


def read_status(data_dir: Path, job: str) -> dict[str, Any] | None:
    """Return the last recorded status for ``job``, or ``None`` if it has never run."""
    path = data_dir / "_status" / f"{job}.json"
    return json.loads(path.read_text()) if path.exists() else None
