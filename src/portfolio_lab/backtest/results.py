"""Backtest results: the generic run artifacts the dashboard renders.

Each run is an immutable directory ``results/runs/<run_id>/`` holding ``meta.json``,
``metrics.json``, ``daily.parquet`` (NAV and diagnostics per session) and
``weights.parquet`` (targets at each rebalance). ``_SUCCESS`` is written last; readers only
list runs that have it, so a crash mid-write never shows up as a partial run.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from portfolio_lab.core.store import write_parquet_atomic

SUCCESS_MARKER = "_SUCCESS"


@dataclass
class RunResult:
    """Everything a backtest produced.

    Args:
        meta: Description of the run (strategy, params, dates, costs, caveats, ...).
        metrics: Summary statistics (see ``backtest.metrics.compute``).
        daily: One row per session: date, nav, ret, turnover, cost, cash, holdings,
            benchmark_ret.
        weights: One row per (rebalance date, symbol) target weight.
    """

    meta: dict[str, Any]
    metrics: dict[str, float]
    daily: pl.DataFrame
    weights: pl.DataFrame


def runs_dir(data_dir: Path) -> Path:
    """Directory holding all run directories."""
    return data_dir / "results" / "runs"


def make_run_id(meta: dict[str, Any], created: datetime) -> str:
    """Return a sortable, unique id: ``<UTC timestamp>-<strategy>-<params hash>``."""
    digest = hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()
    return f"{created:%Y%m%dT%H%M%S}-{meta['strategy']}-{digest[:6]}"


def save_run(result: RunResult, data_dir: Path) -> str:
    """Write a run directory and return its id.

    Args:
        result: The backtest result.
        data_dir: The data directory root.
    """
    created = datetime.now(UTC)
    run_id = make_run_id(result.meta, created)
    path = runs_dir(data_dir) / run_id
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    meta = {**result.meta, "run_id": run_id, "created_at": created.isoformat(timespec="seconds")}
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (path / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    write_parquet_atomic(result.daily, path / "daily.parquet")
    write_parquet_atomic(result.weights, path / "weights.parquet")
    (path / SUCCESS_MARKER).touch()
    return run_id


def list_runs(data_dir: Path) -> list[dict[str, Any]]:
    """Return ``{"meta": ..., "metrics": ...}`` for every complete run, newest first."""
    root = runs_dir(data_dir)
    if not root.exists():
        return []
    runs = []
    for path in sorted(root.iterdir(), reverse=True):
        if (path / SUCCESS_MARKER).exists():
            runs.append(
                {
                    "meta": json.loads((path / "meta.json").read_text()),
                    "metrics": json.loads((path / "metrics.json").read_text()),
                }
            )
    return runs


def load_run(data_dir: Path, run_id: str) -> RunResult:
    """Load a complete run by id.

    Raises:
        FileNotFoundError: If the run does not exist or is incomplete.
    """
    path = runs_dir(data_dir) / run_id
    if not (path / SUCCESS_MARKER).exists():
        raise FileNotFoundError(f"no complete run {run_id!r}")
    return RunResult(
        meta=json.loads((path / "meta.json").read_text()),
        metrics=json.loads((path / "metrics.json").read_text()),
        daily=pl.read_parquet(path / "daily.parquet"),
        weights=pl.read_parquet(path / "weights.parquet"),
    )
