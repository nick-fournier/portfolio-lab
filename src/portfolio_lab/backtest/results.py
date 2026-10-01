"""Backtest results: the generic run artifacts the dashboard renders.

Each run is an immutable directory ``results/runs/<run_id>/`` holding ``meta.json``,
``metrics.json``, ``daily.parquet`` (NAV and diagnostics per session),
``weights.parquet`` (targets at each rebalance) and ``trades.parquet`` (holdings before and
after each trade and at month ends, for after-tax replays; see ``backtest.tax``).
``_SUCCESS`` is written last; readers only list runs that have it, so a crash mid-write
never shows up as a partial run.
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
        trades: The trade log (``backtest.execution.TradeLog``); absent in older runs.
    """

    meta: dict[str, Any]
    metrics: dict[str, float]
    daily: pl.DataFrame
    weights: pl.DataFrame
    trades: pl.DataFrame | None = None


def runs_dir(data_dir: Path) -> Path:
    """Directory holding all run directories."""
    return data_dir / "results" / "runs"


#: Metadata that defines *what* was tested. Two runs agreeing on these are the same
#: configuration, even if they ran on different days, code versions or data.
CONFIG_KEYS = (
    "strategy",
    "params",
    "schedule",
    "start",
    "benchmark",
    "costs",
    "max_weight",
    "delisting_return",
)
#: Metadata that joins the configuration only when present, so older runs keep their ids.
OPTIONAL_CONFIG_KEYS = ("execution",)


def _canonical(value: Any) -> Any:
    """Normalize values so equal configs hash equally (e.g. ``100000`` and ``100000.0``)."""
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def config_id(meta: dict[str, Any]) -> str:
    """Return a short hash identifying a run's configuration (see :data:`CONFIG_KEYS`)."""
    config = _canonical(
        {k: meta.get(k) for k in CONFIG_KEYS}
        | {k: meta[k] for k in OPTIONAL_CONFIG_KEYS if meta.get(k)}
    )
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]


def make_run_id(meta: dict[str, Any], created: datetime) -> str:
    """Return a sortable, unique id: ``<UTC timestamp>-<strategy>-<config hash>``."""
    return f"{created:%Y%m%dT%H%M%S}-{meta['strategy']}-{config_id(meta)[:6]}"


def save_run(result: RunResult, data_dir: Path) -> str:
    """Write a run directory and return its id.

    Args:
        result: The backtest result.
        data_dir: The data directory root.
    """
    created = datetime.now(UTC)
    base_id = run_id = make_run_id(result.meta, created)
    n = 1
    while (runs_dir(data_dir) / run_id).exists():  # same config saved within the same second
        n += 1
        run_id = f"{base_id}-{n}"
    path = runs_dir(data_dir) / run_id
    path.mkdir(parents=True)
    meta = {
        **result.meta,
        "run_id": run_id,
        "config_id": config_id(result.meta),
        "created_at": created.isoformat(timespec="seconds"),
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (path / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    write_parquet_atomic(result.daily, path / "daily.parquet")
    write_parquet_atomic(result.weights, path / "weights.parquet")
    if result.trades is not None:
        write_parquet_atomic(result.trades, path / "trades.parquet")
    (path / SUCCESS_MARKER).touch()
    return run_id


def list_runs(data_dir: Path, latest_only: bool = False) -> list[dict[str, Any]]:
    """Return ``{"meta": ..., "metrics": ...}`` for complete runs, newest first.

    Args:
        data_dir: The data directory root.
        latest_only: Keep only the newest run of each configuration.
    """
    root = runs_dir(data_dir)
    if not root.exists():
        return []
    runs, seen = [], set()
    for path in sorted(root.iterdir(), reverse=True):
        if not (path / SUCCESS_MARKER).exists():
            continue
        meta = json.loads((path / "meta.json").read_text())
        meta["config_id"] = config_id(meta)  # recomputed, so hashing changes apply to old runs
        if latest_only and meta["config_id"] in seen:
            continue
        seen.add(meta["config_id"])
        runs.append({"meta": meta, "metrics": json.loads((path / "metrics.json").read_text())})
    return runs


def prune_runs(data_dir: Path, keep: int = 3) -> list[str]:
    """Delete all but the newest ``keep`` runs of each configuration; return deleted ids."""
    kept: dict[str, int] = {}
    deleted = []
    for run in list_runs(data_dir):
        cid = run["meta"]["config_id"]
        kept[cid] = kept.get(cid, 0) + 1
        if kept[cid] > keep:
            shutil.rmtree(runs_dir(data_dir) / run["meta"]["run_id"])
            deleted.append(run["meta"]["run_id"])
    return deleted


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
        trades=pl.read_parquet(trades) if (trades := path / "trades.parquet").exists() else None,
    )
