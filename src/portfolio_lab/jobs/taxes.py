"""Backtests for the tax calculator: production at several levels of stickiness, plus SPY.

Run on the Sharadar history (1999+). Each variant is production with a different
:class:`~portfolio_lab.backtest.execution.Execution`; a stored run of the same
configuration that has a trade log is reused. Publishing writes, to the main data
directory's ``results/taxes/``:

- ``index.json``: one entry per variant (label, settings, turnover);
- ``<key>.trades.parquet``: the trade log with symbols replaced by anonymous ids;
- ``<key>.nav.parquet``: the daily pre-tax growth of $1.

These are derived backtest results with no prices or tickers, which the Sharadar license
lets us keep and use.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from portfolio_lab.backtest.engine import BacktestConfig, label, run
from portfolio_lab.backtest.execution import Execution
from portfolio_lab.backtest.results import list_runs, load_run, runs_dir, save_run
from portfolio_lab.backtest.tax import CASH
from portfolio_lab.core.config import Settings
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.jobs.tasks import HISTORY_COMPARE_START, PRODUCTION, _attach_runtime
from portfolio_lab.research.panel import Panel
from portfolio_lab.strategies.base import create

log = logging.getLogger(__name__)

#: Weight changes smaller than these are skipped (0 trades every change).
BANDS = (0.0, 0.01, 0.02, 0.05)
#: The variants: every band, with and without deferring short-term gains.
VARIANTS = tuple(Execution(band, defer) for defer in (False, True) for band in BANDS)
#: The benchmark, held throughout.
BENCHMARK = ("buy_hold", {})


def variant_key(execution: Execution) -> str:
    """File-safe key, e.g. ``band2-defer``."""
    return f"band{round(execution.band * 100)}" + ("-defer" if execution.defer_short_gains else "")


def _stored(data_dir: Path, run_label: str) -> str | None:
    """Latest stored run from :data:`HISTORY_COMPARE_START` with this label and a trade log."""
    for stored in list_runs(data_dir):
        meta: dict[str, Any] = stored["meta"]
        has_log = (runs_dir(data_dir) / meta["run_id"] / "trades.parquet").exists()
        start = meta["start"] == str(HISTORY_COMPARE_START)
        if meta.get("label") == run_label and start and has_log:
            return meta["run_id"]
    return None


def tax_runs_task(settings: Settings, publish: Path | None = None) -> dict[str, str]:
    """Run (or reuse) every variant and SPY; optionally publish them (see module docs).

    Returns:
        Run ids by variant key (``spy`` for the benchmark).
    """
    panel = None
    runs: dict[str, str] = {}
    jobs = [(variant_key(e), *PRODUCTION, e) for e in VARIANTS]
    jobs.append(("spy", *BENCHMARK, Execution()))
    for key, name, params, execution in jobs:
        strategy = _attach_runtime(create(name, **params), settings)
        suffix = f" [{execution.describe()}]" if execution.active else ""
        if run_id := _stored(settings.data_dir, label(strategy) + suffix):
            runs[key] = run_id
            continue
        if panel is None:
            panel = Panel.load(settings.data_dir)
        config = BacktestConfig(HISTORY_COMPARE_START, panel.dates[-1], execution=execution)
        result = run(strategy, panel, config)
        runs[key] = save_run(result, settings.data_dir)
        log.info("tax variant %s saved as %s", key, runs[key])
    if publish is not None:
        publish_runs(settings.data_dir, runs, publish)
    return runs


def publish_runs(source: Path, runs: dict[str, str], dest: Path) -> Path:
    """Write the calculator's files for ``runs`` (key -> run id) to ``dest``'s results."""
    folder = DataPaths(dest).taxes
    folder.mkdir(parents=True, exist_ok=True)
    index = []
    for key, run_id in runs.items():
        stored = load_run(source, run_id)
        if stored.trades is None:
            raise ValueError(f"run {run_id} has no trade log")
        symbols = stored.trades.filter(pl.col("key") != CASH)["key"].unique(maintain_order=True)
        ids = {s: f"h{k}" for k, s in enumerate(symbols.to_list())} | {CASH: CASH}
        trades = stored.trades.with_columns(pl.col("key").replace_strict(ids))
        write_parquet_atomic(trades, folder / f"{key}.trades.parquet")
        write_parquet_atomic(stored.daily.select("date", "nav"), folder / f"{key}.nav.parquet")
        execution = stored.meta.get("execution") or asdict(Execution())
        index.append({
            "key": key, "run_id": run_id, "strategy": stored.meta["strategy"],
            "label": stored.meta.get("label"), **execution,
            "turnover": stored.metrics.get("turnover_annual"),
        })  # fmt: skip
    (folder / "index.json").write_text(json.dumps(index, indent=2))
    return folder
