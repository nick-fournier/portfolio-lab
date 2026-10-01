"""Growth series and return horizons shared by the Overview and Compare pages.

Two sources, merged by key: the long comparison (``make_vs_buy/history_*``: funds since
launch and our strategies backtested since 1999) and the live data (each configuration's
latest run, and ``make_vs_buy/growth.parquet`` for funds since 2017). For a key in both,
the longer history wins. Everything is cached until one of its files changes, and charts
use weekly points (the last close of each week), which look the same as daily ones at a
fraction of the size.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from portfolio_lab.backtest.results import list_runs, load_run, runs_dir
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.research.funds import CATEGORIES

HORIZONS = {"1Y": 1, "5Y": 5, "10Y": 10, "20Y": 20}
YEAR = 252
_CACHE: dict[str, tuple[tuple, Any]] = {}


@dataclass(frozen=True)
class Series:
    """One growth-of-$1 series."""

    key: str
    name: str
    category: str
    daily: pl.DataFrame  # date, growth
    run_id: str | None = None


def cached(name: str, paths: list[Path], build: Callable[[], Any]) -> Any:
    """``build()``'s result, rebuilt only when one of ``paths`` changes (mtime or presence)."""
    stamp = tuple(p.stat().st_mtime_ns if p.exists() else None for p in paths)
    hit = _CACHE.get(name)
    if hit is None or hit[0] != stamp:
        _CACHE[name] = (stamp, build())
    return _CACHE[name][1]


def sources(data_dir: Path) -> list[Path]:
    """Files whose change invalidates the series."""
    folder = DataPaths(data_dir).make_vs_buy
    names = ("summary.parquet", "growth.parquet", "history_summary.parquet",
             "history_growth.parquet")  # fmt: skip
    return [runs_dir(data_dir), *(folder / n for n in names)]


def _from_growth(summary: pl.DataFrame, growth: pl.DataFrame) -> dict[str, Series]:
    """Series from a comparison's summary (names, categories, starts) and growth table."""
    full = summary.filter(pl.col("period") == "full").unique("key")
    out = {}
    for key, name, category, start in full.select("key", "name", "category", "start").iter_rows():
        rows = growth.filter((pl.col("key") == key) & (pl.col("date") >= start)).sort("date")
        if rows.height:
            rebased = rows.select("date", (pl.col("growth") / rows["growth"][0]).alias("growth"))
            out[key] = Series(key, name, category, rebased)
    return out


def _live_runs(data_dir: Path) -> dict[str, Series]:
    """Each configuration's latest run, keyed like the comparisons (``ours: <label>``)."""
    out = {}
    for run in list_runs(data_dir, latest_only=True):
        meta = run["meta"]
        daily = load_run(data_dir, meta["run_id"]).daily.select(
            "date", pl.col("nav").alias("growth")
        )
        if meta["strategy"] == "buy_hold":  # a fund held outright: keyed by its symbol
            symbol = meta.get("params", {}).get("symbol") or "SPY"
            out[symbol] = Series(symbol, symbol, "passive", daily, meta["run_id"])
            continue
        label = meta.get("label") or meta["strategy"]
        out[f"ours: {label}"] = Series(f"ours: {label}", label, "ours", daily, meta["run_id"])
    return out


def load(data_dir: Path) -> dict[str, Series]:
    """All series, the longest history winning for each key (see module docs)."""

    def build() -> dict[str, Series]:
        folder = DataPaths(data_dir).make_vs_buy
        merged: dict[str, Series] = {}
        for summary, growth in (("summary", "growth"), ("history_summary", "history_growth")):
            s, g = folder / f"{summary}.parquet", folder / f"{growth}.parquet"
            if s.exists() and g.exists():
                merged |= _from_growth(pl.read_parquet(s), pl.read_parquet(g))
        for key, live in _live_runs(data_dir).items():
            old = merged.get(key)
            if old is None or old.daily["date"].min() >= live.daily["date"].min():
                merged[key] = live
            else:  # keep the long history, but link to the live run's page
                merged[key] = Series(old.key, old.name, old.category, old.daily, live.run_id)
        return merged

    return cached(f"series:{data_dir}", sources(data_dir), build)


def weekly(daily: pl.DataFrame) -> pl.DataFrame:
    """The first row, then the last row of each calendar week."""
    week = pl.col("date").dt.truncate("1w")
    last = daily.with_columns(week.alias("_w")).group_by("_w", maintain_order=True).last()
    return pl.concat([daily.head(1), last.drop("_w")]).unique("date").sort("date")


def horizons(series: Series) -> dict[str, Any]:
    """Annual return over each horizon ending at the last date, plus full-history stats."""
    days = series.daily["date"].to_numpy().astype("datetime64[D]")
    dates = [days[0].item(), days[-1].item()]
    growth = series.daily["growth"].to_numpy()
    end = dates[-1]
    row: dict[str, Any] = {"key": series.key, "name": series.name, "run_id": series.run_id,
                           "category": series.category, "type": CATEGORIES.get(series.category),
                           "since": dates[0]}  # fmt: skip
    for label, years in HORIZONS.items():
        start = end - timedelta(days=round(365.25 * years))
        if dates[0] > start + timedelta(days=7):
            row[label] = None
            continue
        k = int(np.searchsorted(days, np.datetime64(start)))
        row[label] = float((growth[-1] / growth[k]) ** (1 / years) - 1)
    span_years = max((end - dates[0]).days / 365.25, 1e-9)
    row["Max"] = float(growth[-1] ** (1 / span_years) - 1)
    rets = np.diff(growth) / growth[:-1]
    row["sharpe"] = float(rets.mean() / rets.std() * np.sqrt(YEAR)) if rets.std() > 0 else None
    row["worst_drop"] = float((growth / np.maximum.accumulate(growth) - 1).min())
    return row


def table(series: list[Series]) -> list[dict[str, Any]]:
    """Horizon rows, ours first, then by 10-year return (best first, missing last)."""
    rows = [horizons(s) for s in series if s.daily.height > 1]
    order = {c: k for k, c in enumerate(CATEGORIES)}

    def rank(r: dict) -> tuple:
        return (r["category"] != "ours", -(r["10Y"] if r["10Y"] is not None else -9),
                order.get(r["category"], 99), r["name"])  # fmt: skip

    return sorted(rows, key=rank)


def first_date(series: list[Series]) -> date:
    """The earliest date among ``series``."""
    return min(s.daily["date"].min() for s in series)
