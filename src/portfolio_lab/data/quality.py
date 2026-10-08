"""Data quality: checks on the conformed and derived tables, every night.

Two kinds:

- **Hard invariants** (:func:`invariants`): values that cannot be right (duplicate keys, a
  non-positive close, a high below the low, negative volume, a return of -100% or worse,
  negative assets, a listing ending before it starts). Any one stops the nightly chain:
  the derive step and the paper account refuse to run on such data.
- **Measured checks** (:func:`measured`): each metric judged against its own history, not
  a fixed limit. The latest value is flagged when it moves more than any earlier step did,
  or leaves the range of every earlier value by more than that largest step: the latest
  session's traded bars per source, monthly agreement between sources on the days they overlap,
  and per month the stocks covered, each input's coverage and 1st/50th/99th percentiles,
  and the forecasts' count, mean and spread. Flags are shown, not enforced.

:func:`check` runs both, writes ``_status/quality.json`` and appends the metrics to
``DataPaths.quality`` (one row per metric per run).
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import upsert_parquet, write_status
from portfolio_lab.data import reader, schemas

log = logging.getLogger(__name__)

#: Sessions of daily row counts the latest session is judged against.
COUNT_HISTORY = 250
#: Close prices agree when within this share of each other.
AGREE = 1e-3
#: Inputs whose per-month distribution is tracked (the forecaster's and the health score's).
TRACKED = ("mom_12_1", "mom6m", "cfo_to_assets", "fcf_yield", "sales_yield", "earnings_yield",
           "rd_mve", "roa", "accruals", "leverage", "current_ratio", "share_issuance",
           "d_roa", "d_gross_margin", "market_value")  # fmt: skip


class QualityError(RuntimeError):
    """Raised when a hard invariant fails."""


def _source(root: Path, source: str, table: str) -> pl.LazyFrame:
    return reader._scan(DataPaths(root).conformed(source, table))


def invariants(root: Path) -> list[str]:
    """Every hard invariant broken in any source's conformed tables (module docs)."""
    rules = {
        "prices": [
            ("close <= 0", pl.col("close") <= 0),
            ("high < low", pl.col("high") < pl.col("low") * (1 - 1e-6)),
            ("volume < 0", pl.col("volume") < 0),
            ("return <= -100%", pl.col("ret_cc") <= -1),
        ],
        "filings": [("assets < 0", pl.col("assets") < 0)],
        "listings": [("to before from", pl.col("to") < pl.col("from"))],
    }
    broken = []
    for table, checks in rules.items():
        key = list(schemas.KEYS[table])
        for source in reader.sources(root, table):
            lf = _source(root, source, table)
            counts = lf.select(
                (pl.len() - pl.struct(key).n_unique()).alias("duplicate keys"),
                *[expr.sum().alias(name) for name, expr in checks],
            ).collect(engine="streaming")
            broken += [f"{source}/{table}: {n:,} rows with {name}"
                       for name, n in counts.row(0, named=True).items() if n]  # fmt: skip
    return broken


def _judge(name: str, history: list[float], latest: float) -> dict[str, Any]:
    """``latest`` against ``history`` (oldest first): outside its range, or a record step."""
    past = np.array([v for v in history if v is not None and np.isfinite(v)], dtype=float)
    row: dict[str, Any] = {"metric": name, "value": latest, "flag": None}
    if latest is None or not np.isfinite(latest) or len(past) < 12:
        return row
    lo, hi = float(past.min()), float(past.max())
    largest = float(np.abs(np.diff(past)).max())
    step = abs(latest - past[-1])
    # beyond the range by more than the metric has ever moved in one step (a new record
    # by a normal step is how a trending metric grows)
    if latest < lo - largest or latest > hi + largest:
        row["flag"] = f"outside its history ({lo:.4g} to {hi:.4g}) by more than any step"
    elif step > largest:
        row["flag"] = f"moved {step:.4g}, more than any earlier step ({largest:.4g})"
    return row


def measured(root: Path) -> list[dict[str, Any]]:
    """The measured checks' metrics, each judged against its own history (module docs)."""
    paths = DataPaths(root)
    out = []
    for source in reader.sources(root, "prices"):
        # traded bars only: untraded shells come and go (a day without trades has no bar)
        daily = (
            _source(root, source, "prices").filter(pl.col("volume") > 0)
            .group_by("date").agg(pl.len().alias("n"))
            .sort("date").collect(engine="streaming").tail(COUNT_HISTORY + 1)
        )  # fmt: skip
        out.append(_judge(f"{source}/prices traded on the latest day",
                          daily["n"].to_list()[:-1], float(daily["n"][-1])))  # fmt: skip
    out += _agreement(root)
    if paths.features.exists():
        out += _monthly(pl.read_parquet(paths.features))
    forecasts = paths.forecaster / "nine.parquet"
    if forecasts.exists():
        f = pl.read_parquet(forecasts, columns=["date", "forecast"])
        by_month = f.group_by("date").agg(
            pl.len().alias("count"), pl.col("forecast").mean().alias("mean"),
            pl.col("forecast").std().alias("spread"),
        ).sort("date")  # fmt: skip
        for col in ("count", "mean", "spread"):
            values = by_month[col].cast(pl.Float64).to_list()
            out.append(_judge(f"forecasts {col}", values[:-1], values[-1]))
    return out


def _agreement(root: Path) -> list[dict[str, Any]]:
    """Monthly share of overlapping days where the two price sources' closes agree."""
    sources = reader.sources(root, "prices")
    if len(sources) < 2:
        return []
    a, b = (_source(root, s, "prices").select("sid", "date", "close") for s in sources[:2])
    months = (
        a.join(b, on=["sid", "date"], suffix="_b")
        .group_by(pl.col("date").dt.truncate("1mo").alias("month"))
        .agg(((pl.col("close") / pl.col("close_b") - 1).abs() <= AGREE).mean().alias("agree"))
        .sort("month").collect(engine="streaming")
    )  # fmt: skip
    if months.height < 2:
        return []
    values = months["agree"].to_list()
    return [_judge(f"{sources[0]} vs {sources[1]} closes agreeing", values[:-1], values[-1])]


def _monthly(features: pl.DataFrame) -> list[dict[str, Any]]:
    """Per month: stocks, and each tracked input's coverage and percentiles."""
    cols = [c for c in TRACKED if c in features.columns]
    stats = features.group_by("date").agg(
        pl.len().cast(pl.Float64).alias("stocks"),
        *[pl.col(c).is_not_null().mean().alias(f"{c} coverage") for c in cols],
        *[pl.col(c).quantile(q).alias(f"{c} p{round(q * 100)}")
          for c in cols for q in (0.01, 0.5, 0.99)],
    ).sort("date")  # fmt: skip
    out = []
    for name in stats.columns[1:]:
        values = stats[name].to_list()
        out.append(_judge(f"monthly {name}", values[:-1], values[-1]))
    return out


def check(root: Path, stage: str) -> dict[str, Any]:
    """Run both kinds of checks; record them; raise :class:`QualityError` on a hard failure.

    Args:
        root: The data directory.
        stage: What just ran (``conform`` or ``derive``), recorded with the results.
    """
    now = datetime.now(UTC)
    broken = invariants(root)
    metrics = measured(root)
    flags = [f"{m['metric']}: {m['flag']}" for m in metrics if m["flag"]]
    status = {"stage": stage, "hard": broken, "flags": flags, "metrics": len(metrics)}
    write_status(root, "quality", status)
    history = pl.DataFrame(
        [{"at": now, "stage": stage, "metric": m["metric"], "value": m["value"],
          "flag": m["flag"]} for m in metrics],
        schema={"at": pl.Datetime("us", "UTC"), "stage": pl.String, "metric": pl.String,
                "value": pl.Float64, "flag": pl.String},
    )  # fmt: skip
    if history.height:
        upsert_parquet(history, DataPaths(root).quality, ["at", "metric"])
    for flag in flags:
        log.warning("quality: %s", flag)
    if broken:
        raise QualityError("; ".join(broken))
    return status
