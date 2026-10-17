"""Seed averages and Sharpe summaries for the optimizer's walks (moved out of cli.py)."""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl


def sharpe(monthly: np.ndarray) -> float:
    """Annualized Sharpe of monthly returns (NaN under two months)."""
    return float(monthly.mean() / monthly.std() * 12**0.5) if len(monthly) > 1 else float("nan")


def halves(dates: list[date], r: np.ndarray) -> str:
    """Sharpe overall, to 2015 and after, as text."""
    half = next((k for k, x in enumerate(dates) if x >= date(2015, 12, 31)), len(dates))
    return f"{sharpe(r):.2f} (to 2015 {sharpe(r[:half]):.2f}, after {sharpe(r[half:]):.2f})"


def average(runs: list[Path]) -> str:
    """Average the seeds' portfolios and report Sharpe; save the average next to the seeds.

    Each seed's portfolio is first scaled to $1 of gross exposure (longs plus shorts) each
    month, so a seed whose weights grew huge cannot swamp the others. A seed's month is a dud,
    and left out of that month's average, when its training loss or return is not finite or it
    holds nothing. The averaged portfolio (date, symbol, weight; $1 gross per seed averaged) is
    written to ``weights.parquet`` in the seeds' parent folder; its monthly return is the
    average of the good seeds' scaled returns.
    """
    good = (pl.col("loss").is_finite() & pl.col("ret").is_finite()
            & ((pl.col("long") - pl.col("short")) > 0))  # fmt: skip
    logs = [pl.read_parquet(r / "log.parquet")
            .select("date", pl.when(good).then(pl.col("ret") / (pl.col("long") - pl.col("short")))
                    .alias(r.name))
            for r in runs]  # fmt: skip
    joined = logs[0]
    for log in logs[1:]:
        joined = joined.join(log, on="date", how="full", coalesce=True)
    names = [r.name for r in runs]
    duds = int(joined.select(pl.sum_horizontal(pl.col(n).is_null() for n in names)).sum().item())
    joined = joined.filter(pl.any_horizontal(pl.col(n).is_not_null() for n in names)).sort("date")
    each = [sharpe(joined[n].drop_nulls().to_numpy()) for n in names]
    mean = joined.select(pl.mean_horizontal(names)).to_series().to_numpy()
    frames = []
    for r, log in zip(runs, logs, strict=True):
        keep = log.filter(pl.col(r.name).is_not_null()).select("date")
        w = pl.read_parquet(r / "weights.parquet").select("date", "symbol", "weight")
        w = w.join(keep, on="date").with_columns(
            pl.col("weight") / pl.col("weight").abs().sum().over("date")
        )
        frames.append(w.with_columns(pl.lit(r.name).alias("seed")))
    stacked = pl.concat(frames)
    count = stacked.group_by("date").agg(pl.col("seed").n_unique().alias("n"))
    average = (stacked.group_by("date", "symbol").agg(pl.col("weight").sum())
               .join(count, on="date").select("date", "symbol", pl.col("weight") / pl.col("n"))
               .sort("date", "symbol"))  # fmt: skip
    average.write_parquet(runs[0].parent / "weights.parquet")
    return (f"{len(runs)} seeds, each: min {min(each):.2f} median {np.median(each):.2f} "
            f"max {max(each):.2f}; {duds} dud seed-months left out; "
            f"average portfolio: {halves(joined['date'].to_list(), mean)}")  # fmt: skip
