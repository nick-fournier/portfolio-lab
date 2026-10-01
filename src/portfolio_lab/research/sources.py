"""Live data (Alpaca prices + SEC fundamentals) vs Sharadar: where production's gap comes from.

Production earns less on the live data than on Sharadar over the same years. Three
studies split the gap:

1. **Swap test.** Production runs on all four mixes of prices (with the stock pool they
   imply) and fundamentals (the monthly feature panel). The data directories are mixed
   with symlinks, so nothing is copied.
2. **Health inputs.** For the same stock and month, how often each source has each of the
   nine Piotroski inputs, how closely they agree (rank correlation across stocks) and how
   much the healthiest-100 lists overlap.
3. **Timing.** Days since the latest filing, per stock and month: which source sees a
   filing first.

Comparisons use the 400 most liquid stocks on each date (by ``log_adv``) in the Sharadar
panel, production's candidate pool.
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from portfolio_lab.backtest.engine import BacktestConfig, run
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.piotroski import PIOTROSKI, health_scores

#: Production's candidate pool and list size.
POOL, TOP = 400, 100


def mixed_dir(prices: Path, features: Path, out: Path) -> Path:
    """A data directory of symlinks: ``prices``' data with ``features``' feature panel.

    Everything but the feature panel (prices, universe, rates, caches) comes from ``prices``.
    """
    out.mkdir(parents=True, exist_ok=True)
    for item in prices.iterdir():
        if item.name in {"features", "results"}:
            continue
        link = out / item.name
        if not link.exists():
            link.symlink_to(item)
    (out / "features").mkdir(exist_ok=True)
    link = out / "features" / "monthly.parquet"
    link.unlink(missing_ok=True)
    link.symlink_to(features / "features" / "monthly.parquet")
    return out


def swap_test(strategy_factory, dirs: dict[str, Path], scratch: Path, start: date) -> pl.DataFrame:
    """Production on every mix of prices and fundamentals.

    Args:
        strategy_factory: Builds a fresh production strategy.
        dirs: ``{"live": ..., "sharadar": ...}`` data directories.
        scratch: Where the mixed directories go.
        start: First session.

    Returns:
        prices, fundamentals, cagr, sharpe, max_drawdown, turnover per mix.
    """
    rows = []
    for price_name, price_dir in dirs.items():
        panel = None
        for fund_name, fund_dir in dirs.items():
            root = mixed_dir(price_dir, fund_dir, scratch / f"{price_name}-{fund_name}")
            if panel is None:
                panel = Panel.load(root)
            else:  # same prices: swap only the feature panel
                panel.features = pl.read_parquet(root / "features" / "monthly.parquet").sort("date")
            end = min(panel.dates[-1], _last_feature_date(dirs))
            result = run(strategy_factory(), panel, BacktestConfig(start, end))
            m = result.metrics
            rows.append({"prices": price_name, "fundamentals": fund_name, "cagr": m["cagr"],
                         "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                         "turnover": m.get("turnover_annual")})  # fmt: skip
        del panel
    return pl.DataFrame(rows)


def _last_feature_date(dirs: dict[str, Path]) -> date:
    return min(
        pl.scan_parquet(d / "features" / "monthly.parquet").select(pl.col("date").max()).collect()
        .item()
        for d in dirs.values()
    )  # fmt: skip


def _pool(features: pl.DataFrame, start: date) -> pl.DataFrame:
    """date, symbol of the :data:`POOL` most liquid stocks each month from ``start``."""
    return (
        features.filter((pl.col("date") >= start) & pl.col("log_adv").is_not_null())
        .with_columns(pl.col("log_adv").rank("ordinal", descending=True).over("date").alias("_r"))
        .filter(pl.col("_r") <= POOL)
        .select("date", "symbol")
    )


def compare_inputs(live: pl.DataFrame, sharadar: pl.DataFrame, start: date) -> dict:
    """Coverage and agreement of the health inputs, the healthiest-100 overlap and timing.

    Args:
        live: The live feature panel.
        sharadar: Sharadar's feature panel.
        start: First month compared.
    """
    pool = _pool(sharadar, start)
    columns = [*PIOTROSKI, "days_since_filing"]
    left = live.select("date", "symbol", *columns)
    right = sharadar.select("date", "symbol", *columns)
    both = pool.join(left, on=["date", "symbol"], how="left").join(
        right, on=["date", "symbol"], how="left", suffix="_s"
    )
    in_live = pool.join(live.select("date", "symbol"), on=["date", "symbol"], how="semi").height
    inputs = []
    for c in PIOTROSKI:
        a, b = pl.col(c), pl.col(f"{c}_s")
        corr = (
            both.filter(a.is_not_null() & b.is_not_null())
            .group_by("date")
            .agg(pl.corr(a, b, method="spearman").alias("rho"))["rho"]
        )
        inputs.append({
            "input": c,
            "live_has": both[c].is_not_null().mean(),
            "sharadar_has": both[f"{c}_s"].is_not_null().mean(),
            "rank_corr": float(np.nanmedian(corr.to_numpy())) if corr.len() else None,
        })  # fmt: skip
    overlap = []
    for (day,), month in both.group_by("date"):
        live_h = health_scores(month.select("symbol", *PIOTROSKI))
        shar_h = health_scores(
            month.select("symbol", *[pl.col(f"{c}_s").alias(c) for c in PIOTROSKI])
        )
        top = [set(sorted(h, key=h.get, reverse=True)[:TOP]) for h in (live_h, shar_h)]
        overlap.append({"date": day, "overlap": len(top[0] & top[1]) / TOP})
    lag = both.filter(
        pl.col("days_since_filing").is_not_null() & pl.col("days_since_filing_s").is_not_null()
    ).select((pl.col("days_since_filing") - pl.col("days_since_filing_s")).alias("d"))["d"]
    return {
        "pool_in_live": in_live / pool.height,
        "inputs": pl.DataFrame(inputs),
        "overlap": pl.DataFrame(overlap).sort("date"),
        "timing": {
            "median_days_later": float(lag.median()) if lag.len() else None,
            "same": float((lag == 0).mean()) if lag.len() else None,
            "live_later": float((lag > 0).mean()) if lag.len() else None,
            "live_earlier": float((lag < 0).mean()) if lag.len() else None,
        },
    }
