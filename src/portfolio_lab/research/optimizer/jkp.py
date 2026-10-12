"""JKP price-based characteristics for every common stock-month (``jkp_monthly``, ``jkp_daily``).

Runs both modules over the common stocks in chunks (daily prices are about 75 million rows), then
the one cross-sectional step (the zero-trades tie-break). Shares outstanding for turnover come
from the monthly table's market value over the month's last close.
"""

from pathlib import Path

import numpy as np
import polars as pl

from portfolio_lab.data import reader
from portfolio_lab.research.optimizer import jkp_daily, jkp_monthly

#: Daily price columns the characteristics read.
PRICE_COLUMNS = ["high", "low", "close", "volume", "close_adj", "ret_cc"]


def build(root: Path, market_value: pl.DataFrame, chunks: int = 8) -> pl.DataFrame:
    """Price-based JKP characteristics: symbol, ym, then one column each.

    Args:
        root: The data hive.
        market_value: date, symbol, market_value (the monthly table's, for shares outstanding).
        chunks: Groups of stocks processed one at a time (memory).
    """
    factors = jkp_monthly.french(root)
    monthly_factors = jkp_monthly.monthly_factors(factors)
    listings = reader.read(root, "listings")
    sids = listings.filter(pl.col("category") == "common")["sid"].unique().sort().to_numpy()
    first_filing = (
        reader.read(root, "filings", ["period_end"])
        .group_by("sid")
        .agg(pl.col("period_end").min().alias("first_period"))
    )
    mv = market_value.select(
        pl.col("symbol").cast(pl.Int64).alias("sid"),
        jkp_monthly.month_key(pl.col("date")).alias("ym"),
        "market_value",
    )
    frames = []
    for chunk in np.array_split(sids, chunks):
        daily = reader.read(root, "prices", PRICE_COLUMNS, sids=chunk.tolist())
        monthly = jkp_monthly.characteristics(daily, monthly_factors, first_filing)
        shares = (
            monthly.select("sid", "ym", "prc")
            .join(mv, on=["sid", "ym"], how="inner")
            .select("sid", "ym", (pl.col("market_value") / pl.col("prc")).alias("shares"))
        )
        d = jkp_daily.prepare(daily, factors, listings, shares)
        frames.append(
            monthly.join(jkp_daily.characteristics(d, factors), on=["sid", "ym"], how="left")
        )
        del daily, d
    out = jkp_daily.zero_trades(pl.concat(frames, how="diagonal_relaxed"))
    # NaN (numpy regressions, 0/0) means missing; rvol_252d is only a helper (betabab, rmax5_rvol)
    return out.with_columns(
        pl.col(pl.Float64, pl.Float32).fill_nan(None),
        pl.col("sid").cast(pl.String).alias("symbol"),
    ).drop("sid", "rvol_252d")
