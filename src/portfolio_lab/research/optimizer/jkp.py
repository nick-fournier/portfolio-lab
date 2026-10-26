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


def split_adjusted_shares(root: Path) -> pl.DataFrame:
    """sid, ym (the filing month), shares: each filing's share count in today's units.

    The filings keep the count as reported (``conform.sharadar`` undoes later splits); every
    split after the filing multiplies it, as CRSP's shares times its share-adjustment factor.
    """
    f = (
        reader.read(root, "filings", ["shares_out"])
        .filter(pl.col("shares_out") > 0)
        .select("sid", "filed", "shares_out")
        .sort("sid", "filed")
        .unique(["sid", "filed"], keep="last", maintain_order=True)
    )
    splits = reader.read(root, "actions").filter(
        (pl.col("action") == "split") & (pl.col("value") > 0)
    )
    later = (
        f.join(splits.select("sid", pl.col("date").alias("_split"), "value"), on="sid")
        .filter(pl.col("_split") > pl.col("filed"))
        .group_by("sid", "filed")
        .agg(pl.col("value").product().alias("_factor"))
    )
    return (
        f.join(later, on=["sid", "filed"], how="left")
        .select(
            "sid",
            jkp_monthly.month_key(pl.col("filed")).alias("ym"),
            (pl.col("shares_out") * pl.col("_factor").fill_null(1.0)).alias("shares"),
        )
        .unique(["sid", "ym"], keep="last", maintain_order=True)
        .sort("sid", "ym")
    )


def _issuance(m: pl.DataFrame, shares: pl.DataFrame) -> pl.DataFrame:
    """``chcsho_12m`` and ``eqnpo_12m`` (JKP ``market_chars_monthly``, aux 6945-6982).

    Share issuance is the 12-month change in split-adjusted shares outstanding: each month
    takes its latest filing's count (:func:`split_adjusted_shares`). Net payout is the log
    total return minus the log change in market value over 12 months, set to 0 when below
    1e-5 in size.
    """
    m = (
        m.sort("ym")
        .join_asof(shares.sort("ym"), on="ym", by="sid", strategy="backward")
        .sort("sid", "ym")
    )
    count = pl.col("ym").cum_count().over("sid")
    ri = (1 + pl.col("ret").fill_null(0.0)).cum_prod().over("sid")
    m = m.with_columns(ri.alias("_ri"), count.alias("_n"))
    lag = lambda c: pl.col(c).shift(12).over("sid")  # noqa: E731
    me, me12 = pl.col("market_value"), lag("market_value")
    npo = (pl.col("_ri") / lag("_ri")).log() - (me / me12).log()
    ok = (pl.col("_n") > 12) & (pl.col("_ri") > 0) & (lag("_ri") > 0) & (me > 0) & (me12 > 0)
    return m.with_columns(
        pl.when((pl.col("_n") > 12) & (lag("shares") > 0))
        .then(pl.col("shares") / lag("shares") - 1)
        .alias("chcsho_12m"),
        pl.when(ok).then(pl.when(npo.abs() < 1e-5).then(0.0).otherwise(npo)).alias("eqnpo_12m"),
    ).drop("shares", "_ri", "_n", "ret", "prc_adj", "market_value")


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
    issued = split_adjusted_shares(root)
    frames = []
    for chunk in np.array_split(sids, chunks):
        daily = reader.read(root, "prices", PRICE_COLUMNS, sids=chunk.tolist())
        monthly = jkp_monthly.characteristics(daily, monthly_factors, first_filing)
        shares = (
            monthly.select("sid", "ym", "prc")
            .join(mv, on=["sid", "ym"], how="inner")
            .select("sid", "ym", (pl.col("market_value") / pl.col("prc")).alias("shares"))
        )
        monthly = _issuance(monthly.join(mv, on=["sid", "ym"], how="left"), issued)
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
