"""Ownership features: what insiders and institutions have been doing with each stock.

Monthly, point in time (only filings public before the month end):

- ``insider_buy``: open-market purchases by officers, directors and 10% owners over the
  past six months (Form 4 filing dates), as a share of market value.
- ``insider_net``: purchases minus sales over the same window, as a share of market value.
- ``insider_buys``: the number of purchase transactions in the window (log of 1 + count).
  All three are null before the insider data's first full window.
- ``inst_breadth_1y`` / ``inst_breadth_1q``: change in the number of 13F institutions
  holding the stock versus a year and a quarter earlier, counted from 45 days after each
  quarter end, when the filings are due.
"""

from datetime import timedelta

import polars as pl

#: Insider activity window.
INSIDER_DAYS = 182
FEATURES = ("insider_buy", "insider_net", "insider_buys", "inst_breadth_1y", "inst_breadth_1q")


def _asof(grid: pl.DataFrame, table: pl.DataFrame, on: str, shift_days: int) -> pl.DataFrame:
    """Latest ``table`` row per symbol with ``on`` strictly before each date minus a shift."""
    left = grid.with_columns((pl.col("date") - timedelta(days=shift_days)).alias("_at")).sort("_at")
    right = table.with_columns((pl.col(on) + timedelta(days=1)).alias("_at")).sort("_at")
    return left.join_asof(
        right.drop(on), on="_at", by="symbol", strategy="backward", check_sortedness=False
    ).drop("_at")


def _insiders(grid: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """Six-month insider sums from cumulative totals at the date and six months before."""
    cols = ("buy_value", "sell_value", "buys")
    cum = trades.sort("symbol", "filed").select(
        "symbol", "filed", *[pl.col(c).fill_null(0).cum_sum().over("symbol") for c in cols]
    )
    now = _asof(grid, cum, "filed", 0)
    then = _asof(grid, cum, "filed", INSIDER_DAYS).rename({c: f"{c}_then" for c in cols})
    window = now.join(then, on=["date", "symbol"]).with_columns(
        (pl.col(c).fill_null(0) - pl.col(f"{c}_then").fill_null(0)).alias(c) for c in cols
    )
    # Before the data's first full window, "no filings" means unknown, not no activity.
    covered = (pl.col("date") >= trades["filed"].min() + timedelta(days=INSIDER_DAYS)) & (
        pl.col("market_value") > 0
    )
    net = pl.col("buy_value") - pl.col("sell_value")
    return window.select(
        "date", "symbol",
        pl.when(covered).then(pl.col("buy_value") / pl.col("market_value")).alias("insider_buy"),
        pl.when(covered).then(net / pl.col("market_value")).alias("insider_net"),
        pl.when(covered).then(pl.col("buys").log1p()).alias("insider_buys"),
    )  # fmt: skip


def _breadth(grid: pl.DataFrame, holders: pl.DataFrame) -> pl.DataFrame:
    """Change in the number of institutional holders over a year and a quarter."""
    table = holders.select("symbol", "available", "holders")
    joined = _asof(grid, table, "available", 0)
    for name, days in (("1y", 365), ("1q", 91)):
        before = _asof(grid, table, "available", days)
        joined = joined.join(
            before.select("date", "symbol", pl.col("holders").alias(f"holders_{name}")),
            on=["date", "symbol"],
        )
    return joined.select(
        "date", "symbol",
        *[pl.when(pl.col(f"holders_{n}") > 0).then(pl.col("holders") / pl.col(f"holders_{n}") - 1)
          .alias(f"inst_breadth_{n}") for n in ("1y", "1q")],
    )  # fmt: skip


def ownership_features(
    features: pl.DataFrame, trades: pl.DataFrame | None, holders: pl.DataFrame | None
) -> pl.DataFrame:
    """Add the ownership features (see module docs) to the monthly feature panel."""
    grid = features.select("date", "symbol", "market_value")
    if trades is not None:
        features = features.join(_insiders(grid, trades), on=["date", "symbol"], how="left")
    if holders is not None:
        features = features.join(_breadth(grid, holders), on=["date", "symbol"], how="left")
    return features
