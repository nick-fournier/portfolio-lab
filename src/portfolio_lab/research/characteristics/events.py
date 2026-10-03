"""Returns and volume around quarterly reports, per report.

Green, Hand & Zhang measure these around the earnings announcement; Sharadar gives the
SEC filing date instead, which usually follows the announcement, so these are a stand-in:

- ``ear``: summed daily returns from the trading day before the filing to the day after.
- ``aeavol``: average volume over those days relative to the average from 30 to 10
  trading days before the filing, minus 1.
"""

import polars as pl

WINDOW = (-1, 1)
BASELINE = (-30, -10)


def around_filings(prices: pl.LazyFrame, filings: pl.DataFrame) -> pl.DataFrame:
    """``ear`` and ``aeavol`` for each filing.

    Args:
        prices: Daily bars (symbol, date, volume, ret_cc).
        filings: symbol, filed (one row per quarterly report).

    Returns:
        symbol, filed, ear, aeavol.
    """
    daily = (
        prices.select("symbol", "date", "volume", "ret_cc").collect().sort("symbol", "date")
        .with_columns(pl.int_range(pl.len()).over("symbol").alias("k"))
    )  # fmt: skip
    # the filing's trading day: the first session on or after the filing date
    events = (
        filings.select("symbol", "filed").unique().sort("filed")
        .join_asof(daily.select("symbol", "date", "k").sort("date"), left_on="filed",
                   right_on="date", by="symbol", strategy="forward", check_sortedness=False)
        .drop_nulls("k").drop("date")
    )  # fmt: skip
    offsets = pl.DataFrame({"off": list(range(BASELINE[0], WINDOW[1] + 1))})
    rows = (
        events.join(offsets, how="cross")
        .with_columns((pl.col("k") + pl.col("off")).alias("k"))
        .join(daily.select("symbol", "k", "volume", "ret_cc"), on=["symbol", "k"])
    )
    in_window = pl.col("off").is_between(*WINDOW)
    in_base = pl.col("off").is_between(*BASELINE)
    return rows.group_by("symbol", "filed").agg(
        pl.col("ret_cc").filter(in_window).sum().alias("ear"),
        (pl.col("volume").filter(in_window).mean()
         / pl.col("volume").filter(in_base).mean() - 1).alias("aeavol"),
    ).fill_nan(None)  # fmt: skip
