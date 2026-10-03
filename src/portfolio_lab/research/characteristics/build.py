"""Assemble the extra stock inputs onto the monthly feature grid.

Joins ``characteristics.trading``, ``characteristics.accounts`` and
``characteristics.events`` to every (month end, stock) row of the feature panel, point in
time: a filing counts from the day after it was filed and for at most
:data:`MAX_AGE_DAYS`. Then finishes the inputs that need the month's market value or
industry (2-digit SIC): valuation ratios, values minus their industry average (``*_ia``),
industry momentum and sales concentration (``herf``), and sin stocks.
"""

import logging
from datetime import timedelta
from pathlib import Path

import polars as pl

from portfolio_lab.research.characteristics import accounts, events, trading

log = logging.getLogger(__name__)

MAX_AGE_DAYS = 550
#: Sin stocks by SIC: tobacco, beer/wine/liquor (gaming has no clean SIC code).
SIN_SIC = ((2100, 2199), (2080, 2085))
#: Inputs adjusted by subtracting the month's industry average: (output, source).
INDUSTRY_ADJUSTED = (("bm_ia", "book_to_market"), ("cfp_ia", "cf_yield"),
                     ("mve_ia", "market_value"), ("pchcapx_ia", "pchcapx"),
                     ("chatoia", "chato"), ("chpmia", "chpm"), ("tb", "tb_raw"))  # fmt: skip
#: Every input this module adds.
COLUMNS = (
    "mom6m", "mom36m", "chmom", "indmom", "maxret", "retvol", "beta_w", "betasq", "idiovol",
    "pricedelay", "turn", "std_turn", "std_dolvol", "ill", "zerotrade", "baspread", "age",
    "bm_ia", "cfp_ia", "mve_ia", "cashpr", "rd_mve", "operprof", "roic", "roaq", "roeq",
    "roavol", "nincr", "tb", "chtx", "absacc", "pctacc", "stdacc", "stdcf", "cashdebt", "egr",
    "lgr", "invest", "chinv", "grcapx", "pchcapx_ia", "cinvest", "grltnoa", "depr", "pchdepr",
    "rd", "rd_sale", "orgcap", "quick", "pchquick", "cash", "salecash", "saleinv", "salerec",
    "pchsaleinv", "pchsale_pchinvt", "pchsale_pchrect", "pchsale_pchxsga", "pchgm_pchsale",
    "chatoia", "chpmia", "tang", "ear", "aeavol", "rsup", "divi", "divo", "sin", "herf",
)  # fmt: skip


def _asof(grid: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    """Each grid row joined to the latest ``table`` row (symbol, filed, ...) visible then."""
    visible = (pl.col("filed") + timedelta(days=1)).alias("_visible")
    right = table.with_columns(visible).sort("_visible")
    joined = grid.sort("date").join_asof(
        right, left_on="date", right_on="_visible", by="symbol", strategy="backward",
        check_sortedness=False,
    )  # fmt: skip
    stale = (pl.col("date") - pl.col("filed")).dt.total_days() > MAX_AGE_DAYS
    cols = [c for c in table.columns if c not in ("symbol", "filed", "q")]
    return joined.with_columns(pl.when(stale).then(None).otherwise(pl.col(c)).alias(c) for c in cols
                               ).drop("filed", "q", "_visible", strict=False)  # fmt: skip


def build(data_dir: Path, raw: Path) -> pl.DataFrame:
    """The extra inputs for every row of ``data_dir``'s feature panel (see module docs).

    Args:
        data_dir: A Sharadar data directory (features, prices, fundamentals, macro).
        raw: Folder with Sharadar's bulk zips (``fundamentals.zip``).

    Returns:
        date, symbol and :data:`COLUMNS`.
    """
    grid = pl.read_parquet(data_dir / "features" / "monthly.parquet",
                           columns=["date", "symbol", "sic2", "market_value", "book_to_market",
                                    "cf_yield", "mom_12_1"])  # fmt: skip
    daily = data_dir / "prices" / "daily" / "**" / "*.parquet"
    prices = pl.scan_parquet(daily, hive_partitioning=True)
    log.info("trading inputs")
    bars = trading.monthly_bars(prices, grid)
    grid = grid.join(trading.monthly_inputs(bars, grid), on=["date", "symbol"], how="left")
    mat, market, symbols, weeks = trading.weekly_returns(prices)
    betas = trading.regressions(grid.select("date", "symbol"), mat, market, symbols, weeks)
    grid = grid.join(betas, on=["date", "symbol"], how="left")
    del mat
    log.info("statement inputs")
    obs = pl.read_parquet(data_dir / "macro" / "observations.parquet")
    cpi = obs.filter(pl.col("series") == "CPIAUCSL").select("date", pl.col("value").alias("cpi"))
    yearly = accounts.annual(accounts.read_rows(raw, "ART", accounts.ANNUAL), cpi)
    quarterly_rows = accounts.read_rows(raw, "ARQ", accounts.QUARTERLY)
    quarterly = accounts.quarterly(quarterly_rows)
    log.info("filing-date returns")
    around = events.around_filings(prices, quarterly_rows.select("symbol", "filed"))
    grid = _asof(grid, yearly)
    grid = _asof(grid, quarterly)
    grid = _asof(grid, around)
    companies = pl.read_parquet(data_dir / "fundamentals" / "companies.parquet")
    sic = (
        pl.read_parquet(data_dir / "fundamentals" / "tickers.parquet")
        .join(companies.select("cik", "sic"), on="cik")
        .select("symbol", "sic")
        .unique("symbol")
    )
    grid = grid.join(sic, on="symbol", how="left")
    return finish(grid)


def finish(grid: pl.DataFrame) -> pl.DataFrame:
    """Inputs that need the month's market value or industry; then the output columns."""
    # divisions by zero (e.g. sales / inventory with no inventory) are missing, not infinite,
    # so they can't spill into industry averages
    grid = _finite(grid)
    c, mve = pl.col, pl.col("market_value")
    industry = ("date", "sic2")
    in_industry = c("sic2").is_not_null()
    share = c("revenue") / c("revenue").sum().over(industry)
    sin = pl.lit(False)
    for lo, hi in SIN_SIC:
        sin = sin | c("sic").is_between(lo, hi)
    out = grid.with_columns(
        ((mve + c("debtnc").fill_null(0) - c("assets")) / c("cashneq")).alias("cashpr"),
        (c("rnd") / mve).alias("rd_mve"),
        (c("d_revenue_q") / mve).alias("rsup"),
        pl.when(in_industry).then(c("mom_12_1").mean().over(industry)).alias("indmom"),
        pl.when(in_industry).then((share * share).sum().over(industry)).alias("herf"),
        pl.when(c("sic").is_not_null()).then(sin.cast(pl.Float64)).alias("sin"),
        *[pl.when(in_industry).then(c(src) - c(src).mean().over(industry)).alias(name)
          for name, src in INDUSTRY_ADJUSTED],
    )  # fmt: skip
    return _finite(out.select("date", "symbol", *COLUMNS)).sort("date", "symbol")


def _finite(frame: pl.DataFrame) -> pl.DataFrame:
    """Infinite and NaN floats as missing."""
    floats = [n for n, t in frame.schema.items() if t == pl.Float64]
    bad = [pl.when(pl.col(n).is_finite()).then(pl.col(n)).alias(n) for n in floats]
    return frame.with_columns(bad)
