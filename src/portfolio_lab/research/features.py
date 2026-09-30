"""Monthly point-in-time feature panel: one row per eligible stock at every month end.

Each row uses only what was known at that month's close:

- the company's latest filing state (``research.fundamentals``) filed strictly before the
  date and at most :data:`MAX_FILING_AGE_DAYS` old,
- prices up to and including the date.

Market value is shares outstanding (from that filing) times the price on the filing date,
grown by the stock's total return since, which carries it through splits without having to
detect them. Features (all floats; null where an input is missing):

- **Valuation**: earnings, book, cash-flow, free-cash-flow, sales and dividend yields.
- **Quality** (the raw Piotroski inputs and relatives): return on assets, cash flow to
  assets, accruals, gross profitability, operating margin, leverage, current ratio, and
  their year-on-year changes; share issuance, asset and sales growth.
- **Price**: 12-1 month momentum, last month's and last week's return, volatility, beta,
  size, liquidity; distance from the 52-week high, stock-specific volatility (net of the
  market) and the trend in liquidity.
- **Earnings news**: the latest quarter's earnings and revenue change from a year earlier
  (the difference between consecutive trailing-twelve-month figures), the stock's return
  relative to the market around the filing, and days since it.
- **Industry momentum**: the average 1- and 6-month return of the stock's industry.
- ``fscore`` for comparison, and ``*_ind``: key features as percentiles within the stock's
  industry (SIC major group) on that date.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl

from portfolio_lab.core.calendar import rebalance_dates
from portfolio_lab.research.panel import Panel

MAX_FILING_AGE_DAYS = 550
TRADING_DAYS = 252
MONTH = 21
#: Features also expressed as percentiles within the industry.
INDUSTRY_RELATIVE = (
    "earnings_yield", "book_to_market", "cf_yield", "sales_yield", "roa",
    "gross_profitability", "asset_growth", "mom_12_1",
)  # fmt: skip
#: Fewest stocks in an industry on a date for industry percentiles.
MIN_INDUSTRY = 5
#: Every feature column :func:`build_features` produces (with industry data and F-scores).
FEATURES = (
    "earnings_yield", "book_to_market", "cf_yield", "fcf_yield", "sales_yield",
    "dividend_yield", "roa", "cfo_to_assets", "accruals", "gross_profitability",
    "operating_margin", "leverage", "current_ratio", "d_roa", "d_lt_debt", "d_current_ratio",
    "d_gross_margin", "d_asset_turnover", "share_issuance", "asset_growth", "sales_growth",
    "log_size", "mom_12_1", "ret_1m", "volatility", "beta", "log_adv", "fscore",
    "ret_1w", "high_52w", "idio_vol", "liquidity_trend",
    "earnings_surprise", "revenue_surprise", "filing_reaction", "days_since_filing",
    "ind_mom_1m", "ind_mom_6m",
    *(f"{c}_ind" for c in INDUSTRY_RELATIVE),
    # Ownership (``research.ownership``), where insider and 13F data are available.
    "insider_buy", "insider_net", "insider_buys", "inst_breadth_1y", "inst_breadth_1q",
)  # fmt: skip
#: Features computed from prices; the rest describe the business (``FUNDAMENTAL``).
PRICE_FEATURES = (
    "mom_12_1", "ret_1m", "volatility", "beta", "log_adv", "ret_1w", "high_52w", "idio_vol",
    "liquidity_trend", "filing_reaction", "ind_mom_1m", "ind_mom_6m", "mom_12_1_ind",
)  # fmt: skip
FUNDAMENTAL = tuple(f for f in FEATURES if f not in PRICE_FEATURES)
WEEK = 5
HALF_YEAR = 126
#: Sessions before (and one after) the filing counted as its market reaction; earnings
#: releases usually come a few days before the 10-Q or 10-K.
REACTION_SESSIONS = 10


def _finite(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, np.nan)


def _price_features(panel: Panel, index: int, cols: np.ndarray) -> dict[str, np.ndarray]:
    """Price-based features for ``cols`` at session ``index``."""
    ret = panel.field("ret_cc")
    window = ret[index - TRADING_DAYS + 1 : index + 1][:, cols]
    growth = np.nancumprod(1 + np.nan_to_num(window), axis=0)
    spy = np.nan_to_num(ret[index - TRADING_DAYS + 1 : index + 1, panel.symbol_index["SPY"]])
    enough = np.isfinite(window).mean(axis=0) >= 0.8
    filled = np.where(np.isfinite(window), window, 0.0)
    spy_c = spy - spy.mean()
    beta = spy_c @ (filled - filled.mean(axis=0)) / (spy_c @ spy_c)
    residual = (filled - filled.mean(axis=0)) - np.outer(spy_c, beta)
    adv = panel.field("adv")
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "ret_1w": np.where(enough, growth[-1] / growth[-WEEK - 1] - 1, np.nan),
            "ret_6m": np.where(enough, growth[-1] / growth[-HALF_YEAR - 1] - 1, np.nan),
            "high_52w": np.where(enough, growth[-1] / growth.max(axis=0) - 1, np.nan),
            "idio_vol": np.where(enough, residual.std(axis=0), np.nan),
            "liquidity_trend": _finite(np.log(adv[index, cols] / adv[index - HALF_YEAR, cols])),
            # growth is relative to the price just before the window (1.0).
            "mom_12_1": np.where(enough, growth[-MONTH - 1] - 1, np.nan),
            "ret_1m": np.where(enough, growth[-1] / growth[-MONTH - 1] - 1, np.nan),
            "volatility": np.where(enough, np.nanstd(window, axis=0), np.nan),
            "beta": np.where(enough, beta, np.nan),
            "log_adv": np.log(panel.field("adv")[index, cols]),
        }


def _price_grid(panel: Panel, start: date) -> pl.DataFrame:
    """Month-end rows (date, symbol) for eligible stocks, with price features and close."""
    days = [d for d in panel.dates if d >= start]
    close = panel.field("close")
    frames = []
    for day in rebalance_dates(days, "M"):
        i = panel.date_index[day]
        if i < TRADING_DAYS:
            continue
        cols = np.flatnonzero(panel.eligible[i])
        feats = _price_features(panel, i, cols)
        frames.append(
            pl.DataFrame(
                {
                    "date": [day] * len(cols),
                    "symbol": [panel.symbols[j] for j in cols],
                    "_col": cols,
                    "_row": [i] * len(cols),
                    "close": close[i, cols],
                    **feats,
                }
            )
        )
    return pl.concat(frames)


def _total_return_index(panel: Panel) -> np.ndarray:
    """Cumulative adjusted growth per symbol (dates x symbols), 1 before the first bar."""
    return np.cumprod(1 + np.nan_to_num(panel.field("ret_cc")), axis=0)


def _market_value(grid: pl.DataFrame, panel: Panel) -> pl.Series:
    """Shares at filing x price at filing x total return since (see module docs)."""
    index = _total_return_index(panel)
    close = panel.field("close")
    dates = np.array(panel.dates, dtype="datetime64[D]")
    filed = grid["filed"].to_numpy().astype("datetime64[D]")
    ok = ~np.isnat(filed)
    at_filing = np.full(grid.height, -1)
    at_filing[ok] = np.searchsorted(dates, filed[ok], side="right") - 1
    rows, cols = grid["_row"].to_numpy(), grid["_col"].to_numpy()
    valid = at_filing >= 0
    f = np.where(valid, at_filing, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = grid["shares_out"].to_numpy() * close[f, cols] * index[rows, cols] / index[f, cols]
    return pl.Series("market_value", np.where(valid, value, np.nan)).fill_nan(None)


def _fundamental_features() -> list[pl.Expr]:
    """Valuation and quality features from the joined filing state and market value."""
    c, mv = pl.col, pl.col("market_value")

    def ratio(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return pl.when(b > 0).then(a / b)

    gross = pl.coalesce(c("gross_profit"), c("revenue") - c("cost_of_revenue"))
    gross_py = pl.coalesce(c("gross_profit_py"), c("revenue_py") - c("cost_of_revenue_py"))
    lt_debt = pl.when(c("lt_debt").is_null() & c("liabilities").is_not_null()).then(0.0)
    lt_debt = lt_debt.otherwise(c("lt_debt"))
    lt_debt_py = pl.when(c("lt_debt_py").is_null() & c("liabilities_py").is_not_null()).then(0.0)
    lt_debt_py = lt_debt_py.otherwise(c("lt_debt_py"))
    roa, roa_py = ratio(c("net_income"), c("assets")), ratio(c("net_income_py"), c("assets_py"))
    margin, margin_py = ratio(gross, c("revenue")), ratio(gross_py, c("revenue_py"))
    current = ratio(c("assets_cur"), c("liab_cur"))
    current_py = ratio(c("assets_cur_py"), c("liab_cur_py"))
    return [
        ratio(c("net_income"), mv).alias("earnings_yield"),
        ratio(c("equity"), mv).alias("book_to_market"),
        ratio(c("cfo"), mv).alias("cf_yield"),
        ratio(c("cfo") - c("capex"), mv).alias("fcf_yield"),
        ratio(c("revenue"), mv).alias("sales_yield"),
        ratio(c("dividends"), mv).alias("dividend_yield"),
        roa.alias("roa"),
        ratio(c("cfo"), c("assets")).alias("cfo_to_assets"),
        ratio(c("net_income") - c("cfo"), c("assets")).alias("accruals"),
        ratio(gross, c("assets")).alias("gross_profitability"),
        ratio(c("operating_income"), c("revenue")).alias("operating_margin"),
        ratio(c("liabilities"), c("assets")).alias("leverage"),
        current.alias("current_ratio"),
        (roa - roa_py).alias("d_roa"),
        (ratio(lt_debt, c("assets")) - ratio(lt_debt_py, c("assets_py"))).alias("d_lt_debt"),
        (current - current_py).alias("d_current_ratio"),
        (margin - margin_py).alias("d_gross_margin"),
        (ratio(c("revenue"), c("assets")) - ratio(c("revenue_py"), c("assets_py"))).alias(
            "d_asset_turnover"
        ),
        (ratio(c("shares_weighted"), c("shares_weighted_py")) - 1).alias("share_issuance"),
        (ratio(c("assets"), c("assets_py")) - 1).alias("asset_growth"),
        (ratio(c("revenue"), c("revenue_py")) - 1).alias("sales_growth"),
        mv.log().alias("log_size"),
    ]


def _asof(grid: pl.DataFrame, table: pl.DataFrame, max_age: int) -> pl.DataFrame:
    """Join each (symbol, date) to the latest ``table`` row filed strictly before the date."""
    right = table.with_columns(
        (pl.col("filed") + timedelta(days=1)).alias("_visible"),
        pl.col("filed").alias("_filed_right"),
    ).sort("_visible")
    joined = grid.sort("date").join_asof(
        right, left_on="date", right_on="_visible", by="symbol", strategy="backward",
        check_sortedness=False,
    )  # fmt: skip
    too_old = (pl.col("date") - pl.col("_filed_right")).dt.total_days() > max_age
    stale = [c for c in table.columns if c not in ("symbol", "filed")]
    return joined.with_columns(
        [pl.when(too_old).then(None).otherwise(pl.col(c)).alias(c) for c in stale]
    ).drop("_visible", "_filed_right")


def _with_changes(states: pl.DataFrame) -> pl.DataFrame:
    """Add each filing's change in trailing-twelve-month earnings and revenue.

    Consecutive quarters' TTM figures differ by the latest quarter minus the same quarter a
    year earlier, the seasonally adjusted news in the filing. Null unless the previous
    filing covered the quarter before (80 to 100 days earlier).
    """
    states = states.sort("cik", "filed", "accn")
    prev_end = pl.col("period_end").shift(1).over("cik")
    quarter = (pl.col("period_end") - prev_end).dt.total_days().is_between(80, 100)
    return states.with_columns(
        pl.when(quarter).then(pl.col(c) - pl.col(c).shift(1).over("cik")).alias(f"_d_{c}")
        for c in ("net_income", "revenue")
    )


def _filing_reaction(grid: pl.DataFrame, panel: Panel) -> pl.Series:
    """Stock return minus SPY's over the sessions around the latest filing."""
    index = _total_return_index(panel)
    dates = np.array(panel.dates, dtype="datetime64[D]")
    filed = grid["filed"].to_numpy().astype("datetime64[D]")
    ok = ~np.isnat(filed)
    at = np.full(grid.height, -1)
    at[ok] = np.searchsorted(dates, filed[ok], side="right") - 1
    rows, cols = grid["_row"].to_numpy(), grid["_col"].to_numpy()
    end = np.minimum(at + 1, rows)
    begin = at - REACTION_SESSIONS
    valid = ok & (begin >= 0)
    b, e = np.where(valid, begin, 0), np.where(valid, end, 0)
    spy = panel.symbol_index["SPY"]
    with np.errstate(invalid="ignore", divide="ignore"):
        stock = index[e, cols] / index[b, cols] - 1
        market = index[e, spy] / index[b, spy] - 1
    return pl.Series("filing_reaction", np.where(valid, stock - market, np.nan)).fill_nan(None)


def _earnings_news() -> list[pl.Expr]:
    """Earnings and revenue surprises (scaled) and days since the filing."""
    c = pl.col
    return [
        pl.when(c("market_value") > 0)
        .then(c("_d_net_income") / c("market_value"))
        .alias("earnings_surprise"),
        pl.when(c("revenue_py") > 0)
        .then(c("_d_revenue") / c("revenue_py"))
        .alias("revenue_surprise"),
        (c("date") - c("filed")).dt.total_days().cast(pl.Float64).alias("days_since_filing"),
    ]


def build_features(
    panel: Panel,
    states: pl.DataFrame,
    tickers: pl.DataFrame,
    companies: pl.DataFrame | None = None,
    fscores: pl.DataFrame | None = None,
    start: date | None = None,
) -> pl.DataFrame:
    """Monthly feature panel (see module docs).

    Args:
        panel: Prices and eligibility.
        states: Filing states from ``research.fundamentals.filing_states``.
        tickers: symbol -> cik map.
        companies: SEC profiles with ``cik`` and ``sic`` (for industry features).
        fscores: Point-in-time F-scores by symbol (symbol, filed, fscore, n_signals).
        start: First month end (default: the first with a year of price history).

    Returns:
        date, symbol, sic2, market_value, the features, and ``fscore``.
    """
    grid = _price_grid(panel, start or panel.dates[0])
    by_symbol = _with_changes(states).join(tickers, on="cik").drop("accn", "form")
    grid = _asof(grid, by_symbol, MAX_FILING_AGE_DAYS)
    grid = grid.with_columns(_market_value(grid, panel))
    grid = grid.with_columns(_fundamental_features())
    grid = grid.with_columns(_filing_reaction(grid, panel), *_earnings_news())
    if fscores is not None:
        scored = fscores.filter(pl.col("n_signals") >= 8).select("symbol", "filed", "fscore")
        scored = scored.with_columns(pl.col("fscore").cast(pl.Float64))
        grid = _asof(grid, scored, MAX_FILING_AGE_DAYS).drop("filed_right", strict=False)
    if companies is not None:
        grid = grid.join(
            companies.select("cik", (pl.col("sic") // 100).alias("sic2")), on="cik", how="left"
        )
        group = ("date", "sic2")
        enough = pl.col("sic2").is_not_null() & (pl.len().over(group) >= MIN_INDUSTRY)
        grid = grid.with_columns(
            *[
                pl.when(enough)
                .then(pl.col(c).rank().over(group) / pl.col(c).count().over(group))
                .alias(f"{c}_ind")
                for c in INDUSTRY_RELATIVE
            ],
            pl.when(enough).then(pl.col("ret_1m").mean().over(group)).alias("ind_mom_1m"),
            pl.when(enough).then(pl.col("ret_6m").mean().over(group)).alias("ind_mom_6m"),
        )
    lead = ["date", "symbol", *(["sic2"] if "sic2" in grid.columns else []), "market_value"]
    rest = [
        c for c in grid.columns
        if not c.startswith("_")
        and c not in by_symbol.columns
        and c not in (*lead, "close", "ret_6m")
    ]  # fmt: skip
    return grid.select(*lead, *rest).sort("date", "symbol")
