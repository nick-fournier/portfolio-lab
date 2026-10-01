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
- **Price**: 12-1 month momentum, last month's return, volatility, beta, size, liquidity.
- **Financial strength**: distance to default (how many standard deviations of asset value
  stand between the company and its debt, the naive Merton model of Bharath and Shumway,
  2008), the volatility of annual return on assets over the last five fiscal years, and
  interest coverage (operating income over interest expense).
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
    "distance_to_default", "roa_volatility", "interest_coverage",
    *(f"{c}_ind" for c in INDUSTRY_RELATIVE),
)  # fmt: skip
#: Fiscal years of return on assets behind ``roa_volatility`` (at least the minimum).
STABILITY_YEARS, STABILITY_MIN_YEARS = 5, 3
#: Bounds on distance to default and interest coverage; debt-free companies get the top.
MAX_DISTANCE, MAX_COVERAGE = 20.0, 100.0


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
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
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
        _distance_to_default(),
        _interest_coverage(),
        c("_roa_volatility").alias("roa_volatility"),
    ]


def _debt() -> pl.Expr:
    """Debt, counting missing current or long-term debt as none when liabilities exist."""
    c = pl.col
    known = (
        c("liabilities").is_not_null() | c("lt_debt").is_not_null() | c("debt_cur").is_not_null()
    )
    return pl.when(known).then(c("lt_debt").fill_null(0.0) + c("debt_cur").fill_null(0.0))


def _distance_to_default() -> pl.Expr:
    """Naive Merton distance to default over one year (Bharath and Shumway, 2008).

    The default point is short-term debt plus half the long-term debt; debt volatility is
    5% plus a quarter of equity volatility; expected asset growth is taken as zero.
    """
    c = pl.col
    equity = c("market_value")
    point = c("debt_cur").fill_null(0.0) + 0.5 * c("lt_debt").fill_null(0.0)
    sigma_e = c("volatility") * np.sqrt(TRADING_DAYS)
    value = equity + point
    sigma_v = equity / value * sigma_e + point / value * (0.05 + 0.25 * sigma_e)
    distance = ((value / point).log() - 0.5 * sigma_v**2) / sigma_v
    has_inputs = equity.is_not_null() & sigma_e.is_not_null() & _debt().is_not_null()
    return (
        pl.when(has_inputs & (point <= 0)).then(MAX_DISTANCE)
        .when(has_inputs & (equity > 0)).then(distance.clip(-MAX_DISTANCE, MAX_DISTANCE))
        .alias("distance_to_default")
    )  # fmt: skip


def _interest_coverage() -> pl.Expr:
    """Operating income over interest expense; companies without debt get the top."""
    c = pl.col
    no_debt = (_debt() == 0) & (c("interest").is_null() | (c("interest") <= 0))
    coverage = (c("operating_income") / c("interest")).clip(-MAX_COVERAGE, MAX_COVERAGE)
    return (
        pl.when(no_debt).then(MAX_COVERAGE)
        .when(c("interest") > 0).then(coverage)
        .alias("interest_coverage")
    )  # fmt: skip


def roa_stability(states: pl.DataFrame) -> pl.DataFrame:
    """Add ``_roa_volatility`` to each filing state: how much annual ROA has swung.

    (Underscored like the other state columns; the feature is ``roa_volatility``.)

    The standard deviation of annual return on assets over the last
    :data:`STABILITY_YEARS` fiscal years (10-Ks) known when the state was filed.
    """
    annual = (
        states.filter(pl.col("form").str.starts_with("10-K") & (pl.col("assets") > 0))
        .with_columns((pl.col("net_income") / pl.col("assets")).alias("_roa"))
        .drop_nulls("_roa")
        .sort("cik", "period_end", "filed")
        .unique(["cik", "period_end"], keep="last", maintain_order=True)
    )
    annual = annual.with_columns(
        pl.col("_roa")
        .rolling_std(STABILITY_YEARS, min_samples=STABILITY_MIN_YEARS)
        .over("cik")
        .alias("_roa_volatility")
    ).select("cik", pl.col("filed").alias("_stable_filed"), "_roa_volatility")
    return states.sort("filed").join_asof(
        annual.sort("_stable_filed"), left_on="filed", right_on="_stable_filed", by="cik",
        strategy="backward", check_sortedness=False,
    ).drop("_stable_filed")  # fmt: skip


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
    if "interest" not in states.columns:  # states built before interest was extracted
        states = states.with_columns(pl.lit(None, pl.Float64).alias("interest"))
    by_symbol = roa_stability(states).join(tickers, on="cik").drop("accn", "form")
    grid = _asof(grid, by_symbol, MAX_FILING_AGE_DAYS)
    grid = grid.with_columns(_market_value(grid, panel))
    grid = grid.with_columns(_fundamental_features())
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
            pl.when(enough)
            .then(pl.col(c).rank().over(group) / pl.col(c).count().over(group))
            .alias(f"{c}_ind")
            for c in INDUSTRY_RELATIVE
        )
    lead = ["date", "symbol", *(["sic2"] if "sic2" in grid.columns else []), "market_value"]
    rest = [
        c for c in grid.columns
        if not c.startswith("_") and c not in by_symbol.columns and c not in (*lead, "close")
    ]  # fmt: skip
    return grid.select(*lead, *rest).sort("date", "symbol")
