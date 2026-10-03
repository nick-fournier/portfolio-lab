"""Stock inputs from daily prices and volume, per stock per month end.

Modeled on the price- and volume-based characteristics of Green, Hand & Zhang (2017), as
used by Gu, Kelly & Xiu (2020); each uses only data up to the month end:

- From the month's daily bars: ``maxret`` (largest daily return), ``retvol`` (sd of daily
  returns), ``baspread`` (average (high - low) / midpoint, a spread estimate from daily
  ranges), ``ill`` (Amihud illiquidity: average |return| / dollar volume), ``std_dolvol``
  (sd of log daily dollar volume), ``std_turn`` (sd of daily volume / shares),
  ``zerotrade`` (zero-volume days, turnover-adjusted, per 21 trading days).
- From monthly returns: ``mom6m`` (months 2-6 back), ``mom36m`` (months 13-36 back),
  ``chmom`` (months 1-6 back minus months 7-12 back), ``turn`` (average monthly volume
  of the last three months / shares), ``age`` (years since the first traded month).
- From weekly returns over three years (at least 52 weeks) against the equal-weighted
  market: ``beta``, ``betasq``, ``idiovol`` (sd of the residuals) and ``pricedelay``
  (1 - adjusted R² without / with four lagged weeks of the market).

Shares outstanding at the month end are market value / the month's last close (raw),
which carries the filed share count through later splits (``research.features``).
"""

import numpy as np
import polars as pl

#: Weekly-regression window and the fewest weeks it needs.
WEEKS, MIN_WEEKS = 156, 52
#: Lagged market weeks in the price-delay regression.
DELAY_LAGS = 4


def _daily(prices: pl.LazyFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """Daily bars tagged with the month end they belong to (the grid's dates)."""
    ends = grid.select(pl.col("date").alias("end")).unique().with_columns(
        pl.col("end").dt.year().alias("_y"), pl.col("end").dt.month().alias("_m"))  # fmt: skip
    return (
        prices.select("symbol", "date", "high", "low", "close", "volume", "ret_cc")
        .with_columns(pl.col("date").dt.year().alias("_y"), pl.col("date").dt.month().alias("_m"))
        .collect()
        .join(ends, on=["_y", "_m"])
        .drop("_y", "_m")
    )


def monthly_bars(prices: pl.LazyFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """Per symbol and month end: the month's return and its daily-bar statistics.

    Args:
        prices: Daily bars (symbol, date, high, low, close, volume, ret_cc), raw levels.
        grid: The monthly feature grid (date, symbol, market_value).

    Returns:
        symbol, date (month end), ret, volume, maxret, retvol, baspread, ill, std_dolvol,
        std_vol, zero_days, days, close_end (last close of the month).
    """
    d = _daily(prices, grid)
    dollar = pl.col("close") * pl.col("volume")
    traded = pl.col("volume") > 0
    return d.group_by("symbol", pl.col("end").alias("date")).agg(
        ((1 + pl.col("ret_cc").fill_null(0.0)).product() - 1).alias("ret"),
        pl.col("volume").sum().alias("volume"),
        pl.col("close").sort_by("date").last().alias("close_end"),
        pl.col("ret_cc").max().alias("maxret"),
        pl.col("ret_cc").std().alias("retvol"),
        ((pl.col("high") - pl.col("low")) / ((pl.col("high") + pl.col("low")) / 2)).mean()
        .alias("baspread"),
        (pl.col("ret_cc").abs() / dollar).filter(traded).mean().alias("ill"),
        dollar.filter(traded).log().std().alias("std_dolvol"),
        pl.col("volume").std().alias("std_vol"),
        (~traded).sum().alias("zero_days"),
        pl.len().alias("days"),
    )  # fmt: skip


def _product(r: pl.Expr, start: int, stop: int) -> pl.Expr:
    """Compounded return over months ``start``..``stop`` back (0 = this month); null if any is."""
    terms = [(1 + r.shift(k)) for k in range(start, stop + 1)]
    out = terms[0]
    for t in terms[1:]:
        out = out * t
    return out - 1


def monthly_inputs(bars: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """Momentum variants, turnover and age from :func:`monthly_bars`, per grid row.

    Months are consecutive month ends; a month a stock has no bars counts as missing, so
    windows that include it are null.
    """
    ends = sorted(grid["date"].unique().to_list())
    order = pl.DataFrame({"date": ends, "_k": range(len(ends))})
    full = (
        bars.select("symbol").unique().join(order, how="cross")
        .join(bars, on=["symbol", "date"], how="left")
        .join(grid.select("date", "symbol", "market_value"), on=["date", "symbol"], how="left")
        .with_columns((pl.col("market_value") / pl.col("close_end")).alias("shares"))
        .sort("symbol", "_k")
    )  # fmt: skip
    r = pl.col("ret")
    first = pl.when(pl.col("days").is_not_null()).then(pl.col("_k")).min().over("symbol")
    vol3 = (pl.col("volume").shift(0) + pl.col("volume").shift(1) + pl.col("volume").shift(2)) / 3
    out = full.with_columns(
        _product(r, 1, 5).over("symbol").alias("mom6m"),
        _product(r, 12, 35).over("symbol").alias("mom36m"),
        (_product(r, 0, 5) - _product(r, 6, 11)).over("symbol").alias("chmom"),
        (vol3.over("symbol") / pl.col("shares")).alias("turn"),
        ((pl.col("_k") - first) / 12).alias("age"),
        (pl.col("std_vol") / pl.col("shares")).alias("std_turn"),
    ).with_columns(
        ((pl.col("zero_days") + 1 / (pl.col("volume") / pl.col("shares")) / 480_000)
         * 21 / pl.col("days")).alias("zerotrade"),
    )  # fmt: skip
    keep = ["mom6m", "mom36m", "chmom", "turn", "std_turn", "zerotrade", "age", "maxret",
            "retvol", "baspread", "ill", "std_dolvol"]  # fmt: skip
    return grid.select("date", "symbol").join(
        out.select("date", "symbol", *keep), on=["date", "symbol"], how="left"
    )


def weekly_returns(prices: pl.LazyFrame) -> tuple[np.ndarray, np.ndarray, list[str], list]:
    """Weekly returns (weeks x symbols) and the equal-weighted market's weekly return.

    Weeks end on Friday; a symbol's week is missing if it had no bars that week.
    """
    w = (
        prices.select("symbol", "date", "ret_cc")
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("symbol", "week")
        .agg(((1 + pl.col("ret_cc").fill_null(0.0)).product() - 1).alias("r"))
        .collect()
    )
    weeks = sorted(w["week"].unique().to_list())
    symbols = sorted(w["symbol"].unique().to_list())
    wi = {d: k for k, d in enumerate(weeks)}
    si = {s: k for k, s in enumerate(symbols)}
    mat = np.full((len(weeks), len(symbols)), np.nan, dtype=np.float64)
    mat[[wi[d] for d in w["week"]], [si[s] for s in w["symbol"]]] = w["r"].to_numpy()
    mat[np.abs(mat) > 5] = np.nan  # bad prints
    market = np.nanmean(mat, axis=1)
    return mat, market, symbols, weeks


def _adj_r2(ssr: np.ndarray, sst: np.ndarray, n: np.ndarray, p: int) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1 - (ssr / (n - p)) / (sst / (n - 1))


def regressions(
    grid: pl.DataFrame, mat: np.ndarray, market: np.ndarray, symbols: list[str], weeks: list
) -> pl.DataFrame:
    """beta, betasq, idiovol, pricedelay per grid row (see module docs)."""
    col = {s: k for k, s in enumerate(symbols)}
    week_end = np.array(weeks, dtype="datetime64[D]") + 6
    lags = np.column_stack(
        [np.r_[np.full(k, np.nan), market[:-k]] for k in range(1, DELAY_LAGS + 1)]
    )
    rows = []
    for (day,), month in grid.group_by("date"):
        stop = int(np.searchsorted(week_end, np.datetime64(day, "D"), side="right"))
        lo = max(0, stop - WEEKS)
        names = [s for s in month["symbol"].to_list() if s in col]
        if stop - lo < MIN_WEEKS or not names:
            continue
        y = mat[lo:stop, [col[s] for s in names]]
        m = market[lo:stop]
        ok = np.isfinite(y) & np.isfinite(m)[:, None]
        n = ok.sum(0).astype(float)
        yz, mz = np.where(ok, y, 0.0), np.where(ok, m[:, None], 0.0)
        sm, sy = mz.sum(0), yz.sum(0)
        smm, smy, syy = (mz * mz).sum(0), (mz * yz).sum(0), (yz * yz).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = (smy - sm * sy / n) / (smm - sm * sm / n)
            alpha = (sy - beta * sm) / n
            sst = syy - sy * sy / n
            ssr = syy - alpha * sy - beta * smy
            idiovol = np.sqrt(ssr / (n - 1))
        r2_1 = _adj_r2(ssr, sst, n, 2)
        # market plus lagged market weeks: batched normal equations over the valid weeks
        x = np.column_stack([np.ones(stop - lo), m, lags[lo:stop]])
        ok5 = ok & np.isfinite(x).all(1)[:, None]
        xz = np.nan_to_num(x)
        w5 = ok5.astype(float)
        xtx = np.einsum("ws,wi,wj->sij", w5, xz, xz)
        xty = np.einsum("ws,wi,ws->si", w5, xz, np.where(ok5, y, 0.0))
        n5 = w5.sum(0)
        y5 = np.where(ok5, y, 0.0)
        good = n5 >= MIN_WEEKS
        b = np.full((len(names), x.shape[1]), np.nan)
        if good.any():
            ridge = 1e-12 * np.eye(x.shape[1])  # keeps singular windows solvable
            b[good] = np.linalg.solve(xtx[good] + ridge, xty[good][..., None])[..., 0]
        ssr5 = (y5 * y5).sum(0) - np.einsum("si,si->s", b, xty)
        sst5 = (y5 * y5).sum(0) - y5.sum(0) ** 2 / n5
        r2_5 = _adj_r2(ssr5, sst5, n5, x.shape[1])
        enough = n >= MIN_WEEKS
        with np.errstate(invalid="ignore", divide="ignore"):
            delay = 1 - r2_1 / r2_5
        rows.append(pl.DataFrame({
            "date": [day] * len(names), "symbol": names,
            "beta_w": np.where(enough, beta, np.nan), "idiovol": np.where(enough, idiovol, np.nan),
            "pricedelay": np.where(enough & good, delay, np.nan),
        }))  # fmt: skip
    out = pl.concat(rows).with_columns(
        (pl.col("beta_w") ** 2).alias("betasq"), pl.all().exclude("date", "symbol").fill_nan(None)
    )
    return grid.select("date", "symbol").join(out, on=["date", "symbol"], how="left")
