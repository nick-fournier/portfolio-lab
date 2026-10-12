"""JKP characteristics built from monthly returns, ported from ``bkelly-lab/jkp-data``.

Jensen, Kelly & Pedersen (2023) define the 153 characteristics the paper's transformer reads.
This module ports the ones built from monthly returns (``market_chars_monthly``, ``market_beta``,
``residual_momentum``, ``firm_age`` in ``src/jkp/data/aux_functions.py``, commit 666e896),
line for line where our data allows:

- ``ret_{j}_{i}``: compound return over months t-j+1 .. t-i, from a return index in which a
  missing month counts as zero; only month t-i must have a return.
- ``seas_{s}_{e}an`` / ``na``: the average of the monthly returns at lags 12k-1 (same calendar
  month as t+1) / of the other lags in years s..e back; any missing month in the window makes
  both missing.
- ``beta_60m``: slope of monthly excess returns on the market over months t-59..t (at least 36),
  after dropping months with an exactly zero return and winsorizing excess returns across stocks
  each month at 0.1% / 99.9%.
- ``resff3_12_1`` / ``resff3_6_1``: one three-factor regression over months t-35..t (at least 24),
  then mean / standard deviation of its residuals over t-11..t-1 / t-5..t-1.
- ``prc`` (month-end close), ``age`` (months since first seen).

Departures from JKP, forced by our data (each flagged where it happens):

- Monthly returns compound our daily total returns within the calendar month; no delisting
  return is added (a stock's own past returns, so delistings rarely matter here).
- Factors are Ken French's daily US factors (CRSP value-weighted market, SMB, HML; daily T-bill)
  instead of JKP's own per-country versions; monthly factors compound the daily ones.
- ``age`` starts at the first price or filing we have; our prices begin in December 1997, so
  older companies are understated.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

#: Ken French's daily three factors (``ingest/raw/french`` under the data root).
FRENCH = Path("ingest") / "raw" / "french" / "F-F_Research_Data_Factors_daily_CSV.zip"
#: (j, i) momentum and reversal windows (JKP main.py).
MOMENTUM = ((1, 0), (3, 1), (6, 1), (9, 1), (12, 1), (12, 7), (60, 12))
#: (s, e) seasonality windows, in years.
SEASONS = ((1, 1), (2, 5), (6, 10), (11, 15), (16, 20))


def month_key(e: pl.Expr) -> pl.Expr:
    """A date's month as one integer (year * 12 + month - 1)."""
    return e.dt.year().cast(pl.Int32) * 12 + e.dt.month().cast(pl.Int32) - 1


def french(root: Path) -> pl.DataFrame:
    """Ken French's daily factors as fractions: date, mktrf, smb, hml, rf."""
    with zipfile.ZipFile(root / FRENCH) as archive:
        text = archive.read(archive.namelist()[0]).decode()
    rows = [line for line in text.splitlines() if line[:8].strip().isdigit() and "," in line]
    frame = pl.read_csv(
        io.StringIO("\n".join(rows)),
        has_header=False,
        new_columns=["d", "mktrf", "smb", "hml", "rf"],
    )
    return frame.select(
        pl.col("d").cast(pl.String).str.strptime(pl.Date, "%Y%m%d").alias("date"),
        *(pl.col(c).cast(pl.Float64) / 100 for c in ("mktrf", "smb", "hml", "rf")),
    )


def monthly_factors(daily_factors: pl.DataFrame) -> pl.DataFrame:
    """Monthly factors compounded from the daily ones: ym, mktrf, smb, hml, rf.

    The market's excess return is the compounded total market return minus the compounded
    T-bill (French publishes monthly files too; we only keep the daily one).
    """
    compound = lambda e: (1 + e).product() - 1  # noqa: E731
    return (
        daily_factors.group_by(month_key(pl.col("date")).alias("ym"))
        .agg(
            compound(pl.col("mktrf") + pl.col("rf")).alias("mkt"),
            compound(pl.col("rf")).alias("rf"),
            compound(pl.col("smb")).alias("smb"),
            compound(pl.col("hml")).alias("hml"),
        )
        .select("ym", (pl.col("mkt") - pl.col("rf")).alias("mktrf"), "smb", "hml", "rf")
        .sort("ym")
    )


def monthly_returns(daily: pl.DataFrame) -> pl.DataFrame:
    """One row per stock and calendar month, with no gaps between its first and last month.

    Args:
        daily: sid, date, close, ret_cc (our daily total return).

    Returns:
        sid, ym, ``ret`` (compounded daily returns; null in a month with no bars) and ``prc``
        (the month's last close), sorted by sid and month.
    """
    months = (
        daily.with_columns(month_key(pl.col("date")).alias("ym"))
        .sort("sid", "date")
        .group_by("sid", "ym")
        .agg(
            ((1 + pl.col("ret_cc").fill_null(0.0)).product() - 1).alias("ret"),
            pl.col("close").drop_nulls().last().alias("prc"),
        )
    )
    grid = (
        months.group_by("sid")
        .agg(pl.int_ranges(pl.col("ym").min(), pl.col("ym").max() + 1).first().alias("ym"))
        .explode("ym")
    )
    return grid.join(months, on=["sid", "ym"], how="left").sort("sid", "ym")


def momentum_and_seasonality(months: pl.DataFrame) -> pl.DataFrame:
    """``ret_{j}_{i}`` and ``seas_{s}_{e}an/na`` for every stock-month (module docs)."""
    ri = (1 + pl.col("ret").fill_null(0.0)).cum_prod().over("sid")
    count = pl.col("ym").cum_count().over("sid")  # 1-based row number within the stock
    frame = months.with_columns(ri.alias("ri"), count.alias("count"))
    out = []
    for j, i in MOMENTUM:
        ok = (
            (pl.col("ri").shift(j).over("sid") != 0)
            & (pl.col("count") > j)
            & pl.col("ret").shift(i).over("sid").is_not_null()
        )
        value = pl.col("ri").shift(i).over("sid") / pl.col("ri").shift(j).over("sid") - 1
        out.append(pl.when(ok).then(value).alias(f"ret_{j}_{i}"))
    for s, e in SEASONS:
        lags = range((s - 1) * 12, e * 12)
        annual = range(s * 12 - 1, e * 12, 12)
        total = pl.sum_horizontal([pl.col("ret").shift(k).over("sid") for k in lags]).alias("x")
        # sum_horizontal skips nulls; JKP's '+' propagates them: any missing lag -> missing
        missing = pl.any_horizontal([pl.col("ret").shift(k).over("sid").is_null() for k in lags])
        ann = pl.sum_horizontal([pl.col("ret").shift(k).over("sid") for k in annual])
        ok = (pl.col("count") >= e * 12) & ~missing
        n_ann, n_all = len(annual), len(lags)
        out.append(pl.when(ok).then(ann / n_ann).alias(f"seas_{s}_{e}an"))
        out.append(pl.when(ok).then((total - ann) / (n_all - n_ann)).alias(f"seas_{s}_{e}na"))
    return frame.select("sid", "ym", *out)


def _winsorized_excess(months: pl.DataFrame, factors: pl.DataFrame) -> pl.DataFrame:
    """Monthly excess returns for the regressions (JKP ``prep_data_factor_regs``).

    Months with an exactly zero or missing return are dropped; excess returns are clipped at
    the month's 0.1% / 99.9% quantiles across all stocks (JKP: QUANTILE_DISC, all securities).
    """
    data = months.join(factors, on="ym", how="inner").filter(
        pl.col("ret").is_not_null() & (pl.col("ret") != 0) & pl.col("mktrf").is_not_null()
    )
    exc = pl.col("ret") - pl.col("rf")
    lo = exc.quantile(0.001, interpolation="lower").over("ym")
    hi = exc.quantile(0.999, interpolation="higher").over("ym")
    return data.with_columns(exc.clip(lo, hi).alias("ret_exc"))


def beta_60m(months: pl.DataFrame, factors: pl.DataFrame) -> pl.DataFrame:
    """Market beta over months t-59..t, at least 36 months (module docs)."""
    data = _winsorized_excess(months, factors).select("sid", "ym", "ret_exc", "mktrf")
    grid = months.select("sid", "ym").join(data, on=["sid", "ym"], how="left")
    x, y = pl.col("mktrf"), pl.col("ret_exc")
    has = y.is_not_null()
    sums = {
        name: pl.when(has).then(e).otherwise(0.0).rolling_sum(60, min_samples=1).over("sid")
        for name, e in {"n": pl.lit(1.0), "sx": x, "sy": y, "sxy": x * y, "sxx": x * x}.items()
    }
    g = grid.with_columns(**sums)
    cov = pl.col("sxy") - pl.col("sx") * pl.col("sy") / pl.col("n")
    var = pl.col("sxx") - pl.col("sx") ** 2 / pl.col("n")
    return g.select("sid", "ym", pl.when(pl.col("n") >= 36).then(cov / var).alias("beta_60m"))


def residual_momentum(months: pl.DataFrame, factors: pl.DataFrame) -> pl.DataFrame:
    """``resff3_12_1`` and ``resff3_6_1`` (module docs)."""
    data = _winsorized_excess(months, factors).filter(
        pl.col("smb").is_not_null() & pl.col("hml").is_not_null()
    )
    grid = months.select("sid", "ym").join(
        data.select("sid", "ym", "ret_exc", "mktrf", "hml", "smb"), on=["sid", "ym"], how="left"
    )
    out = []
    for (sid,), group in grid.group_by("sid"):
        g = group.sort("ym")
        y = g["ret_exc"].to_numpy().astype(float)
        x = np.column_stack([np.ones(len(g)), g.select("mktrf", "hml", "smb").to_numpy()])
        r12 = np.full(len(g), np.nan)
        r6 = np.full(len(g), np.nan)
        if len(g) >= 36:
            ok = np.isfinite(y)
            yw = sliding_window_view(np.where(ok, y, 0.0), 36)  # (T-35, 36), ends at t
            xw = sliding_window_view(np.where(ok[:, None], x, 0.0), 36, axis=0)  # (T-35, 4, 36)
            mw = sliding_window_view(ok, 36)
            enough = mw.sum(axis=1) >= 24
            xtx = np.einsum("tim,tjm->tij", xw, xw)
            xty = np.einsum("tim,tm->ti", xw, yw)
            solvable = enough & (np.linalg.matrix_rank(xtx) == 4)
            b = np.full((len(yw), 4), np.nan)
            b[solvable] = np.linalg.solve(xtx[solvable], xty[solvable][..., None])[..., 0]
            res = np.where(mw, yw - np.einsum("tim,ti->tm", xw, b), np.nan)
            for incl, target in ((12, r12), (6, r6)):
                part = res[:, 36 - incl : 35]  # months t-incl+1 .. t-1 (position 35 is month t)
                n = np.isfinite(part).sum(axis=1)
                mean = np.nanmean(np.where(n[:, None] > 0, part, 0.0), axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    std = np.nanstd(part, axis=1, ddof=1)
                    target[35:] = np.where(solvable & (n > 1) & (std > 0), mean / std, np.nan)
        out.append(
            pl.DataFrame(
                {"sid": [sid] * len(g), "ym": g["ym"], "resff3_12_1": r12, "resff3_6_1": r6}
            )
        )
    return pl.concat(out)


def age(months: pl.DataFrame, first_filing: pl.DataFrame) -> pl.DataFrame:
    """Months since the stock was first seen (JKP ``firm_age``, with our data's start).

    First seen = the earlier of its first month with a price and December of the year before
    its first filing's period end (JKP's Compustat rule).

    Args:
        months: From :func:`monthly_returns`.
        first_filing: sid, ``first_period`` (date of its earliest filing's period end).
    """
    first = months.group_by("sid").agg(pl.col("ym").min().alias("first_price"))
    first = first.join(
        first_filing.select(
            "sid",
            ((pl.col("first_period").dt.year() - 1) * 12 + 11).cast(pl.Int32).alias("first_acc"),
        ),
        on="sid",
        how="left",
    )
    start = pl.min_horizontal("first_price", "first_acc")
    return (
        months.select("sid", "ym")
        .join(first, on="sid")
        .select("sid", "ym", (pl.col("ym") - start).alias("age"))
    )


def characteristics(
    daily: pl.DataFrame, factors: pl.DataFrame, first_filing: pl.DataFrame
) -> pl.DataFrame:
    """Every monthly-return characteristic for every stock-month: sid, ym, then the columns."""
    months = monthly_returns(daily)
    frames = [
        months.select("sid", "ym", "prc"),
        momentum_and_seasonality(months),
        beta_60m(months, factors),
        residual_momentum(months, factors),
        age(months, first_filing),
    ]
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on=["sid", "ym"], how="left")
    return out
