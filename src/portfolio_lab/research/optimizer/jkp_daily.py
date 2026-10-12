"""JKP characteristics built from daily data, ported from ``bkelly-lab/jkp-data``.

Ports ``prepare_daily``, the rolling daily statistics (aux_functions.py lines 9253-9723),
``finish_daily_chars`` and ``bidask_hl`` (commit 666e896). JKP's ``_21d / _126d / _252d /
_1260d`` windows are calendar months: 1, 6, 12 and 60 months ending with month t, needing at
least 15, 60, 120 and 750 trading days. Each statistic here is built from per-stock-month sums
that are added up over the window, which is the same arithmetic.

- Return statistics use daily excess returns (total return minus the daily T-bill) on days the
  market traded: ``rvol``, ``rmax1/5`` (total returns), ``rskew``, CAPM residual volatility,
  skewness and coskewness (1 month), CAPM residual volatility (12 months), three-factor residual
  volatility (ddof 3, as JKP) and skewness, downside beta (12 months, at least 60 down days),
  Dimson beta (1 month; lagged, same-day and next-day market, next day within the month),
  market correlation of 3-day summed returns (60 months, at least 750 days) and the
  Frazzini-Pedersen beta = correlation x stock volatility / market volatility (12 months, no
  shrinkage, as JKP).
- Trading statistics: dollar volume and its variability, Amihud illiquidity, turnover and its
  variability (6 months); zero-volume days x 21 with a turnover tie-break (1, 6, 12 months);
  the Corwin-Schultz high-low spread (1 month, more than 10 valid days).
- ``prc_highprc_252d``: the latest adjusted close over the highest in 12 months.

Departures from JKP, forced by our data:

- Factors are Ken French's daily US factors and daily T-bill, not JKP's own.
- Adjusted prices use our split-and-dividend-adjusted close (JKP divides by the share factor
  only), so the 52-week high ratio is slightly lower for dividend payers.
- Turnover's shares outstanding are the month's market value over its last close (our market
  value carries the latest filing's share count through splits); JKP uses daily CRSP shares.
- The NASDAQ volume adjustment uses our listing history's exchange; the zero-trades tie-break
  ranks turnover across the stocks we cover, not JKP's global file.
"""

import numpy as np
import polars as pl

from portfolio_lab.research.optimizer.jkp_monthly import month_key

#: Window lengths in months and their minimum days (JKP config.py ROLLING_DAILY_SPECS).
WINDOWS = {1: 15, 6: 60, 12: 120, 60: 750}


def _nasdaq_divisor() -> pl.Expr:
    """JKP's NASDAQ volume adjustment: /2 before Feb 2001, /1.8 in 2001, /1.6 in 2002-2003."""
    return (
        pl.when(~pl.col("nasdaq"))
        .then(1.0)
        .when(pl.col("date") < pl.date(2001, 2, 1))
        .then(2.0)
        .when(pl.col("date") <= pl.date(2001, 12, 31))
        .then(1.8)
        .when(pl.col("date") <= pl.date(2003, 12, 31))
        .then(1.6)
        .otherwise(1.0)
    )


def prepare(
    daily: pl.DataFrame, factors: pl.DataFrame, listings: pl.DataFrame, shares: pl.DataFrame
) -> pl.DataFrame:
    """Daily rows on market trading days with everything the statistics need.

    Args:
        daily: sid, date, high, low, close, volume, close_adj, ret_cc.
        factors: From ``jkp_monthly.french``.
        listings: sid, from, to, exchange (our listing history).
        shares: sid, ym, shares (shares outstanding for the month).
    """
    daily = daily.with_columns(pl.col("sid").cast(pl.Int64))
    listings = listings.with_columns(pl.col("sid").cast(pl.Int64))
    shares = shares.with_columns(pl.col("sid").cast(pl.Int64))
    on_nasdaq = (
        daily.select("sid", "date")
        .sort("date")
        .join_asof(
            listings.select("sid", pl.col("from").alias("date"), "to", "exchange").sort("date"),
            on="date",
            by="sid",
            strategy="backward",
        )
        .select(
            "sid",
            "date",
            (
                (pl.col("exchange") == "NASDAQ")(pl.col("exchange") == "NASDAQ")
                & (pl.col("date") <= pl.col("to"))(
                    pl.col("to").is_null() | (pl.col("date") <= pl.col("to"))
                )
            )
            .fill_null(False)
            .alias("nasdaq"),
        )
    )
    d = daily.join(factors, on="date", how="inner").join(on_nasdaq, on=["sid", "date"], how="left")
    d = d.with_columns(month_key(pl.col("date")).alias("ym"), pl.col("nasdaq").fill_null(False))
    tvol = pl.col("volume") / _nasdaq_divisor()
    adj = pl.col("close_adj") / pl.col("close")
    d = d.with_columns(
        (pl.col("ret_cc") - pl.col("rf")).alias("ret_exc"),
        tvol.alias("tvol"),
        (pl.col("close").abs() * tvol).alias("dolvol_d"),
        pl.col("close_adj").alias("prc_adj"),
        (pl.col("high") * adj).alias("high_adj"),
        (pl.col("low") * adj).alias("low_adj"),
    )
    d = d.join(shares, on=["sid", "ym"], how="left").with_columns(
        pl.when(pl.col("shares") > 0).then(pl.col("tvol") / pl.col("shares")).alias("turnover_d")
    )
    return d.sort("sid", "date")


def _ols_by_month(d: pl.DataFrame, y: str, xs: list[str]) -> pl.DataFrame:
    """Per stock-month OLS of ``y`` on an intercept and ``xs``: sid, ym, n, b0..bk."""
    cols = ["one", *xs]
    data = d.with_columns(pl.lit(1.0).alias("one")).drop_nulls([y, *xs])
    sums = data.group_by("sid", "ym").agg(
        pl.len().alias("n"),
        *((pl.col(a) * pl.col(b)).sum().alias(f"{a}*{b}") for a in cols for b in cols),
        *((pl.col(a) * pl.col(y)).sum().alias(f"{a}*y") for a in cols),
    )
    k = len(cols)
    xtx = sums.select([f"{a}*{b}" for a in cols for b in cols]).to_numpy().reshape(-1, k, k)
    xty = sums.select([f"{a}*y" for a in cols]).to_numpy()
    b = np.full((len(sums), k), np.nan)
    ok = np.abs(np.linalg.det(xtx)) > 1e-18
    b[ok] = np.linalg.solve(xtx[ok], xty[ok][..., None])[..., 0]
    return sums.select("sid", "ym", "n").with_columns(pl.Series(f"b{i}", b[:, i]) for i in range(k))


def one_month(d: pl.DataFrame, market_days: pl.DataFrame) -> pl.DataFrame:
    """The 1-month statistics, one row per stock-month (module docs)."""
    r = d.filter(pl.col("ret_exc").is_not_null())
    g = ["sid", "ym"]
    m = pl.col("mktrf")
    beta = pl.cov("ret_exc", "mktrf").over(g) / m.var().over(g)
    alpha = pl.col("ret_exc").mean().over(g) - beta * m.mean().over(g)
    capm = r.with_columns(
        (pl.col("ret_exc") - alpha - beta * m).alias("res"), (m - m.mean().over(g)).alias("dm")
    )
    stats = capm.group_by(g).agg(
        pl.col("ret_exc").count().alias("n"),
        pl.col("ret_exc").std().alias("rvol_21d"),
        pl.col("ret_cc").max().alias("rmax1_21d"),
        pl.col("ret_cc").top_k(5).mean().alias("rmax5_21d"),
        pl.col("ret_exc").skew(bias=False).alias("rskew_21d"),
        pl.col("res").std().alias("ivol_capm_21d"),
        pl.col("res").skew(bias=False).alias("iskew_capm_21d"),
        (
            (pl.col("res") * pl.col("dm") ** 2).mean()
            / ((pl.col("res") ** 2).mean().sqrt() * (pl.col("dm") ** 2).mean())
        ).alias("coskew_21d"),
    )
    # three-factor residuals: count checked on all rows, regression on rows with factors
    ff3 = _ols_by_month(r, "ret_exc", ["mktrf", "smb", "hml"])
    res = (
        r.join(ff3, on=g)
        .filter(pl.col("smb").is_not_null() & pl.col("hml").is_not_null())
        .with_columns(
            (
                pl.col("ret_exc")
                - pl.col("b0")
                - pl.col("b1") * pl.col("mktrf")
                - pl.col("b2") * pl.col("smb")
                - pl.col("b3") * pl.col("hml")
            ).alias("res")
        )
    )
    ff3_stats = res.group_by(g).agg(
        pl.col("res").std(ddof=3).alias("ivol_ff3_21d"),
        pl.col("res").skew(bias=False).alias("iskew_ff3_21d"),
    )
    # Dimson: lag crosses months, lead stays within the month (so the month's last day drops)
    dims = r.join(market_days, on="date", how="inner")
    dim = _ols_by_month(dims, "ret_exc", ["mktrf_lg1", "mktrf", "mktrf_ld1"]).select(
        *g, (pl.col("b1") + pl.col("b2") + pl.col("b3")).alias("beta_dimson_21d")
    )
    out = stats.join(ff3_stats, on=g, how="left").join(dim, on=g, how="left")
    keep = [
        "rvol_21d",
        "rmax1_21d",
        "rmax5_21d",
        "rskew_21d",
        "ivol_capm_21d",
        "iskew_capm_21d",
        "coskew_21d",
        "ivol_ff3_21d",
        "iskew_ff3_21d",
        "beta_dimson_21d",
    ]
    return out.select(*g, *(pl.when(pl.col("n") >= 15).then(pl.col(c)).alias(c) for c in keep))


def month_sums(d: pl.DataFrame) -> pl.DataFrame:
    """Per stock-month sums that the multi-month statistics add up over their windows."""
    has_r = pl.col("ret_exc").is_not_null()
    r, m = pl.col("ret_exc"), pl.col("mktrf")
    down = has_r & (m < 0)
    r3 = (r + r.shift(1) + r.shift(2)).over("sid")  # 3-day sums by the stock's own rows
    m3 = (m + m.shift(1) + m.shift(2)).over("sid")
    d = d.with_columns(r3.alias("r3"), m3.alias("m3"))
    ok3 = pl.col("r3").is_not_null() & pl.col("m3").is_not_null()
    s = lambda cond, e: pl.when(cond).then(e).otherwise(0.0).sum()  # noqa: E731
    dv = pl.col("dolvol_d")
    return d.group_by("sid", "ym").agg(
        s(has_r, 1.0).alias("n"),
        s(has_r, r).alias("sr"),
        s(has_r, m).alias("sm"),
        s(has_r, r * r).alias("srr"),
        s(has_r, m * m).alias("smm"),
        s(has_r, r * m).alias("srm"),
        s(down, 1.0).alias("dn"),
        s(down, r).alias("dsr"),
        s(down, m).alias("dsm"),
        s(down, m * m).alias("dsmm"),
        s(down, r * m).alias("dsrm"),
        s(ok3, 1.0).alias("n3"),
        s(ok3, pl.col("r3")).alias("s3r"),
        s(ok3, pl.col("m3")).alias("s3m"),
        s(ok3, pl.col("r3") ** 2).alias("s3rr"),
        s(ok3, pl.col("m3") ** 2).alias("s3mm"),
        s(ok3, pl.col("r3") * pl.col("m3")).alias("s3rm"),
        s(dv.is_not_null(), 1.0).alias("nd"),
        s(dv.is_not_null(), dv).alias("sd"),
        s(dv.is_not_null(), dv * dv).alias("sdd"),
        s(has_r & (dv > 0), pl.col("ret_cc").abs() / dv * 1e6).alias("sami"),
        s(has_r & (dv > 0), 1.0).alias("nami"),
        s(pl.col("tvol").is_not_null(), 1.0).alias("nt"),
        s(pl.col("tvol") == 0, 1.0).alias("nzero"),
        s(pl.col("turnover_d").is_not_null(), 1.0).alias("nto"),
        s(pl.col("turnover_d").is_not_null(), pl.col("turnover_d")).alias("sto"),
        s(pl.col("turnover_d").is_not_null(), pl.col("turnover_d") ** 2).alias("stoo"),
        s(has_r & pl.col("prc_adj").is_not_null(), 1.0).alias("np"),
        pl.col("prc_adj").filter(has_r).max().alias("pmax"),
        pl.col("prc_adj").filter(has_r).drop_nulls().last().alias("plast"),
    )


def _std(n: pl.Expr, s: pl.Expr, ss: pl.Expr) -> pl.Expr:
    return ((ss - s * s / n) / (n - 1)).sqrt()


def multi_month(sums: pl.DataFrame) -> pl.DataFrame:
    """The 6-, 12- and 60-month statistics from :func:`month_sums` (module docs)."""
    grid = (
        sums.group_by("sid")
        .agg(pl.int_ranges(pl.col("ym").min(), pl.col("ym").max() + 1).first().alias("ym"))
        .explode("ym")
    )
    g = grid.join(sums, on=["sid", "ym"], how="left").sort("sid", "ym")
    value_cols = [c for c in sums.columns if c not in ("sid", "ym", "pmax", "plast")]
    g = g.with_columns(
        pl.col(value_cols).fill_null(0.0), pl.col("plast").forward_fill().over("sid")
    )
    w = {
        k: {c: pl.col(c).rolling_sum(k, min_samples=1).over("sid") for c in value_cols}
        for k in (6, 12, 60)
    }
    a6, a12, a60 = w[6], w[12], w[60]
    rvol12 = _std(a12["n"], a12["sr"], a12["srr"])
    mvol12 = _std(a12["n"], a12["sm"], a12["smm"])
    cov12 = a12["srm"] - a12["sr"] * a12["sm"] / a12["n"]
    var12 = a12["smm"] - a12["sm"] ** 2 / a12["n"]
    beta12 = cov12 / var12
    # residual r - beta*m (no alpha): its sample std
    res_ss = (a12["srr"] - a12["sr"] ** 2 / a12["n"]) - 2 * beta12 * cov12 + beta12**2 * var12
    dcov = a12["dsrm"] - a12["dsr"] * a12["dsm"] / a12["dn"]
    dvar = a12["dsmm"] - a12["dsm"] ** 2 / a12["dn"]
    c3 = a60["s3rm"] - a60["s3r"] * a60["s3m"] / a60["n3"]
    v3r = a60["s3rr"] - a60["s3r"] ** 2 / a60["n3"]
    v3m = a60["s3mm"] - a60["s3m"] ** 2 / a60["n3"]
    corr = c3 / (v3r * v3m).sqrt()
    pmax12 = pl.col("pmax").rolling_max(12, min_samples=1).over("sid")
    ok12, ok6 = a12["n"] >= 120, a6["n"] >= 60
    mean_to = a6["sto"] / a6["nto"]
    mean_dv = a6["sd"] / a6["nd"]
    out = g.select(
        "sid",
        "ym",
        pl.when(ok12).then(rvol12).alias("rvol_252d"),
        pl.when(ok12).then((res_ss / (a12["n"] - 1)).sqrt()).alias("ivol_capm_252d"),
        pl.when(ok12 & (a12["dn"] >= 60)).then(dcov / dvar).alias("betadown_252d"),
        pl.when(a60["n3"] >= 750).then(corr).alias("corr_1260d"),
        pl.when(ok12 & (a60["n3"] >= 750)).then(corr * rvol12 / mvol12).alias("betabab_1260d"),
        pl.when(ok12 & (a12["np"] >= 120)).then(pl.col("plast") / pmax12).alias("prc_highprc_252d"),
        pl.when(ok6 & (a6["nd"] >= 60)).then(a6["sami"] / a6["nami"]).alias("ami_126d"),
        pl.when(a6["nd"] >= 60).then(mean_dv).alias("dolvol_126d"),
        pl.when((a6["nd"] >= 60) & (mean_dv != 0))
        .then(_std(a6["nd"], a6["sd"], a6["sdd"]) / mean_dv)
        .alias("dolvol_var_126d"),
        pl.when(a6["nto"] >= 60).then(mean_to).alias("turnover_126d"),
        pl.when((a6["nto"] >= 60) & (mean_to != 0))
        .then(_std(a6["nto"], a6["sto"], a6["stoo"]) / mean_to)
        .alias("turnover_var_126d"),
        *(
            pl.when(w_["nt"] >= WINDOWS[k]).then(w_["nzero"] / w_["nt"] * 21).alias(f"zt_{k}")
            for k, w_ in ((1, {c: pl.col(c) for c in value_cols}), (6, a6), (12, a12))
        ),
        *(
            pl.when(w_["nto"] > 0).then(w_["sto"] / w_["nto"]).alias(f"to_{k}")
            for k, w_ in ((1, {c: pl.col(c) for c in value_cols}), (6, a6), (12, a12))
        ),
    )
    return out


def zero_trades(frame: pl.DataFrame) -> pl.DataFrame:
    """``zero_trades_21d/126d/252d``: zero-volume share x 21 plus JKP's turnover tie-break.

    The tie-break is each stock's turnover rank (highest first, ties averaged) over the count
    of stocks with turnover in that month, divided by 100; null turnover gives null.
    """
    cols = []
    for k, name in ((1, "21d"), (6, "126d"), (12, "252d")):
        to = pl.col(f"to_{k}")
        rank = to.rank(descending=True, method="average").over("ym") / to.count().over("ym")
        cols.append(
            pl.when(to.is_not_null())
            .then(pl.col(f"zt_{k}") + rank / 100)
            .alias(f"zero_trades_{name}")
        )
    return frame.with_columns(cols).drop([c for c in frame.columns if c[:3] in ("zt_", "to_")])


def bidask(d: pl.DataFrame) -> pl.DataFrame:
    """``bidaskhl_21d``: Corwin-Schultz from adjusted highs and lows (JKP ``bidask_hl``)."""
    hi, lo, p = pl.col("high_adj"), pl.col("low_adj"), pl.col("prc_adj")
    x = d.select("sid", "date", "ym", "tvol", hi.alias("h"), lo.alias("l"), p.alias("p"))
    bad = (
        (pl.col("tvol") == 0)
        | (pl.col("l") <= 0)
        | (pl.col("h") <= 0)
        | (pl.col("l") == pl.col("h"))
    )
    x = x.with_columns(
        pl.when(bad).then(None).otherwise(pl.col("h")).alias("h"),
        pl.when(bad).then(None).otherwise(pl.col("l")).alias("l"),
    )
    valid = (pl.col("l") > 0) & (pl.col("h") > pl.col("l"))
    x = x.with_columns(
        pl.when(valid).then(pl.col("l")).forward_fill().shift(1).over("sid").alias("lr"),
        pl.when(valid).then(pl.col("h")).forward_fill().shift(1).over("sid").alias("hr"),
    )
    # an invalid day takes the last valid range, moved to cover the day's close
    lr, hr, p_ = pl.col("lr"), pl.col("hr"), pl.col("p")
    x = x.with_columns(
        pl.when(valid)
        .then(pl.col("l"))
        .when((p_ >= lr) & (p_ <= hr))
        .then(lr)
        .when(p_ < lr)
        .then(p_)
        .otherwise(lr + (p_ - hr))
        .alias("l"),
        pl.when(valid)
        .then(pl.col("h"))
        .when((p_ >= lr) & (p_ <= hr))
        .then(hr)
        .when(p_ < lr)
        .then(hr - (lr - p_))
        .otherwise(p_)
        .alias("h"),
    )
    x = x.with_columns(
        pl.when(pl.col("h") / pl.col("l") > 8).then(None).otherwise(pl.col("l")).alias("l"),
        pl.when(pl.col("h") / pl.col("l") > 8).then(None).otherwise(pl.col("h")).alias("h"),
    )
    prev_p = pl.col("p").shift(1).over("sid")
    h1, l1 = pl.col("h").shift(1).over("sid"), pl.col("l").shift(1).over("sid")
    h_t = (
        pl.when((prev_p > 0) & (prev_p < pl.col("l")))
        .then(pl.col("h") - (pl.col("l") - prev_p))
        .when(prev_p > pl.col("h"))
        .then(prev_p)
        .otherwise(pl.col("h"))
    )
    l_t = (
        pl.when((prev_p > 0) & (prev_p < pl.col("l")))
        .then(prev_p)
        .when(prev_p > pl.col("h"))
        .then(pl.col("l") + (prev_p - pl.col("h")))
        .otherwise(pl.col("l"))
    )
    x = x.with_columns(h_t.alias("ht"), l_t.alias("lt"), h1.alias("h1"), l1.alias("l1"))
    h2 = pl.max_horizontal("ht", "h1")
    l2 = pl.min_horizontal("lt", "l1")
    beta = (pl.col("ht") / pl.col("lt")).log() ** 2 + (pl.col("h1") / pl.col("l1")).log() ** 2
    gamma = (h2 / l2).log() ** 2
    const = 3 - 2 * np.sqrt(2)
    alpha = ((np.sqrt(2) - 1) * beta.sqrt()) / const - (gamma / const).sqrt()
    spread = 2 * (alpha.exp() - 1) / (1 + alpha.exp())
    ok = (pl.col("lt") > 0) & (pl.col("l1") > 0) & (l2 > 0)
    x = x.with_columns(pl.when(ok).then(pl.max_horizontal(spread, pl.lit(0.0))).alias("s"))
    return x.group_by("sid", "ym").agg(
        pl.when(pl.col("s").count() > 10).then(pl.col("s").mean()).alias("bidaskhl_21d")
    )


def market_days(factors: pl.DataFrame) -> pl.DataFrame:
    """Each market day's lagged and next-day market return (next day within the month only)."""
    f = factors.sort("date").with_columns(month_key(pl.col("date")).alias("ym"))
    return f.select(
        "date",
        pl.col("mktrf").shift(1).alias("mktrf_lg1"),
        pl.col("mktrf").shift(-1).over("ym").alias("mktrf_ld1"),
    )


def characteristics(d: pl.DataFrame, factors: pl.DataFrame) -> pl.DataFrame:
    """Every daily characteristic for every stock-month: sid, ym, then the columns.

    ``zero_trades_*`` still needs :func:`zero_trades` across all stocks (it ranks them).
    """
    one = one_month(d, market_days(factors))
    multi = multi_month(month_sums(d))
    out = multi.join(one, on=["sid", "ym"], how="left").join(
        bidask(d), on=["sid", "ym"], how="left"
    )
    return out.with_columns((pl.col("rmax5_21d") / pl.col("rvol_252d")).alias("rmax5_rvol_21d"))
