"""Stock inputs from financial statements, per filing, from Sharadar's raw fundamentals.

Modeled on the accounting characteristics of Green, Hand & Zhang (2017). Two kinds of rows,
both as reported (no restatements) and dated by the filing date:

- ``ART``: trailing-twelve-month figures at every quarterly filing ("annual" items),
  compared with the same figures one and two years earlier.
- ``ARQ``: the quarter's own figures, compared with earlier quarters (up to 16 back).

Items that need the month's market value or industry (valuation ratios, industry
adjustments, concentration) are finished in ``characteristics.build``. Field names follow
Sharadar's (``netinc``, ``assets``, ...); outflows (``capex``, ``ncfdiv``) are made positive.
"""

import zipfile
from pathlib import Path

import polars as pl

#: Sharadar fields read for the annual (trailing-twelve-month) rows.
ANNUAL = ("netinc", "equity", "ncfo", "ncfdiv", "liabilities", "assets", "revenue", "cor",
          "sgna", "intexp", "rnd", "depamor", "ppnenet", "inventory", "receivables",
          "cashneq", "assetsc", "liabilitiesc", "debtc", "debtnc", "capex", "taxexp", "ebit",
          "intangibles", "payables")  # fmt: skip
#: Sharadar fields read for the quarterly rows.
QUARTERLY = ("netinc", "assets", "revenue", "equity", "taxexp", "ppnenet", "cashneq",
             "assetsc", "liabilitiesc", "debtc")  # fmt: skip
OUTFLOWS = ("capex", "ncfdiv")
#: Quarters of history for the volatility inputs, and the fewest they need.
HISTORY, MIN_HISTORY = 16, 8
#: Organization capital: SG&A depreciation rate, starting-stock divisor, and years summed.
ORG_DEPRECIATION, ORG_START, ORG_YEARS = 0.15, 0.25, 10


def read_rows(raw: Path, dimension: str, fields: tuple[str, ...]) -> pl.DataFrame:
    """One row per ticker and fiscal quarter for ``dimension`` (``ART`` or ``ARQ``).

    Returns:
        symbol, filed (filing date), q (quarter number: year * 4 + quarter - 1 of the
        period's calendar date) and the fields as floats.
    """
    cols = ["ticker", "dimension", "calendardate", "date", *fields]
    with zipfile.ZipFile(raw / "fundamentals.zip") as z:
        f = pl.read_csv(z.open(z.namelist()[0]).read(), columns=cols, infer_schema_length=0)
    cal = pl.col("calendardate").str.to_date()
    f = (
        f.filter(pl.col("dimension") == dimension)
        .select(
            pl.col("ticker").alias("symbol"),
            pl.col("date").str.to_date().alias("filed"),
            (cal.dt.year() * 4 + (cal.dt.month() - 1) // 3).alias("q"),
            *[pl.col(c).cast(pl.Float64, strict=False) for c in fields],
        )
        .with_columns(pl.col(c).abs() for c in OUTFLOWS if c in fields)
    )
    # one row per quarter: the earliest filing of it (as first reported)
    return f.sort("filed").unique(["symbol", "q"], keep="first").sort("symbol", "q")


def _filled(rows: pl.DataFrame) -> pl.DataFrame:
    """Rows on a gap-free quarter index per symbol, missing quarters as nulls.

    Shifting by k rows then means k quarters.
    """
    span = rows.group_by("symbol").agg(pl.col("q").min().alias("lo"), pl.col("q").max().alias("hi"))
    grid = span.with_columns(pl.int_ranges("lo", pl.col("hi") + 1).alias("q")).explode("q")
    return grid.select("symbol", "q").join(rows, on=["symbol", "q"], how="left").sort("symbol", "q")


def _lag(name: str, k: int) -> pl.Expr:
    return pl.col(name).shift(k).over("symbol")


def _pct(now: pl.Expr, before: pl.Expr) -> pl.Expr:
    return (now - before) / before


def annual(rows: pl.DataFrame, cpi: pl.DataFrame) -> pl.DataFrame:
    """Annual-item inputs per ART filing (see module docs).

    Args:
        rows: :func:`read_rows` for ``ART`` with :data:`ANNUAL`.
        cpi: date, cpi (consumer price index level), for organization capital.
    """
    f = _filled(rows)
    c = pl.col
    y1 = {n: _lag(n, 4) for n in ANNUAL}  # one year earlier
    y2 = {n: _lag(n, 8) for n in ANNUAL}  # two years earlier
    avg_at = (c("assets") + y1["assets"]) / 2
    ib, ib1 = c("netinc"), y1["netinc"]
    accruals = ib - c("ncfo")
    quick, quick1 = ((c("assetsc") - c("inventory")) / c("liabilitiesc"),
                     (y1["assetsc"] - y1["inventory"]) / y1["liabilitiesc"])  # fmt: skip
    saleinv, saleinv1 = c("revenue") / c("inventory"), y1["revenue"] / y1["inventory"]
    depr, depr1 = c("depamor") / c("ppnenet"), y1["depamor"] / y1["ppnenet"]
    gm, gm1 = c("revenue") - c("cor"), y1["revenue"] - y1["cor"]
    rd_at, rd_at1 = c("rnd") / c("assets"), y1["rnd"] / y1["assets"]
    sales_growth = _pct(c("revenue"), y1["revenue"])
    out = f.with_columns(
        _pct(c("equity"), y1["equity"]).alias("egr"),
        _pct(c("liabilities"), y1["liabilities"]).alias("lgr"),
        ((c("revenue") - c("cor") - c("sgna").fill_null(0) - c("intexp").fill_null(0))
         / y1["equity"]).alias("operprof"),
        (c("ebit") / (c("equity") + c("liabilities") - c("cashneq"))).alias("roic"),
        ((ib + c("depamor")) / ((c("liabilities") + y1["liabilities"]) / 2)).alias("cashdebt"),
        (accruals / avg_at).abs().alias("absacc"),
        (accruals / pl.when(ib == 0).then(0.01).otherwise(ib.abs())).alias("pctacc"),
        ((c("inventory") - y1["inventory"]) / avg_at).alias("chinv"),
        # long-term net operating assets: PP&E and intangibles (other items not in Sharadar)
        ((c("ppnenet") - y1["ppnenet"] + c("intangibles").fill_null(0)
          - y1["intangibles"].fill_null(0) + c("depamor")) / avg_at).alias("grltnoa"),
        (((c("ppnenet") - y1["ppnenet"]) + (c("inventory") - y1["inventory"]))
         / y1["assets"]).alias("invest"),
        _pct(c("capex"), y2["capex"]).alias("grcapx"),
        _pct(c("capex"), y1["capex"]).alias("pchcapx"),
        depr.alias("depr"), _pct(depr, depr1).alias("pchdepr"),
        # R&D up more than 5% (of assets); 0 when it did not rise or there is no prior R&D
        pl.when(c("rnd").is_not_null()).then(
            (_pct(rd_at, rd_at1).fill_nan(None) > 0.05).fill_null(False).cast(pl.Float64)
        ).alias("rd"),
        (c("rnd") / c("revenue")).alias("rd_sale"),
        quick.alias("quick"), _pct(quick, quick1).alias("pchquick"),
        (c("revenue") / c("cashneq")).alias("salecash"),
        saleinv.alias("saleinv"),
        (c("revenue") / c("receivables")).alias("salerec"),
        _pct(saleinv, saleinv1).alias("pchsaleinv"),
        (sales_growth - _pct(c("inventory"), y1["inventory"])).alias("pchsale_pchinvt"),
        (sales_growth - _pct(c("receivables"), y1["receivables"])).alias("pchsale_pchrect"),
        (sales_growth - _pct(c("sgna"), y1["sgna"])).alias("pchsale_pchxsga"),
        (_pct(gm, gm1) - sales_growth).alias("pchgm_pchsale"),
        (c("revenue") / avg_at - y1["revenue"] / ((y1["assets"] + y2["assets"]) / 2))
        .alias("chato"),
        (ib / c("revenue") - ib1 / y1["revenue"]).alias("chpm"),
        ((c("cashneq") + 0.715 * c("receivables") + 0.547 * c("inventory")
          + 0.535 * c("ppnenet")) / c("assets")).alias("tang"),
        pl.when((c("taxexp") > 0) & (ib <= 0)).then(1.0)
        .otherwise(c("taxexp") / 0.35 / ib).alias("tb_raw"),
        ((c("ncfdiv") > 0) & (y1["ncfdiv"].fill_null(0) == 0)).cast(pl.Float64).alias("divi"),
        ((c("ncfdiv").fill_null(0) == 0) & (y1["ncfdiv"] > 0)).cast(pl.Float64).alias("divo"),
    )  # fmt: skip
    out = _orgcap(out, cpi).with_columns((c("orgcap_stock") / avg_at).alias("orgcap"))
    keep = ["egr", "lgr", "operprof", "roic", "cashdebt", "absacc", "pctacc", "chinv", "grltnoa",
            "invest",
            "grcapx", "pchcapx", "depr", "pchdepr", "rd", "rd_sale", "quick", "pchquick",
            "salecash", "saleinv", "salerec", "pchsaleinv", "pchsale_pchinvt",
            "pchsale_pchrect", "pchsale_pchxsga", "pchgm_pchsale", "chato", "chpm", "tang",
            "tb_raw", "divi", "divo", "orgcap", "revenue", "rnd", "capex", "debtnc", "assets",
            "cashneq"]  # fmt: skip
    return out.filter(c("filed").is_not_null()).select("symbol", "filed", "q", *keep)


def _orgcap(f: pl.DataFrame, cpi: pl.DataFrame) -> pl.DataFrame:
    """Capitalized SG&A, as ``orgcap_stock``.

    Real SG&A summed with 15% yearly depreciation over up to :data:`ORG_YEARS` years, the
    oldest year standing in for the stock before it.
    """
    quarter_end = pl.date(pl.col("q") // 4, (pl.col("q") % 4) * 3 + 3, 1).dt.month_end()
    f = f.with_columns(quarter_end.alias("_end")).sort("_end").join_asof(
        cpi.sort("date"), left_on="_end", right_on="date", strategy="backward"
    ).sort("symbol", "q")  # fmt: skip
    real = (pl.col("sgna") / pl.col("cpi")).alias("_real")
    f = f.with_columns(real)
    stock, oldest = pl.lit(0.0), pl.lit(None, dtype=pl.Float64)
    for k in range(ORG_YEARS):
        term = _lag("_real", 4 * k)
        stock = stock + (1 - ORG_DEPRECIATION) ** k * term.fill_null(0)
        # the oldest year starts the stock at SG&A / ORG_START instead of SG&A
        start = (1 - ORG_DEPRECIATION) ** k * term * (1 / ORG_START - 1)
        oldest = pl.when(term.is_not_null()).then(start).otherwise(oldest)
    return f.with_columns(
        pl.when(pl.col("_real").is_not_null())
        .then(stock + oldest.fill_null(0))
        .alias("orgcap_stock")
    ).drop("_end", "_real", "cpi", "date", strict=False)


def quarterly(rows: pl.DataFrame) -> pl.DataFrame:
    """Quarterly-item inputs per ARQ filing (see module docs).

    Args:
        rows: :func:`read_rows` for ``ARQ`` with :data:`QUARTERLY`.
    """
    f = _filled(rows)
    c = pl.col
    lag = _lag
    f = f.with_columns(
        (c("netinc") / lag("assets", 1)).alias("roaq"),
        (c("netinc") / lag("equity", 1)).alias("roeq"),
        ((c("taxexp") - lag("taxexp", 4)) / lag("assets", 4)).alias("chtx"),
        (c("revenue") - lag("revenue", 4)).alias("d_revenue_q"),
        (c("cashneq") / c("assets")).alias("cash"),
        ((c("ppnenet") - lag("ppnenet", 1)) / c("revenue")).alias("_inv"),
    )
    sales = pl.when(c("revenue") > 0).then(c("revenue")).otherwise(0.01)

    def change(name: str) -> pl.Expr:
        return c(name) - lag(name, 1)

    working = (change("assetsc") - change("cashneq")) - (change("liabilitiesc") - change("debtc"))
    sacc = working / sales
    f = f.with_columns(sacc.alias("_sacc")).with_columns(
        (c("netinc") / sales - c("_sacc")).alias("_scf"),
        (c("_inv") - (lag("_inv", 1) + lag("_inv", 2) + lag("_inv", 3)) / 3).alias("cinvest"),
    )
    up = [(lag("netinc", k) > lag("netinc", k + 1)).fill_null(False) for k in range(8)]
    run, nincr = pl.lit(True), pl.lit(0)
    for u in up:
        run = run & u
        nincr = nincr + run.cast(pl.Int32)
    f = f.with_columns(nincr.cast(pl.Float64).alias("nincr"))
    for name, src in (("roavol", "roaq"), ("stdacc", "_sacc"), ("stdcf", "_scf")):
        hist = pl.concat_list([lag(src, k) for k in range(HISTORY)])
        f = f.with_columns(
            pl.when(hist.list.drop_nulls().list.len() >= MIN_HISTORY)
            .then(hist.list.drop_nulls().list.std()).alias(name))  # fmt: skip
    keep = ["roaq", "roeq", "chtx", "d_revenue_q", "cash", "cinvest", "nincr", "roavol",
            "stdacc", "stdcf"]  # fmt: skip
    return f.filter(c("filed").is_not_null()).select("symbol", "filed", "q", *keep)
