"""JKP accounting characteristics from our filings (ported from ``bkelly-lab/jkp-data``).

Pipeline (JKP ``create_acc_chars`` + ``combine_ann_qtr_chars``; spec tmp/jkp_fund_spec.md):

1. Two panels per company on a monthly grid from its first to last fiscal period: **annual**
   (our trailing-twelve-month row of each 10-K, i.e. the fiscal year) and **quarterly** (every
   trailing-twelve-month row: TTM flows, quarter-end balance sheet, plus the single-quarter
   revenue, earnings and operating cash flow). Lags are grid-row shifts (12 = a year).
2. JKP's helper variables (:func:`helpers`) and characteristics (:mod:`.jkp_fund_chars`).
3. Publication: a filing is used from 4 months after its period end until the next filing of
   the same panel becomes usable, for at most 18 months.
4. Combine: each month takes the quarterly value if the annual one is missing or the quarterly
   filing is newer, else the annual one.
5. Ratios to market equity use the current month's market value (``*_me``, ``*_mev``,
   ``eq_dur``, ``ival_me``, ``aliq_mat``); ``kz_index`` uses market value at the period end.

Our filings map to Compustat items one to one where they exist (``ITEMS``). Missing items, as
in the spec: preferred stock, deferred taxes, minority interest, long- and short-term
investments set to 0 (or left out of a coalesce); income before extraordinary items = net
income; pretax income = net income + taxes; EBITDA = operating income + depreciation; gross
PP&E = net PP&E; EBIT = operating income (Compustat's EBIT is operating). The original version
of each filing is used (the earliest filed), so later restatements never leak backwards.
Not built (no data): employee growth, sales per employee, Altman Z, long- and short-term
investment changes, equity issuance and payout ratios.
"""

from pathlib import Path

import polars as pl

from portfolio_lab.data import reader
from portfolio_lab.research.optimizer import jkp_fund_chars as fc
from portfolio_lab.research.optimizer.jkp_monthly import month_key

C = pl.col
#: Compustat item -> our filing column.
ITEMS = {
    "at": "assets",
    "act": "assets_cur",
    "lct": "liab_cur",
    "lt": "liabilities",
    "dltt": "lt_debt",
    "dlc": "debt_cur",
    "seq": "equity",
    "che": "cash",
    "invt": "inventory",
    "rect": "receivables",
    "ppent": "ppe",
    "intan": "intangibles",
    "ap": "payables",
    "sale": "revenue",
    "cogs": "cost_of_revenue",
    "xsga": "sga",
    "gp": "gross_profit",
    "oiadp": "operating_income",
    "dp": "depreciation",
    "xint": "interest_expense",
    "txt": "tax",
    "ni": "net_income",
    "xrd": "rnd",
    "oancf": "cfo",
    "capx": "capex",
    "dv": "dividends",
}
#: Months after the period end before a filing is used, and the most months it stays in use.
LAG_TO_PUBLIC, MAX_DATA_LAG = 4, 18


def _sum_sas(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """JKP ``sum_sas``: a missing side counts as 0 unless both are missing."""
    return pl.when(a.is_null() & b.is_null()).then(None).otherwise(a.fill_null(0) + b.fill_null(0))


def filings(root: Path) -> pl.DataFrame:
    """Original (earliest filed) version of every filing row, with its month."""
    f = reader.read(root, "filings").sort("filed")
    f = f.unique(["sid", "period_end", "period"], keep="first", maintain_order=True)
    return f.with_columns(month_key(C("period_end")).alias("ym")).sort("sid", "ym", "period_end")


def _grid(rows: pl.DataFrame) -> pl.DataFrame:
    """Rows on a gap-free monthly grid per company (one row per month; the later period wins)."""
    rows = rows.unique(["sid", "ym"], keep="last", maintain_order=True)
    span = rows.group_by("sid").agg(
        pl.int_ranges(C("ym").min(), C("ym").max() + 1).first().alias("ym")
    )
    grid = span.explode("ym").join(rows, on=["sid", "ym"], how="left").sort("sid", "ym")
    return grid.with_columns(
        C("ym").cum_count().over("sid").alias("count"),
        C("period_end").is_not_null().alias("data_available"),
    )


def panels(f: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The annual and quarterly panels (module docs), Compustat-named, on monthly grids."""
    items = [C(ours).alias(cs) for cs, ours in ITEMS.items()]
    ttm = f.filter(C("period") == "ttm").select("sid", "ym", "period_end", "form", *items)
    annual = _grid(ttm.filter(C("form").str.starts_with("10-K")).drop("form"))
    q = f.filter(C("period") == "q").select(
        "sid",
        "period_end",
        C("revenue").alias("sale_qtr"),
        C("net_income").alias("ni_qtr"),
        C("cfo").alias("ocf_qtr"),
    )
    quarterly = _grid(ttm.drop("form").join(q, on=["sid", "period_end"], how="left"))
    return annual, quarterly


def helpers(p: pl.DataFrame) -> pl.DataFrame:
    """JKP ``add_helper_vars`` on our items (aux 4654-4928; substitutions in module docs)."""
    p = p.with_columns(pl.when(C(c) >= 0).then(C(c)).alias(c) for c in ("at", "sale", "dv", "che"))
    p = p.with_columns(
        C("sale").alias("sale_x"),
        _sum_sas(C("dltt"), C("dlc")).alias("debt_x"),
        (C("cogs") + C("xsga")).alias("opex_x"),
        C("dv").alias("div_x"),
        C("act").alias("ca_x"),
        C("lct").alias("cl_x"),
        pl.lit(0.0).alias("fna_x"),
        (C("ppent") + C("invt")).alias("ppeinv_x"),
        C("ppent").alias("ppegt"),
        pl.coalesce(C("gp"), C("sale") - C("cogs")).alias("gp_x"),
        pl.coalesce(C("seq"), C("at") - C("lt")).alias("seq_x"),
        C("ni").alias("ni_x"),
        C("ni").alias("nix_x"),
        (C("ni") + C("txt")).alias("pi_x"),
    )
    p = p.with_columns(
        (C("ca_x") - C("che")).alias("coa_x"),
        (C("cl_x") - C("dlc").fill_null(0)).alias("col_x"),
        (C("lt") - C("cl_x") - C("dltt")).alias("ncol_x"),
        C("debt_x").alias("fnl_x"),
        (C("debt_x") - C("che").fill_null(0)).alias("netdebt_x"),
        pl.coalesce(C("oiadp") + C("dp"), C("sale_x") - C("opex_x"), C("gp_x") - C("xsga")).alias(
            "ebitda_x"
        ),
        pl.coalesce(C("at"), C("seq_x") + C("dltt") + C("lct").fill_null(0)).alias("at_x"),
    )
    p = p.with_columns(
        (C("coa_x") - C("col_x")).alias("cowc_x"),
        (C("col_x") + C("ncol_x")).alias("ol_x"),
        (C("fna_x") - C("fnl_x")).alias("nfna_x"),
        pl.when(C("seq_x") > 0).then(C("seq_x")).alias("be_x"),
        pl.when((C("netdebt_x") + C("seq_x")) > 0).then(C("netdebt_x") + C("seq_x")).alias("bev_x"),
        pl.coalesce(C("oiadp"), C("ebitda_x") - C("dp")).alias("ebit_x"),
        (C("ebitda_x") + C("xrd").fill_null(0)).alias("op_x"),
        (C("ebitda_x") - C("xint")).alias("ope_x"),
        (C("at_x") - C("ca_x")).alias("ncoa_x"),
        (
            C("che")
            + 0.75 * (C("ca_x") - C("che"))
            + 0.5 * (C("at_x") - C("ca_x") - C("intan").fill_null(0))
        ).alias("aliq_x"),
        # no other-assets / other-liabilities items: lnoa ~ (at - act) - (lt - lct - dltt) + dp
        ((C("at") - C("act")) - (C("lt") - C("lct") - C("dltt")) + C("dp")).alias("lnoa_x"),
    )
    p = p.with_columns(
        (C("coa_x") + C("ncoa_x")).alias("oa_x"), (C("ncoa_x") - C("ncol_x")).alias("nncoa_x")
    )
    p = p.with_columns((C("oa_x") - C("ol_x")).alias("noa_x"))
    d12 = lambda c: C(c) - fc.lag(c, 12)  # noqa: E731
    yr = C("count") > 12
    p = p.with_columns(
        pl.when(yr)
        .then(pl.coalesce(C("ni_x") - C("oancf"), d12("cowc_x") + d12("nncoa_x")))
        .alias("oacc_x"),
        pl.when(yr).then(_sum_sas(d12("dltt"), d12("dlc"))).alias("dbnetis_x"),
    )
    p = p.with_columns(
        pl.when(yr).then(C("oacc_x") + d12("nfna_x")).alias("tacc_x"),
        pl.coalesce(C("oancf"), C("ni_x") - C("oacc_x"), C("ni_x") + C("dp")).alias("ocf_x"),
        (C("ebitda_x") + C("xrd").fill_null(0) - C("oacc_x")).alias("cop_x"),
    )
    return p.with_columns((C("ocf_x") - C("capx")).alias("fcf_x"))


def characteristics(p: pl.DataFrame, quarterly: bool) -> pl.DataFrame:
    """All panel-level characteristics (``.jkp_fund_chars``) on a helper-filled panel."""
    p = p.with_columns(fc.standard(p))
    p = fc.ocf_at_chg1(p)
    p = fc.persistence(p)
    p = fc.duration_pieces(p)
    p = fc.intrinsic_value(p)
    if quarterly:
        p = fc.niq_changes(p.with_columns(fc.quarterly_only(p)))
    return p


def publish(p: pl.DataFrame, keep: list[str]) -> pl.DataFrame:
    """Each filing row's characteristics on every month it is in use (module docs)."""
    rows = p.filter(C("data_available")).select("sid", C("ym").alias("data_ym"), *keep)
    start = C("data_ym") + LAG_TO_PUBLIC
    nxt = start.shift(-1).over("sid") - 1
    end = pl.min_horizontal(nxt.fill_null(10**9), C("data_ym") + MAX_DATA_LAG)
    rows = rows.with_columns(start.alias("start"), end.alias("end")).filter(C("end") >= C("start"))
    return (
        rows.with_columns(pl.int_ranges("start", C("end") + 1).alias("ym"))
        .explode("ym")
        .drop("start", "end")
    )


def combine(annual: pl.DataFrame, quarterly: pl.DataFrame, chars: list[str]) -> pl.DataFrame:
    """Annual base; the quarterly value wins when the annual is missing or older (module docs)."""
    q = quarterly.rename({c: f"{c}__q" for c in [*chars, "data_ym"] if c in quarterly.columns})
    j = annual.join(q, on=["sid", "ym"], how="left")
    out = []
    for c in chars:
        if c not in annual.columns:
            out.append(C(f"{c}__q").alias(c))
        elif f"{c}__q" in j.columns:
            newer = C(f"{c}__q").is_not_null() & (C("data_ym__q") > C("data_ym"))
            out.append(pl.when(C(c).is_null() | newer).then(C(f"{c}__q")).otherwise(C(c)).alias(c))
        else:
            out.append(C(c))
    return j.select("sid", "ym", *out)


#: Panel-level columns carried through publication (characteristics and what later steps need).
VALUATION_INPUTS = (
    "at_x",
    "be_x",
    "debt_x",
    "netdebt_x",
    "sale_x",
    "ni_x",
    "ocf_x",
    "fcf_x",
    "xrd",
    "div_x",
    "bev_x",
    "ebitda_x",
    "aliq_x",
    "_dur_pv",
    "_dur_tpv",
    "_ival",
)


def valuation(frame: pl.DataFrame) -> pl.DataFrame:
    """Ratios to the current month's market equity ``me`` (JKP ``scale_me``; module docs).

    ``div12m_me`` uses the filings' trailing-twelve-month dividends (JKP sums the dividends in
    the price feed over 12 months).
    """
    me = pl.when(C("me") > 0).then(C("me"))
    mev = C("me") + C("netdebt_x")
    mev = pl.when(mev > 0).then(mev)
    ratios = {
        "at_me": "at_x",
        "be_me": "be_x",
        "debt_me": "debt_x",
        "netdebt_me": "netdebt_x",
        "sale_me": "sale_x",
        "ni_me": "ni_x",
        "ocf_me": "ocf_x",
        "fcf_me": "fcf_x",
        "rd_me": "xrd",
        "div12m_me": "div_x",
    }
    k = 10 + 1.12 / 0.12
    eq_dur = C("_dur_tpv") / me + k * (me - C("_dur_pv")) / me
    mat = C("at_x") - C("be_x") + C("me")
    mat = pl.when(mat > 0).then(mat)
    frame = frame.sort("sid", "ym").with_columns(mat.alias("_mat"))
    return frame.with_columns(
        *((C(v) / me).alias(name) for name, v in ratios.items()),
        (C("bev_x") / mev).alias("bev_mev"),
        (C("ebitda_x") / mev).alias("ebitda_mev"),
        pl.when(eq_dur > 0).then(eq_dur).alias("eq_dur"),
        (C("_ival") / me).alias("ival_me"),
        (C("aliq_x") / C("_mat").shift(12).over("sid")).alias("aliq_mat"),
    ).drop(*VALUATION_INPUTS, "_mat", "me")


def build(root: Path, market_value: pl.DataFrame) -> pl.DataFrame:
    """Every buildable JKP accounting characteristic per company and month: sid, ym, columns.

    Args:
        root: The data hive.
        market_value: sid, ym, me (each month's market value; the monthly table's).
    """
    f = filings(root)
    annual, quarterly = panels(f)
    me = market_value.select("sid", "ym", C("me").alias("me_fiscal"))
    out = {}
    for name, panel in (("annual", annual), ("quarterly", quarterly)):
        p = helpers(panel).join(me, on=["sid", "ym"], how="left")
        p = characteristics(p, quarterly=name == "quarterly")
        base = set(annual.columns) | set(quarterly.columns)
        keep = [
            c
            for c in p.columns
            if c not in base
            and not c.endswith("_x")
            and c != "me_fiscal"
            and c not in ("data_available",)
        ] + list(VALUATION_INPUTS)
        keep = list(dict.fromkeys(c for c in keep if c in p.columns))
        out[name] = publish(p, keep)
    chars = list(
        dict.fromkeys(
            c for frame in out.values() for c in frame.columns if c not in ("sid", "ym", "data_ym")
        )
    )
    combined = combine(out["annual"], out["quarterly"], chars)
    return valuation(combined.join(market_value, on=["sid", "ym"], how="left"))
