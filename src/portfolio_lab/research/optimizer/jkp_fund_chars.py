"""JKP accounting characteristics on a filing panel (ported from ``bkelly-lab/jkp-data``).

Every function here works on one panel from :mod:`.jkp_fund` (annual or quarterly), a grid with
one row per company and month, sorted, where ``count`` is the row number within the company.
Lags are row shifts on that grid, as in JKP (``shift(12)`` = a year earlier). Formulas follow
aux_functions.py (commit 666e896); line numbers are in tmp/jkp_fund_spec.md. Helper variables
(``*_x``) come from :func:`.jkp_fund.helpers`.
"""

import polars as pl

C = pl.col


def lag(e: pl.Expr | str, k: int) -> pl.Expr:
    """``e`` ``k`` grid rows earlier within the company."""
    e = C(e) if isinstance(e, str) else e
    return e.shift(k).over("sid")


def _count(k: int) -> pl.Expr:
    return C("count") > k


def div(n: pl.Expr, d: pl.Expr, mode: int = 1) -> pl.Expr:
    """JKP ``safe_div``: 1 d!=0; 2 n/|d| for d!=0; 3 d>0; 8 n>0 and d>0."""
    if mode == 2:
        return pl.when(d != 0).then(n / d.abs())
    if mode == 3:
        return pl.when(d > 0).then(n / d)
    if mode == 8:
        return pl.when((n > 0) & (d > 0)).then(n / d)
    return pl.when(d != 0).then(n / d)


def lagdiv(n: pl.Expr, d: str, k: int = 12) -> pl.Expr:
    """``n`` over ``d`` ``k`` rows earlier (modes 4 and 9: count > k and the lag > 0)."""
    return pl.when(_count(k) & (lag(d, k) > 0)).then(n / lag(d, k))


def growth(x: str, h: int) -> pl.Expr:
    """``var_growth``: x / x(h) - 1 when x(h) > 0."""
    return pl.when((lag(x, h) > 0) & _count(h)).then(C(x) / lag(x, h) - 1)


def chg_assets(x: str, h: int = 12) -> pl.Expr:
    """``chg_to_assets``: change over h months over current assets."""
    return pl.when((C("at_x") > 0) & _count(h)).then((C(x) - lag(x, h)) / C("at_x"))


def chg_lagassets(x: str) -> pl.Expr:
    """``chg_to_lagassets``: 12-month change over assets a year ago."""
    return pl.when((lag("at_x", 12) > 0) & _count(12)).then((C(x) - lag(x, 12)) / lag("at_x", 12))


def chg_sumassets(x: str) -> pl.Expr:
    """``chg_to_avgassets``: 12-month change over assets now plus a year ago (a sum, as JKP)."""
    den = C("at_x") + lag("at_x", 12)
    return pl.when((den > 0) & _count(12)).then((C(x) - lag(x, 12)) / den)


def chg_exp(x: str) -> pl.Expr:
    """``chg_to_exp``: x over the average of x one and two years ago, minus 1."""
    s = lag(x, 12) + lag(x, 24)
    return pl.when((s > 0) & _count(24)).then(C(x) / (s / 2) - 1)


def _std_at(e: pl.Expr, lags: range, min_n: int) -> pl.Expr:
    """Sample standard deviation of ``e`` at the given row lags, if at least ``min_n`` exist."""
    vals = [lag(e, k) if k else e for k in lags]
    n = pl.sum_horizontal([v.is_not_null().cast(pl.Int32) for v in vals])
    mean = pl.sum_horizontal(vals) / n
    ss = pl.sum_horizontal([(v - mean) ** 2 for v in vals])
    return pl.when(n >= min_n).then((ss / (n - 1)).sqrt())


def standard(frame: pl.DataFrame) -> list[pl.Expr]:
    """Characteristics built the same way on the annual and the quarterly panel."""
    x = {
        "at_gr1": growth("at_x", 12),
        "sale_gr1": growth("sale_x", 12),
        "sale_gr3": growth("sale_x", 36),
        "debt_gr3": growth("debt_x", 36),
        "inv_gr1": growth("invt", 12),
        "capx_gr1": growth("capx", 12),
        "capx_gr2": growth("capx", 24),
        "capx_gr3": growth("capx", 36),
        "be_gr1a": chg_assets("be_x"),
        "inv_gr1a": chg_assets("invt"),
        "coa_gr1a": chg_assets("coa_x"),
        "col_gr1a": chg_assets("col_x"),
        "cowc_gr1a": chg_assets("cowc_x"),
        "ncoa_gr1a": chg_assets("ncoa_x"),
        "ncol_gr1a": chg_assets("ncol_x"),
        "nncoa_gr1a": chg_assets("nncoa_x"),
        "fnl_gr1a": chg_assets("fnl_x"),
        "nfna_gr1a": chg_assets("nfna_x"),
        "tax_gr1a": chg_assets("txt"),
        "noa_gr1a": chg_lagassets("noa_x"),
        "ppeinv_gr1a": chg_lagassets("ppeinv_x"),
        "lnoa_gr1a": chg_sumassets("lnoa_x"),
        "noa_at": lagdiv(C("noa_x"), "at_x"),
        "oaccruals_at": div(C("oacc_x"), C("at_x")),
        "oaccruals_ni": div(C("oacc_x"), C("nix_x"), 2),
        "taccruals_at": div(C("tacc_x"), C("at_x")),
        "taccruals_ni": div(C("tacc_x"), C("nix_x"), 2),
        "dbnetis_at": div(C("dbnetis_x"), C("at_x")),
        "gp_at": div(C("gp_x"), C("at_x")),
        "gp_atl1": lagdiv(C("gp_x"), "at_x"),
        "op_at": div(C("op_x"), C("at_x"), 3),
        "op_atl1": lagdiv(C("op_x"), "at_x"),
        "cop_at": div(C("cop_x"), C("at_x")),
        "cop_atl1": lagdiv(C("cop_x"), "at_x"),
        "ope_be": div(C("ope_x"), C("be_x")),
        "ope_bel1": lagdiv(C("ope_x"), "be_x"),
        "ni_be": div(C("ni_x"), C("be_x")),
        "ebit_bev": div(C("ebit_x"), C("bev_x")),
        "ebit_sale": div(C("ebit_x"), C("sale_x")),
        "sale_bev": div(C("sale_x"), C("bev_x")),
        "opex_at": div(C("opex_x"), C("at_x")),
        "ocf_at": div(C("ocf_x"), C("at_x"), 3),
        "pi_nix": div(C("pi_x"), C("nix_x"), 8),
        "cash_at": div(C("che"), C("at_x"), 3),
        "rd_sale": div(C("xrd"), C("sale_x")),
        "aliq_at": lagdiv(C("aliq_x"), "at_x"),
        "at_be": div(C("at_x"), C("be_x")),
        "tangibility": div(
            C("che") + 0.715 * C("rect") + 0.547 * C("invt") + 0.535 * C("ppegt"), C("at_x")
        ),
        "dsale_dinv": chg_exp("sale_x") - chg_exp("invt"),
        "dsale_drec": chg_exp("sale_x") - chg_exp("rect"),
        "dgp_dsale": chg_exp("gp_x") - chg_exp("sale_x"),
        "dsale_dsga": chg_exp("sale_x") - chg_exp("xsga"),
        # helpers kept for the composite scores (dropped after them)
        "_ni_at": div(C("ni_x"), C("at_x")),
        "_gp_sale": div(C("gp_x"), C("sale_x")),
        "_debt_at": div(C("debt_x"), C("at_x")),
    }
    at_avg = pl.when(lag("at_x", 12).is_not_null()).then((C("at_x") + lag("at_x", 12)) / 2)
    x["at_turnover"] = pl.when(_count(12)).then(div(C("sale_x"), at_avg.fill_null(C("at_x"))))
    rd = [lag("xrd", k) if k else C("xrd") for k in (0, 12, 24, 36, 48)]
    rd5 = rd[0] + 0.8 * rd[1] + 0.6 * rd[2] + 0.4 * rd[3] + 0.2 * rd[4]
    x["rd5_at"] = pl.when(_count(48) & (C("at_x") > 0)).then(rd5 / C("at_x"))
    cs = div(C("capx"), C("sale_x"), 3)
    lags = lag(cs, 12) + lag(cs, 24) + lag(cs, 36)
    x["capex_abn"] = pl.when(_count(36) & (lags != 0)).then(cs / (lags / 3) - 1)
    for name, num, den in (
        ("_gpoa_ch5", "gp_x", "at_x"),
        ("_roe_ch5", "ni_x", "be_x"),
        ("_roa_ch5", "ni_x", "at_x"),
        ("_cfoa_ch5", "ocf_x", "at_x"),
        ("_gmar_ch5", "gp_x", "sale_x"),
    ):
        ratio = pl.when(C(den) > 0).then(C(num) / C(den))
        x[name] = pl.when(_count(60)).then(ratio - lag(ratio, 60))
    x["_roe_be_std"] = pl.when(_count(48)).then(
        _std_at(div(C("ni_x"), C("be_x")), range(0, 49, 12), 5)
    )
    out = [e.alias(k) for k, e in x.items()]
    out += composites()
    return out


def ocf_at_chg1(frame: pl.DataFrame) -> pl.DataFrame:
    """``ocf_at_chg1`` (needs ``ocf_at`` first)."""
    return frame.with_columns(
        pl.when(_count(12)).then(C("ocf_at") - lag("ocf_at", 12)).alias("ocf_at_chg1")
    )


def composites() -> list[pl.Expr]:
    """F-score, O-score, KZ index, earnings variability, persistence, duration and value pieces."""
    at12 = lag("at_x", 12)
    roa = pl.when(at12 > 0).then(C("ni_x") / at12)
    croa = pl.when(at12 > 0).then(C("ocf_x") / at12)
    lev = div(C("dltt"), C("at_x"), 3)
    liq = div(C("ca_x"), C("cl_x"), 3)
    gm = div(C("gp_x"), C("sale_x"))
    aturn = pl.when(at12 > 0).then(C("sale_x") / at12)
    parts = [
        roa > 0,
        croa > 0,
        (roa - lag(roa, 12)) > 0,
        (croa - roa) > 0,
        (lev - lag(lev, 12)) < 0,
        (liq - lag(liq, 12)) > 0,
        pl.lit(True),  # equity issuance: JKP coalesces missing sstk to 0, so always a point
        (gm - lag(gm, 12)) > 0,
        (aturn - lag(aturn, 12)) > 0,
    ]
    f_score = pl.when(_count(24)).then(pl.sum_horizontal([p.cast(pl.Int32) for p in parts]))
    f_score = pl.when(pl.all_horizontal([p.is_not_null() for p in parts])).then(f_score)
    at, lt, ca = C("at_x"), C("lt"), C("ca_x")
    nix, nix12 = C("nix_x"), lag("nix_x", 12)
    o = (
        -1.32
        - 0.407 * at.log()
        + 6.03 * C("debt_x") / at
        - 1.43 * (ca - C("cl_x")) / at
        + 0.076 * C("cl_x") / ca
        - 1.72 * (C("lt") > at).cast(pl.Float64)
        - 2.37 * nix / at
        - 1.83 * (C("pi_x") + C("dp")) / lt
        + 0.285 * ((nix < 0) & (nix12 < 0)).cast(pl.Float64)
        - 0.52 * (nix - nix12) / (nix.abs() + nix12.abs())
    )
    o_score = pl.when((at > 0) & (ca > 0) & (lt > 0) & _count(12)).then(o)
    ppe12 = lag("ppent", 12)
    kz = (
        -1.002 * (C("ni_x") + C("dp")) / ppe12
        + 0.283 * (at + C("me_fiscal") - C("be_x")) / at
        + 3.139 * C("debt_x") / (C("debt_x") + C("seq_x"))
        - 39.368 * C("div_x") / ppe12
        - 1.315 * C("che") / ppe12
    )
    kz_index = pl.when(
        _count(12) & (ppe12 > 0) & (at > 0) & ((C("debt_x") + C("seq_x")) != 0)
    ).then(kz)
    roa6 = pl.when(at12 != 0).then(C("ni_x") / at12)
    croa6 = pl.when(at12 != 0).then(C("ocf_x") / at12)
    sd_roa, sd_croa = _std_at(roa6, range(0, 49, 12), 5), _std_at(croa6, range(0, 49, 12), 5)
    variability = pl.when(_count(60) & (sd_croa > 0)).then(sd_roa / sd_croa)
    return [
        f_score.cast(pl.Float64).alias("f_score"),
        o_score.alias("o_score"),
        kz_index.alias("kz_index"),
        variability.alias("earnings_variability"),
    ]


def persistence(frame: pl.DataFrame) -> pl.DataFrame:
    """``ni_ar1`` / ``ni_ivol``: AR(1) of earnings over assets over 5 yearly pairs (all needed)."""
    y = pl.when(C("at_x") > 0).then(C("ni_x") / C("at_x"))
    f = frame.with_columns(y.alias("_y")).with_columns(lag("_y", 12).alias("_x"))
    ys = [lag("_y", k) if k else C("_y") for k in (0, 12, 24, 36, 48)]
    xs = [lag("_x", k) if k else C("_x") for k in (0, 12, 24, 36, 48)]
    ok = pl.all_horizontal([v.is_not_null() for v in ys + xs])
    my, mx = pl.sum_horizontal(ys) / 5, pl.sum_horizontal(xs) / 5
    sxy = pl.sum_horizontal([(a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)])
    sxx = pl.sum_horizontal([(a - mx) ** 2 for a in xs])
    slope = sxy / sxx
    resid = [b - (my - slope * mx) - slope * a for a, b in zip(xs, ys, strict=True)]
    ivol = (pl.sum_horizontal([r**2 for r in resid]) / 4).sqrt()  # residual mean is 0; ddof 1
    return f.with_columns(
        pl.when(ok & (sxx > 0)).then(slope).alias("ni_ar1"),
        pl.when(ok & (sxx > 0)).then(ivol).alias("ni_ivol"),
    ).drop("_y", "_x")


def duration_pieces(frame: pl.DataFrame) -> pl.DataFrame:
    """Equity duration's market-free sums (``eq_dur`` divides by market equity later).

    Dechow-Sloan-Soliman cash distributions over 10 years: ROE and growth revert to 12% and 6%
    at speeds 0.57 and 0.24, discounted at 12% (JKP ``equity_duration_cd``).
    """
    be12, s12 = lag("be_x", 12), lag("sale_x", 12)
    roe = pl.when(be12 > 1).then(C("ni_x") / be12)
    g = pl.when(s12 > 1).then(C("sale_x") / s12 - 1)
    be = C("be_x")
    pv, tpv, ok = pl.lit(0.0), pl.lit(0.0), pl.lit(True)
    for t in range(1, 11):
        g = 0.06 * (1 - 0.24) + 0.24 * g
        roe = 0.12 * (1 - 0.57) + 0.57 * roe
        cd = be * (roe - g)
        be = be * (1 + g)
        ok = ok & (be >= 0)
        pv = pv + cd / 1.12**t
        tpv = tpv + t * cd / 1.12**t
    return frame.with_columns(
        pl.when(ok).then(pv).alias("_dur_pv"), pl.when(ok).then(tpv).alias("_dur_tpv")
    )


def intrinsic_value(frame: pl.DataFrame) -> pl.DataFrame:
    """Frankel-Lee intrinsic value (``ival_me`` divides it by market equity later)."""
    nix, be, be12 = C("nix_x"), C("be_x"), lag("be_x", 12)
    po = pl.when(nix > 0).then(C("div_x") / nix).otherwise(C("div_x") / (0.06 * C("at_x")))
    roe = pl.when(_count(12) & ((be + be12) > 0)).then(nix / ((be + be12) / 2))
    be1 = (1 + (1 - po) * roe) * be
    iv = be + (roe - 0.12) / 1.12 * be + (roe - 0.12) / (1.12 * 0.12) * be1
    return frame.with_columns(pl.when(iv > 0).then(iv).alias("_ival"))


def quarterly_only(frame: pl.DataFrame) -> list[pl.Expr]:
    """Characteristics from single-quarter items (quarterly panel only)."""
    sq, nq, oq = C("sale_qtr"), C("ni_qtr"), C("ocf_qtr")
    out = {
        "saleq_gr1": pl.when(_count(12) & (lag(sq, 12) > 0)).then(sq / lag(sq, 12) - 1),
        "niq_at": lagdiv(nq, "at_x", 3),
        "niq_be": lagdiv(nq, "be_x", 3),
        "ocfq_saleq_std": pl.when(_count(45)).then(
            _std_at(pl.when(sq > 0).then(oq / sq), range(0, 46, 3), 8)
        ),
        "_roeq_be_std": pl.when(_count(57)).then(_std_at(div(nq, C("be_x")), range(0, 58, 3), 12)),
    }
    for name, e in (("saleq_su", sq), ("niq_su", nq)):
        chg = e - lag(e, 12)
        vals = [lag(chg, k) for k in range(3, 25, 3)]  # the 8 quarters before this one
        n = pl.sum_horizontal([v.is_not_null().cast(pl.Int32) for v in vals])
        mean = pl.sum_horizontal(vals) / n
        sd = (pl.sum_horizontal([(v - mean) ** 2 for v in vals]) / (n - 1)).sqrt()
        out[name] = pl.when(_count(36) & (n > 6) & (sd != 0)).then((chg - mean) / sd)
    inc = pl.when(C("ni_x").is_not_null() & lag("ni_x", 12).is_not_null()).then(
        (C("ni_x") > lag("ni_x", 12)).cast(pl.Int32)
    )
    seq = [lag(inc, k) if k else inc for k in range(0, 22, 3)]
    run, alive = pl.lit(0), pl.lit(1)
    for v in seq:
        alive = alive * v
        run = run + alive
    ok = pl.all_horizontal([v.is_not_null() for v in seq]) & (C("count") >= 33)
    out["ni_inc8q"] = pl.when(ok).then(run.cast(pl.Float64))
    return [e.alias(k) for k, e in out.items()]


def niq_changes(frame: pl.DataFrame) -> pl.DataFrame:
    """``niq_at_chg1`` / ``niq_be_chg1`` (need ``niq_at`` / ``niq_be`` first)."""
    return frame.with_columns(
        pl.when(_count(12)).then(C("niq_at") - lag("niq_at", 12)).alias("niq_at_chg1"),
        pl.when(_count(12)).then(C("niq_be") - lag("niq_be", 12)).alias("niq_be_chg1"),
    )
