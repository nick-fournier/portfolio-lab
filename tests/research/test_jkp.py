from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from portfolio_lab.data import reader
from portfolio_lab.research.optimizer import jkp, jkp_daily, jkp_fund, jkp_monthly, jkp_scores
from portfolio_lab.research.optimizer import jkp_fund_chars as fc


def _months(rets: list[float | None], sid: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sid": [sid] * len(rets),
            "ym": list(range(24000, 24000 + len(rets))),
            "ret": rets,
            "prc": [10.0] * len(rets),
        }
    )


def test_momentum_compounds_months_t_minus_j_plus_1_to_t_minus_i():
    rets = [0.01 * (k + 1) for k in range(14)]
    out = jkp_monthly.momentum_and_seasonality(_months(rets)).row(13, named=True)
    t = 13  # last month
    expected_12_1 = np.prod([1 + rets[k] for k in range(t - 11, t)]) - 1  # t-11 .. t-1
    expected_12_7 = np.prod([1 + rets[k] for k in range(t - 11, t - 6)]) - 1  # t-11 .. t-7
    assert out["ret_12_1"] == pytest.approx(expected_12_1)
    assert out["ret_12_7"] == pytest.approx(expected_12_7)
    assert out["ret_1_0"] == pytest.approx(rets[t])
    assert out["ret_60_12"] is None  # needs more than 60 months


def test_a_missing_month_inside_the_momentum_window_counts_as_zero_but_t_minus_1_is_required():
    rets = [0.02] * 14
    rets[5] = None
    out = jkp_monthly.momentum_and_seasonality(_months(rets)).row(13, named=True)
    assert out["ret_12_1"] == pytest.approx(1.02**10 - 1)  # 11 months, one of them missing
    rets[12] = None  # month t-1
    out = jkp_monthly.momentum_and_seasonality(_months(rets)).row(13, named=True)
    assert out["ret_12_1"] is None


def test_seasonality_annual_is_lag_11_and_non_annual_the_other_lags_including_month_t():
    rets = [0.001 * k for k in range(12)]  # lag 0 = last = 0.011, lag 11 = first = 0.0
    out = jkp_monthly.momentum_and_seasonality(_months(rets)).row(11, named=True)
    assert out["seas_1_1an"] == pytest.approx(rets[0])
    assert out["seas_1_1na"] == pytest.approx(np.mean(rets[1:]))
    rets[3] = None
    out = jkp_monthly.momentum_and_seasonality(_months(rets)).row(11, named=True)
    assert out["seas_1_1an"] is None and out["seas_1_1na"] is None


def test_beta_60m_recovers_the_slope_and_needs_36_months():
    rng = np.random.default_rng(0)
    m = rng.normal(0.01, 0.04, 70)
    months = _months(list(1.5 * m + 0.003), sid=7)
    factors = pl.DataFrame({"ym": months["ym"], "mktrf": m, "smb": 0.0, "hml": 0.0, "rf": 0.0})
    out = jkp_monthly.beta_60m(months, factors).sort("ym")
    assert out["beta_60m"][34] is None
    assert out["beta_60m"][-1] == pytest.approx(1.5, abs=1e-6)


def _daily(n_days: int = 40) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(1)
    days = [date(2020, 1, 1) + timedelta(days=k) for k in range(n_days)]
    mkt = rng.normal(0, 0.01, n_days)
    ret = 0.8 * mkt + rng.normal(0, 0.01, n_days)
    close = 10 * np.cumprod(1 + ret)
    volume = np.where(np.arange(n_days) % 5 == 0, 0.0, 1000.0)
    daily = pl.DataFrame(
        {
            "sid": 1,
            "date": days,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
            "close_adj": close,
            "ret_cc": ret,
        }
    )
    factors = pl.DataFrame(
        {
            "date": days,
            "mktrf": mkt,
            "smb": rng.normal(0, 0.005, n_days),
            "hml": rng.normal(0, 0.005, n_days),
            "rf": 0.0,
        }
    )
    return daily, factors


def test_one_month_statistics_match_their_definitions():
    daily, factors = _daily()
    listings = pl.DataFrame(
        {"sid": [1], "from": [date(2019, 1, 1)], "to": [date(2030, 1, 1)], "exchange": ["NYSE"]}
    )
    shares = pl.DataFrame({"sid": [1, 1], "ym": [2020 * 12, 2020 * 12 + 1], "shares": [1e5, 1e5]})
    d = jkp_daily.prepare(daily, factors, listings, shares)
    jan = d.filter(pl.col("ym") == 2020 * 12)
    out = jkp_daily.one_month(d, jkp_daily.market_days(factors)).filter(pl.col("ym") == 2020 * 12)
    assert out["rvol_21d"][0] == pytest.approx(jan["ret_exc"].std())
    assert out["rmax1_21d"][0] == pytest.approx(jan["ret_cc"].max())
    r, m = jan["ret_exc"].to_numpy(), jan["mktrf"].to_numpy()
    beta = np.cov(r, m)[0, 1] / m.var(ddof=1)
    res = r - (r.mean() - beta * m.mean()) - beta * m
    assert out["ivol_capm_21d"][0] == pytest.approx(res.std(ddof=1))
    sums = jkp_daily.month_sums(d).filter(pl.col("ym") == 2020 * 12)
    assert sums["nzero"][0] == pytest.approx((jan["volume"] == 0).sum())


def test_corwin_schultz_for_a_constant_range_with_no_drift_is_about_the_log_range():
    days = [date(2021, 3, 1) + timedelta(days=k) for k in range(20)]
    d = pl.DataFrame(
        {
            "sid": 1,
            "date": days,
            "ym": 2021 * 12 + 2,
            "tvol": 1000.0,
            "high_adj": 10.1,
            "low_adj": 9.9,
            "prc_adj": 10.0,
        }
    )
    out = jkp_daily.bidask(d)
    # beta = 2 L^2, gamma = L^2 with L = ln(high/low): alpha = L, spread = 2(e^L - 1)/(1 + e^L)
    a = np.log(10.1 / 9.9)
    assert out["bidaskhl_21d"][0] == pytest.approx(2 * (np.exp(a) - 1) / (1 + np.exp(a)))


def test_a_filing_is_used_from_4_months_after_its_period_until_the_next_one_at_most_18():
    # filings at months 100 and 112 (annual); the second ends at 112 + 18
    p = pl.DataFrame(
        {
            "sid": [1] * 13,
            "ym": list(range(100, 113)),
            "data_available": [k in (0, 12) for k in range(13)],
            "x": [1.0] + [None] * 11 + [2.0],
        }
    )
    out = jkp_fund.publish(p, ["x"])
    first = out.filter(pl.col("data_ym") == 100)["ym"].to_list()
    second = out.filter(pl.col("data_ym") == 112)["ym"].to_list()
    assert first == list(range(104, 116))  # until the next filing's start (116) minus 1
    assert second == list(range(116, 131))  # 112 + 18 = 130


def test_combine_takes_the_quarterly_value_when_the_annual_is_missing_or_older():
    ann = pl.DataFrame(
        {"sid": [1, 1, 1], "ym": [10, 11, 12], "data_ym": [5, 5, 5], "a": [1.0, None, 1.0]}
    )
    qtr = pl.DataFrame(
        {
            "sid": [1, 1, 1],
            "ym": [10, 11, 12],
            "data_ym": [5, 5, 8],
            "a": [2.0, 2.0, 3.0],
            "q_only": [7.0, 7.0, 7.0],
        }
    )
    out = jkp_fund.combine(ann, qtr, ["a", "q_only"]).sort("ym")
    assert out["a"].to_list() == [1.0, 2.0, 3.0]  # tie -> annual; missing -> quarterly; newer
    assert out["q_only"].to_list() == [7.0, 7.0, 7.0]


def test_growth_needs_a_positive_lagged_value_and_enough_history():
    p = pl.DataFrame(
        {"sid": [1] * 14, "count": list(range(1, 15)), "x": [0.0, 2.0] + [1.0] * 10 + [3.0, 4.0]}
    )
    out = p.with_columns(fc.growth("x", 12).alias("g"))["g"].to_list()
    assert out[12] is None  # lag (row 0) is 0
    assert out[13] == pytest.approx(1.0)  # 4 / 2 - 1


def test_mispricing_ranks_within_the_month_and_needs_ten_stocks():
    n = 12
    t = pl.DataFrame(
        {
            "date": [date(2020, 1, 31)] * n,
            "o_score": [float(k) for k in range(n)],
            "ret_12_1": [None] * n,
            "gp_at": [None] * n,
            "niq_at": [None] * n,
        }
    )
    for c in (
        "chcsho_12m",
        "eqnpo_12m",
        "oaccruals_at",
        "noa_at",
        "at_gr1",
        "ppeinv_gr1a",
        "gp_at",
        "ni_be",
        "_ni_at",
        "ocf_at",
        "_gp_sale",
        "_gpoa_ch5",
        "_roe_ch5",
        "_roa_ch5",
        "_cfoa_ch5",
        "_gmar_ch5",
        "betabab_1260d",
        "_debt_at",
        "_roeq_be_std",
        "_roe_be_std",
    ):
        if c not in t.columns:
            t = t.with_columns(pl.lit(None, pl.Float64).alias(c))
    out = jkp_scores.scores(t)
    # o_score: high value -> low rank; the highest o_score gets rank 1/12
    assert out["mispricing_perf"].to_list()[-1] == pytest.approx(1 / n)
    assert out["mispricing_mgmt"].null_count() == n  # fewer than 3 inputs present
    assert not any(c.startswith("_") for c in out.columns)


def test_financial_assets_cash_and_operating_assets_count_investments_as_jkp_does():
    """fna = ivst + ivao, che = cash + ivst (Compustat's cash), ncoa = at - act - ivao."""
    row = dict.fromkeys(jkp_fund.ITEMS, 0.0) | {
        "sid": 1, "ym": 100, "count": 1, "at": 100.0, "act": 40.0, "che": 5.0, "ivst": 3.0,
        "ivao": 10.0, "dltt": 20.0, "dlc": 2.0, "lt": 50.0, "lct": 15.0,
    }  # fmt: skip
    out = jkp_fund.helpers(pl.DataFrame([row])).row(0, named=True)
    assert (out["fna_x"], out["che"], out["ncoa_x"]) == (13.0, 8.0, 50.0)
    assert out["nfna_x"] == 13.0 - 22.0 and out["coa_x"] == 32.0 and out["netdebt_x"] == 14.0
    none = jkp_fund.helpers(pl.DataFrame([row | {"ivst": None, "ivao": None}])).row(0, named=True)
    assert (none["fna_x"], none["che"], none["ncoa_x"]) == (0.0, 5.0, 60.0)  # missing counts 0


def test_share_issuance_is_the_change_in_split_adjusted_shares_and_ignores_dividends(tmp_path):
    """100 shares filed at month 0, a 2:1 split at month 6, 220 filed at month 12: +10%."""
    filed = [date(2020, 1, 15), date(2021, 1, 15)]
    reader.write(tmp_path, "sharadar", "filings", pl.DataFrame(
        {"sid": [1, 1], "filed": filed, "period_end": [date(2019, 12, 31), date(2020, 12, 31)],
         "period": ["ttm", "ttm"], "shares_out": [100.0, 220.0]}))  # fmt: skip
    reader.write(tmp_path, "sharadar", "actions", pl.DataFrame(
        {"sid": [1], "date": [date(2020, 7, 1)], "action": ["split"], "value": [2.0]}))  # fmt: skip
    shares = jkp.split_adjusted_shares(tmp_path)
    assert shares["shares"].to_list() == [200.0, 220.0]
    first = pl.select(jkp_monthly.month_key(pl.lit(date(2020, 1, 31)))).item()
    months = pl.DataFrame({"sid": [1] * 13, "ym": list(range(first, first + 13)),
                           "ret": [0.01] * 13, "prc_adj": [1.0] * 13,
                           "market_value": [1000.0 * 1.01**k for k in range(13)]})  # fmt: skip
    out = jkp._issuance(months, shares)
    assert out["chcsho_12m"][-1] == pytest.approx(0.10)  # the split and the 1%/month returns cancel
    assert out["eqnpo_12m"][-1] == 0.0  # market value grew with the total return: no net payout
