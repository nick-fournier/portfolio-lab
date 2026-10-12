from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.optimizer import jkp_daily, jkp_monthly


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
