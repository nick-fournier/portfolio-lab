from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from portfolio_lab.core.calendar import rebalance_dates
from portfolio_lab.research.features import (
    MAX_COVERAGE,
    MAX_DISTANCE,
    MAX_FILING_AGE_DAYS,
    _distance_to_default,
    _interest_coverage,
    build_features,
    roa_stability,
)
from portfolio_lab.research.fundamentals import CONCEPTS

from ..strategies.test_strategies import _poisoned

SYMBOLS = [f"S{i:02d}" for i in range(12)]


def _states(dates, symbols=SYMBOLS, step=63):
    """A filing every ``step`` sessions per company, with simple growing fundamentals."""
    rows = []
    for k, _ in enumerate(symbols):
        for n, i in enumerate(range(5, len(dates), step)):
            row = {c + s: None for s in ("", "_py") for c in CONCEPTS}
            row |= {
                "cik": k, "accn": f"{k}-{n}", "form": "10-Q", "filed": dates[i],
                "period_end": dates[i] - timedelta(days=40), "shares_out": 1e6 * (k + 1),
                "net_income": 10.0 + n + k, "net_income_py": 8.0 + k, "equity": 500.0 + n,
                "assets": 1000.0 + 10 * n, "assets_py": 1000.0, "revenue": 300.0 + k,
                "revenue_py": 280.0, "cfo": 12.0 + k, "liabilities": 400.0,
            }  # fmt: skip
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


@pytest.fixture
def inputs(make_panel):
    panel = make_panel(symbols=SYMBOLS, days=400)
    tickers = pl.DataFrame({"symbol": SYMBOLS, "cik": range(len(SYMBOLS))})
    companies = pl.DataFrame({"cik": range(len(SYMBOLS)), "sic": [3570] * 6 + [6022] * 6})
    return panel, _states(panel.dates), tickers, companies


def _month_end(panel, after=260):
    days = rebalance_dates(panel.dates[after:-5], "M")
    return panel.date_index[days[1]]


def test_features_are_point_in_time(inputs):
    panel, states, tickers, companies = inputs
    i = _month_end(panel)
    day = panel.dates[i]
    start = panel.dates[0]
    clean = build_features(panel, states, tickers, companies, start=start).filter(
        pl.col("date") == day
    )
    # Garbage after the date, rewritten filings from the date on, and one filed that day.
    changed = states.with_columns(
        pl.when(pl.col("filed") >= day).then(1e12).otherwise(pl.col("net_income"))
        .alias("net_income")
    )  # fmt: skip
    extra = (
        changed.filter(pl.col("cik") == 0)
        .head(1)
        .with_columns(pl.lit(day).alias("filed"), pl.lit(-1e9).alias("equity"))
    )
    poisoned = build_features(
        _poisoned(panel, i), pl.concat([changed, extra]), tickers, companies, start=start
    ).filter(pl.col("date") == day)
    assert clean.height == len(SYMBOLS)
    assert clean.equals(poisoned)


def test_market_value_and_ratios(inputs):
    panel, states, tickers, companies = inputs
    i = _month_end(panel)
    day = panel.dates[i]
    row = (
        build_features(panel, states, tickers, companies, start=panel.dates[0])
        .filter((pl.col("date") == day) & (pl.col("symbol") == "S03"))
        .row(0, named=True)
    )
    filing = (
        states.filter((pl.col("cik") == 3) & (pl.col("filed") < day)).tail(1).row(0, named=True)
    )
    f = panel.date_index[filing["filed"]]
    j = panel.symbol_index["S03"]
    growth = np.prod(1 + np.nan_to_num(panel.field("ret_cc")[f + 1 : i + 1, j]))
    expected = filing["shares_out"] * panel.field("close")[f, j] * growth
    assert row["market_value"] == pytest.approx(expected)
    assert row["earnings_yield"] == pytest.approx(filing["net_income"] / expected)
    assert row["roa"] == pytest.approx(filing["net_income"] / filing["assets"])
    assert row["sic2"] == 35 and 0 < row["earnings_yield_ind"] <= 1


def test_stale_filings_are_ignored(inputs):
    panel, states, tickers, companies = inputs
    first_only = states.filter(pl.col("accn").str.ends_with("-0"))
    out = build_features(panel, first_only, tickers, companies, start=panel.dates[0])
    age = (out["date"] - panel.dates[5]).dt.total_days()
    assert (
        out.filter(age > MAX_FILING_AGE_DAYS)["earnings_yield"].null_count()
        == out.filter(age > MAX_FILING_AGE_DAYS).height
    )


def test_financial_strength_features():
    rows = pl.DataFrame({
        "market_value": [100.0, 100.0, 100.0, None],
        "volatility": [0.02, 0.02, 0.02, 0.02],
        "lt_debt": [0.0, 100.0, 100.0, None],
        "debt_cur": [0.0, 0.0, 50.0, None],
        "liabilities": [10.0, 120.0, 170.0, None],
        "operating_income": [5.0, 30.0, 30.0, 1.0],
        "interest": [None, 6.0, 10.0, None],
    })  # fmt: skip
    out = rows.select(_distance_to_default(), _interest_coverage())
    dd, cov = out["distance_to_default"].to_list(), out["interest_coverage"].to_list()
    assert dd[0] == MAX_DISTANCE and cov[0] == MAX_COVERAGE  # no debt
    assert 0 < dd[2] < dd[1] < MAX_DISTANCE  # more debt due soon: closer to default
    assert cov[1:3] == pytest.approx([5.0, 3.0])
    assert dd[3] is None and cov[3] is None  # unknown debt


def test_roa_stability_uses_only_known_annual_filings():
    rows = [(1, "10-K", date(2000 + y, 3, 1), date(1999 + y, 12, 31), roa * 100, 100.0)
            for y, roa in enumerate([0.10, 0.12, 0.08, 0.30])]  # fmt: skip
    rows.append((1, "10-Q", date(2003, 5, 1), date(2003, 3, 31), 5.0, 100.0))
    states = pl.DataFrame(rows, schema=["cik", "form", "filed", "period_end", "net_income",
                                        "assets"], orient="row")  # fmt: skip
    out = roa_stability(states).sort("filed")["_roa_volatility"].to_list()
    assert out[:2] == [None, None]  # fewer than three years
    assert out[2] == pytest.approx(0.02)  # std of 0.10, 0.12, 0.08
    assert out[3] == pytest.approx(pl.Series([0.10, 0.12, 0.08, 0.30]).std())
    assert out[4] == out[3]  # a 10-Q carries the latest annual figure
