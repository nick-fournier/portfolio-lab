from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from portfolio_lab.core.calendar import rebalance_dates
from portfolio_lab.research.features import MAX_FILING_AGE_DAYS, build_features
from portfolio_lab.research.fundamentals import CONCEPTS

from ..strategies.test_strategies import _poisoned

SYMBOLS = [f"S{i:02d}" for i in range(12)]


def _states(dates, symbols=SYMBOLS, step=63):
    """A filing every ``step`` sessions per company, with simple growing fundamentals."""
    rows = []
    for k, _ in enumerate(symbols):
        for n, i in enumerate(range(5, len(dates), step)):
            row = {p + c: None for p in ("", "prior_") for c in CONCEPTS}
            row |= {
                "cik": k, "accn": f"{k}-{n}", "form": "10-Q", "filed": dates[i],
                "period_end": dates[i] - timedelta(days=40), "shares_out": 1e6 * (k + 1),
                "net_income": 10.0 + n + k, "prior_net_income": 8.0 + k, "equity": 500.0 + n,
                "assets": 1000.0 + 10 * n, "prior_assets": 1000.0, "revenue": 300.0 + k,
                "prior_revenue": 280.0, "cfo": 12.0 + k, "liabilities": 400.0,
            }  # fmt: skip
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


@pytest.fixture
def inputs(make_panel):
    panel = make_panel(symbols=SYMBOLS, days=400)
    tickers = pl.DataFrame({"symbol": SYMBOLS, "cik": range(len(SYMBOLS))})
    industry = pl.DataFrame({"symbol": SYMBOLS, "sic": [3570] * 6 + [6022] * 6})
    return panel, _states(panel.dates).join(tickers, on="cik"), industry


def _month_end(panel, after=260):
    days = rebalance_dates(panel.dates[after:-5], "M")
    return panel.date_index[days[1]]


def test_features_are_point_in_time(inputs):
    panel, states, industry = inputs
    i = _month_end(panel)
    day = panel.dates[i]
    start = panel.dates[0]
    clean = build_features(panel, states, industry, start=start).filter(pl.col("date") == day)
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
        _poisoned(panel, i), pl.concat([changed, extra]), industry, start=start
    ).filter(pl.col("date") == day)
    assert clean.height == len(SYMBOLS)
    assert clean.equals(poisoned)


def test_market_value_and_ratios(inputs):
    panel, states, industry = inputs
    i = _month_end(panel)
    day = panel.dates[i]
    row = (
        build_features(panel, states, industry, start=panel.dates[0])
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
    panel, states, industry = inputs
    first_only = states.filter(pl.col("accn").str.ends_with("-0"))
    out = build_features(panel, first_only, industry, start=panel.dates[0])
    age = (out["date"] - panel.dates[5]).dt.total_days()
    assert (
        out.filter(age > MAX_FILING_AGE_DAYS)["earnings_yield"].null_count()
        == out.filter(age > MAX_FILING_AGE_DAYS).height
    )
