from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.conditions import (
    caution_dial,
    conditional_ic,
    current_conditions,
)

DATES = [date(2024, m, 28) for m in range(1, 13)]


def _env():
    # VIX percentile: calm (0.1) for the first six months, stressed (0.9) after.
    return pl.DataFrame({"date": DATES, "vix_pct": [0.1] * 6 + [0.9] * 6})


def test_conditional_ic_splits_by_the_condition_on_each_date():
    scores = pl.DataFrame(
        {"signal": "roa", "pool": "all", "horizon": 21, "date": DATES,
         "ic": [0.05, 0.03, 0.04, 0.06, 0.05, 0.05, -0.01, 0.0, -0.02, 0.01, -0.01, 0.0]}
    )  # fmt: skip
    out = conditional_ic(scores, _env())
    assert out["condition"].unique().to_list() == ["VIX (vs history)"]
    calm, stressed = out.row(0, named=True), out.row(1, named=True)
    assert (calm["bucket"], stressed["bucket"]) == ("calm", "stressed")
    assert calm["mean_ic"] == pytest.approx(0.28 / 6)
    assert calm["months"] == 6


def test_caution_dial_measures_the_market_ahead():
    days = [date(2024, 1, 1) + np.timedelta64(k, "D").astype(object) for k in range(400)]
    market = np.full(400, 0.001)
    market[200:] = -0.001  # the market falls in the second half
    index = {d: k for k, d in enumerate(days)}
    env = pl.DataFrame({"date": [days[10], days[199]], "vix_pct": [0.1, 0.9]})
    dial = caution_dial(env, days, market, index).filter(pl.col("months_ahead") == 3)
    calm, stressed = dial.row(0, named=True), dial.row(1, named=True)
    assert calm["mean_return"] == pytest.approx(1.001**63 - 1)
    assert stressed["mean_return"] < 0 and stressed["mean_drawdown"] < 0


def test_current_conditions():
    now = current_conditions(_env())
    assert now == [{"condition": "VIX (vs history)", "bucket": "stressed", "value": 0.9}]
