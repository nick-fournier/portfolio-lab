from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.funds import FUNDS, PAIRS, compare

DAYS = [date(2020, 1, 1) + timedelta(days=k) for k in range(600)]


def _series(ret, start=0):
    return pl.DataFrame({"date": DAYS[start:], "ret": ret[start:]})


def test_compare_full_and_common_periods():
    rng = np.random.default_rng(0)
    spy = rng.normal(0.0004, 0.01, len(DAYS))
    ours = 2 * spy  # twice the market's moves: beta 2
    series = {
        "SPY": ("S&P 500", "passive", _series(spy)),
        "ours: x": ("x", "ours", _series(ours)),
        "LATE": ("A late fund", "active_etf", _series(spy, start=300)),
    }
    summary, growth = compare(series, None, DAYS[0])
    full = {r["key"]: r for r in summary.filter(pl.col("period") == "full").to_dicts()}
    common = {r["key"]: r for r in summary.filter(pl.col("period") == "common").to_dicts()}
    assert full["SPY"]["start"] == DAYS[0] and full["LATE"]["start"] == DAYS[300]
    assert {r["start"] for r in common.values()} == {DAYS[300]}  # everyone from the latest start
    assert full["ours: x"]["beta"] == pytest.approx(2.0, abs=0.01)
    assert full["SPY"]["alpha"] == pytest.approx(0.0, abs=1e-9)
    last = growth.filter(pl.col("date") == DAYS[-1])
    spy_growth = last.filter(pl.col("key") == "SPY")["growth"][0]
    assert spy_growth == pytest.approx(np.prod(1 + spy[300:]))


def test_every_pair_names_a_registered_fund():
    symbols = {f.symbol for f in FUNDS}
    assert set(PAIRS.values()) <= symbols
    assert len(symbols) == len(FUNDS)  # no duplicates


def test_young_funds_do_not_shorten_the_common_period():
    rng = np.random.default_rng(1)
    spy = rng.normal(0.0004, 0.01, len(DAYS))
    series = {
        "SPY": ("S&P 500", "passive", _series(spy)),
        "ours: x": ("x", "ours", _series(spy * 1.5)),
        "YOUNG": ("A young fund", "active_etf", _series(spy, start=450)),
    }
    summary, _ = compare(series, None, DAYS[0], ours="ours: x")
    common = {r["key"]: r for r in summary.filter(pl.col("period") == "common").to_dicts()}
    assert common["SPY"]["start"] == DAYS[0]  # not pushed back to the young fund's launch
    assert common["YOUNG"]["start"] == DAYS[450]  # measured from its own start
    same = summary.filter(pl.col("period") == "ours_since:YOUNG").row(0, named=True)
    assert same["key"] == "ours: x" and same["start"] == DAYS[450]
