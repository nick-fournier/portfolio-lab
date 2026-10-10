from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

from portfolio_lab.research.forecaster import nine
from portfolio_lab.strategies.base import create
from portfolio_lab.strategies.meanvar.nine import NineInputs

SYMBOLS = [f"S{k}" for k in range(6)]


def _month(k: int) -> date:
    return date(2010 + k // 12, k % 12 + 1, 28)


def _inputs(tmp_path, months=40, seed=0) -> NineInputs:
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(months):
        for s in SYMBOLS:
            rows.append({"date": _month(m), "symbol": s, "actual": rng.normal(0, 0.05),
                         "forecast": rng.normal(0, 0.01), "spy12": 0.1,
                         **{t: rng.normal() for t in nine.TERMS}})  # fmt: skip
    pl.DataFrame(rows).write_parquet(tmp_path / nine.FILE)
    return NineInputs(tmp_path)


def _prices(seed=1, days=260):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2012-01-02", periods=days)
    return pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.02, (days, len(SYMBOLS))), axis=0)),
                        index=index, columns=SYMBOLS)  # fmt: skip


def test_expected_returns_follow_grinolds_rule_with_the_ic_known_by_then(tmp_path):
    inputs = _inputs(tmp_path)
    ics = [0.02 * (m % 5) for m in range(40)]
    inputs._ic = pl.DataFrame({"date": [_month(m) for m in range(40)], "ic": ics})
    asof = _month(30) + timedelta(days=2)
    mu = inputs.expected(asof, _prices(), risk_free=0.03)
    ic = float(np.mean(ics[:30]))  # months before this forecast's month
    forecast = pd.Series(inputs.forecasts(asof))
    vol = _prices().pct_change().std() * 252**0.5
    z = (forecast - forecast.mean()) / forecast.std()
    assert np.allclose(mu, (0.03 + 12**0.5 * ic * vol * z).reindex(mu.index))
    assert inputs.expected(_month(30) + timedelta(days=20), _prices(), 0.03).empty  # stale forecast
    inputs._ic = inputs._ic.head(5)
    assert inputs.expected(asof, _prices(), 0.03).empty  # too few graded months


def test_the_forecaster_starts_at_its_first_forecast(tmp_path):
    _inputs(tmp_path)
    strategy = create("meanvar", expected="nine", forecaster_dir=tmp_path)
    assert strategy.first_decision() == _month(0)
    assert create("meanvar").first_decision() is None
