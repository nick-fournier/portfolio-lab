from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from portfolio_lab.research.forecaster import nine
from portfolio_lab.strategies.meanvar.nine import (
    MIN_SLOPE_MONTHS,
    NineInputs,
    ar1_annual,
    ar1_forecast_sum,
)

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
    slopes = [{"date": _month(m), **{t: rng.normal(0, 0.01) for t in nine.TERMS}, "center": 0.08}
              for m in range(months)]  # fmt: skip
    pl.DataFrame(slopes).write_parquet(tmp_path / nine.SLOPES)
    return NineInputs(tmp_path)


def _prices(seed=1, days=260):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2012-01-02", periods=days)
    return pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.02, (days, len(SYMBOLS))), axis=0)),
                        index=index, columns=SYMBOLS)  # fmt: skip


def test_expected_returns_keep_the_forecasts_order_with_the_ar1_spread(tmp_path):
    inputs = _inputs(tmp_path)
    asof = _month(30) + timedelta(days=2)
    mu = inputs.expected(asof, _prices(), risk_free=0.03)
    assert mu.std() == pytest.approx(ar1_annual(_prices()).std())
    forecast = pd.Series(inputs.forecasts(asof))
    vol = _prices().pct_change().std() * 252**0.5
    raw = ((forecast - forecast.mean()) / forecast.std() * vol).reindex(mu.index)
    assert np.allclose((mu - 0.03) / raw, ((mu - 0.03) / raw).iloc[0])  # T-bill + k * raw
    assert inputs.expected(_month(30) + timedelta(days=20), _prices(), 0.03).empty


def test_ar1_sum_matches_iterating_the_fitted_recursion():
    rng = np.random.default_rng(4)
    r = np.zeros(500)
    for t in range(1, 500):
        r[t] = 0.0004 + 0.3 * r[t - 1] + rng.normal(0, 0.01)
    phi, c = np.polyfit(r[:-1], r[1:], 1)
    path, last = [], r[-1]
    for _ in range(21):
        last = c + phi * last
        path.append(last)
    assert ar1_forecast_sum(r, 21) == pytest.approx(sum(path))


def test_covariance_is_symmetric_positive_and_uses_only_known_months(tmp_path):
    inputs = _inputs(tmp_path)
    prices = _prices()
    cov = inputs.covariance(_month(35), prices)
    assert list(cov.index) == SYMBOLS and np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > 0
    early = inputs.covariance(_month(MIN_SLOPE_MONTHS - 2), prices)  # too few payoffs: price only
    log_returns = np.log(prices).diff().dropna()
    assert np.diag(early) == pytest.approx(log_returns.var().to_numpy() * 252, rel=0.2)
    # rewriting months on or after the decision date changes nothing
    changed = pl.read_parquet(tmp_path / nine.FILE).with_columns(
        pl.when(pl.col("date") >= _month(35)).then(9.9).otherwise(pl.col("actual")).alias("actual")
    )
    changed.write_parquet(tmp_path / nine.FILE)
    assert np.allclose(NineInputs(tmp_path).covariance(_month(35), prices), cov)
