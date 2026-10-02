from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from portfolio_lab.strategies.meanvar.expected import (
    MIN_MONTHS,
    ForecastCheck,
    PooledModel,
    james_stein,
)


def test_james_stein_pulls_noisy_stocks_harder():
    mu = pd.Series([0.5, -0.1, 0.2, 0.1, 0.0, 0.3])
    calm = james_stein(mu, pd.Series(0.1, index=mu.index))
    noisy = james_stein(mu, pd.Series(0.6, index=mu.index))
    assert calm.mean() == pytest.approx(mu.mean()) == pytest.approx(noisy.mean())
    assert (calm - mu.mean()).abs().sum() > (noisy - mu.mean()).abs().sum()
    assert calm.rank().equals(mu.rank())  # shrinking keeps the order


class _View:
    """Monthly features where next month's return is 2% x the rank of ``mom_12_1``."""

    def __init__(self, months: int, n: int = 80, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dates = [date(2000, 1, 31) + timedelta(days=31 * k) for k in range(months)]
        self.asof = self.dates[-1]
        rows, self.returns = [], {}
        for d in self.dates:
            mom = rng.normal(size=n)
            rank = pd.Series(mom).rank(pct=True).to_numpy() - 0.5
            noise = rng.normal(0, 0.05, size=n)
            self.returns[d] = dict(
                zip([f"S{k}" for k in range(n)], 0.01 + 0.02 * rank + noise, strict=True)
            )
            rows += [(d, f"S{k}", mom[k], rng.normal(), float(n - k)) for k in range(n)]
        self.table = pl.DataFrame(
            rows, schema=["date", "symbol", "mom_12_1", "ret_1m", "log_adv"], orient="row"
        )

    def feature_history(self, columns, since=None):
        rows = self.table.filter(pl.col("date") <= self.asof)
        if since is not None:
            rows = rows.filter(pl.col("date") >= since)
        return rows.with_columns(
            *[pl.lit(0.0).alias(c) for c in columns if c not in rows.columns]
        ).select("date", "symbol", *columns)

    def period_returns(self, symbols, start, end):
        return pd.Series({s: self.returns[start][s] for s in symbols})


def test_pooled_model_learns_a_shared_signal():
    view = _View(MIN_MONTHS + 12)
    model = PooledModel("pooled_top")
    symbols = [f"S{k}" for k in range(80)]
    mu = model.predict(view, symbols)
    latest = view.table.filter(pl.col("date") == view.asof)
    corr = pd.Series(mu.to_numpy()).corr(
        pd.Series(latest["mom_12_1"].to_numpy()), method="spearman"
    )
    assert corr > 0.95  # the planted momentum effect, learned across stocks
    assert PooledModel("pooled_top").predict(_View(MIN_MONTHS - 5), symbols) is None


def test_forecast_check_grades_the_previous_month():
    view = _View(3)
    check = ForecastCheck()
    view.asof = view.dates[0]
    first = pd.Series({s: view.returns[view.dates[0]][s] * 12 for s in view.returns[view.dates[0]]})
    check.record(view, first)  # perfect foresight of the next month
    check.record(view, first)  # a second call on the same date is ignored
    view.asof = view.dates[1]
    check.record(view, first)
    [graded] = check.log
    assert graded["ic"] == pytest.approx(1.0) and graded["slope"] == pytest.approx(1.0)
