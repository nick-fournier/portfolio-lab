from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.forecasting import grade_months, summarize


def _forecasts(noise: float, months: int = 24, n: int = 200, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(months):
        day = date(2010 + m // 12, m % 12 + 1, 28)
        signal = rng.normal(0, 0.02, n)
        actual = signal + rng.normal(0, noise, n)
        rows += [(day, f"S{k}", signal[k], actual[k] - actual.mean()) for k in range(n)]
    return pl.DataFrame(rows, schema=["date", "symbol", "forecast", "actual"], orient="row")


def test_perfect_forecasts_grade_perfectly():
    months = grade_months(_forecasts(noise=0.0))
    assert months["ic"].min() == pytest.approx(1.0)
    assert months["slope"].median() == pytest.approx(1.0, abs=0.05)
    assert months["r2"].mean() == pytest.approx(1.0, abs=0.01)
    assert (months["spread"] > 0).all()


def test_noisy_forecasts_have_realistic_slope_and_low_r2():
    months = grade_months(_forecasts(noise=0.10))
    assert 0.5 < months["slope"].median() < 1.5
    assert months["r2"].mean() < 0.1
    summary = summarize(months, split=date(2011, 1, 1))
    assert summary["all"]["months"] == 24 and summary["first_half"]["months"] == 12
    assert summary["all"]["ic"] > 0
