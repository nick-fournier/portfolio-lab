from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.forecasting import blend, calibrate, grade_months, summarize


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


def test_calibration_learns_the_scale_from_earlier_years():
    overconfident = _forecasts(noise=0.02, months=72).with_columns(
        (pl.col("forecast") * 4).alias("forecast"), pl.lit(0.0).alias("size")
    )
    calibrated = calibrate(overconfident)
    assert calibrated["date"].min().year == 2013  # three earlier years needed
    assert calibrated["factor"].mean() == pytest.approx(0.25, rel=0.15)
    assert grade_months(calibrated)["slope"].median() == pytest.approx(1.0, abs=0.2)


def test_blend_learns_to_favor_the_better_model():
    good = _forecasts(noise=0.02, months=72).with_columns(pl.lit(0.0).alias("size"))
    rng = np.random.default_rng(1)
    junk = good.with_columns(pl.Series("forecast", rng.normal(size=good.height)))
    half = grade_months(blend(good, junk, 0.5))["ic"].mean()
    learned = blend(good, junk, None)
    assert learned.filter(pl.col("date").dt.year() >= 2013)["weight"].min() >= 0.9
    assert grade_months(learned)["ic"].mean() > half
