import numpy as np
import pandas as pd
import pytest

from portfolio_lab.research import stress
from portfolio_lab.research.stress import StressGauge, absorption, panic, tilt


def test_tilt_is_zero_below_the_band_and_one_above():
    assert tilt(0.4, 0.5, 0.9) == 0.0
    assert tilt(0.7, 0.5, 0.9) == pytest.approx(0.5)
    assert tilt(0.95, 0.5, 0.9) == 1.0


def test_gauge_is_neutral_until_it_has_history(monkeypatch):
    monkeypatch.setattr(stress, "MIN_HISTORY", 3)
    gauge = StressGauge()
    levels = [gauge.update({"x": v})[0] for v in (1.0, 2.0, 3.0, 4.0, 5.0, 0.5)]
    assert levels[:4] == [0.5] * 4  # input needs 3 readings, then the composite needs 3
    assert levels[4] == 1.0  # the highest reading so far
    assert levels[5] < 0.5  # the lowest


def test_absorption_is_higher_when_stocks_move_together():
    rng = np.random.default_rng(0)
    common = rng.normal(size=(250, 1))
    together = pd.DataFrame(common + 0.2 * rng.normal(size=(250, 20)))
    apart = pd.DataFrame(rng.normal(size=(250, 20)))
    assert absorption(together) > 0.8 > absorption(apart)


def test_panic_is_zero_at_a_high_and_positive_after_a_fall():
    calm = pd.Series([0.001] * 600)
    fall = pd.concat([calm, pd.Series([-0.02] * 40)], ignore_index=True)
    assert panic(calm) == pytest.approx(0.0)
    assert panic(fall) > 0.0
