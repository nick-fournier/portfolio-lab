import os
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_lab.backtest.engine import BacktestConfig, _params, describe, label, run
from portfolio_lab.research.dataview import DataView
from portfolio_lab.strategies.base import create
from portfolio_lab.strategies.meanvar import forecast as forecast_mod
from portfolio_lab.strategies.meanvar.forecast import Forecaster, ForecastSpec, forecast_one
from portfolio_lab.strategies.meanvar.optimize import _within_bounds, optimize

GROWTH = 0.0004  # daily


def _walk(n=252, seed=0, drift=GROWTH, vol=0.01):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + rng.normal(drift, vol, n))


def test_historical_mean_is_annualized_geometric_growth():
    prices = 100 * (1 + GROWTH) ** np.arange(253)
    mu = forecast_one(prices, ForecastSpec("historical_mean"))
    assert mu == pytest.approx((1 + GROWTH) ** 252 - 1)


def test_legacy_arima_recovers_simulated_coefficients():
    # ARIMA(3,2,0) == AR(3) on second differences; least squares recovers the coefficients
    # and its forecast equals iterating them forward and integrating twice.
    rng = np.random.default_rng(0)
    true = np.array([-0.5, -0.3, -0.1])
    d2 = np.zeros(5000)
    for t in range(3, 5000):
        d2[t] = true @ d2[t - 3 : t][::-1] + rng.normal(0, 0.1)
    prices = 100 + np.cumsum(np.cumsum(d2))
    lags = np.column_stack([d2[2 - k : len(d2) - 1 - k] for k in range(3)])
    coef, *_ = np.linalg.lstsq(lags, d2[3:], rcond=None)
    np.testing.assert_allclose(coef, true, atol=0.03)

    hist = list(d2[-3:])
    for _ in range(5):
        hist.append(float(coef @ np.array(hist[-1:-4:-1])))
    diff1, level = prices[-1] - prices[-2], prices[-1]
    for x in hist[3:]:
        diff1 += x
        level += diff1
    assert forecast_mod.ar_diff_forecast(prices, 5) == pytest.approx(level)


def test_legacy_arima_is_finite_bounded_and_deterministic():
    results = [forecast_one(_walk(seed=s), ForecastSpec("arima320_price")) for s in range(20)]
    assert all(np.isfinite(mu) and -0.99 <= mu <= 5.0 for mu in results)
    assert results == [
        forecast_one(_walk(seed=s), ForecastSpec("arima320_price")) for s in range(20)
    ]


def test_forecast_rejects_bad_inputs():
    assert np.isnan(forecast_one(_walk(10), ForecastSpec()))  # too short
    assert np.isnan(forecast_one(np.r_[_walk(100), -1.0], ForecastSpec()))  # non-positive
    with pytest.raises(ValueError, match="unknown forecast model"):
        ForecastSpec("crystal_ball")


def test_forecaster_caches_in_memory_and_on_disk(tmp_path, monkeypatch):
    calls = []
    real = forecast_mod.forecast_one

    def counting(prices, spec):
        calls.append(1)
        return real(prices, spec) if prices[0] > 50 else np.nan

    monkeypatch.setattr(forecast_mod, "forecast_one", counting)
    windows = {"AAA": _walk(seed=1), "BBB": _walk(seed=2), "BAD": _walk(seed=3) / 10}
    spec = ForecastSpec("historical_mean")
    asof = date(2024, 1, 31)

    first = Forecaster(spec, tmp_path).forecast(asof, windows)
    assert set(first) == {"AAA", "BBB"}  # the failed fit is omitted...
    assert len(calls) == 3
    again = Forecaster(spec, tmp_path).forecast(asof, windows)  # new instance: loads the cache
    assert again == first and len(calls) == 3  # ...and cached, so it isn't refit
    other_model = Forecaster(ForecastSpec("arima320_price"), tmp_path)
    other_model.forecast(asof, {"AAA": windows["AAA"]})
    assert len(calls) == 4  # different model configuration, different cache


def _prices(n_symbols=12, n=252):
    data = {f"S{i:02d}": _walk(n, seed=i, drift=0.0002 * i) for i in range(n_symbols)}
    return pd.DataFrame(data, index=pd.bdate_range("2023-01-02", periods=n))


def test_optimize_respects_bounds_and_budget():
    prices = _prices()
    mu = pd.Series({s: 0.05 + 0.02 * i for i, s in enumerate(prices.columns)})
    weights = optimize(mu, prices, risk_free=0.03, max_weight=0.2)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    assert max(weights.values()) <= 0.2 + 1e-6
    assert set(weights) <= set(prices.columns)


def test_optimize_falls_back_when_max_sharpe_is_infeasible():
    prices = _prices()
    mu = pd.Series(-0.05, index=prices.columns)  # every forecast below the risk-free rate
    weights = optimize(mu, prices, risk_free=0.05, max_weight=0.2)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_optimize_edge_cases():
    prices = _prices(n_symbols=3)
    mu = pd.Series({"S00": 0.1, "S01": 0.2, "S02": 0.3})
    weights = optimize(mu, prices, risk_free=0.0, max_weight=0.1)  # cap raised to 1/3
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    assert optimize(mu[["S00"]], prices, risk_free=0.0) == {}
    with pytest.raises(ValueError, match="unknown objective"):
        optimize(mu, prices, risk_free=0.0, objective="yolo")


def test_meanvar_strategy_on_panel(make_panel):
    panel = make_panel(symbols=[f"S{i:02d}" for i in range(12)])
    strategy = create("meanvar", model="historical_mean", max_weight=0.2)
    view = DataView(panel, 200)
    weights = strategy.target_weights(view)
    assert weights and sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    assert set(weights) <= set(view.eligible())
    assert max(weights.values()) <= 0.2 + 1e-6


def test_meanvar_params_exclude_runtime_fields(tmp_path):
    strategy = create("meanvar", top_n="50")
    strategy.cache_dir, strategy.workers = tmp_path, 4
    params = _params(strategy)
    assert params["top_n"] == 50
    assert "cache_dir" not in params and "workers" not in params


@dataclass
class Closing:
    schedule: str = "M"
    name: str = "closing"
    closed: bool = False

    def target_weights(self, view):
        return {}

    def close(self):
        self.closed = True


def test_engine_closes_strategies(make_panel):
    panel = make_panel()
    strategy = Closing()
    run(strategy, panel, BacktestConfig(panel.dates[10], panel.dates[40]))
    assert strategy.closed


def test_pool_workers_are_pinned_to_one_math_thread(monkeypatch):
    # Workers start lazily on the first task, so the cap must still be set afterwards;
    # without it each worker starts a thread per core (measured ~9x slower on orange).
    for var in forecast_mod._THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    forecaster = Forecaster(ForecastSpec("historical_mean"), workers=2)
    forecaster._executor()
    try:
        assert all(os.environ[var] == "1" for var in forecast_mod._THREAD_ENV_VARS)
    finally:
        forecaster.close()


def test_solver_tolerance_is_cleaned_up():
    # e.g. a solver returning weights that sum to 1.0026 and slightly exceed the cap
    cleaned = _within_bounds({"A": 0.1003, "B": 0.5, "C": 0.4023, "D": 0.0}, cap=0.5)
    assert sum(cleaned.values()) == pytest.approx(1.0)
    assert max(cleaned.values()) <= 0.5 and "D" not in cleaned
    small = {"A": 0.3, "B": 0.3}
    assert _within_bounds(small, cap=0.5) == small  # under budget: left alone (cash)


def test_label_shows_only_non_default_params():
    assert label(create("meanvar")) == "meanvar"
    assert label(create("meanvar", model="arima320_price", top_n=50)) == (
        "meanvar (model=arima320_price, top_n=50)"
    )


def test_describe_summary_only():
    summary = describe(create("meanvar"), summary_only=True)
    assert summary.startswith("Mean-variance optimization") and "momentum" not in summary
    assert "momentum" in describe(create("meanvar"))
