import numpy as np
import polars as pl
import pytest

from portfolio_lab.backtest.costs import CostModel
from portfolio_lab.backtest.metrics import compute
from portfolio_lab.backtest.results import RunResult, list_runs, load_run, runs_dir, save_run


def test_metrics_constant_return():
    r = np.full(252, 0.001)
    m = compute(r, np.zeros(252))
    assert m["cagr"] == pytest.approx(1.001**252 - 1)
    assert m["total_return"] == pytest.approx(1.001**252 - 1)
    assert m["volatility"] == pytest.approx(0)
    assert m["max_drawdown"] == 0


def test_metrics_drawdown_and_beta():
    r = np.array([0.1, -0.5, 0.2])
    m = compute(
        r,
        np.zeros(3),
        benchmark=r / 2,
        turnover=np.array([1.0, 0, 0]),
        holdings=np.array([2, 2, 1]),
    )
    assert m["max_drawdown"] == pytest.approx(0.55 / 1.1 - 1)
    assert m["max_drawdown_days"] == 2
    assert m["beta"] == pytest.approx(2.0)
    assert m["avg_holdings"] == pytest.approx(5 / 3)
    assert m["turnover_annual"] == pytest.approx(1.0 / (3 / 252))


def test_cost_buckets():
    model = CostModel(
        half_spread_bps=5, impact_buckets=((0.01, 0.0), (float("inf"), 50.0)), notional=1e6
    )
    trades = np.array([0.1, -0.1, 0.1])
    adv = np.array([1e9, 1e6, np.nan])  # tiny participation, 10% participation, unknown
    expected = 0.1 * 5e-4 + 0.1 * 55e-4 + 0.1 * 55e-4
    assert model.cost(trades, adv) == pytest.approx(expected)
    assert model.cost(np.zeros(3), adv) == 0


def _result():
    daily = pl.DataFrame({"date": ["2024-01-03"], "nav": [1.01], "ret": [0.01]})
    weights = pl.DataFrame({"date": ["2024-01-02"], "symbol": ["A"], "weight": [1.0]})
    return RunResult({"strategy": "fixed", "params": {"k": 1}}, {"cagr": 0.1}, daily, weights)


def test_save_list_load_roundtrip(tmp_path):
    run_id = save_run(_result(), tmp_path)
    assert run_id.endswith(tuple("0123456789abcdef")) and "-fixed-" in run_id
    (runs_dir(tmp_path) / "incomplete").mkdir()  # no _SUCCESS: must be ignored
    runs = list_runs(tmp_path)
    assert [r["meta"]["run_id"] for r in runs] == [run_id]
    loaded = load_run(tmp_path, run_id)
    assert loaded.metrics == {"cagr": 0.1}
    assert loaded.daily.equals(_result().daily)
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path, "incomplete")
