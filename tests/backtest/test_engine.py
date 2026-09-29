from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest

from portfolio_lab.backtest.costs import CostModel
from portfolio_lab.backtest.engine import BacktestConfig, run, validate_weights
from portfolio_lab.core.calendar import sessions
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import Panel

DAYS = sessions(date(2024, 1, 2), date(2024, 1, 8))  # 5 sessions
FLAT_COST = CostModel(half_spread_bps=10, impact_buckets=((float("inf"), 0.0),))


@dataclass
class Fixed:
    weights: dict
    schedule: str = "D"
    name: str = "fixed"

    def target_weights(self, view):
        return dict(self.weights)


def _panel(ret_cc, ret_co, rf=0.0, fell_to_otc=(), traded=None):
    """Two stocks A, B plus a price-only SPY, all eligible from day 0 (closes follow ret_cc)."""
    shape = (len(DAYS), 3)
    fields = {
        "close": np.where(np.isfinite(np.array(ret_cc, dtype=float)), 10.0, np.nan),
        "ret_cc": np.array(ret_cc, dtype=float),
        "ret_co": np.array(ret_co, dtype=float),
        "adv": np.full(shape, 1e12),
    }
    eligible = np.zeros(shape, dtype=bool)
    eligible[:, :2] = True
    rf = np.full(len(DAYS), rf)
    return Panel(
        DAYS,
        ["A", "B", "SPY"],
        fields,
        eligible,
        rf,
        ["A", "B"],
        fell_to_otc=fell_to_otc,
        traded=traded,
    )


def test_golden_nav_with_gap_intraday_drift_and_costs():
    cc = [[0, 0, 0], [0.10, -0.05, 0], [0.02, 0.04, 0], [0, 0, 0], [0, 0, 0]]
    co = [[0, 0, 0], [0.01, 0.00, 0], [0.03, -0.01, 0], [0, 0, 0], [0, 0, 0]]
    panel = _panel(cc, co)
    config = BacktestConfig(DAYS[0], DAYS[2], costs=FLAT_COST)
    result = run(Fixed({"A": 0.5, "B": 0.5}), panel, config)

    # Day 1: buy 50/50 at the open (turnover 1, 10 bps), then earn the open-to-close move.
    nav0 = 1 - 0.001
    a1 = nav0 * 0.5 * (1.10 / 1.01)
    b1 = nav0 * 0.5 * (0.95 / 1.00)
    nav1 = a1 + b1
    # Day 2: gap, rebalance the drifted weights back to 50/50, then intraday.
    a_open, b_open = a1 * 1.03, b1 * 0.99
    nav_open = a_open + b_open
    turnover2 = abs(0.5 - a_open / nav_open) + abs(0.5 - b_open / nav_open)
    nav_after = nav_open * (1 - turnover2 * 0.001)
    nav2 = nav_after * 0.5 * (1.02 / 1.03) + nav_after * 0.5 * (1.04 / 0.99)

    daily = result.daily
    assert daily["date"].to_list() == DAYS[1:3]
    np.testing.assert_allclose(daily["nav"].to_numpy(), [nav1, nav2], rtol=1e-12)
    np.testing.assert_allclose(daily["turnover"].to_numpy(), [1.0, turnover2], rtol=1e-12)
    np.testing.assert_allclose(daily["ret"].to_numpy(), [nav1 - 1, nav2 / nav1 - 1], rtol=1e-12)


def test_cash_earns_risk_free():
    zeros = [[0, 0, 0]] * len(DAYS)
    panel = _panel(zeros, zeros, rf=0.001)
    result = run(Fixed({"A": 0.5}), panel, BacktestConfig(DAYS[0], DAYS[-1], costs=FLAT_COST))
    nav = (1 - 0.0005) * (0.5 + 0.5 * 1.001)  # day 1: half in A, half cash earning 0.1%
    assert result.daily["nav"][0] == pytest.approx(nav)
    assert result.daily["cash"][0] == pytest.approx(0.5 * 1.001 * (1 - 0.0005) / nav)


def test_held_name_without_bars_is_liquidated():
    cc = [[0, 0, 0]] + [[np.nan, 0.01, 0]] * (len(DAYS) - 1)
    co = [[0, 0, 0]] + [[np.nan, 0.0, 0]] * (len(DAYS) - 1)
    panel = _panel(cc, co)
    config = BacktestConfig(DAYS[0], DAYS[-1], costs=FLAT_COST, max_missing_days=3)
    result = run(Fixed({"A": 0.5, "B": 0.5}, schedule="M"), panel, config)
    assert result.metrics["forced_liquidations"] == 1
    assert result.daily["holdings"].to_list() == [2, 2, 1, 1]


@pytest.mark.parametrize(
    ("fell_to_otc", "delisting_return", "a_exit"),
    [((), -0.30, 1.0), (("A",), -0.30, 0.70), (("A",), -1.0, 0.0), (("A",), 0.0, 1.0)],
)
def test_delisting_return_only_for_names_that_fell_to_otc(fell_to_otc, delisting_return, a_exit):
    # A's last bar is day 1; it never trades again.
    cc = [[0, 0, 0], [0.0, 0.0, 0]] + [[np.nan, 0.0, 0]] * (len(DAYS) - 2)
    co = [[0, 0, 0], [0.0, 0.0, 0]] + [[np.nan, 0.0, 0]] * (len(DAYS) - 2)
    panel = _panel(cc, co, fell_to_otc=fell_to_otc)
    assert panel.last_bar.tolist() == [1, len(DAYS) - 1, len(DAYS) - 1]
    config = BacktestConfig(
        DAYS[0], DAYS[-1], costs=FLAT_COST, max_missing_days=2, delisting_return=delisting_return
    )
    result = run(Fixed({"A": 0.5, "B": 0.5}, schedule="M"), panel, config)
    nav0 = 1 - 0.001
    assert result.daily["nav"][-1] == pytest.approx(nav0 * 0.5 * (1 + a_exit))
    assert result.metrics["forced_liquidations"] == 1
    assert result.metrics["otc_delistings"] == (1 if fell_to_otc else 0)
    assert result.meta["delisting_return"] == delisting_return


def test_gap_before_later_bars_exits_at_last_price_even_if_otc():
    # A misses days 1-2 but trades again on day 3: a halt, not a delisting.
    cc = [[0, 0, 0], [np.nan, 0, 0], [np.nan, 0, 0], [0.0, 0, 0], [0.0, 0, 0]]
    panel = _panel(cc, cc, fell_to_otc=("A",))
    config = BacktestConfig(DAYS[0], DAYS[-1], costs=FLAT_COST, max_missing_days=2)
    result = run(Fixed({"A": 0.5, "B": 0.5}, schedule="M"), panel, config)
    assert result.metrics["otc_delistings"] == 0
    assert result.daily["nav"][-1] == pytest.approx(1 - 0.001)


def test_frozen_zero_volume_bars_count_as_missing():
    # A halts after day 1 but keeps printing zero-volume bars at a frozen price.
    zeros = [[0, 0, 0]] * len(DAYS)
    traded = np.ones((len(DAYS), 3), dtype=bool)
    traded[2:, 0] = False
    panel = _panel(zeros, zeros, fell_to_otc=("A",), traded=traded)
    assert panel.last_bar[0] == 1
    config = BacktestConfig(DAYS[0], DAYS[-1], costs=FLAT_COST, max_missing_days=2)
    result = run(Fixed({"A": 0.5, "B": 0.5}, schedule="M"), panel, config)
    assert result.metrics["otc_delistings"] == 1
    assert result.daily["nav"][-1] == pytest.approx((1 - 0.001) * 0.5 * 1.7)


def test_benchmark_and_metadata():
    zeros = [[0, 0, 0.01]] * len(DAYS)
    panel = _panel(zeros, [[0, 0, 0]] * len(DAYS))
    result = run(Fixed({"SPY": 1.0}), panel, BacktestConfig(DAYS[0], DAYS[-1], costs=FLAT_COST))
    assert result.daily["benchmark_ret"].to_list() == [0.01] * 4
    assert result.meta["strategy"] == "fixed"
    assert result.meta["description"] == ""  # Fixed has no docstring
    assert result.meta["start"] == DAYS[0] and result.meta["end"] == DAYS[-1]
    assert any("Delisted" in c for c in result.meta["caveats"])
    assert result.weights["symbol"].unique().to_list() == ["SPY"]


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"C": 0.5}, "not allowed"),
        ({"A": 0.7, "B": 0.7}, "sum"),
        ({"A": -0.1}, "outside"),
        ({"A": float("nan")}, "outside"),
        ({"A": 0.9}, "outside"),  # above max_weight
    ],
)
def test_validate_weights_rejects(weights, message):
    zeros = [[0, 0, 0]] * len(DAYS)
    panel = _panel(zeros, zeros)
    with pytest.raises(ValueError, match=message):
        validate_weights(weights, DataView(panel, 1), panel, max_weight=0.8)


def test_validate_weights_allows_benchmark_and_cash():
    zeros = [[0, 0, 0]] * len(DAYS)
    panel = _panel(zeros, zeros)
    vector = validate_weights({"SPY": 0.3, "A": 0.2}, DataView(panel, 1), panel, max_weight=1.0)
    assert vector.tolist() == [0.2, 0.0, 0.3]
