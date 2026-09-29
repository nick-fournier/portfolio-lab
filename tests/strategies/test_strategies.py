import numpy as np
import pytest

import portfolio_lab.strategies  # noqa: F401  (registers built-in strategies)
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import Panel
from portfolio_lab.strategies.base import REGISTRY, Composed, create
from portfolio_lab.strategies.construct import equal_weight, top_k_equal

ASOF = 150


def _poisoned(panel: Panel, after: int, seed: int = 1) -> Panel:
    """Copy of ``panel`` with every row after ``after`` replaced by garbage."""
    rng = np.random.default_rng(seed)
    fields = {}
    for name, array in panel.fields.items():
        garbage = array.copy()
        garbage[after + 1 :] = rng.choice(
            [np.nan, 1e9, -0.99, 0.0], size=garbage[after + 1 :].shape
        )
        fields[name] = garbage
    eligible = panel.eligible.copy()
    eligible[after + 1 :] = rng.random(eligible[after + 1 :].shape) > 0.5
    rf = panel.rf_daily.copy()
    rf[after + 1 :] = 1.0
    return Panel(panel.dates, panel.symbols, fields, eligible, rf, panel.universe)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_no_strategy_can_see_the_future(name, make_panel):
    """Corrupting every row after the decision date must not change any strategy's weights."""
    panel = make_panel(symbols=[f"S{i:02d}" for i in range(12)])
    clean = create(name).target_weights(DataView(panel, ASOF))
    poisoned = create(name).target_weights(DataView(_poisoned(panel, ASOF), ASOF))
    assert clean == poisoned
    assert clean, f"{name} produced no weights on the test panel"


def test_equal_weight_strategy_top_n(make_panel):
    panel = make_panel(symbols=[f"S{i:02d}" for i in range(12)])
    view = DataView(panel, ASOF)
    weights = create("equal_weight", top_n=3).target_weights(view)
    assert set(weights) == set(view.top_liquid(3))
    assert sum(weights.values()) == pytest.approx(1.0)


def test_buy_hold_waits_for_a_price(make_panel):
    panel = make_panel()
    assert create("buy_hold").target_weights(DataView(panel, ASOF)) == {"SPY": 1.0}
    assert create("buy_hold", symbol="NOPE").target_weights(DataView(panel, ASOF)) == {}


def test_construct_helpers_and_composed(make_panel):
    assert equal_weight(["A", "B", "A"]) == {"A": 0.5, "B": 0.5}
    assert equal_weight([]) == {}
    assert top_k_equal({"A": 1.0, "B": 3.0, "C": 3.0}, 2) == {"B": 0.5, "C": 0.5}

    strategy = Composed(
        name="momentum_top2",
        schedule="M",
        signal=lambda view: view.returns(20, view.eligible()).sum().to_dict(),
        construct=lambda scores, view: top_k_equal(scores, 2),
    )
    weights = strategy.target_weights(DataView(make_panel(), ASOF))
    assert len(weights) == 2 and sum(weights.values()) == pytest.approx(1.0)


def test_unknown_strategy():
    with pytest.raises(KeyError, match="unknown strategy"):
        create("nope")
