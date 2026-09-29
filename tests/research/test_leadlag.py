from datetime import date

import numpy as np
import pytest

from portfolio_lab.core.calendar import sessions
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.leadlag import estimate, period_returns, predict
from portfolio_lab.research.panel import Panel
from portfolio_lab.signals.leadlag import LeadLag, month_start_view

DAYS = sessions(date(2023, 1, 3), date(2024, 12, 31))


def _panel(returns, symbols):
    returns = np.asarray(returns, dtype=float)
    fields = {
        "close": np.full(returns.shape, 10.0),
        "ret_cc": returns,
        "ret_co": np.zeros_like(returns),
        "adv": np.tile(np.arange(returns.shape[1], 0, -1, dtype=float) * 1e6, (len(returns), 1)),
    }
    eligible = np.ones(returns.shape, dtype=bool)
    return Panel(DAYS[: len(returns)], symbols, fields, eligible, np.zeros(len(returns)), symbols)


@pytest.fixture
def follower_panel():
    """LEAD is noise; FOL copies LEAD's return one day later; the rest are noise."""
    rng = np.random.default_rng(0)
    n, others = 300, 8
    lead = rng.normal(0, 0.02, n)
    fol = np.r_[0.0, lead[:-1]] * 0.8 + rng.normal(0, 0.005, n)
    noise = rng.normal(0, 0.02, (n, others))
    symbols = ["LEAD", "FOL", *[f"N{i}" for i in range(others)]]
    return _panel(np.column_stack([lead, fol, noise]), symbols)


def test_network_finds_the_leader_and_predicts_its_follower(follower_panel):
    view = DataView(follower_panel, 280)
    network = estimate(view, horizon=1, window=252, k=2)
    fol = network.symbols.index("FOL")
    assert network.symbols[network.leaders[fol][np.argmax(np.abs(network.weights[fol]))]] == "LEAD"
    assert network.weights[fol].max() > 0.7
    # Tomorrow FOL should move with LEAD's move today.
    nxt = DataView(follower_panel, 281)
    scores = predict(network, nxt, horizon=1)
    lead_today = follower_panel.field("ret_cc")[281, 0]
    assert np.sign(scores["FOL"]) == np.sign(lead_today)


def test_period_returns_compound_blocks_ending_at_asof(follower_panel):
    view = DataView(follower_panel, 19)
    weekly = period_returns(view, ["LEAD"], periods=4, horizon=5)
    daily = follower_panel.field("ret_cc")[:20, 0]
    assert weekly.shape == (4, 1)
    assert weekly[-1, 0] == pytest.approx(np.prod(1 + daily[15:20]) - 1)


def test_market_mode(follower_panel):
    network = estimate(DataView(follower_panel, 280), mode="market")
    assert network.leaders.shape == (len(network.symbols), 1)
    assert set(predict(network, DataView(follower_panel, 281), 1)) == set(network.symbols)
    with pytest.raises(ValueError, match="mode"):
        estimate(DataView(follower_panel, 280), mode="nope")


def test_month_start_view_and_monthly_reestimation(follower_panel):
    i = next(j for j, d in enumerate(DAYS) if d == date(2024, 1, 10))
    at = month_start_view(DataView(follower_panel, i))
    assert at.asof == date(2023, 12, 29)  # last session of December
    signal = LeadLag(horizon=1)
    signal.score(DataView(follower_panel, i), ["FOL"])
    signal.score(DataView(follower_panel, i + 1), ["FOL"])
    assert list(signal._cache) == [date(2023, 12, 29)]  # same month, same model


def test_earlier_view_never_moves_forward(follower_panel):
    view = DataView(follower_panel, 50)
    assert view.earlier(10).index == 40 and view.earlier(500).index == 0
    with pytest.raises(ValueError):
        view.earlier(-1)
