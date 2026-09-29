from datetime import date

import numpy as np
import polars as pl
import pytest

import portfolio_lab.signals  # noqa: F401  (registers built-in signals)
from portfolio_lab.core.calendar import sessions
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.scoreboard import (
    HORIZON,
    evaluate,
    forward_returns,
    score_date,
    summarize,
)
from portfolio_lab.signals.base import REGISTRY, create

from ..strategies.test_strategies import ASOF, _poisoned

SYMBOLS = [f"S{i:02d}" for i in range(24)]


def _panel(ret_cc, fell_to_otc=()):
    """Panel from a (days x symbols) return array; closes exist where returns do."""
    ret_cc = np.array(ret_cc, dtype=float)
    days = sessions(date(2024, 1, 2), date(2024, 12, 31))[: ret_cc.shape[0]]
    symbols = [f"X{j}" for j in range(ret_cc.shape[1])]
    fields = {
        "close": np.where(np.isfinite(ret_cc), 10.0, np.nan),
        "ret_cc": ret_cc,
        "ret_co": np.zeros_like(ret_cc),
        "adv": np.full(ret_cc.shape, 1e9),
    }
    eligible = np.ones(ret_cc.shape, dtype=bool)
    rf = np.zeros(len(days))
    return Panel(days, symbols, fields, eligible, rf, symbols, fell_to_otc=fell_to_otc)


def test_forward_returns_compound_and_apply_delisting_return():
    ret = [[0, 0, 0], [0.1, 0.1, 0.1], [0.1, np.nan, np.nan], [0, np.nan, np.nan]]
    panel = _panel(ret, fell_to_otc=("X2",))
    fwd = forward_returns(panel, 0, horizon=3, delisting_return=-0.3)
    # X0 compounds; X1 was acquired (no bars after day 1, exits at its last price);
    # X2 fell to OTC with its last trade inside the window.
    np.testing.assert_allclose(fwd, [1.1 * 1.1 - 1, 0.1, 1.1 * 0.7 - 1])


def test_score_date_perfect_and_inverted_rankings():
    panel = _panel(np.zeros((2, 30)))
    forward = np.linspace(-0.1, 0.1, 30)
    perfect = {f"X{j}": float(j) for j in range(30)}
    result = score_date(perfect, forward, panel)
    assert result["ic"] == pytest.approx(1.0)
    assert result["top"] > 0 > result["bottom"]
    assert score_date({k: -v for k, v in perfect.items()}, forward, panel)["ic"] == pytest.approx(
        -1.0
    )
    assert score_date(dict(list(perfect.items())[:5]), forward, panel) is None  # too few


def test_tied_scores_share_a_quintile():
    panel = _panel(np.zeros((2, 30)))
    forward = np.arange(30, dtype=float)
    scores = {f"X{j}": 9.0 if j >= 20 else 5.0 for j in range(30)}  # like F-scores
    result = score_date(scores, forward, panel)
    assert result["top"] == pytest.approx(np.mean(range(20, 30)))
    assert result["bottom"] == pytest.approx(np.mean(range(20)))


def test_evaluate_uses_month_ends_with_complete_forward_windows(make_panel):
    panel = make_panel(symbols=SYMBOLS, days=400)
    rows = evaluate(create("momentum"), "momentum", panel, panel.dates[260], pools=["all"])
    assert rows["signal"].unique().to_list() == ["momentum"]
    assert all(panel.date_index[d] + HORIZON < len(panel.dates) for d in rows["date"])
    months = [(d.year, d.month) for d in rows["date"]]
    assert len(months) == len(set(months)) >= 4
    assert rows["n"].min() == len(SYMBOLS)


def test_summarize():
    scores = pl.DataFrame(
        {
            "signal": ["a"] * 4,
            "pool": ["all"] * 4,
            "date": [date(2024, m, 28) for m in range(1, 5)],
            "n": [100] * 4,
            "ic": [0.1, 0.0, 0.1, 0.0],
            "top": [0.02] * 4,
            "bottom": [0.01] * 4,
        }
    )
    row = summarize(scores).row(0, named=True)
    assert row["months"] == 4 and row["hit"] == 0.5
    assert row["mean_ic"] == pytest.approx(0.05)
    assert row["ic_t"] == pytest.approx(0.05 / np.std([0.1, 0, 0.1, 0], ddof=1) * 2)
    assert row["spread"] == pytest.approx(0.12)
    assert row["top"] == pytest.approx(1.02**12 - 1)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_no_signal_can_see_the_future(name, make_panel):
    """Corrupting every row after the decision date must not change any signal's scores."""
    panel = make_panel(symbols=SYMBOLS[:12], days=400)
    clean = create(name).score(DataView(panel, ASOF), SYMBOLS[:12])
    poisoned = create(name).score(DataView(_poisoned(panel, ASOF), ASOF), SYMBOLS[:12])
    assert clean == poisoned
    assert clean, f"{name} produced no scores on the test panel"
