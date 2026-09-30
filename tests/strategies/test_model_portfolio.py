import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.dataview import DataView
from portfolio_lab.strategies.base import create
from portfolio_lab.strategies.model_portfolio import model_scores

from ..research.test_risk import N_DAYS, PER_SECTOR, sector_panel


@pytest.fixture
def panel():
    """The two-sector panel with predictions: higher-numbered stocks rated higher."""
    p = sector_panel()
    months = p.features["date"].unique().sort()
    n = 2 * PER_SECTOR
    p.predictions = pl.DataFrame(
        [(d, f"S{k:02d}", k / n) for d in months for k in range(n)],
        schema=["date", "symbol", "score"], orient="row",
    )  # fmt: skip
    return p


def test_exclusion_and_selection(panel):
    view = DataView(panel, N_DAYS - 1)
    scores = model_scores(view, view.eligible())
    assert max(scores, key=scores.get) == "S39" and min(scores, key=scores.get) == "S00"
    kept = create("model_portfolio", weighting="exclude_only", exclude=0.25).target_weights(view)
    assert set(kept) == {f"S{k:02d}" for k in range(10, 40)}  # the weakest quarter dropped
    top = create("model_portfolio", weighting="equal", hold=5).target_weights(view)
    assert create("model_portfolio").schedule == "M"
    assert set(top) == {"S35", "S36", "S37", "S38", "S39"}


def test_optimizer_respects_caps_and_beta(panel):
    view = DataView(panel, N_DAYS - 1)
    strategy = create(
        "model_portfolio", weighting="optimized", hold=30, max_weight=0.08, sector_cap=0.6
    )
    weights = strategy.target_weights(view)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert max(weights.values()) <= 0.08 + 1e-6
    tech = sum(w for s, w in weights.items() if int(s[1:]) >= PER_SECTOR)
    assert tech <= 0.6 + 1e-6  # the higher-rated sector is capped
    assert set(strategy.last_signals) == set(weights)


def test_band_smooths_small_changes_only(panel):
    strategy = create("model_portfolio", weighting="equal", hold=4, band=0.01)
    strategy._previous = {"S39": 0.255, "S38": 0.245, "S00": 0.5}
    weights = strategy.target_weights(DataView(panel, N_DAYS - 1))
    assert "S00" not in weights  # no longer selected: sold
    assert weights["S39"] / weights["S38"] == pytest.approx(0.255 / 0.245)  # kept, small change
    assert sum(weights.values()) == pytest.approx(1.0)
    assert np.isclose(weights["S37"], weights["S36"])
