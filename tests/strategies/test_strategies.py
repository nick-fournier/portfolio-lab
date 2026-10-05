import numpy as np
import polars as pl
import pytest

import portfolio_lab.strategies  # noqa: F401  (registers built-in strategies)
import portfolio_lab.strategies.meanvar.strategy as mv
from portfolio_lab.research import stress
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import Panel
from portfolio_lab.strategies.base import REGISTRY, Composed, create
from portfolio_lab.strategies.construct import equal_weight, top_k_equal
from portfolio_lab.strategies.explain import KEYS

ASOF = 300  # long enough for 12-month lookbacks (panels below have 400 sessions)


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
    asof = panel.dates[after]
    fundamentals = panel.fundamentals
    if fundamentals is not None:  # rescore filings made from asof on, and add future ones
        dtype = fundamentals.schema["fscore"]
        future = fundamentals.filter(pl.col("filed") >= asof).with_columns(
            pl.lit(0, dtype=dtype).alias("fscore")
        )
        extra = fundamentals.with_columns(
            pl.lit(asof).alias("filed"), pl.lit(9, dtype=dtype).alias("fscore")
        )
        fundamentals = pl.concat([fundamentals.filter(pl.col("filed") < asof), future, extra])
    poisoned = Panel(
        panel.dates, panel.symbols, fields, eligible, rf, panel.universe, fundamentals=fundamentals
    )
    if panel.environment is not None:  # garbage in every environment row after asof
        numeric = [c for c, t in panel.environment.schema.items() if t.is_numeric()]
        poisoned.environment = panel.environment.with_columns(
            pl.when(pl.col("date") > asof).then(-1e9).otherwise(pl.col(c)).alias(c) for c in numeric
        )
    if panel.features is not None:  # garbage in every feature row dated after asof
        numeric = [c for c, t in panel.features.schema.items() if t.is_numeric()]
        poisoned.features = panel.features.with_columns(
            pl.when(pl.col("date") > asof).then(-1e9).otherwise(pl.col(c)).alias(c) for c in numeric
        )
    return poisoned


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_no_strategy_can_see_the_future(name, make_panel):
    """Corrupting every row after the decision date must not change any strategy's weights."""
    panel = make_panel(symbols=[f"S{i:02d}" for i in range(12)], days=400)
    clean = create(name).target_weights(DataView(panel, ASOF))
    poisoned = create(name).target_weights(DataView(_poisoned(panel, ASOF), ASOF))
    assert clean == poisoned
    assert clean, f"{name} produced no weights on the test panel"


def test_meanvar_risk_gauge_cannot_see_the_future(make_panel, monkeypatch):
    """With the gauge's history built over several rebalances, the future still can't leak."""
    monkeypatch.setattr(stress, "MIN_HISTORY", 3)
    panel = make_panel(symbols=[f"S{i:02d}" for i in range(12)], days=400)
    panel.environment = pl.DataFrame(
        {
            "date": panel.dates[::21],
            "financial_conditions": np.linspace(-1, 1, len(panel.dates[::21])),
        }
    )
    rebalances = list(range(ASOF - 6 * 21, ASOF + 1, 21))
    runs = []
    for source in (panel, _poisoned(panel, ASOF)):
        strategy = create("meanvar", model="historical_mean", risk_gauge=True)
        weights = [strategy.target_weights(DataView(source, i)) for i in rebalances]
        runs.append((weights[-1], strategy.diagnostics()["gauge"]))
    assert runs[0] == runs[1]
    assert len(runs[0][1]) == len(rebalances) and runs[0][0]


def test_equal_weight_strategy_top_n(make_panel):
    panel = make_panel(symbols=[f"S{i:02d}" for i in range(12)], days=400)
    view = DataView(panel, ASOF)
    weights = create("equal_weight", top_n=3).target_weights(view)
    assert set(weights) == set(view.top_liquid(3))
    assert sum(weights.values()) == pytest.approx(1.0)


def test_buy_hold_waits_for_a_price(make_panel):
    panel = make_panel(days=400)
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
    weights = strategy.target_weights(DataView(make_panel(days=400), ASOF))
    assert len(weights) == 2 and sum(weights.values()) == pytest.approx(1.0)


def test_unknown_strategy():
    with pytest.raises(KeyError, match="unknown strategy"):
        create("nope")


def test_piotroski_holds_high_scores_equally(make_panel):
    panel = make_panel(symbols=[f"S{i:02d}" for i in range(6)], days=400)
    weights = create("piotroski").target_weights(DataView(panel, ASOF))
    assert weights == {
        "S00": pytest.approx(1 / 3),
        "S02": pytest.approx(1 / 3),
        "S04": pytest.approx(1 / 3),
    }


def test_meanvar_fscore_filter_limits_candidates(make_panel):
    panel = make_panel(symbols=[f"S{i:02d}" for i in range(12)], days=400)
    view = DataView(panel, ASOF)
    weights = create(
        "meanvar", model="historical_mean", min_fscore=8, max_weight=0.5
    ).target_weights(view)
    assert weights and set(weights) <= {f"S{i:02d}" for i in range(0, 12, 2)}  # the 9-scorers


def test_meanvar_healthy_share_keeps_the_healthiest(make_panel):
    panel = make_panel(symbols=[f"S{i:02d}" for i in range(12)], days=400)
    view = DataView(panel, ASOF)
    weights = create(
        "meanvar", model="historical_mean", healthy_share=0.5, max_weight=0.5
    ).target_weights(view)
    assert weights and set(weights) <= {f"S{i:02d}" for i in range(6, 12)}  # healthier half


def test_meanvar_quarterly_health_refresh_reuses_the_set_within_a_quarter(make_panel, monkeypatch):
    calls = []
    real = mv.health_scores
    monkeypatch.setattr(mv, "health_scores", lambda f: calls.append(1) or real(f))
    panel = make_panel(symbols=[f"S{i:02d}" for i in range(12)], days=400)
    strategy = create("meanvar", model="historical_mean", healthy_share=0.5,
                      health_schedule="Q")  # fmt: skip
    quarter = lambda i: (panel.dates[i].year, (panel.dates[i].month - 1) // 3)  # noqa: E731
    same = next(i for i in range(ASOF + 1, len(panel.dates)) if quarter(i) == quarter(ASOF))
    other = next(i for i in range(ASOF + 1, len(panel.dates)) if quarter(i) != quarter(ASOF))
    chosen = strategy._healthy_set(DataView(panel, ASOF), None)
    assert strategy._healthy_set(DataView(panel, same), None) == chosen
    assert len(calls) == 1  # reused within the quarter
    strategy._healthy_set(DataView(panel, other), None)
    assert len(calls) == 2  # recomputed in the next quarter


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_strategy_explains_itself(name, make_panel):
    """Run pages need plain-language text for every step, and example values for holdings."""
    strategy = create(name)
    text = strategy.explain()
    assert set(text) == set(KEYS)
    assert all(isinstance(v, str) and len(v) > 10 for v in text.values()), text
    columns = getattr(strategy, "example_columns", {})
    if columns:
        panel = make_panel(symbols=[f"S{i:02d}" for i in range(12)], days=400)
        weights = strategy.target_weights(DataView(panel, ASOF))
        assert set(strategy.last_signals) == {s for s, w in weights.items() if w > 0}
        assert all(set(v) == set(columns) for v in strategy.last_signals.values())


def test_meanvar_forecast_pool_takes_the_best_forecasts(make_panel, monkeypatch):
    panel = make_panel(symbols=[f"S{i:02d}" for i in range(12)], days=400)
    view = DataView(panel, ASOF)
    strategy = create("meanvar", model="historical_mean", forecast_pool=12, top_n=4,
                      expected="forecaster", max_weight=0.5)  # fmt: skip
    forecast = {f"S{i:02d}": i / 100 for i in range(12)}  # S11 best
    monkeypatch.setattr(strategy, "_learned_at", lambda v: forecast)
    weights = strategy.target_weights(view)
    assert weights and set(weights) <= {"S08", "S09", "S10", "S11"}


def test_learned_expected_returns_add_the_level():
    import pandas as pd  # noqa: PLC0415

    from portfolio_lab.strategies.meanvar.learned import expected_returns  # noqa: PLC0415

    model_mu = pd.Series({"A": 0.10, "B": 0.20, "C": 0.30})
    forecast = {"A": 0.01, "B": -0.01}  # C has no forecast: left out
    rel = expected_returns(forecast, model_mu, 0.04, excess=False)
    assert rel.to_dict() == pytest.approx({"A": 0.15 + 0.12, "B": 0.15 - 0.12})
    exc = expected_returns(forecast, model_mu, 0.04, excess=True)
    assert exc.to_dict() == pytest.approx({"A": 0.04 + 0.12, "B": 0.04 - 0.12})


def test_learned_forecasts_use_the_latest_recent_month(tmp_path, monkeypatch):
    from datetime import date  # noqa: PLC0415

    from portfolio_lab.strategies.meanvar import learned  # noqa: PLC0415

    lf = learned.LearnedForecasts(tmp_path)
    by_date = {date(2020, 1, 31): {"A": 1.0}, date(2020, 2, 28): {"A": 2.0}}
    monkeypatch.setattr(lf, "_load", lambda: by_date)
    assert lf.at(date(2020, 3, 2)) == {"A": 2.0}
    assert lf.at(date(2020, 3, 20)) == {}  # too stale
