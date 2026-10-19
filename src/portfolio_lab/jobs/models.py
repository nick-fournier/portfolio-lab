"""The models the lab runs: the two production classes, their Kelly versions, the baselines.

Class 1 and class 2 are production: drawn on the Overview and Compare charts, and the paper
account follows one of them (:data:`PAPER_STRATEGY`). Their Kelly versions weigh the same
candidates for long-run growth instead of return per unit of risk; they are drawn beside
production so the trade-off (more return, much deeper drops) stays in view.

Titles name the three choices that set a model apart, in order: where its expected returns
come from (momentum: the trailing one-year return; linear: the nine-term forecaster), how its
candidates are filtered (health: the tapered healthiest of the 400 most liquid), and what the
optimizer maximizes (max-Sharpe, or Kelly, with how hard it bets when not full Kelly).
"""

import json
from typing import Any

from portfolio_lab.backtest.engine import label as strategy_label
from portfolio_lab.strategies.base import create

#: Class 1, "Momentum · health · max-Sharpe": meanvar on the healthiest
#: (continuous F-score) of the most liquid stocks, monthly, with weight limits tapering near
#: the edges of both lists instead of hard cutoffs (``strategies.meanvar.soft``), weighted for
#: the best expected return per unit of risk (maximum Sharpe, ``strategies.meanvar.optimize``),
#: holding the minimum-variance mix in bear markets and equal weights in rebounds (see
#: ``strategies.meanvar`` and ``research.regimes``). Max-Sharpe again since 2026-10-18, after
#: twelve days of Kelly (:data:`CLASS_1_KELLY`).
CLASS_1: tuple[str, dict[str, Any]] = (
    "meanvar",
    {"health_rank_pool": 400, "soften": "taper", "bear_defense": True, "rebound": "equal",
     "objective": "max_sharpe", "title": "Momentum · health · max-Sharpe"},
)  # fmt: skip
#: Class 2, "Linear · health · max-Sharpe": the same pool, objective and bear switch as
#: :data:`CLASS_1`, with the nine-term forecasts in Grinold's form (T-bill + sqrt(12) x IC x
#: volatility x z, honest expected returns) and the price covariance, no rebound switch
#: (``forecasters.grinold``).
CLASS_2: tuple[str, dict[str, Any]] = (
    "meanvar",
    {"health_rank_pool": 400, "soften": "taper", "bear_defense": True, "objective": "max_sharpe",
     "expected": "linear", "title": "Linear · health · max-Sharpe"},
)  # fmt: skip
#: Class 1 weighted for the highest expected long-run growth (full Kelly: expected return minus
#: half the variance), production from 2026-10-06 to 2026-10-18. With trailing returns as its
#: expected returns it fills the 10% caps with the past year's biggest winners: 1999-2026,
#: 23.1% a year against max-Sharpe's 19.8%, but 34% volatility against 22% and a 66% worst
#: drop against 34%.
CLASS_1_KELLY: tuple[str, dict[str, Any]] = (
    "meanvar",
    {**CLASS_1[1], "objective": "kelly", "title": "Momentum · health · Kelly"},
)
#: Class 2 betting 16 times as hard as full Kelly (risk aversion 1/16), the best of 1 .. 1/64
#: and 0 on the scratch hive 2005-2026 and a monthly walk-forward pick (#74). From its first
#: trade in January 2000 its worst drop is 63%, against max-Sharpe's 38%.
CLASS_2_KELLY: tuple[str, dict[str, Any]] = (
    "meanvar",
    {
        **CLASS_2[1],
        "objective": "kelly",
        "risk_aversion": 1 / 16,
        "title": "Linear · health · Kelly x16",
    },
)
#: The production models, drawn on the Overview and Compare charts against SPY and the best
#: funds.
MODELS: tuple[tuple[str, dict[str, Any]], ...] = (CLASS_1, CLASS_2)
#: The production models' higher-risk versions, drawn beside them (every other strategy is
#: listed in the tables but starts hidden on the charts).
KELLY_MODELS: tuple[tuple[str, dict[str, Any]], ...] = (CLASS_1_KELLY, CLASS_2_KELLY)
#: The model the paper account follows, and production wherever one model stands for the lab
#: (the Compare page's head-to-heads with young funds, the tax calculator): class 2 from the
#: 2026-10-30 rebalance on, class 1 before (same Sharpe since 2000, class 2 about 2% a year
#: ahead; the paper account tests its forecaster pipeline before any real money).
PAPER_STRATEGY: tuple[str, dict[str, Any]] = CLASS_2
#: Backtests the scheduler refreshes weekly: the four models only (SPY and the funds come
#: from the fund comparison; earlier baselines can still be run by hand, ``plab backtest``).
SCHEDULED_BACKTESTS: tuple[tuple[str, dict[str, Any]], ...] = (
    *MODELS,
    *KELLY_MODELS,
)


def run_label(meta: dict[str, Any]) -> str:
    """A saved run's display name: its configuration's current title, else its saved label.

    Configurations are matched by their untitled label, which spells out every non-default
    parameter (a title is not one), so renaming a model renames its earlier runs too.
    """
    titled = {}
    for name, params in (*MODELS, *KELLY_MODELS):
        untitled = {k: v for k, v in params.items() if k != "title"}
        titled[strategy_label(create(name, **untitled))] = strategy_label(create(name, **params))
    try:
        saved = strategy_label(create(meta["strategy"], **meta.get("params", {})))
    except (KeyError, TypeError, ValueError):  # a retired strategy or parameter
        saved = None
    return titled.get(saved) or meta.get("label") or meta["strategy"]


def model_record(name: str, params: dict[str, Any]) -> str:
    """A model's identity as saved with each paper rebalance; :func:`run_label` reads it back."""
    untitled = {k: v for k, v in params.items() if k != "title"}
    label = strategy_label(create(name, **params))
    return json.dumps({"strategy": name, "params": untitled, "label": label})
