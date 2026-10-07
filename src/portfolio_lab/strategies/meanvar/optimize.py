"""Mean-variance portfolio optimization with PyPortfolioOpt.

Ports the legacy optimizer with its bugs fixed: covariance is estimated from *daily*
prices with ``frequency=252`` (the legacy code applied the daily default to monthly
prices), expected returns and the risk-free rate are both annual, and missing data is
handled per symbol instead of dropping every date on which any symbol is missing.
"""

import logging

import pandas as pd
from pypfopt import EfficientFrontier, objective_functions
from pypfopt.exceptions import OptimizationError
from pypfopt.risk_models import CovarianceShrinkage

log = logging.getLogger(__name__)

#: ``kelly``: the highest expected long-run growth, expected return minus half the variance
#: (quadratic utility at risk aversion 1, without the L2 regularizer). It concentrates and
#: takes more risk than ``max_sharpe``, and compounds faster.
OBJECTIVES = ("max_sharpe", "min_volatility", "max_quadratic_utility", "kelly")
TRADING_DAYS = 252
#: Weights below this are dropped; the dropped sliver is held as cash.
WEIGHT_CUTOFF = 1e-4


def _within_bounds(weights: dict[str, float], cap: float | dict[str, float]) -> dict[str, float]:
    """Clip solver output to ``[0, cap]`` (per symbol) and scale it to sum to at most 1.

    Solvers satisfy constraints only to a tolerance, e.g. returning weights summing to
    1.0026; the backtest engine rightly rejects anything over budget.
    """
    limit = cap if isinstance(cap, dict) else dict.fromkeys(weights, cap)
    clipped = {s: min(float(w), limit[s]) for s, w in weights.items() if w > 0}
    total = sum(clipped.values())
    return {s: w / total for s, w in clipped.items()} if total > 1 else clipped


def _solve(ef: EfficientFrontier, objective: str, risk_free: float, risk_aversion: float) -> None:
    """Run the requested objective on ``ef`` (raises if infeasible)."""
    if objective == "max_sharpe":
        ef.max_sharpe(risk_free_rate=risk_free)
    elif objective == "kelly":
        ef.max_quadratic_utility(risk_aversion=1.0)
    elif objective == "max_quadratic_utility":
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_quadratic_utility(risk_aversion=risk_aversion)
    else:
        ef.min_volatility()


def optimize(
    mu: pd.Series,
    prices: pd.DataFrame,
    risk_free: float,
    objective: str = "max_sharpe",
    max_weight: float = 0.10,
    risk_aversion: float = 1.0,
    caps: dict[str, float] | None = None,
    cov: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Long-only mean-variance weights for the symbols in ``mu``.

    Args:
        mu: Annualized expected return per symbol.
        prices: Daily adjusted prices (dates x symbols) covering ``mu``'s symbols.
        risk_free: Annual risk-free rate.
        objective: One of :data:`OBJECTIVES`.
        max_weight: Cap on any single weight (raised to ``1/n`` if infeasibly low).
        risk_aversion: Risk aversion for ``max_quadratic_utility``.
        caps: Per-symbol caps instead of ``max_weight`` (scaled up together if they sum
            to less than 1, so a fully invested portfolio stays feasible).
        cov: Annual covariance to use instead of the prices' shrunk sample covariance.

    Returns:
        Weights summing to 1 (less sub-cutoff slivers left in cash), or an empty dict (all
        cash) if no solution is found. If max-Sharpe is infeasible (e.g. every expected
        return is below the risk-free rate), falls back to minimum volatility.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")
    symbols = [s for s in mu.index if s in prices.columns]
    if len(symbols) < 2:
        return {}
    if cov is None:
        cov = CovarianceShrinkage(prices[symbols], frequency=TRADING_DAYS).ledoit_wolf()
    cov = cov.loc[symbols, symbols]
    if caps is None:
        cap = dict.fromkeys(symbols, max(max_weight, 1.0 / len(symbols)))
    else:
        total = sum(caps.get(s, 0.0) for s in symbols)
        grow = max(1.0, 1.0 / total) if total > 0 else 1.0
        cap = {s: min(1.0, caps.get(s, 0.0) * grow) for s in symbols}

    for attempt in dict.fromkeys((objective, "min_volatility")):
        bounds = [(0.0, cap[s]) for s in symbols]
        ef = EfficientFrontier(mu[symbols], cov, weight_bounds=bounds)
        try:
            _solve(ef, attempt, risk_free, risk_aversion)
        except (OptimizationError, ValueError) as exc:
            log.debug("%s failed (%s); falling back", attempt, exc)
            continue
        # No rounding; weights below the cutoff are dropped and their sliver stays in cash.
        weights = ef.clean_weights(cutoff=WEIGHT_CUTOFF, rounding=None)
        return _within_bounds(weights, cap)
    log.warning("no feasible portfolio for %d symbols; holding cash", len(symbols))
    return {}
