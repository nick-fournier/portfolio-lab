"""Model portfolio: many modest calls held together, likely underperformers excluded.

The long value portfolio (M11). Built in layers, so variants show what each adds:

1. **Exclude**: drop the ``exclude`` share of candidates the models rate least likely to
   beat the median (the most reliable part of the models).
2. **Select**: of the rest, keep the ``hold`` highest-rated.
3. **Weight**: equal weights, or an optimizer that tilts toward stronger calls while
   controlling risk with the factor risk model (``research.risk``), capping each stock and
   sector and keeping market beta in a band. A no-trade band leaves small changes alone.

The defaults: monthly, the 100 best-rated, optimized with a turnover penalty.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import cvxpy as cp
import numpy as np
import polars as pl

from portfolio_lab.core.calendar import Frequency
from portfolio_lab.research import risk
from portfolio_lab.research.dataview import DataView
from portfolio_lab.strategies import explain
from portfolio_lab.strategies.base import Weights, register
from portfolio_lab.strategies.construct import equal_weight

WEIGHTINGS = ("exclude_only", "equal", "optimized")
#: Assumed skill of the model ranking (information coefficient), for scaling expected
#: edges in the optimizer: alpha = IC x volatility x score z-score (Grinold's rule).
ASSUMED_IC = 0.05


def model_scores(view: DataView, symbols: Sequence[str]) -> dict[str, float]:
    """Each symbol's forecast score as a percentile (0 to 1) among ``symbols``."""
    preds = view.predictions(symbols)
    if preds.is_empty() or "score" not in preds.columns:
        return {}
    ranked = preds.drop_nulls("score").with_columns(
        (pl.col("score").rank() / pl.col("score").count()).alias("score")
    )
    return dict(zip(ranked["symbol"], ranked["score"], strict=True))


def _zscore(values: np.ndarray) -> np.ndarray:
    sd = values.std()
    return (values - values.mean()) / sd if sd > 0 else np.zeros(len(values))


@register("model_portfolio")
@dataclass
class ModelPortfolio:
    """Model portfolio: exclude likely underperformers, hold the best-rated, weigh the risk.

    Args:
        pool: Candidates: the ``pool`` most liquid eligible stocks (``None`` for all).
        exclude: Share of candidates dropped as likely underperformers.
        hold: Number of stocks held (after exclusion), unless ``weighting`` is
            ``exclude_only``.
        weighting: ``exclude_only`` (equal weights on everything not excluded; the
            default), ``equal`` (equal weights on the ``hold`` best) or ``optimized``
            (risk-aware tilt).
        max_weight: Cap on any one stock (``optimized``).
        sector_cap: Cap on any one sector (``optimized``).
        beta_band: Allowed range of portfolio market beta (``optimized``).
        risk_aversion: Trade-off between expected edge and risk (``optimized``).
        band: No-trade band: weights moving less than this keep their previous value.
        turnover_cost: Cost charged in the optimizer per unit of weight traded (``optimized``),
            so it only trades when the expected edge outweighs the cost.
        schedule: Rebalance frequency.
    """

    pool: int | None = None
    exclude: float = 0.3
    hold: int = 100
    weighting: str = "optimized"
    max_weight: float = 0.05
    sector_cap: float = 0.25
    beta_band: tuple[float, float] = (0.9, 1.1)
    risk_aversion: float = 2.0
    band: float = 0.005
    turnover_cost: float = 0.002
    schedule: Frequency = "M"
    name: str = "model_portfolio"
    example_columns: ClassVar[dict[str, str]] = {"Model score (percentile)": "pct"}
    _previous: dict = field(default_factory=dict, repr=False, metadata={"param": False})

    def __post_init__(self) -> None:
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")
        self.pool = int(self.pool) if self.pool is not None else None
        self.hold, self.exclude = int(self.hold), float(self.exclude)
        self.beta_band = tuple(float(b) for b in self.beta_band)
        self.last_signals: dict[str, dict[str, float]] = {}

    def explain(self) -> dict[str, str]:
        """Plain-language description of a run (see ``strategies.explain``)."""
        how = {
            "exclude_only": "equal weights on every remaining candidate",
            "equal": f"equal weights on the {self.hold} best-rated",
            "optimized": f"a risk-aware mix of the {self.hold} best-rated",
        }[self.weighting]
        construction = {
            "exclude_only": "Every candidate not excluded is held in equal weights: this "
            "variant measures what excluding likely underperformers alone is worth.",
            "equal": f"Holds the {self.hold} best-rated remaining stocks in equal weights.",
            "optimized": (
                f"Takes the {self.hold} best-rated remaining stocks and chooses weights that "
                "balance expected edge against risk, using a factor risk model (how stocks "
                "move together through the market, sectors, size, value, momentum, "
                f"volatility and profitability). No stock above {self.max_weight:.0%}, no "
                f"sector above {self.sector_cap:.0%}, market beta between "
                f"{self.beta_band[0]:g} and {self.beta_band[1]:g}."
            ),
        }[self.weighting]
        return {
            "summary": (
                f"Drops the {self.exclude:.0%} of stocks the models rate least likely to beat "
                f"the median, then holds {how}."
            ),
            "candidates": explain.candidates(self.pool),
            "signal": (
                "The gradient-boosted tree model's predicted return rank for the next month, "
                "learned from every eligible stock's value, quality and price traits plus the "
                "market environment, as a percentile among the candidates. The model is "
                "retrained each January on earlier data only, so every rating is out of sample."
            ),
            "construction": construction
            + f" Weights that would move by less than {self.band:.1%} are left alone, to "
            "avoid needless trading.",
            "drivers": (
                "Whether many small, fairly independent edges add up: each stock's call is only "
                "a little better than a coin flip, so the result depends on breadth and on the "
                "exclusion of likely underperformers, the models' most reliable signal."
            ),
            "related": (
                "The model counterpart of equal_weight: the same pool with the weakest calls "
                "removed. Compare the exclude_only, equal and optimized variants to see what "
                "each layer adds."
            ),
        }

    def target_weights(self, view: DataView) -> Weights:
        """Exclude, select and weight as configured (see module docs)."""
        candidates = view.top_liquid(self.pool) if self.pool else view.eligible()
        scores = model_scores(view, candidates)
        if not scores:
            self.last_signals = {}
            return {}
        ranked = sorted(scores, key=lambda s: -scores[s])
        kept = ranked[: max(1, round(len(ranked) * (1 - self.exclude)))]
        if self.weighting == "exclude_only":
            weights = equal_weight(kept)
        else:
            chosen = kept[: self.hold]
            weights = equal_weight(chosen)
            if self.weighting == "optimized":
                weights = self._optimize(view, chosen, scores) or weights
        weights = self._apply_band(weights)
        column = next(iter(self.example_columns))
        self.last_signals = {s: {column: scores[s]} for s in weights}
        return weights

    def _optimize(
        self, view: DataView, chosen: list[str], scores: dict[str, float]
    ) -> Weights | None:
        """Risk-aware weights for ``chosen`` (``None`` if the model or solver fails)."""
        model = risk.fit(view, view.eligible())
        if model is None:
            return None
        names = [s for s in chosen if s in set(model.symbols)]
        if len(names) < 2:
            return None
        cov = model.cov(names)
        vol = np.sqrt(np.diag(cov))
        alpha = ASSUMED_IC * vol * _zscore(np.array([scores[s] for s in names]))
        index = {s: k for k, s in enumerate(model.symbols)}
        sectors = [model.sectors[index[s]] for s in names]
        betas = view.features(names, ["beta"])
        beta_of = dict(zip(betas["symbol"], betas["beta"], strict=True)) if betas.height else {}
        beta = np.array([beta_of.get(s) or 1.0 for s in names])
        w = cp.Variable(len(names))
        cap = max(self.max_weight, 1.0 / len(names))
        constraints = [cp.sum(w) == 1, w >= 0, w <= cap]
        for sec in set(sectors):
            members = [k for k, s in enumerate(sectors) if s == sec]
            if len(members) * cap > self.sector_cap:
                constraints.append(cp.sum(w[members]) <= self.sector_cap)
        banded = [*constraints, beta @ w >= self.beta_band[0], beta @ w <= self.beta_band[1]]
        previous = np.array([self._previous.get(s, 0.0) for s in names])
        objective = cp.Maximize(
            alpha @ w
            - self.risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
            - self.turnover_cost * cp.norm1(w - previous)
        )
        for attempt in (banded, constraints):  # drop the beta band if infeasible
            problem = cp.Problem(objective, attempt)
            try:
                problem.solve()
            except cp.error.SolverError:
                continue
            if w.value is not None and problem.status in ("optimal", "optimal_inaccurate"):
                values = np.clip(w.value, 0, None)
                values /= values.sum()
                return {s: float(v) for s, v in zip(names, values, strict=True) if v > 1e-4}
        return None

    def _apply_band(self, weights: Weights) -> Weights:
        """Smooth small changes for stocks that stay selected; renormalize to 1.

        A stock kept from the previous rebalance whose target moved by less than ``band``
        keeps its previous weight. Stocks that dropped out are sold and new ones bought.
        """
        if self._previous and self.band > 0:
            merged = {
                s: self._previous[s]
                if s in self._previous and abs(w - self._previous[s]) < self.band
                else w
                for s, w in weights.items()
            }
            total = sum(merged.values())
            weights = {s: w / total for s, w in merged.items()} if total > 0 else {}
        self._previous = dict(weights)
        return weights
