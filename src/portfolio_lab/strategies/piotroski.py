"""Piotroski value screen: hold financially strong companies by their latest F-score."""

from dataclasses import dataclass

from portfolio_lab.core.calendar import Frequency
from portfolio_lab.research.dataview import DataView
from portfolio_lab.strategies.base import Weights, register
from portfolio_lab.strategies.construct import equal_weight


@register("piotroski")
@dataclass
class Piotroski:
    """Piotroski screen: equal-weight companies whose latest F-score is 8 or 9.

    The F-score counts nine signs of improving financial health in a company's latest
    annual report (profitability, cash flow, falling leverage, liquidity, no dilution,
    margins, efficiency). This is the original optimizer's stock filter, computed from SEC
    filings as of the date each filing became public. Banks and foreign (IFRS) filers lack
    the inputs and are skipped.

    Args:
        pool: Candidates: the ``pool`` most liquid eligible stocks (``None`` for all eligible).
        min_score: Minimum F-score held.
        min_signals: Minimum number of the nine signals that must be computable.
        schedule: Rebalance frequency.
    """

    pool: int | None = None
    min_score: int = 8
    min_signals: int = 8
    schedule: Frequency = "M"
    name: str = "piotroski"

    def __post_init__(self) -> None:
        self.pool = int(self.pool) if self.pool is not None else None
        self.min_score, self.min_signals = int(self.min_score), int(self.min_signals)

    def target_weights(self, view: DataView) -> Weights:
        """Equal weights over candidates scoring at least ``min_score``."""
        candidates = view.top_liquid(self.pool) if self.pool else view.eligible()
        scores = view.fscores(candidates, min_signals=self.min_signals)
        return equal_weight(sorted(s for s, f in scores.items() if f >= self.min_score))
