"""Piotroski value screen: hold financially strong companies by their latest F-score."""

from dataclasses import dataclass
from typing import ClassVar

from portfolio_lab.core.calendar import Frequency
from portfolio_lab.research.dataview import DataView
from portfolio_lab.strategies import explain
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

    example_columns: ClassVar[dict[str, str]] = {"F-score": "int"}

    def __post_init__(self) -> None:
        self.pool = int(self.pool) if self.pool is not None else None
        self.min_score, self.min_signals = int(self.min_score), int(self.min_signals)
        self.last_signals: dict[str, dict[str, float]] = {}

    def explain(self) -> dict[str, str]:
        """Plain-language description of a run (see ``strategies.explain``)."""
        return {
            "summary": (
                f"Equal-weights companies with a Piotroski F-score of {self.min_score}+ among "
                + ("all eligible stocks." if self.pool is None else f"the {self.pool} most liquid.")
            ),
            "candidates": explain.candidates(self.pool)
            + f" Only companies with at least {self.min_signals} of the 9 F-score signals "
            "computable count; banks and foreign (IFRS) filers usually lack the inputs.",
            "signal": (
                "The F-score from the company's latest annual report filed before the rebalance "
                "date: one point each for positive profit, positive cash flow, rising return on "
                "assets, cash flow above profit, falling debt, rising liquidity, no new shares, "
                "rising gross margin and rising asset turnover (0 to 9). It measures financial "
                "health and improvement, not price, so it is a quality screen rather than a "
                "forecast."
            ),
            "construction": (
                f"Holds every candidate scoring {self.min_score} or more in equal weights."
            ),
            "drivers": (
                "Whether financially strong, improving companies beat the rest of the pool. "
                "Holdings change mostly when new annual reports arrive."
            ),
            "related": (
                "The same score filters meanvar's candidates in meanvar (min_fscore). "
                "equal_weight with the same pool is its control."
            ),
        }

    def target_weights(self, view: DataView) -> Weights:
        """Equal weights over candidates scoring at least ``min_score``."""
        candidates = view.top_liquid(self.pool) if self.pool else view.eligible()
        scores = view.fscores(candidates, min_signals=self.min_signals)
        held = sorted(s for s, f in scores.items() if f >= self.min_score)
        self.last_signals = {s: {"F-score": float(scores[s])} for s in held}
        return equal_weight(held)
