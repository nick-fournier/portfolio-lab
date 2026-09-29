"""Cross-sectional momentum: hold the stocks that rose most over the past year."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from portfolio_lab.core.calendar import Frequency
from portfolio_lab.research.dataview import DataView
from portfolio_lab.strategies import explain
from portfolio_lab.strategies.base import Weights, register
from portfolio_lab.strategies.construct import top_k_equal


def momentum_scores(
    view: DataView, symbols: Sequence[str], lookback: int = 252, skip: int = 21
) -> dict[str, float]:
    """Each symbol's return from ``lookback`` sessions ago to ``skip`` sessions ago.

    Symbols without a price at the start of the window are left out.
    """
    # prices(lookback + 1): the first row is the price `lookback` sessions ago.
    prices = view.prices(lookback + 1, symbols)
    if len(prices) <= lookback:
        return {}  # not enough history yet
    start, end = prices.iloc[0], prices.iloc[-1 - skip]
    return (end / start - 1).dropna().to_dict()


@register("momentum")
@dataclass
class Momentum:
    """Momentum: equal-weight the biggest winners of the past year, skipping the last month.

    Each month it ranks candidate stocks by their return over the past 12 months excluding
    the most recent month (the standard definition, since the last month tends to reverse)
    and holds the top ones in equal weights. It tests whether mean-variance's edge is just
    momentum; like any momentum strategy it is flattered by survivorship bias in free data.

    Args:
        pool: Candidates: the ``pool`` most liquid eligible stocks (``None`` for all eligible).
        hold: Number of top-ranked stocks held.
        lookback: Sessions in the ranking window (about 12 months).
        skip: Most recent sessions left out of the ranking (about 1 month).
        schedule: Rebalance frequency.
    """

    pool: int | None = 100
    hold: int = 20
    lookback: int = 252
    skip: int = 21
    schedule: Frequency = "M"
    name: str = "momentum"

    def __post_init__(self) -> None:
        self.pool = int(self.pool) if self.pool is not None else None
        self.hold, self.lookback, self.skip = int(self.hold), int(self.lookback), int(self.skip)
        if not 0 <= self.skip < self.lookback:
            raise ValueError("skip must be at least 0 and shorter than lookback")

    example_columns: ClassVar[dict[str, str]] = {"Return, 12 months ago → 1 month ago": "pct"}

    def explain(self) -> dict[str, str]:
        """Plain-language description of a run (see ``strategies.explain``)."""
        pool = "all eligible stocks" if self.pool is None else f"the {self.pool} most liquid stocks"
        months, skip = round(self.lookback / 21), round(self.skip / 21)
        return {
            "summary": (
                f"Equal-weights the {self.hold} biggest winners of the past {months} months "
                f"(skipping the last {skip}) among {pool}."
            ),
            "candidates": explain.candidates(self.pool),
            "signal": (
                f"Each stock's return from {months} months ago to {skip} month(s) ago. The most "
                "recent month is skipped because very recent winners tend to give some of it "
                "back (short-term reversal). This is extrapolation: the bet is that past "
                "winners keep winning, not a forecast of anything new."
            ),
            "construction": (
                f"Buys the {self.hold} highest-ranked stocks in equal weights "
                f"({1 / self.hold:.0%} each), ignoring volatility and how the stocks move together."
            ),
            "drivers": (
                "Whether last year's biggest winners keep winning. It concentrates in whatever "
                "theme is hot, so it swings hard: high market sensitivity and deep drawdowns "
                "when leadership turns."
            ),
            "related": (
                "Same candidates as meanvar (pool=100): meanvar uses the full trailing year "
                "including the last month and weights by expected return, volatility and "
                "co-movement; momentum equal-weights the top 20 and skips the last month. "
                "equal_weight (top_n=100) holds all 100 candidates equally, the control for both."
            ),
        }

    def target_weights(self, view: DataView) -> Weights:
        """Equal weights on the ``hold`` stocks with the highest skip-month return."""
        candidates = view.top_liquid(self.pool) if self.pool else view.eligible()
        scores = momentum_scores(view, candidates, self.lookback, self.skip)
        weights = top_k_equal(scores, self.hold)
        column = next(iter(self.example_columns))
        self.last_signals = {s: {column: scores[s]} for s in weights}
        return weights
