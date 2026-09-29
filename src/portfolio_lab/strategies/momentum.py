"""Cross-sectional momentum: hold the stocks that rose most over the past year."""

from collections.abc import Sequence
from dataclasses import dataclass

from portfolio_lab.core.calendar import Frequency
from portfolio_lab.research.dataview import DataView
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

    def target_weights(self, view: DataView) -> Weights:
        """Equal weights on the ``hold`` stocks with the highest skip-month return."""
        candidates = view.top_liquid(self.pool) if self.pool else view.eligible()
        return top_k_equal(momentum_scores(view, candidates, self.lookback, self.skip), self.hold)
