"""Baseline strategies every other strategy is judged against."""

from dataclasses import dataclass

from portfolio_lab.core.calendar import Frequency
from portfolio_lab.research.dataview import DataView
from portfolio_lab.strategies.base import Weights, register
from portfolio_lab.strategies.construct import equal_weight


@register("equal_weight")
@dataclass
class EqualWeightEligible:
    """Hold every eligible stock (or the ``top_n`` most liquid) in equal weights.

    Args:
        top_n: If set, only the ``top_n`` eligible stocks by dollar volume.
        schedule: Rebalance frequency.
    """

    top_n: int | None = None
    schedule: Frequency = "M"
    name: str = "equal_weight"

    def __post_init__(self) -> None:
        self.top_n = int(self.top_n) if self.top_n is not None else None

    def target_weights(self, view: DataView) -> Weights:
        """Equal weights over the eligible (or most liquid eligible) stocks."""
        symbols = view.top_liquid(self.top_n) if self.top_n else view.eligible()
        return equal_weight(symbols)


@register("buy_hold")
@dataclass
class BuyHold:
    """Hold a single symbol (by default SPY) at 100%.

    Args:
        symbol: What to hold; benchmark ETFs are allowed.
        schedule: Rebalance frequency (a single holding never drifts, so this only
            matters when the symbol has no data yet).
    """

    symbol: str = "SPY"
    schedule: Frequency = "M"
    name: str = "buy_hold"

    def target_weights(self, view: DataView) -> Weights:
        """All-in on ``symbol`` once it has a price; cash before that."""
        closes = view.close([self.symbol])
        has_price = self.symbol in closes.index and bool(closes.notna().all())
        return {self.symbol: 1.0} if has_price else {}
