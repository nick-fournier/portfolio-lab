"""Baseline strategies every other strategy is judged against."""

from dataclasses import dataclass

from portfolio_lab.core.calendar import Frequency
from portfolio_lab.research.dataview import DataView
from portfolio_lab.strategies.base import Weights, register
from portfolio_lab.strategies.construct import equal_weight


@register("equal_weight")
@dataclass
class EqualWeightEligible:
    """Equal-weight baseline: every eligible stock, equally weighted, rebalanced monthly.

    It shows what owning the market evenly earns, with no stock picking at all. With
    ``top_n``, only the most liquid eligible stocks are held.

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
    """Benchmark: buy and hold one symbol, by default SPY (the S&P 500).

    Any strategy has to beat this, after costs, to be worth its complexity.

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
