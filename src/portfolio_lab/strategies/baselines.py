"""Baseline strategies every other strategy is judged against."""

from dataclasses import dataclass

from portfolio_lab.core.calendar import Frequency
from portfolio_lab.research.dataview import DataView
from portfolio_lab.strategies import explain
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

    def explain(self) -> dict[str, str]:
        """Plain-language description of a run (see ``strategies.explain``)."""
        pool = "every eligible stock" if self.top_n is None else f"the {self.top_n} most liquid"
        return {
            "summary": f"Holds {pool} in equal weights: the no-skill baseline.",
            "candidates": explain.candidates(self.top_n),
            "signal": "None. Every candidate is held; there is no ranking.",
            "construction": "Equal weights: each candidate gets the same share (1/N).",
            "drivers": (
                "How the average stock in the pool did. Anything a stock-picking strategy "
                "earns beyond this is what its picking added."
            ),
            "related": (
                "The control for strategies that pick from the same pool: with top_n=100 it "
                "holds all of the candidates that momentum, meanvar and piotroski (pool=100) "
                "choose from."
            ),
        }

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

    def explain(self) -> dict[str, str]:
        """Plain-language description of a run (see ``strategies.explain``)."""
        return {
            "summary": f"Buys {self.symbol} and holds it: the benchmark to beat.",
            "candidates": (
                f"Only {self.symbol}"
                + (
                    ", the fund tracking the S&P 500 (the 500 largest US companies)."
                    if self.symbol == "SPY"
                    else "."
                )
            ),
            "signal": "None: nothing is ranked or predicted.",
            "construction": f"100% in {self.symbol} (cash until it has a price).",
            "drivers": f"Simply how {self.symbol} did.",
            "related": "Every strategy is measured against this, after costs.",
        }

    def target_weights(self, view: DataView) -> Weights:
        """All-in on ``symbol`` once it has a price; cash before that."""
        closes = view.close([self.symbol])
        has_price = self.symbol in closes.index and bool(closes.notna().all())
        return {self.symbol: 1.0} if has_price else {}
