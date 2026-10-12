"""A strategy that holds a walk's weights, so the backtest engine can grade them with costs."""

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from portfolio_lab.core.calendar import Frequency
from portfolio_lab.research.dataview import DataView
from portfolio_lab.strategies.base import Weights


@dataclass
class Replay:
    """Each month end, hold the weights the walk chose at the latest month end up to then.

    Names no longer eligible at the decision date, and weights below ``min_weight``, are
    dropped and the rest scaled back up to sum to 1 (names were eligible when chosen; the
    engine refuses ineligible names).

    Args:
        weights: The walk's ``weights`` frame (date, symbol, weight).
        min_weight: Smallest weight held (0: hold every candidate the softmax gives weight).
        name: Strategy name for the run.
    """

    weights: pl.DataFrame = field(metadata={"param": False})
    min_weight: float = 0.0
    name: str = "optimizer"
    schedule: Frequency = "M"

    def __post_init__(self) -> None:
        self._by_date: dict[date, Weights] = {
            d: dict(zip(g["symbol"], g["weight"], strict=True))
            for (d,), g in self.weights.group_by("date")
        }
        self._dates = sorted(self._by_date)

    def target_weights(self, view: DataView) -> Weights:
        """The latest chosen weights, restricted to names eligible now."""
        k = bisect_right(self._dates, view.asof)
        if k == 0:
            return {}
        chosen = self._by_date[self._dates[k - 1]]
        ok = set(view.eligible())
        kept = {s: w for s, w in chosen.items() if s in ok and w > max(self.min_weight, 0.0)}
        total = sum(kept.values())
        return {s: w / total for s, w in kept.items()} if total > 0 else {}
