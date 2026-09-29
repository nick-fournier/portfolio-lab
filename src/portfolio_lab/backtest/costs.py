"""Trading cost model: half-spread plus size-dependent market impact.

Cost of a rebalance, as a fraction of portfolio value, is the sum over names of
``|weight change| x (half_spread + impact)``. Impact grows with participation: the dollar
trade as a share of the name's trailing median daily dollar volume, at a configurable
portfolio notional. Deterministic, so backtests are reproducible.
"""

from dataclasses import dataclass, field

import numpy as np

BPS = 1e-4

#: (max participation, impact in bps) buckets, checked in order.
DEFAULT_IMPACT_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.001, 0.0),
    (0.01, 5.0),
    (0.05, 20.0),
    (float("inf"), 50.0),
)


@dataclass(frozen=True)
class CostModel:
    """Costs charged on each rebalance.

    Args:
        half_spread_bps: Cost per unit of turnover for crossing half the spread.
        impact_buckets: Participation thresholds and their impact cost in bps.
        notional: Portfolio size in dollars, used to size trades against volume.
    """

    half_spread_bps: float = 5.0
    impact_buckets: tuple[tuple[float, float], ...] = field(default=DEFAULT_IMPACT_BUCKETS)
    notional: float = 100_000.0

    def cost(self, trades: np.ndarray, adv: np.ndarray, nav: float = 1.0) -> float:
        """Return the cost of a rebalance as a fraction of portfolio value.

        Args:
            trades: Weight changes per name (target minus current), any sign.
            adv: Trailing median daily dollar volume per name; NaN means unknown, which is
                charged the highest impact bucket.
            nav: Portfolio value relative to the starting notional.
        """
        size = np.abs(trades)
        dollars = size * nav * self.notional
        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.where(adv > 0, dollars / adv, np.inf)
        impact = np.full(size.shape, self.impact_buckets[-1][1])
        for threshold, bps in reversed(self.impact_buckets):
            impact = np.where(participation <= threshold, bps, impact)
        return float(np.sum(size * (self.half_spread_bps + impact)) * BPS)
