"""Soft cutoffs for picking meanvar's candidates: no stock is all-in at #400 and out at #401.

Production takes the :data:`TOP` healthiest of the :data:`POOL` most liquid stocks, so a
stock's fate can hinge on a tiny difference in volume or health right at either line.
Three softer ways to choose (``MeanVar.soften``), each returning candidate sets with
optional per-stock weight caps:

- ``average``: the same selection at every pool in :data:`POOLS` and list size in
  :data:`TOPS`; the strategy optimizes each and averages the weights, so a stock near an
  edge gets a partial weight (it is in some of the sets, not all).
- ``taper``: one set, the :data:`TAPER_HEALTH` healthiest of the :data:`TAPER_LIQUIDITY`
  most liquid, with each stock's weight cap shrinking linearly to zero across those ranges
  (liquidity ranks 300-500, health ranks 100-150 among the 500, the equivalent of 80-120
  among 400).
- ``sticky``: production's selection with hysteresis: a stock enters the pool at liquidity
  rank 400 but leaves only below 500, and enters the list at health rank 100 but leaves
  only below 130.
"""

from collections.abc import Callable, Sequence

from portfolio_lab.research.dataview import DataView

POOL, TOP = 400, 100
POOLS = (300, 350, 400, 450, 500)
TOPS = (80, 100, 120)
TAPER_LIQUIDITY = (300, 500)
TAPER_HEALTH = (100, 150)
STICKY_POOL = (400, 500)
STICKY_TOP = (100, 130)
MODES = ("average", "taper", "sticky")

#: Candidates and, for ``taper``, each one's weight cap as a share of the usual cap.
CandidateSet = tuple[list[str], dict[str, float] | None]
Health = Callable[[DataView, list[str]], dict[str, float]]


def fade(rank: int, lo: int, hi: int) -> float:
    """1 up to rank ``lo``, falling linearly to 0 at rank ``hi`` (ranks start at 1)."""
    return 1.0 if rank <= lo else max(0.0, (hi - rank) / (hi - lo))


def _healthiest(health: Health, view: DataView, pool: list[str]) -> list[str]:
    scores = health(view, pool)
    return sorted(scores, key=lambda s: -scores[s])


class SoftSelector:
    """Chooses candidate sets for one :data:`MODES` mode; ``sticky`` keeps state.

    Args:
        mode: One of :data:`MODES`.
        health: The strategy's health score of a pool (higher is healthier).
    """

    def __init__(self, mode: str, health: Health):
        if mode not in MODES:
            raise ValueError(f"unknown soften mode {mode!r}; choose from {MODES}")
        self.mode = mode
        self.health = health
        self._pool: set[str] = set()
        self._candidates: set[str] = set()

    def sets(self, view: DataView, among: Sequence[str] | None) -> list[CandidateSet]:
        """The candidate sets for this rebalance (see module docs)."""
        among = list(among) if among is not None else None
        if self.mode == "average":
            order = view.top_liquid(max(POOLS), among=among)
            out = []
            for pool in POOLS:
                ranked = _healthiest(self.health, view, order[:pool])
                out += [(ranked[:top], None) for top in TOPS]
            return out
        if self.mode == "taper":
            order = view.top_liquid(TAPER_LIQUIDITY[1], among=among)
            liquidity = {s: k + 1 for k, s in enumerate(order)}
            ranked = _healthiest(self.health, view, order)[: TAPER_HEALTH[1]]
            shares = {
                s: fade(liquidity[s], *TAPER_LIQUIDITY) * fade(k + 1, *TAPER_HEALTH)
                for k, s in enumerate(ranked)
            }
            shares = {s: f for s, f in shares.items() if f > 0}
            return [(list(shares), shares)]
        order = view.top_liquid(STICKY_POOL[1], among=among)
        enter, leave = STICKY_POOL
        pool = order[:enter] + [s for s in order[enter:leave] if s in self._pool]
        ranked = _healthiest(self.health, view, pool)
        enter, leave = STICKY_TOP
        candidates = ranked[:enter] + [s for s in ranked[enter:leave] if s in self._candidates]
        self._pool, self._candidates = set(pool), set(candidates)
        return [(candidates, None)]
