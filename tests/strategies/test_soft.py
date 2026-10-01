import pytest

from portfolio_lab.strategies.meanvar.soft import POOLS, TOPS, SoftSelector, fade


class _View:
    """Liquidity order is the symbol number; health is set per test."""

    def __init__(self, n: int):
        self.symbols = [f"S{k:03d}" for k in range(n)]

    def top_liquid(self, n, among=None):
        return self.symbols[:n]


def _health(order):
    """Health ranking: ``order`` first (healthiest), the rest after."""
    return lambda view, pool: {s: (2.0 if s in order else 1.0) - pool.index(s) * 1e-6 for s in pool}


def test_fade():
    assert fade(300, 300, 500) == 1.0
    assert fade(400, 300, 500) == pytest.approx(0.5)
    assert fade(500, 300, 500) == 0.0


def test_average_makes_one_set_per_pool_and_size():
    sets = SoftSelector("average", _health([])).sets(_View(600), None)
    assert len(sets) == len(POOLS) * len(TOPS)
    assert {len(c) for c, _ in sets} == set(TOPS)


def test_taper_shrinks_caps_near_the_edges():
    [(candidates, shares)] = SoftSelector("taper", _health([])).sets(_View(600), None)
    assert shares["S000"] == 1.0  # most liquid and healthiest
    assert 0 < shares[candidates[-1]] < 1
    assert len(candidates) < 150


def test_sticky_keeps_members_until_they_fall_past_the_exit():
    view = _View(600)
    selector = SoftSelector("sticky", _health(["S350"]))
    first = selector.sets(view, None)[0][0]
    assert "S350" in first
    # 120 new names jump ahead: S350 falls to liquidity rank 471, past the entry line (400)
    # but inside the exit line (500).
    view.symbols = view.symbols[:300] + [f"X{k}" for k in range(120)] + view.symbols[300:]
    second = selector.sets(view, None)[0][0]
    assert "S350" in second
    fresh = SoftSelector("sticky", _health(["S350"])).sets(view, None)[0][0]
    assert "S350" not in fresh
