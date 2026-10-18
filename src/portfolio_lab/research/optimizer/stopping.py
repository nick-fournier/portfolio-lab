"""Held-out months in each month's training, scored after every pass (2026-10-15).

With ``watch`` (research.optimizer.walk) each month holds out ``holdout`` of its training
months, a random block of consecutive months or (``scatter``) single months drawn anywhere,
trains on the rest one pass at a time and scores the held-out months' loss after each pass.
Every pass runs and the last is kept: the score curve is a reading of how training went, not a
rule that acts on it (the walk logs it as ``held_curve``, and as ``held_loss`` its best mean
over :data:`SMOOTH` consecutive passes). The held-out months move every month, so over time
every month is trained on. :class:`Guard` undoes a pass that leaves the model broken.
"""

import copy
import math

import numpy as np

#: Consecutive passes the logged best held-out loss is averaged over (one pass's score jumps
#: around far more than its trend moves; grid9).
SMOOTH = 3


def held_block(train: list[int], holdout: int, rng: np.random.Generator) -> tuple[list, list]:
    """A random block of ``holdout`` consecutive months of ``train``: (months to fit, block).

    With too few months to spare a block (``holdout`` + 2 or fewer), nothing is held out.
    """
    if holdout <= 0 or len(train) <= holdout + 2:
        return train, []
    a = int(rng.integers(len(train) - holdout + 1))
    return train[:a] + train[a + holdout :], train[a : a + holdout]


def held_months(train: list[int], holdout: int, rng: np.random.Generator,
                scatter: bool = False) -> tuple[list, list]:  # fmt: skip
    """``holdout`` months of ``train`` held out: one block (:func:`held_block`) or scattered.

    ``scatter``: single months drawn at random anywhere in ``train`` (in month order), so no
    one stretch of the market (a crash) decides alone; same too-few-months rule as the block.
    """
    if not scatter:
        return held_block(train, holdout, rng)
    if holdout <= 0 or len(train) <= holdout + 2:
        return train, []
    pick = set(rng.choice(len(train), holdout, replace=False).tolist())
    return ([k for i, k in enumerate(train) if i not in pick],
            [k for i, k in enumerate(train) if i in pick])  # fmt: skip


def best_smoothed(curve: list[float]) -> float:
    """The lowest mean of :data:`SMOOTH` consecutive finite scores (``inf``: no such window)."""
    best = math.inf
    for p in range(len(curve)):
        recent = [v for v in curve[max(p + 1 - SMOOTH, 0) : p + 1] if math.isfinite(v)]
        value = sum(recent) / SMOOTH if len(recent) == SMOOTH else math.nan
        if math.isfinite(value) and value < best:
            best = value
    return best


def fit_watched(one_pass, score, held: bool, passes: int) -> tuple[list, int, float, list[float]]:
    """Run ``one_pass(p)`` for every pass ``p`` (0-based); with ``held``, ``score()`` after each.

    ``score()`` grades the held-out months (lower is better). Returns the last pass's training
    losses, the passes run, :func:`best_smoothed` of the scores and every pass's score (without
    held-out months, ``held`` False: ``nan`` and an empty curve).
    """
    losses: list = []
    curve: list[float] = []
    for p in range(passes):
        losses = one_pass(p)
        if held:
            curve.append(score())
    return losses, passes, best_smoothed(curve) if held else math.nan, curve


class Guard:
    """Rolls back a pass that leaves the model broken; the passes after it run at half rate.

    A pass whose weights come out not finite (a blown-up step, bench1-3) is undone: model,
    Adam and ``average`` go back to where the pass began, ``rollbacks`` counts it and
    ``scale`` (the factor on every later pass's rate) halves, so a month never ends broken.
    """

    def __init__(self, model, adam, average=None):
        self.model, self.adam, self.average = model, adam, average
        self.scale, self.rollbacks = 1.0, 0

    def __call__(self, one_pass, p: int) -> list:
        """Run ``one_pass(p)``; its losses, or none if it was rolled back."""
        saved = (copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.adam.state_dict()),
                 self.average.state() if self.average is not None else None)  # fmt: skip
        losses = one_pass(p)
        if all(bool(q.isfinite().all()) for q in self.model.parameters()):
            return losses
        self.model.load_state_dict(saved[0])
        self.adam.load_state_dict(saved[1])
        if self.average is not None:
            self.average.load(saved[2])
        self.scale, self.rollbacks = self.scale / 2, self.rollbacks + 1
        return []
