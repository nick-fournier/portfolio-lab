"""Early stopping for each month's training (2026-10-15).

A fixed number of passes is wrong both ways: a calm month needs few passes from last month's
model, a shock month more, and too many overfit (grid9: held-out loss bottoms near pass 15
from a cold start while the book's size balloons). Instead, each month holds out a random
block of ``holdout`` consecutive training months, trains on the rest one pass at a time,
scores the block after each pass (its Sharpe, net of costs with costs: free of the book's
size, which the squared loss is not, so one crash month cannot dominate), and stops once the
score has not improved for ``patience`` passes (at most ``most``). The model and Adam then go
back to the best pass. The block moves every month, so over time every month is trained on.
"""

import copy
import math

import numpy as np


def held_block(train: list[int], holdout: int, rng: np.random.Generator) -> tuple[list, list]:
    """A random block of ``holdout`` consecutive months of ``train``: (months to fit, block).

    With too few months to spare a block (``holdout`` + 2 or fewer), nothing is held out.
    """
    if len(train) <= holdout + 2:
        return train, []
    a = int(rng.integers(len(train) - holdout + 1))
    return train[:a] + train[a + holdout :], train[a : a + holdout]


def fit_stopped(
    model, adam, one_pass, score, held: bool, most: int, patience: int
) -> tuple[list, int, float]:
    """Train by ``one_pass(p)`` until ``score()`` stops rising (module docs).

    ``one_pass(p)`` runs pass ``p`` (0-based) and returns its losses; ``score()`` grades the
    held-out block (higher is better). Returns the kept pass's training losses, the passes
    kept and the best score (without a block, ``held`` False: ``most`` passes, ``nan``).
    """
    losses: list = []
    if not held:
        for p in range(most):
            losses = one_pass(p)
        return losses, most, math.nan
    best, best_at, kept = -math.inf, 0, None
    for p in range(most):
        losses = one_pass(p)
        value = score()
        if math.isfinite(value) and value > best:
            best, best_at = value, p + 1
            kept = (copy.deepcopy(model.state_dict()), copy.deepcopy(adam.state_dict()), losses)
        elif p + 1 - best_at >= patience:
            break
    if kept is None:  # never finite: keep the last pass
        return losses, p + 1, math.nan
    model.load_state_dict(kept[0])
    adam.load_state_dict(kept[1])
    return kept[2], best_at, best
