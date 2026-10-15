"""Early stopping for each month's training (2026-10-15).

A fixed number of passes is wrong both ways: a calm month needs about one pass from last
month's model, a shock month needs many, and a cold start overfits past ~15 passes (held-out
loss rises while the training loss keeps falling and the book's size balloons; grid9). Instead,
each month holds out a random block of ``holdout`` consecutive training months, trains on the
rest one pass at a time, scores the block after each pass, and stops once its loss has not
improved for ``patience`` passes (at most ``most`` passes); the model and Adam are then rolled
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
    model, adam, one_pass, held_loss, held: list[int], most: int, patience: int
) -> tuple[list, int, float]:
    """Train by ``one_pass(first)`` until ``held_loss(held)`` stops improving (module docs).

    ``one_pass(first)`` runs one pass and returns its losses (``first``: the month's first
    pass). Returns the best pass's training losses, the passes kept and the best held-out loss
    (without a block: ``most`` passes, ``nan``).
    """
    if not held:
        losses = []
        for p in range(most):
            losses = one_pass(p == 0)
        return losses, most, math.nan
    best, best_at, kept, losses = math.inf, 0, None, []
    for p in range(most):
        losses = one_pass(p == 0)
        value = held_loss(held)
        if math.isfinite(value) and value < best:
            best, best_at = value, p + 1
            kept = (copy.deepcopy(model.state_dict()), copy.deepcopy(adam.state_dict()), losses)
        elif p + 1 - best_at >= patience:
            break
    if kept is None:  # never finite: keep the last pass
        return losses, most, math.nan
    model.load_state_dict(kept[0])
    adam.load_state_dict(kept[1])
    return kept[2], best_at, best
