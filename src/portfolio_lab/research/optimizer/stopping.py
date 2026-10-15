"""Early stopping for each month's training (2026-10-15).

A fixed number of passes is wrong both ways: a calm month needs few passes from last month's
model, a shock month more, and too many overfit (grid9: from a cold start the held-out loss
bottoms near pass 15-20 and then rises while the book's size balloons). Instead, each month
holds out a random block of ``holdout`` consecutive training months, trains on the rest one
pass at a time and scores the block's loss after each pass. The score jumps around from pass
to pass far more than its trend moves (grid9: +-2 in held-out Sharpe against a trend of ~0.1 a
pass; run32 stopped on the first dip), so the rule reads a smoothed score, the mean over the
last ``smooth`` passes: training stops once that has not improved for ``patience`` passes (at
most ``most``), and the model and Adam go back to the last pass of the best smoothed window.
The block moves every month, so over time every month is trained on.
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


def fit_stopped(  # noqa: PLR0913, PLR0917 - the rule's pieces
    model, adam, one_pass, score, held: bool, most: int, patience: int, smooth: int = 3,
    average=None,
) -> tuple[list, int, float, list[float]]:  # fmt: skip
    """Train by ``one_pass(p)`` until the smoothed ``score()`` stops falling (module docs).

    ``one_pass(p)`` runs pass ``p`` (0-based) and returns its losses; ``score()`` grades the
    held-out block (lower is better); ``average`` (research.optimizer.averaging) is rolled back
    with the model. Returns the kept pass's training losses, the passes kept, the best
    smoothed score and every pass's raw score (without a block, ``held`` False: ``most``
    passes, ``nan``, empty).
    """
    losses: list = []
    if not held:
        for p in range(most):
            losses = one_pass(p)
        return losses, most, math.nan, []
    best, best_at, kept, curve = math.inf, 0, None, []
    for p in range(most):
        losses = one_pass(p)
        curve.append(score())
        recent = [v for v in curve[-smooth:] if math.isfinite(v)]
        value = sum(recent) / smooth if len(recent) == smooth else math.nan
        if math.isfinite(value) and value < best:
            best, best_at = value, p + 1
            kept = (copy.deepcopy(model.state_dict()), copy.deepcopy(adam.state_dict()), losses,
                    average.state() if average is not None else None)  # fmt: skip
        elif p + 1 - max(best_at, smooth) >= patience:
            break
    if kept is None:  # never a finite smoothed score: keep the last pass
        return losses, p + 1, math.nan, curve
    model.load_state_dict(kept[0])
    adam.load_state_dict(kept[1])
    if average is not None:
        average.load(kept[3])
    return kept[2], best_at, best, curve
