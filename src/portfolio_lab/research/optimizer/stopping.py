"""Held-out months in each month's training, scored after every pass (2026-10-15).

With ``watch`` (research.optimizer.walk) each month holds out ``holdout`` of its training
months, a random block of consecutive months or (``scatter``) single months drawn anywhere,
trains on the rest one pass at a time and scores the held-out months after each pass. By
default every pass runs and the last is kept: the score curve is a reading of how training
went, not a rule that acts on it (the walk logs it as ``held_curve``, and as ``held_loss`` its
best mean over :data:`SMOOTH` consecutive passes). With ``keep_best`` (2026-10-19) the score
is the held-out ranking (:func:`rank_corr` of each held-out book with its month's returns, the
sign flipped so that lower is better) and :class:`Keeper` acts on it: the model, its Adam and
its parameter average are kept as they stood at the pass with the best smoothed score, and
training stops once ``keep_patience`` passes have gone by without a new best (the held-out
ranking peaks at pass 3-13 and then decays as the training months are memorized; stage 1,
2026-10-19). The held-out months move every month, so over time every month is trained on.
:class:`Guard` undoes a pass that leaves the model broken.
"""

import copy
import math

import numpy as np
import torch

#: Consecutive passes a held-out score is averaged over (one pass's score jumps around far
#: more than its trend moves; grid9).
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
                scatter: bool = False, every: int = 0) -> tuple[list, list]:  # fmt: skip
    """``holdout`` months of ``train`` held out: one block (:func:`held_block`) or scattered.

    ``scatter``: single months drawn at random anywhere in ``train`` (in month order), so no
    one stretch of the market (a crash) decides alone; same too-few-months rule as the block.
    ``every`` (2026-10-19): instead, every ``every``-th month end (by its position in the
    walk's months) is held out, the same months whichever month is training, so a model that
    continues from month to month never trains on them and their score stays out of sample;
    ``holdout``, ``rng`` and ``scatter`` are then not used.
    """
    if every:
        return [k for k in train if k % every], [k for k in train if not k % every]
    if not scatter:
        return held_block(train, holdout, rng)
    if holdout <= 0 or len(train) <= holdout + 2:
        return train, []
    pick = set(rng.choice(len(train), holdout, replace=False).tolist())
    return ([k for i, k in enumerate(train) if i not in pick],
            [k for i, k in enumerate(train) if i in pick])  # fmt: skip


def rank_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rank correlation of ``a`` with ``b`` (Spearman; equal values ranked in position order)."""
    ra = torch.argsort(torch.argsort(a, stable=True), stable=True).to(a.dtype)
    rb = torch.argsort(torch.argsort(b, stable=True), stable=True).to(a.dtype)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return (ra * rb).sum() / (ra.norm() * rb.norm()).clamp(min=1e-12)


def smoothed_at(curve: list[float], p: int) -> float:
    """The mean of the :data:`SMOOTH` scores ending at pass ``p`` (``nan`` unless all finite)."""
    recent = curve[max(p + 1 - SMOOTH, 0) : p + 1]
    if len(recent) < SMOOTH or not all(math.isfinite(v) for v in recent):
        return math.nan
    return sum(recent) / SMOOTH


def best_smoothed(curve: list[float]) -> float:
    """The lowest :func:`smoothed_at` over the curve (``inf``: no full window of finite scores)."""
    best = math.inf
    for p in range(len(curve)):
        value = smoothed_at(curve, p)
        if math.isfinite(value) and value < best:
            best = value
    return best


class Keeper:
    """Keeps the model as it stood at the pass with the best smoothed score; says when to stop.

    After each pass's score :meth:`see` smooths the curve (:func:`smoothed_at`); a new best
    snapshots the model, its Adam and its ``average`` (as :class:`Guard` does), and
    ``patience`` passes without a new best is the signal to stop. :meth:`restore` puts the
    snapshot back once training is over; with no finite score nothing was kept and the model
    stays as it ended (``best_pass`` -1).
    """

    def __init__(self, model, adam, average=None, patience: int = 4):
        self.model, self.adam, self.average, self.patience = model, adam, average, patience
        self.best, self.best_pass, self.saved = math.inf, -1, None

    def see(self, p: int, curve: list[float]) -> bool:
        """Note pass ``p``'s smoothed score; True once ``patience`` passes brought no new best."""
        value = smoothed_at(curve, p)
        if math.isfinite(value) and value < self.best:
            self.best, self.best_pass = value, p
            self.saved = (copy.deepcopy(self.model.state_dict()),
                          copy.deepcopy(self.adam.state_dict()),
                          self.average.state() if self.average is not None else None)  # fmt: skip
        return self.best_pass >= 0 and p - self.best_pass >= self.patience

    def restore(self) -> None:
        """Put the best pass's model, Adam and average back (nothing kept: leave them)."""
        if self.saved is None:
            return
        self.model.load_state_dict(self.saved[0])
        self.adam.load_state_dict(self.saved[1])
        if self.average is not None:
            self.average.load(self.saved[2])


def fit_watched(one_pass, score, held: bool, passes: int,
                keeper: Keeper | None = None) -> tuple[list, int, float, list[float]]:  # fmt: skip
    """Run ``one_pass(p)`` for every pass ``p`` (0-based); with ``held``, ``score()`` after each.

    ``score()`` grades the held-out months (lower is better). With a ``keeper`` (and ``held``)
    it sees every score, the passes end when it says so and the pass it kept is restored.
    Returns the last pass's training losses, the passes run, :func:`best_smoothed` of the
    scores and every pass's score (without held-out months, ``held`` False: ``nan`` and an
    empty curve).
    """
    losses: list = []
    curve: list[float] = []
    run = 0
    for p in range(passes):
        losses = one_pass(p)
        run = p + 1
        if held:
            curve.append(score())
            if keeper is not None and keeper.see(p, curve):
                break
    if keeper is not None and held:
        keeper.restore()
    return losses, run, best_smoothed(curve) if held else math.nan, curve


class Guard:
    """Rolls back a pass that leaves the model broken; the passes after it run at half rate.

    A pass whose weights come out not finite (a blown-up step, bench1-3) is undone: model,
    Adam, ``average`` and ``book`` (the pricing's book table, refreshed at every step, so a
    broken model's steps leave it not finite too) go back to where the pass began,
    ``rollbacks`` counts it and ``scale`` (the factor on every later pass's rate) halves, so a
    month never ends broken.
    """

    def __init__(self, model, adam, average=None, book: dict | None = None):
        self.model, self.adam, self.average, self.book = model, adam, average, book
        self.scale, self.rollbacks = 1.0, 0

    def __call__(self, one_pass, p: int) -> list:
        """Run ``one_pass(p)``; its losses, or none if it was rolled back."""
        book = None if self.book is None else {k: w.clone() for k, w in self.book.items()}
        saved = (copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.adam.state_dict()),
                 self.average.state() if self.average is not None else None, book)  # fmt: skip
        losses = one_pass(p)
        if all(bool(q.isfinite().all()) for q in self.model.parameters()):
            return losses
        self.model.load_state_dict(saved[0])
        self.adam.load_state_dict(saved[1])
        if self.average is not None:
            self.average.load(saved[2])
        if self.book is not None:  # in place: the pricing holds this table
            self.book.clear()
            self.book.update(saved[3])
        self.scale, self.rollbacks = self.scale / 2, self.rollbacks + 1
        return []
