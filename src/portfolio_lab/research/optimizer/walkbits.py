"""Small pieces of the walk (research.optimizer.walk): its random streams, starting book, model."""

import numpy as np
import torch

from portfolio_lab.research.optimizer import holdings, realism


def starting(settings, pricing, months: list, data: list, t: int, last,
             traded: int) -> tuple:  # fmt: skip
    """(book held going into t, month it was traded): the ``start_book`` when none was traded."""
    if traded == t - 1 or settings.start_book == "guess" or pricing is None:
        return last, traded
    market = pricing.market[t - 1] if pricing.market is not None else None
    first = realism.start_book(settings.start_book, market, len(months[t - 1].symbols))
    return first.to(data[t][0]), t - 1


def walk_streams(settings) -> tuple[np.random.Generator, np.random.Generator]:
    """The walk's (month order, held-out draw) random streams: the seed's."""
    rng = np.random.default_rng(settings.seed)
    hold = rng if settings.holdout_seed is None else np.random.default_rng(settings.holdout_seed)
    return rng, hold


def traded(model, average):
    """The model as it traded: with an ``average``, the averaged parameters put in for good."""
    if average is not None:
        average.install()
    return model


def unit_start(model, data: list, pricing, months: list, settings) -> float:
    """Scale a fresh model so its book starts at gross 1 (``unit_start``, 2026-10-19).

    The book is the readout (``model.lam``) times the last block's output, so dividing the
    readout by the raw gross of the month before the first traded month (holdings from an
    empty book: cash) makes that month's book exactly gross 1. Returns the gross divided by.
    """
    picks = [t for t, m in enumerate(months) if m.date >= settings.start
             and (settings.only is None or m.date in settings.only)]  # fmt: skip
    k = max((picks[0] if picks else len(months) - 1) - 1, 0)
    with torch.no_grad():
        held = pricing.held() if pricing is not None and pricing.book is not None else None
        w = realism.book(model, holdings.inputs(data, pricing, held, k), pricing, k)
        gross = float(w.abs().sum())
        if gross > 0:
            model.lam.div_(gross)
    return gross
