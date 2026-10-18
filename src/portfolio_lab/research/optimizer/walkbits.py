"""Small pieces of the walk (research.optimizer.walk): its random streams, starting book, model."""

import numpy as np

from portfolio_lab.research.optimizer import realism


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
