"""The strategy contract and registry.

A strategy maps a point-in-time :class:`~portfolio_lab.research.dataview.DataView` to target
portfolio weights: ``{symbol: weight}``, long-only, summing to at most 1 (the remainder is
held as cash). The backtest engine calls :meth:`Strategy.target_weights` on each rebalance
date of the strategy's ``schedule``, strictly in date order.

Strategies built from a scoring model and a portfolio rule can use :class:`Composed`
(``signal`` then ``construct``), which is how forecasters and the lead-lag detector plug in.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from portfolio_lab.core.calendar import Frequency
from portfolio_lab.research.dataview import DataView

Weights = dict[str, float]


class Strategy(Protocol):
    """Anything with a name, a rebalance schedule and a ``target_weights`` method."""

    name: str
    schedule: Frequency

    def target_weights(self, view: DataView) -> Weights:
        """Return target weights given data up to the view's decision date."""
        ...


@dataclass
class Composed:
    """A strategy built from a signal (scores per symbol) and a portfolio construction rule.

    Args:
        name: Strategy name.
        schedule: Rebalance frequency.
        signal: Maps a view to ``{symbol: score}`` (e.g. expected returns).
        construct: Maps scores and the view to weights.
    """

    name: str
    schedule: Frequency
    signal: Callable[[DataView], dict[str, float]]
    construct: Callable[[dict[str, float], DataView], Weights]

    def target_weights(self, view: DataView) -> Weights:
        """Score the universe, then turn the scores into weights."""
        return self.construct(self.signal(view), view)


#: Strategy factories by name. Each takes keyword parameters (as strings or values).
REGISTRY: dict[str, Callable[..., Strategy]] = {}


def register(name: str) -> Callable[[Callable[..., Strategy]], Callable[..., Strategy]]:
    """Class/function decorator adding a strategy factory to :data:`REGISTRY`."""

    def decorator(factory: Callable[..., Strategy]) -> Callable[..., Strategy]:
        if name in REGISTRY:
            raise ValueError(f"strategy {name!r} registered twice")
        REGISTRY[name] = factory
        return factory

    return decorator


def create(name: str, **params: Any) -> Strategy:
    """Instantiate a registered strategy by name.

    Raises:
        KeyError: If no strategy is registered under ``name``.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
