"""The signal contract and registry.

A signal maps a point-in-time :class:`~portfolio_lab.research.dataview.DataView` and a list
of candidate symbols to scores, ``{symbol: score}``, where a higher score predicts a higher
return. Only the ranking matters, so a signal can be a return forecast, a quality score or
anything else. The scoreboard (``research.scoreboard``) measures how well each ranking
predicted the returns that followed.
"""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from portfolio_lab.research.dataview import DataView

Scores = dict[str, float]


class Signal(Protocol):
    """Anything with a name and a ``score`` method."""

    name: str

    def score(self, view: DataView, symbols: Sequence[str]) -> Scores:
        """Score ``symbols`` using data up to the view's decision date."""
        ...


#: Signal factories by name. Each takes keyword parameters.
REGISTRY: dict[str, Callable[..., Signal]] = {}


def register(name: str) -> Callable[[Callable[..., Signal]], Callable[..., Signal]]:
    """Class decorator adding a signal factory to :data:`REGISTRY`."""

    def decorator(factory: Callable[..., Signal]) -> Callable[..., Signal]:
        if name in REGISTRY:
            raise ValueError(f"signal {name!r} registered twice")
        REGISTRY[name] = factory
        return factory

    return decorator


def create(name: str, **params: Any) -> Signal:
    """Instantiate a registered signal by name.

    Raises:
        KeyError: If no signal is registered under ``name``.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown signal {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
