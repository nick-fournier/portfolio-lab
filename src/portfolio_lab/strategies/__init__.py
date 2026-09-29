"""Strategies: each maps a point-in-time ``DataView`` to target portfolio weights.

Importing this package registers every built-in strategy in ``base.REGISTRY``.
"""

from portfolio_lab.strategies import (
    baselines,  # noqa: F401  (registers strategies)
    momentum,  # noqa: F401
    piotroski,  # noqa: F401
)
from portfolio_lab.strategies.meanvar import strategy as _meanvar  # noqa: F401
