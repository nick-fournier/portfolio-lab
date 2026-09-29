"""Signals: each scores stocks at a point in time, so its predictive power can be measured.

Importing this package registers every built-in signal in ``base.REGISTRY``.
"""

from portfolio_lab.signals import (
    builtin,  # noqa: F401  (registers signals)
    leadlag,  # noqa: F401
)
