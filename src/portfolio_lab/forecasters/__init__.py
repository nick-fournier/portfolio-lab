"""Forecasters: each stock's expected return, by class of increasing sophistication.

The strategy (``strategies.meanvar``: pool, optimizer, caps, bear switch) is shared; a
forecaster supplies only the expected returns it optimizes on.

- ``trailing``: class 1, production: each stock's trailing one-year return.
- ``nine``: class 2, the Forecaster: least squares on nine terms, refit every month on all
  earlier months; ``grinold`` turns its forecasts into expected returns (Grinold's rule).
- ``baseline``, ``grade`` and ``report``: graded on unseen months, for the Forecasts page.

Class 3 (a transformer) is researched under ``research`` and moves here once it earns it.
"""
