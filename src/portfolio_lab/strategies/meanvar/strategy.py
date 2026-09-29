"""The mean-variance strategy: forecast returns, then optimize on the efficient frontier."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from portfolio_lab.core.calendar import Frequency
from portfolio_lab.research.dataview import DataView
from portfolio_lab.strategies.base import Weights, register
from portfolio_lab.strategies.meanvar.forecast import Forecaster, ForecastSpec
from portfolio_lab.strategies.meanvar.optimize import optimize

#: Share of the lookback window a symbol needs data for to be considered.
MIN_COVERAGE = 0.95


@register("meanvar")
@dataclass
class MeanVar:
    """Mean-variance optimization: forecast returns, then maximize return per unit of risk.

    Each month it takes the most liquid eligible stocks, forecasts their returns with the
    chosen model, estimates how they move together from a year of daily prices, and solves
    for the maximum-Sharpe portfolio (no stock above 10%). This rebuilds the original
    optimizer's core, with its units and data handling fixed. With trailing-return
    forecasts it concentrates in recent winners (a momentum tilt), the style most
    flattered by survivorship bias in free data, so treat its results as an upper bound.

    Args:
        model: Return model: ``arima_logret`` (default), ``arima320_price`` (the legacy
            model) or ``historical_mean`` (no forecast).
        objective: ``max_sharpe``, ``min_volatility`` or ``max_quadratic_utility``.
        top_n: Candidates: the most liquid eligible stocks.
        lookback: Sessions of history for forecasts and covariance.
        horizon: Forecast horizon in sessions.
        max_weight: Cap on any single weight.
        schedule: Rebalance frequency.
        cache_dir: Forecast cache root, set by the runner (not a strategy parameter).
        workers: Processes used for model fits (not a strategy parameter).
    """

    model: str = "arima_logret"
    objective: str = "max_sharpe"
    top_n: int = 100
    lookback: int = 252
    horizon: int = 21
    max_weight: float = 0.10
    schedule: Frequency = "M"
    name: str = "meanvar"
    cache_dir: Path | None = field(default=None, repr=False, metadata={"param": False})
    workers: int = field(default=1, metadata={"param": False})

    def __post_init__(self) -> None:
        self.top_n, self.lookback, self.horizon = (
            int(self.top_n),
            int(self.lookback),
            int(self.horizon),
        )
        self.max_weight = float(self.max_weight)
        self._forecaster: Forecaster | None = None

    def _get_forecaster(self) -> Forecaster:
        """Create the forecaster lazily, once the runner has set ``cache_dir``/``workers``."""
        if self._forecaster is None:
            spec = ForecastSpec(self.model, self.horizon, self.lookback)
            self._forecaster = Forecaster(spec, self.cache_dir, self.workers)
        return self._forecaster

    def target_weights(self, view: DataView) -> Weights:
        """Forecast the most liquid eligible stocks and optimize their weights."""
        prices = view.prices(self.lookback, view.top_liquid(self.top_n))
        coverage = prices.notna().mean()
        prices = prices.loc[:, coverage >= MIN_COVERAGE].ffill().dropna()
        if prices.shape[1] < 2 or len(prices) < 30:
            return {}
        windows = {s: prices[s].to_numpy() for s in prices.columns}
        mu = pd.Series(self._get_forecaster().forecast(view.asof, windows), dtype=float)
        return optimize(mu, prices, view.risk_free(), self.objective, self.max_weight)

    def close(self) -> None:
        """Release the forecaster's worker pool."""
        if self._forecaster is not None:
            self._forecaster.close()
