"""Built-in signals: the return forecasts behind mean-variance, and classic anomalies.

The anomalies are the bar a new model has to clear: momentum (past winners keep winning),
short-term reversal (last month's losers bounce), low volatility (calm stocks earn about
as much as volatile ones) and Piotroski quality.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from portfolio_lab.forecasters.trailing import Forecaster, ForecastSpec, price_windows
from portfolio_lab.research.dataview import DataView
from portfolio_lab.signals.base import Scores, register
from portfolio_lab.strategies.momentum import momentum_scores


@register("forecast")
@dataclass
class Forecast:
    """Expected return from one of mean-variance's forecast models.

    Args:
        model: ``historical_mean`` or ``arima320_price``.
        horizon: Forecast horizon in sessions.
        lookback: Sessions of history fitted.
        cache_dir: Forecast cache root, set by the runner (not a parameter).
        workers: Processes used for model fits (not a parameter).
    """

    model: str = "historical_mean"
    horizon: int = 21
    lookback: int = 252
    name: str = "forecast"
    cache_dir: Path | None = field(default=None, repr=False, metadata={"param": False})
    workers: int = field(default=1, metadata={"param": False})

    def __post_init__(self) -> None:
        self._forecaster: Forecaster | None = None

    def score(self, view: DataView, symbols: Sequence[str]) -> Scores:
        """Annualized expected return per symbol."""
        if self._forecaster is None:
            spec = ForecastSpec(self.model, self.horizon, self.lookback)
            self._forecaster = Forecaster(spec, self.cache_dir, self.workers)
        prices = price_windows(view, symbols, self.lookback)
        if len(prices) < 30:
            return {}
        windows = {s: prices[s].to_numpy() for s in prices.columns}
        return self._forecaster.forecast(view.asof, windows)

    def close(self) -> None:
        """Release the forecaster's worker pool."""
        if self._forecaster is not None:
            self._forecaster.close()


@register("momentum")
@dataclass
class Momentum:
    """Return over the past year, skipping the most recent month.

    Args:
        lookback: Sessions in the window (about 12 months).
        skip: Most recent sessions left out (about 1 month).
        horizon: Sessions ahead it predicts.
    """

    lookback: int = 252
    skip: int = 21
    horizon: int = 21
    name: str = "momentum"

    def score(self, view: DataView, symbols: Sequence[str]) -> Scores:
        """Skip-month momentum per symbol."""
        return momentum_scores(view, symbols, self.lookback, self.skip)


@register("reversal")
@dataclass
class Reversal:
    """Short-term reversal: minus the return over the past ``lookback`` sessions.

    Args:
        lookback: Sessions in the window.
        horizon: Sessions ahead it predicts (use ``lookback`` for a like-for-like test).
    """

    lookback: int = 21
    horizon: int = 21
    name: str = "reversal"

    def score(self, view: DataView, symbols: Sequence[str]) -> Scores:
        """Negative recent return per symbol."""
        return {s: -r for s, r in momentum_scores(view, symbols, self.lookback, 0).items()}


@register("low_vol")
@dataclass
class LowVol:
    """Low volatility: minus the standard deviation of daily returns.

    Args:
        lookback: Sessions of returns.
        horizon: Sessions ahead it predicts.
    """

    lookback: int = 252
    horizon: int = 21
    name: str = "low_vol"

    def score(self, view: DataView, symbols: Sequence[str]) -> Scores:
        """Negative daily volatility per symbol (needs 80% of the window)."""
        returns = view.returns(self.lookback, symbols)
        enough = returns.notna().sum() >= 0.8 * self.lookback
        vol = returns.loc[:, enough].std()
        return {s: -float(v) for s, v in vol.items() if np.isfinite(v)}


@register("fscore")
@dataclass
class FScore:
    """Piotroski F-score from the latest annual report filed before the decision date.

    Args:
        horizon: Sessions ahead it predicts.
    """

    horizon: int = 21
    name: str = "fscore"

    def score(self, view: DataView, symbols: Sequence[str]) -> Scores:
        """F-score (0-9) per symbol that has one."""
        return {s: float(f) for s, f in view.fscores(symbols).items()}


@register("feature")
@dataclass
class Feature:
    """One column of the monthly feature panel (``research.features``) as a signal.

    Args:
        column: Feature name, e.g. ``earnings_yield``.
        horizon: Sessions ahead it predicts.
    """

    column: str = "earnings_yield"
    horizon: int = 21
    name: str = "feature"

    def score(self, view: DataView, symbols: Sequence[str]) -> Scores:
        """The feature's latest value per symbol (missing values left out)."""
        rows = view.features(symbols, [self.column]).drop_nulls()
        return dict(zip(rows["symbol"], rows[self.column].cast(float), strict=True))
