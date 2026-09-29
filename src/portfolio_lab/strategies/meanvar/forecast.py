"""Expected-return forecasts for mean-variance optimization.

Every model returns an **annualized** expected return from a window of daily adjusted
prices, so optimizer inputs are in consistent units (the legacy optimizer mixed quarterly
forecasts with an annual risk-free rate):

- ``arima320_price``: ARIMA(3,2,0) on price levels, the legacy model (kept for comparison;
  it extrapolates recent trend).
- ``arima_logret``: ARIMA(1,0,1) on daily log returns, summed over the horizon.
- ``historical_mean``: trailing geometric mean return, the textbook input with no forecast.

Fits run in a process pool and are cached per (symbol, decision date), so re-running a
backtest or refreshing it weekly only fits new dates. Failed fits are cached as NaN so
they are not retried.
"""

import logging
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from portfolio_lab.core.store import upsert_parquet

log = logging.getLogger(__name__)

MODELS = ("arima320_price", "arima_logret", "historical_mean")
TRADING_DAYS = 252
#: Forecasts are clipped to this annual range, keeping extreme extrapolations from
#: destabilizing the optimizer (the per-name weight cap limits their influence anyway).
MU_BOUNDS = (-0.99, 5.0)
#: Bump when model code changes, so cached forecasts from old code are not reused.
MODEL_VERSION = 1
#: Thread-count variables for numpy's math backends, pinned to 1 in worker processes.
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class ForecastSpec:
    """Which model to fit, how far ahead to forecast, and on how much history.

    Args:
        model: One of :data:`MODELS`.
        horizon: Forecast horizon in sessions.
        lookback: Sessions of history fitted.
    """

    model: str = "arima_logret"
    horizon: int = 21
    lookback: int = TRADING_DAYS

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown forecast model {self.model!r}; choose from {MODELS}")

    @property
    def cache_id(self) -> str:
        """Identifier of this model configuration, used to partition the cache."""
        return f"{self.model}-h{self.horizon}-l{self.lookback}-v{MODEL_VERSION}"


def forecast_one(prices: np.ndarray, spec: ForecastSpec) -> float:
    """Annualized expected return for one symbol, or NaN if the model fails.

    Args:
        prices: Adjusted price levels, oldest first, no missing values.
        spec: The model configuration.
    """
    from statsmodels.tsa.arima.model import ARIMA  # noqa: PLC0415 - heavy; loaded in workers

    if len(prices) < 30 or not np.all(prices > 0):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if spec.model == "historical_mean":
                annual = (prices[-1] / prices[0]) ** (TRADING_DAYS / (len(prices) - 1)) - 1
            elif spec.model == "arima320_price":
                fit = ARIMA(prices, order=(3, 2, 0)).fit()
                if not fit.mle_retvals.get("converged", True):
                    return np.nan
                expected = fit.forecast(steps=spec.horizon)[-1]
                annual = (expected / prices[-1]) ** (TRADING_DAYS / spec.horizon) - 1
            else:  # arima_logret
                fit = ARIMA(np.diff(np.log(prices)), order=(1, 0, 1)).fit()
                if not fit.mle_retvals.get("converged", True):
                    return np.nan
                log_return = fit.forecast(steps=spec.horizon).sum()
                annual = np.exp(log_return * TRADING_DAYS / spec.horizon) - 1
        except (ValueError, np.linalg.LinAlgError):
            return np.nan
    return float(np.clip(annual, *MU_BOUNDS)) if np.isfinite(annual) else np.nan


def _forecast_task(args: tuple[str, np.ndarray, ForecastSpec]) -> tuple[str, float]:
    """Pool worker: forecast one symbol."""
    symbol, prices, spec = args
    return symbol, forecast_one(prices, spec)


class Forecaster:
    """Forecasts expected returns for many symbols, with a pool and an on-disk cache.

    Args:
        spec: The model configuration.
        cache_dir: Root of the forecast cache (``<data>/cache/forecasts``); ``None``
            disables on-disk caching.
        workers: Worker processes; 1 runs inline (used by tests).
    """

    def __init__(self, spec: ForecastSpec, cache_dir: Path | None = None, workers: int = 1):
        self.spec = spec
        self.workers = workers
        self._path = cache_dir / f"model={spec.cache_id}" / "data.parquet" if cache_dir else None
        self._cache: dict[tuple[str, date], float] = {}
        if self._path and self._path.exists():
            cached = pl.read_parquet(self._path)
            self._cache = {(s, d): m for s, d, m in cached.iter_rows()}
        self._pool: ProcessPoolExecutor | None = None

    def _executor(self) -> ProcessPoolExecutor:
        """Start the worker pool on first use (spawned, so workers don't copy the panel).

        Each worker is limited to one BLAS/OpenMP thread: otherwise every process starts a
        thread per core and the pool oversubscribes the CPU (measured ~9x slower on orange).
        The variables must be in the environment when workers start, which is lazily on the
        first task, so they are left set. That doesn't affect this process, whose numpy is
        already initialized.
        """
        if self._pool is None:
            for var in _THREAD_ENV_VARS:
                os.environ[var] = "1"
            self._pool = ProcessPoolExecutor(
                self.workers, mp_context=multiprocessing.get_context("spawn")
            )
        return self._pool

    def forecast(self, asof: date, windows: dict[str, np.ndarray]) -> dict[str, float]:
        """Annualized expected returns for each symbol's price window at ``asof``.

        Symbols whose model fails are omitted.
        """
        todo = [(s, p, self.spec) for s, p in windows.items() if (s, asof) not in self._cache]
        if todo:
            if self.workers > 1 and len(todo) > 1:
                results = list(self._executor().map(_forecast_task, todo, chunksize=4))
            else:
                results = [_forecast_task(t) for t in todo]
            for symbol, mu in results:
                self._cache[(symbol, asof)] = mu
            self._persist([(s, asof, mu) for s, mu in results])
        out = {s: self._cache[(s, asof)] for s in windows}
        return {s: mu for s, mu in out.items() if np.isfinite(mu)}

    def _persist(self, rows: list[tuple[str, date, float]]) -> None:
        """Append new forecasts to the on-disk cache."""
        if self._path is None or not rows:
            return
        frame = pl.DataFrame(rows, schema=["symbol", "asof", "mu"], orient="row")
        upsert_parquet(frame, self._path, ["symbol", "asof"])

    def close(self) -> None:
        """Shut down the worker pool, if one was started."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
