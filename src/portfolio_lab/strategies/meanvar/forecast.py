"""Expected-return forecasts for mean-variance optimization.

Every model returns an **annualized** expected return from a window of daily adjusted
prices, so optimizer inputs are in consistent units (the legacy optimizer mixed quarterly
forecasts with an annual risk-free rate):

- ``arima320_price``: ARIMA(3,2,0) on price levels, the legacy model (kept for comparison;
  it extrapolates recent trend). It is an AR(3) on second differences with no constant, so
  it is fitted by least squares; this matches statsmodels' maximum-likelihood fit (checked
  to 3 decimals on the coefficients) without its occasional convergence failures.
- ``ar1_logret``: AR(1) on daily log returns, fitted by least squares (closed form, so it
  can't fail to converge and gives the same answer on every CPU), forecast analytically
  over the horizon. It replaced ARIMA(1,0,1) fitted by maximum likelihood, whose optimizer
  failed on ~5% of real windows and converged differently on x86 and arm64.
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

MODELS = ("arima320_price", "ar1_logret", "historical_mean")
TRADING_DAYS = 252
#: Forecasts are clipped to this annual range, keeping extreme extrapolations from
#: destabilizing the optimizer (the per-name weight cap limits their influence anyway).
MU_BOUNDS = (-0.99, 5.0)
#: Bump when model code changes, so cached forecasts from old code are not reused.
#: v2: ARIMA(3,2,0) fitted by least squares instead of statsmodels' MLE.
MODEL_VERSION = 2
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

    model: str = "ar1_logret"
    horizon: int = 21
    lookback: int = TRADING_DAYS

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown forecast model {self.model!r}; choose from {MODELS}")

    @property
    def cache_id(self) -> str:
        """Identifier of this model configuration, used to partition the cache."""
        return f"{self.model}-h{self.horizon}-l{self.lookback}-v{MODEL_VERSION}"


def ar1_forecast_sum(returns: np.ndarray, horizon: int) -> float:
    """Sum of the next ``horizon`` returns forecast by an AR(1) fitted with least squares.

    Fits ``r[t] = c + phi * r[t-1]``; with long-run mean ``mu = c / (1 - phi)`` the k-step
    forecast is ``mu + phi**k * (r[-1] - mu)``, so the sum over ``k = 1..horizon`` is
    ``horizon * mu + (r[-1] - mu) * phi * (1 - phi**horizon) / (1 - phi)``. ``phi`` is
    clipped to keep the process stationary.
    """
    phi, c = np.polyfit(returns[:-1], returns[1:], 1)
    phi = float(np.clip(phi, -0.99, 0.99))
    mu = c / (1 - phi)
    return float(horizon * mu + (returns[-1] - mu) * phi * (1 - phi**horizon) / (1 - phi))


def ar_diff_forecast(prices: np.ndarray, horizon: int, order: int = 3, d: int = 2) -> float:
    """Price level ``horizon`` steps ahead from an ARIMA(order, d, 0) fitted by least squares.

    Differences the prices ``d`` times, regresses each difference on its ``order`` lags (no
    constant, as in statsmodels' ARIMA with ``d > 0``), iterates the fitted recursion
    forward, then integrates back to a price level. Only ``d = 2`` integration is needed
    here (the legacy model).
    """
    if d != 2:
        raise ValueError("only second differences (d=2) are supported")
    series = np.diff(prices, n=d)
    n = len(series)
    lags = np.column_stack([series[order - 1 - k : n - 1 - k] for k in range(order)])
    coef, *_ = np.linalg.lstsq(lags, series[order:], rcond=None)
    history = list(series[-order:])
    for _ in range(horizon):
        history.append(float(np.dot(coef, history[-1 : -order - 1 : -1])))
    diff1, level = prices[-1] - prices[-2], prices[-1]
    for second_diff in history[order:]:
        diff1 += second_diff
        level += diff1
    return float(level)


def forecast_one(prices: np.ndarray, spec: ForecastSpec) -> float:
    """Annualized expected return for one symbol, or NaN if the model fails.

    Args:
        prices: Adjusted price levels, oldest first, no missing values.
        spec: The model configuration.
    """
    if len(prices) < 30 or not np.all(prices > 0):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if spec.model == "historical_mean":
                annual = (prices[-1] / prices[0]) ** (TRADING_DAYS / (len(prices) - 1)) - 1
            elif spec.model == "arima320_price":
                expected = ar_diff_forecast(prices, spec.horizon)
                if expected <= 0:
                    return float(MU_BOUNDS[0])  # extrapolated through zero: maximally bearish
                annual = (expected / prices[-1]) ** (TRADING_DAYS / spec.horizon) - 1
            else:  # ar1_logret
                log_return = ar1_forecast_sum(np.diff(np.log(prices)), spec.horizon)
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
