"""Part 2: boosted trees on what part 1 misses.

XGBoost on the stock inputs (missing kept as missing; the trees learn where it goes) plus
the market inputs (``env_*`` and ``mkt_*``), fitted to part 1's residual on the capped
target. Squared loss, so a leaf's weight is its row count: ``min_child_weight`` 2,000 means
at least 2,000 stock-months per leaf.

On ``cuda`` the inputs can be GPU arrays (CuPy), so the trees bin them on the GPU: each
month's bin edges still come from that month's training rows only.
"""

import gc

import numpy as np
import xgboost

PARAMS = {
    "learning_rate": 0.05, "n_estimators": 400, "grow_policy": "lossguide", "max_leaves": 31,
    "max_depth": 0, "min_child_weight": 2000, "reg_lambda": 1.0, "max_bin": 255,
    "tree_method": "hist", "random_state": 0,
}  # fmt: skip


class Trees:
    """Fitted trees and the input columns they use (ones with any value in training)."""

    def __init__(
        self,
        x: np.ndarray,
        residual: np.ndarray,
        device: str = "cpu",
        threads: int = 6,
        sample: float = 1.0,
        seed: int = 0,
    ):
        """Fit on ``x`` (one row per stock-month, NaN for missing) against ``residual``.

        Args:
            x: Training inputs (NumPy, or CuPy on ``cuda``).
            residual: Part 1's residual on the capped target.
            device: ``cpu`` or ``cuda``.
            threads: CPU threads.
            sample: Share of rows and of columns each tree draws (1 = all).
            seed: Random seed for that draw.
        """
        self.keep = np.flatnonzero(_host(np.isfinite(x).any(axis=0)))
        params = PARAMS | {"subsample": sample, "colsample_bytree": sample, "random_state": seed}
        self.model = xgboost.XGBRegressor(**params, device=device, n_jobs=threads)
        self.model.fit(x[:, self.keep], residual)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The correction for each row of ``x``."""
        return _host(self.model.predict(x[:, self.keep])).astype(np.float64)


class TreesPart:
    """Part 2 as boosted trees, refit from scratch every month (``walk.run``'s ``part2``).

    Args:
        device: ``cpu`` or ``cuda`` (inputs then stay on the GPU and are binned there).
        threads: CPU threads.
        sample: Share of rows and of columns each tree draws (1 = all).
        seed: Random seed for that draw.
        dispersion: Add last month's return dispersion to the market inputs.
        yearly: Refit only when forecasting January (part 1 still refits monthly); walk sets
            ``now`` to the forecast month before each fit.
    """

    def __init__(self, device: str = "cpu", threads: int = 6, sample: float = 1.0, seed: int = 0,
                 dispersion: bool = False, yearly: bool = False, stock_only: bool = False):  # fmt: skip
        self.device, self.dispersion, self.yearly, self.now = device, dispersion, yearly, None
        self.columns = [] if stock_only else None  # [] = no market inputs
        self.settings = {"threads": threads, "sample": sample, "seed": seed}
        self.trees: Trees | None = None
        self._pool = None

    def arrays(self, x: np.ndarray):
        """``x`` where the fits read it (on the GPU for ``cuda``) and a converter for targets."""
        if self.device != "cuda":
            return x, np.asarray
        import cupy  # noqa: PLC0415 - GPU only; not installed on orange

        # CuPy keeps freed blocks in its own pool, out of XGBoost's reach: they are handed
        # back after every month (:meth:`release`) or the GPU fills up within a few years
        self._pool = cupy.get_default_memory_pool()
        return cupy.asarray(x), cupy.asarray

    def fit(self, x, residual, years: np.ndarray) -> None:
        """Fit this month's trees (``years``: each row's calendar year, unused here)."""
        if self.yearly and self.trees is not None and self.now.month != 1:
            return
        self.trees = Trees(x, residual, self.device, **self.settings)

    def predict(self, x) -> np.ndarray:
        """This month's correction for each row of ``x``."""
        return self.trees.predict(x)

    def release(self) -> None:
        """Free this month's trees (and the GPU memory they used); yearly trees are kept."""
        if not self.yearly:
            self.trees = None
        if self._pool is not None:
            gc.collect()
            self._pool.free_all_blocks()


def _host(a) -> np.ndarray:
    """A NumPy array of ``a`` (copied from the GPU if it is a CuPy array)."""
    return a.get() if hasattr(a, "get") else np.asarray(a)
