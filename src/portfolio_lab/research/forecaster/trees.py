"""Part 2: boosted trees on what part 1 misses.

XGBoost on the stock inputs (missing kept as missing; the trees learn where it goes) plus
the market inputs (``env_*`` and ``mkt_*``), fitted to part 1's residual on the capped
target. Squared loss, so a leaf's weight is its row count: ``min_child_weight`` 2,000 means
at least 2,000 stock-months per leaf.

On ``cuda`` the inputs can be GPU arrays (CuPy), so the trees bin them on the GPU: each
month's bin edges still come from that month's training rows only.
"""

import numpy as np
import xgboost

PARAMS = {
    "learning_rate": 0.05, "n_estimators": 400, "grow_policy": "lossguide", "max_leaves": 31,
    "max_depth": 0, "min_child_weight": 2000, "reg_lambda": 1.0, "max_bin": 255,
    "tree_method": "hist", "random_state": 0,
}  # fmt: skip


class Trees:
    """Fitted trees and the input columns they use (ones with any value in training)."""

    def __init__(self, x: np.ndarray, residual: np.ndarray, device: str = "cpu", threads: int = 6):
        """Fit on ``x`` (one row per stock-month, NaN for missing) against ``residual``.

        Args:
            x: Training inputs (NumPy, or CuPy on ``cuda``).
            residual: Part 1's residual on the capped target.
            device: ``cpu`` or ``cuda``.
            threads: CPU threads.
        """
        self.keep = np.flatnonzero(_host(np.isfinite(x).any(axis=0)))
        self.model = xgboost.XGBRegressor(**PARAMS, device=device, n_jobs=threads)
        self.model.fit(x[:, self.keep], residual)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The correction for each row of ``x``."""
        return _host(self.model.predict(x[:, self.keep])).astype(np.float64)


def _host(a) -> np.ndarray:
    """A NumPy array of ``a`` (copied from the GPU if it is a CuPy array)."""
    return a.get() if hasattr(a, "get") else np.asarray(a)
