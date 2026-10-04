"""Part 2 as a stack (experiment): linear + nets as the main model, trees on what it misses.

Each month: the nets (``conditioned.CondNetsPart``, all market inputs) are trained on the
target itself; the main model is the linear part and the nets, each centered within the
month, averaged 50/50; the trees (``trees.Trees``) are fitted to the main model's residual
over all earlier months. ``walk.run`` reports the main model as ``linear`` and the trees as
``correction``, so ``walk.combine`` chooses the trees' strength walk-forward as usual.
"""

import gc

import numpy as np

from portfolio_lab.research.forecaster.conditioned import CondNetsPart
from portfolio_lab.research.forecaster.trees import Trees


class StackPart:
    """Linear + nets, then trees on the residual (module docs)."""

    columns = None  # every market input
    dispersion = True

    def __init__(self, device: str = "cpu", threads: int = 6, market_penalty: float = 1.0,
                 seed: int = 0):  # fmt: skip
        self.device, self.threads = device, threads
        self.nets = CondNetsPart(device, arch="bilinear", exposures=8,
                                 market_penalty=market_penalty, seed=seed, factors=-1)  # fmt: skip
        self.row_months: np.ndarray | None = None
        self.linear_rows: np.ndarray | None = None  # set by walk: part 1 on the fit rows
        self.linear_here: np.ndarray | None = None  # and on the month forecast
        self.base_here: np.ndarray | None = None
        self.trees: Trees | None = None
        self.held_err = None

    def arrays(self, x: np.ndarray):
        return x, np.asarray

    def _center(self, f: np.ndarray, month: np.ndarray) -> np.ndarray:
        sums = np.bincount(month, weights=f)
        counts = np.bincount(month)
        return f - (sums / np.maximum(counts, 1))[month]

    def _nets_rows(self, x: np.ndarray, month: np.ndarray) -> np.ndarray:
        """The nets' forecast for every fit row, centered within each month."""
        cuts = np.flatnonzero(np.diff(month)) + 1
        return np.concatenate([self.nets.predict(part) for part in np.split(x, cuts)])

    def fit(self, x: np.ndarray, residual: np.ndarray, years: np.ndarray) -> None:
        month = self.row_months[: len(x)]
        self.nets.row_months = self.row_months
        y = np.asarray(residual) + self.linear_rows
        self.nets.fit(x, y, years)
        self.held_err = self.nets.held_err
        base = 0.5 * self._center(self.linear_rows, month) + 0.5 * self._nets_rows(x, month)
        rest = y - base
        if self.device == "cuda":
            import cupy  # noqa: PLC0415 - GPU only

            self.trees = Trees(cupy.asarray(x), cupy.asarray(rest), self.device, self.threads)
        else:
            self.trees = Trees(x, rest, self.device, self.threads)

    def predict(self, x: np.ndarray) -> np.ndarray:
        lin = self.linear_here - self.linear_here.mean()
        self.base_here = 0.5 * lin + 0.5 * self.nets.predict(x)
        if self.device == "cuda":
            import cupy  # noqa: PLC0415 - GPU only

            return self.trees.predict(cupy.asarray(x))
        return self.trees.predict(x)

    def release(self) -> None:
        self.trees = None
        self.nets.release()
        if self.device == "cuda":
            import cupy  # noqa: PLC0415 - GPU only

            gc.collect()
            cupy.get_default_memory_pool().free_all_blocks()
