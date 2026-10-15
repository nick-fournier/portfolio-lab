"""Gradient clipping per step: a fixed limit, or one that follows the model's own steps.

A shock month (March 2020 entering the window) gives gradients 10^7-10^11 times the usual
size, and one such step wrecks the model for months (run33). Clipping caps a step's size (the
gradient's norm over every parameter) and keeps its direction. A fixed limit (Pascanu et al.
2013) needs a number: 1.0 clipped 95% of all steps here (typical size ~17, run34), changing
every step rather than catching the spikes. :class:`Clipper` with ``"auto"`` instead clips at
a percentile of the recent steps' sizes (AutoClip, Seetharaman et al. 2020): the 95th of the
last ``window`` steps leaves normal training alone and cuts the spikes. The history stores the
sizes after clipping, so a spike cannot raise its own bar, and the first ``warmup`` steps are
not clipped (nothing to compare against yet).
"""

import math
from collections import deque

import torch


class Clipper:
    """Clip each step's gradient (module docs); call it after ``backward``.

    Args:
        clip: ``None`` (no clipping), a positive limit, or ``"auto"``.
        percentile: With ``"auto"``, the percentile of recent sizes used as the limit.
        window: Recent steps the percentile is taken over.
        warmup: Steps before ``"auto"`` starts clipping.
    """

    def __init__(
        self, clip: float | str | None, percentile: float = 95.0, window: int = 500,
        warmup: int = 20,
    ):  # fmt: skip
        if clip == "auto":
            self.fixed = None
        elif clip is None:
            self.fixed = math.inf
        elif isinstance(clip, int | float) and clip > 0:
            self.fixed = float(clip)
        else:
            raise ValueError(f"clip must be None, a positive number or 'auto', not {clip!r}")
        self.q, self.warmup = percentile / 100, warmup
        self.history: deque[float] = deque(maxlen=window)

    def __call__(self, parameters) -> torch.Tensor:
        """Clip the parameters' gradients; returns (size before clipping, limit) as a tensor.

        With a fixed limit nothing is read back from the device; ``"auto"`` reads each size.
        """
        if self.fixed is not None:
            total = torch.nn.utils.clip_grad_norm_(parameters, self.fixed)
            return torch.stack([total, total.new_tensor(self.fixed)])
        limit = self._limit()
        total = torch.nn.utils.clip_grad_norm_(parameters, limit)
        size = float(total)
        if math.isfinite(size):
            self.history.append(min(size, limit))
        return torch.stack([total, total.new_tensor(limit)])

    def _limit(self) -> float:
        if len(self.history) < self.warmup:
            return math.inf
        ranked = sorted(self.history)
        return ranked[min(len(ranked) - 1, math.ceil(self.q * len(ranked)) - 1)]
