"""Gradient clipping per step: a fixed limit, or one that follows the model's own steps.

A shock month (March 2020 entering the window) gives gradients 10^7-10^11 times the usual
size, and one such step wrecks the model for months (run33). Clipping caps a step's size (the
gradient's norm over every parameter) and keeps its direction. A fixed limit (Pascanu et al.
2013) needs a number: 1.0 clipped 95% of all steps here (typical size ~17, run34), changing
every step rather than catching the spikes. :class:`Clipper` with ``"auto"`` instead sets
the limit from the model's own healthy steps: once :meth:`arm` is called (the walk does so
after the first month, since the cold start's early steps are a start-up transient with
sizes far above healthy training's, run39), the ``percentile`` of the next ``calibrate``
steps' sizes (not clipped themselves), then frozen. A bar that keeps
following recent sizes (AutoClip, Seetharaman et al. 2020) tracks the symptom: when training
goes bad the gradients grow for months (15 -> 250 through 2022, run37) and the bar followed
them into the thousands while the fit faded; the same bar frozen at its 2019 value held the
fit through 2022 (run38).
"""

import math

import torch


class Clipper:
    """Clip each step's gradient (module docs); call it after ``backward``.

    Args:
        clip: ``None`` (no clipping), a positive limit, or ``"auto"``.
        percentile: With ``"auto"``, the percentile of the calibration steps' sizes.
        calibrate: Steps whose sizes set the limit, counted from :meth:`arm`.
    """

    def __init__(self, clip: float | str | None, percentile: float = 95.0, calibrate: int = 500):
        if clip == "auto":
            self.fixed = None
        elif clip is None:
            self.fixed = math.inf
        elif isinstance(clip, int | float) and clip > 0:
            self.fixed = float(clip)
        else:
            raise ValueError(f"clip must be None, a positive number or 'auto', not {clip!r}")
        self.q, self.calibrate = percentile / 100, calibrate
        self.history: list[float] = []
        self.armed = False

    def arm(self) -> None:
        """Start calibrating (``"auto"`` only; before this nothing is clipped or recorded)."""
        self.armed = True

    def __call__(self, parameters) -> torch.Tensor:
        """Clip the parameters' gradients; returns (size before clipping, limit) as a tensor.

        With a fixed limit nothing is read back from the device; ``"auto"`` reads each size
        while calibrating.
        """
        if self.fixed is not None:
            total = torch.nn.utils.clip_grad_norm_(parameters, self.fixed)
            return torch.stack([total, total.new_tensor(self.fixed)])
        total = torch.nn.utils.clip_grad_norm_(parameters, math.inf)
        if not self.armed:
            return torch.stack([total, total.new_tensor(math.inf)])
        size = float(total)
        if math.isfinite(size):
            self.history.append(size)
        if len(self.history) >= self.calibrate:
            ranked = sorted(self.history)
            self.fixed = ranked[min(len(ranked) - 1, math.ceil(self.q * len(ranked)) - 1)]
        return torch.stack([total, total.new_tensor(math.inf)])
