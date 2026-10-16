"""Parameter averaging: trade a running average of the model, not its latest step (2026-10-15).

Each training step jolts the model, and one month's fit can sit on a noisy spot (held-out
loss curves swing from pass to pass, grid9). An exponential moving average of the parameters
(Polyak averaging; Izmailov et al. 2018 for the deep-learning case) smooths that out: after
every step the average moves a share ``1 - decay`` of the way toward the latest parameters,
so it spans roughly the last 1 / (1 - decay) steps (``decay`` 0.998: ~500 steps, about one
month's training). Training itself is unchanged; the average is what the walk holds, and
early stopping rolls it back with the model. The average starts at zero and is divided by
1 - decay^n (Adam's bias correction), so the first month's random start is not in it.
"""

from contextlib import contextmanager

import torch


class Average:
    """The moving average of ``model``'s parameters (module docs).

    Args:
        model: The model being trained; its parameters are read after every step.
        decay: Share of the average kept at each step (0.998: ~500 steps).
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0 < decay < 1:
            raise ValueError(f"decay must be between 0 and 1, not {decay!r}")
        self.decay = decay
        self.params = list(model.parameters())
        self.sums = [torch.zeros_like(p) for p in self.params]
        self.steps = 0

    def update(self) -> None:
        """Fold the parameters' current values in (call after each optimizer step)."""
        torch._foreach_mul_(self.sums, self.decay)
        torch._foreach_add_(self.sums, [p.detach() for p in self.params], alpha=1 - self.decay)
        self.steps += 1

    def values(self) -> list[torch.Tensor]:
        """The averaged parameters, bias-corrected."""
        scale = 1 / (1 - self.decay**self.steps) if self.steps else 1.0
        return [s * scale for s in self.sums]

    def state(self) -> tuple[list[torch.Tensor], int]:
        """A snapshot, for rolling back."""
        return [s.clone() for s in self.sums], self.steps

    def load(self, state: tuple[list[torch.Tensor], int]) -> None:
        """Restore a snapshot from :meth:`state`."""
        for s, saved in zip(self.sums, state[0], strict=True):
            s.copy_(saved)
        self.steps = state[1]

    def install(self) -> None:
        """Put the averaged parameters in place for good (training is over)."""
        with torch.no_grad():
            for p, a in zip(self.params, self.values(), strict=True):
                p.copy_(a)

    @contextmanager
    def swapped(self):
        """Run the model with the averaged parameters in place, then put its own back."""
        saved = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for p, a in zip(self.params, self.values(), strict=True):
                p.copy_(a)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, s in zip(self.params, saved, strict=True):
                    p.copy_(s)
