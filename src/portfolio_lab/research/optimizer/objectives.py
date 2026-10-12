"""Long-only objectives graded on a batch of consecutive months (2026-10-12 screen).

The paper's loss, (1 - R)^2 one month at a time, balances return against risk only when the
portfolio's size is free; with long-only weights summing to 1 it rewards raw return and the
weights collapse onto a few stocks. Each objective here grades the monthly returns ``R`` (one
per month of the batch, in date order) of the portfolios the model picked; training minimizes
it. ``gamma`` is the risk appetite where there is one; the defaults make the risk term about
as large as the return term for a diversified small-stock portfolio (monthly mean ~1%,
volatility ~6%, worst month ~-8%, worst 12-month drop ~10%).

- ``sharpe``: -mean(R) / std(R).
- ``meanvar``: -mean(R) + gamma / 2 * var(R).
- ``log``: -mean(log(1 + R)), compounded growth.
- ``downside``: -mean(R) + gamma / 2 * mean(min(R, 0)^2), only losing months count as risk.
- ``cvar``: -mean(R) + gamma * (average loss over the worst 10% of months, at least one).
- ``drawdown``: -mean(R) + gamma * the batch's deepest peak-to-trough drop.
"""

import math

import torch

#: Default gamma per objective (module docs); objectives without one are absent.
GAMMA = {"meanvar": 5.0, "downside": 10.0, "cvar": 0.1, "drawdown": 0.1}
OBJECTIVES = ("sharpe", "meanvar", "log", "downside", "cvar", "drawdown")


def grade(objective: str, r: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
    """The loss of monthly portfolio returns ``r`` (date order) under ``objective``."""
    g = GAMMA.get(objective, 0.0) if gamma is None else gamma
    mean = r.mean()
    if objective == "sharpe":
        return -mean / (r.std() + 1e-6)
    if objective == "meanvar":
        return -mean + g / 2 * r.var()
    if objective == "log":
        return -torch.log1p(r.clamp(min=-0.99)).mean()
    if objective == "downside":
        return -mean + g / 2 * (r.clamp(max=0) ** 2).mean()
    if objective == "cvar":
        worst = torch.topk(-r, max(1, math.ceil(0.1 * len(r)))).values
        return -mean + g * worst.mean()
    if objective == "drawdown":
        wealth = torch.cumprod(1 + r, dim=0)
        peak = torch.cummax(wealth, dim=0).values
        return -mean + g * (1 - wealth / peak).max()
    raise ValueError(f"unknown objective {objective!r}; known: {OBJECTIVES}")
