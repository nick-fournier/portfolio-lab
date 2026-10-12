"""The paper's linear benchmark (BSV): weights linear in the characteristics, ridge in closed form.

Weights w_t = X_t lam, so the portfolio's return is F_{t+1} . lam with F_{t+1} = X_t' R_{t+1}
(one long-short "factor" return per characteristic). Minimizing the paper's loss
sum (1 - F_{t+1} . lam)^2 + z |lam|^2 over the training months is a ridge regression of 1 on
the factor returns (paper eqs. 11-12):

    lam = (sum F F' + z I)^-1 sum F

over the paper's grid z = 10^i, i = -10..3 (sec. 4.3). The penalty is picked two ways:

- ``best``: the one whose out-of-sample Sharpe is highest over the whole test period. This is
  how the paper picks the BSV penalty ("using the full sample", sec. 4.3); it looks ahead.
- ``loo``: each month, the one with the smallest leave-one-out error over that month's
  training months (how the paper picks the linear-attention penalty); no look-ahead.
"""

import numpy as np

from portfolio_lab.research.optimizer.data import Month

#: The paper's penalty grid.
GRID = tuple(10.0**i for i in range(-10, 4))


def _sharpe(r: np.ndarray) -> float:
    return float(r.mean() / r.std() * 12**0.5)


def walk(months: list[Month], window: int = 60, start_index: int | None = None) -> dict:
    """Out-of-sample monthly returns of the ridge benchmark for every penalty (module docs).

    Args:
        months: From :func:`.data.months`, ascending.
        window: Training months before each test month.
        start_index: First test month's position (default: ``window``).

    Returns:
        ``dates``; ``returns`` (test months x penalties); ``loo`` (the walk-forward pick's
        returns); ``loo_z`` (its penalty each month); ``best_z`` and ``sharpe`` per penalty.
    """
    factors = [m.x.T.astype(np.float64) @ np.nan_to_num(m.r.astype(np.float64)) for m in months]
    known = [bool(np.isfinite(m.r).all()) for m in months]
    first = window if start_index is None else max(start_index, window)
    dates, rets, loo, loo_z = [], [], [], []
    for t in range(first, len(months)):
        if not known[t]:
            continue
        a = np.array([factors[k] for k in range(t - window, t) if known[k]])  # months x D
        gram, ones = a.T @ a, a.sum(axis=0)
        f_next = months[t].x.T.astype(np.float64) @ months[t].r.astype(np.float64)
        row, errors = [], []
        for z in GRID:
            inv = np.linalg.inv(gram + z * np.eye(len(gram)))
            lam = inv @ ones
            row.append(float(f_next @ lam))
            hat = np.einsum("ij,jk,ik->i", a, inv, a)  # leave-one-out shortcut
            errors.append(float((((1 - a @ lam) / (1 - hat)) ** 2).mean()))
        k = int(np.argmin(errors))
        dates.append(months[t].date)
        rets.append(row)
        loo.append(row[k])
        loo_z.append(GRID[k])
    returns = np.array(rets)
    sharpe = [_sharpe(returns[:, j]) for j in range(len(GRID))]
    return {"dates": dates, "returns": returns, "loo": np.array(loo), "loo_z": loo_z,
            "best_z": GRID[int(np.argmax(sharpe))], "sharpe": sharpe}  # fmt: skip
