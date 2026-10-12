"""Walk-forward training: at each month end, train on earlier months, then pick the portfolio.

At month end t, the months whose next-month returns are already known are the ones before
t. The training months are the ``window`` months before t (all earlier months if ``window``
is ``None``). The first test month trains a fresh model for ``first_epochs`` passes over its
training months; every later month continues from the previous month's model (and Adam's
state) for ``warm_epochs`` passes over its own training months. Each pass visits the training
months in a shuffled order and takes one Adam step per month (paper, Algorithm 2). Then the
model's weights for month t's candidates are the portfolio held over the next month.
"""

import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
import torch

from portfolio_lab.research.optimizer.data import Month
from portfolio_lab.research.optimizer.model import Optimizer, loss


@dataclass(frozen=True)
class Settings:
    """One walk's settings (each one agreed before a run).

    Args:
        model: ``linear``, ``mlp`` or ``transformer`` (model.MODELS).
        blocks: Blocks for ``mlp`` / ``transformer``.
        hidden: Feed-forward hidden width.
        window: Training months before each test month (``None``: all earlier months).
        first_epochs: Passes over the training months for the first test month.
        warm_epochs: Passes for every later test month, continuing from the last model.
        lr: Adam's learning rate.
        seed: Random seed (initial weights and month order).
        start: First test month end.
    """

    model: str
    blocks: int = 1
    hidden: int = 256
    window: int | None = 60
    first_epochs: int = 100
    warm_epochs: int = 5
    lr: float = 1e-3
    seed: int = 0
    start: date = date(2003, 12, 31)


def _tensors(m: Month, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(m.x).to(device), torch.from_numpy(m.r).to(device)


def walk(
    months: list[Month], settings: Settings, device: str = "cpu"
) -> tuple[pl.DataFrame, pl.DataFrame, Optimizer]:
    """Train and pick portfolios month by month (module docs).

    Args:
        months: From :func:`.data.months`, ascending.
        settings: The walk's settings.
        device: ``cuda`` or ``cpu``.

    Returns:
        ``weights`` (date, symbol, weight) for every test month; ``log`` (date, ret = the
        portfolio's next-month return over the T-bill before costs, effective number of
        holdings = 1 / sum of squared weights, largest weight, mean training loss over the
        last pass, seconds); and the final model.
    """
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    model = Optimizer(settings.model, months[0].x.shape[1], settings.blocks, settings.hidden)
    model.to(device)
    adam = torch.optim.Adam(model.parameters(), lr=settings.lr)
    data = [_tensors(m, device) for m in months]
    known = [bool(np.isfinite(m.r).all()) for m in months]
    weights, log, trained = [], [], False
    for t, month in enumerate(months):
        lo = 0 if settings.window is None else t - settings.window
        if month.date < settings.start or lo < 0:
            continue
        clock = time.perf_counter()
        train = [k for k in range(lo, t) if known[k]]
        losses = []
        for _ in range(settings.warm_epochs if trained else settings.first_epochs):
            losses = []
            for k in rng.permutation(train):
                adam.zero_grad()
                value = loss(model, *data[k])
                value.backward()
                adam.step()
                losses.append(value.item())
        trained = True
        with torch.no_grad():
            w = model.weights(data[t][0]).cpu().numpy().astype(np.float64)
        ret = float(w @ month.r) if known[t] else None
        weights.append(pl.DataFrame({"date": [month.date] * len(w), "symbol": month.symbols,
                                     "weight": w}))  # fmt: skip
        log.append({"date": month.date, "ret": ret, "holdings": float(1 / (w**2).sum()),
                    "largest": float(w.max()), "loss": float(np.mean(losses)),
                    "seconds": time.perf_counter() - clock})  # fmt: skip
    return pl.concat(weights), pl.DataFrame(log), model
