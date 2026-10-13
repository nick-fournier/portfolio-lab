"""Walk-forward training: at each month end, train on earlier months, then pick the portfolio.

At month end t, the months whose next-month returns are already known are the ones before
t. The training months are the ``window`` months before t (all earlier months if ``window``
is ``None``). The first test month trains a fresh model for ``first_epochs`` passes over its
training months; every later month continues from the previous month's model (and Adam's
state) for ``warm_epochs`` passes over its own training months. Each pass visits the training
months in a shuffled order and takes one Adam step per month (paper, Algorithm 2). Then the
model's weights for month t's candidates are the portfolio held over the next month.

With an ``objective`` (``objectives``) a step grades a batch of ``batch`` consecutive training
months instead of one: each pass splits the training months into such blocks from a random
offset and visits them in shuffled order, so every month still goes through the model once per
pass.

With ``subsets`` (stocks the portfolio may hold, by month end) each test month also gets a
portfolio picked from the subset alone, its rows fed to the model without the other stocks
(their ranks across all candidates kept), so attention sees only the subset.
``subset_train`` sets how the model that picks it was trained: ``None``, on every candidate;
``"only"``, on the subset's rows alone in every training month; ``"finetune"``, on every
candidate, then a copy given ``finetune_epochs`` more passes over the subset's rows (fresh
Adam, its own shuffle, so the main model is unchanged).
"""

import copy
import time
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

import numpy as np
import polars as pl
import torch

from portfolio_lab.research.optimizer import costs, objectives
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
        long_only: Softmax weights instead of the paper's unconstrained ones (model docs).
        fast_attention: bfloat16 fused attention (model.Block).
        scale: Long-only with a learned overall size m (model.Optimizer); the saved weights are
            the mix (summing to 1), and m is logged.
        clip: Largest total gradient size per step (``None``: no clipping). Not in the paper;
            added after deeper models diverged (2026-10-11).
        cap: Long-only's largest weight in any stock (``None``: no cap; model.capped_softmax).
        names: Long-only's effective names held (model.spread_softmax).
        objective: Grade batches of months with this (``objectives``; ``None``: the paper's
            loss, one month per step).
        batch: Consecutive months per step with an ``objective``.
        gamma: The objective's risk appetite (``None``: ``objectives.GAMMA``).
        subset_train: ``None``, ``"only"`` or ``"finetune"`` (module docs; needs subsets).
        finetune_epochs: Passes over the subset's rows with ``"finetune"``.
        spread: Cost per dollar traded in the loss (``costs``; 0: none). Paper loss only.
        short_fee: Yearly cost per dollar shorted in the loss (``costs``; 0: none).
        costs: A per-stock cost table (``costs.align``) replacing both flat rates; the walk
            gets it aligned as ``stock_costs``. Recorded here as its path.
        softplus: Long-only weights = softplus(score) / N, size free (model.Optimizer); saved as
            the mix like ``scale``, the total logged as m.
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
    long_only: bool = False
    fast_attention: bool = False
    clip: float | None = 1.0
    scale: bool = False
    cap: float | None = None
    names: float | None = None
    objective: str | None = None
    batch: int = 12
    gamma: float | None = None
    softplus: bool = False
    subset_train: str | None = None
    finetune_epochs: int = 5
    spread: float = 0.0
    short_fee: float = 0.0
    costs: str | None = None


def _tensors(m: Month, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(m.x).to(device), torch.from_numpy(m.r).to(device)


def _steps(train: list[int], settings: Settings, rng: np.random.Generator) -> list[list[int]]:
    """One pass's training steps: single months (paper) or blocks of consecutive months."""
    if settings.objective is None:
        return [[int(k)] for k in rng.permutation(train)]
    offset = int(rng.integers(settings.batch)) if len(train) > settings.batch else 0
    cuts = [0, *range(offset or settings.batch, len(train), settings.batch), len(train)]
    blocks = [train[a:b] for a, b in pairwise(cuts) if b - a >= 3]
    return [blocks[i] for i in rng.permutation(len(blocks))]


Pair = tuple[torch.Tensor, torch.Tensor]


def _train(  # noqa: PLR0913, PLR0917 - one training run's pieces
    model: Optimizer,
    adam: torch.optim.Optimizer,
    source: list[Pair | None],
    train: list[int],
    epochs: int,
    settings: Settings,
    rng: np.random.Generator,
    links: list[costs.Link | None] | None = None,
    stock: list[Pair] | None = None,
) -> list[torch.Tensor]:
    """``epochs`` passes over the ``train`` months of ``source``; the last pass's losses.

    With ``links`` (``costs.links`` of the months) the loss uses the return net of costs, the
    weights the month before computed by the same model without a gradient; with ``stock``
    (each month's per-stock trade cost and borrow fee) each stock pays its own rates.
    """
    losses = []
    for _ in range(epochs):
        losses = []  # kept on the device: reading each one would wait for the GPU
        for step in _steps(train, settings, rng):
            adam.zero_grad()
            if links is not None:
                k = step[0]
                with torch.no_grad():
                    before = model.weights(source[k - 1][0]) if k > 0 else None
                spread, fee, spread_before = settings.spread, settings.short_fee, None
                if stock is not None:
                    (spread, fee), spread_before = stock[k], stock[k - 1][0] if k > 0 else None
                net = costs.net_return(model.weights(source[k][0]), source[k][1], before,
                                       links[k], spread, fee, spread_before)  # fmt: skip
                value = (1 - net) ** 2
            elif settings.objective is None:
                value = loss(model, *source[step[0]])
            else:
                r = torch.stack([model.weights(source[k][0]) @ source[k][1] for k in step])
                value = objectives.grade(settings.objective, r, settings.gamma)
            value.backward()
            if settings.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.clip)
            adam.step()
            losses.append(value.detach())
    return losses


def _subset_rows(
    months: list[Month], data: list[Pair], subsets: dict[date, set[str]] | None
) -> tuple[list[list[int]], list[Pair | None]]:
    """Each month's rows in its subset, and those rows' tensors (``None`` if none)."""
    rows: list[list[int]] = [[] for _ in months]
    for k, m in enumerate(months):
        members = (subsets or {}).get(m.date, set())
        rows[k] = [i for i, symbol in enumerate(m.symbols) if symbol in members]
    return rows, [(x[r], y[r]) if r else None for (x, y), r in zip(data, rows, strict=True)]


def _pricing(
    months: list[Month],
    settings: Settings,
    device: str,
    subsets: dict | None,
    stock_costs: list[tuple[np.ndarray, np.ndarray]] | None,
) -> tuple[list[costs.Link | None] | None, list[Pair] | None]:
    """The loss's cost inputs: ``costs.links`` and per-stock cost tensors (or ``None``).

    Links are ``None`` without costs; the tensors are ``None`` without ``settings.costs``.
    Checks the settings agree.
    """
    if settings.subset_train and subsets is None:
        raise ValueError("subset_train needs subsets")
    if settings.spread <= 0 and settings.short_fee <= 0 and settings.costs is None:
        return None, None
    if settings.objective or settings.subset_train:
        raise ValueError("costs need the paper loss on every candidate")
    if settings.costs is None:
        return costs.links(months, device), None
    if stock_costs is None:
        raise ValueError("settings.costs needs stock_costs")
    tensors = [(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device))
               for a, b in stock_costs]  # fmt: skip
    return costs.links(months, device), tensors


def walk(
    months: list[Month],
    settings: Settings,
    device: str = "cpu",
    subsets: dict[date, set[str]] | None = None,
    stock_costs: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, Optimizer]:
    """Train and pick portfolios month by month (module docs).

    Args:
        months: From :func:`.data.months`, ascending.
        settings: The walk's settings.
        device: ``cuda`` or ``cpu``.
        subsets: Also pick each test month's portfolio from only these stocks, by month end
            (module docs).
        stock_costs: Each month's ``(trade_cost, borrow_fee)`` per candidate (``costs.align``),
            used in the loss when ``settings.costs`` is set.

    Returns:
        ``weights`` (date, symbol, weight) for every test month; ``log`` (date, ret = the
        portfolio's next-month return over the T-bill before costs, ``long`` and ``short`` =
        the sums of the positive and negative weights, ``largest`` = the largest absolute
        weight's share of the sum of absolute weights, mean training loss over the last pass,
        seconds; with ``subsets``, ``ret_subset`` and ``stocks_subset``); and the final model.
        With ``subsets`` the weights also have ``subset_weight`` (null outside the subset).
    """
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    model = Optimizer(settings.model, months[0].x.shape[1], settings.blocks, settings.hidden,
                      settings.long_only, settings.fast_attention, settings.scale,
                      settings.cap, settings.names, settings.softplus)  # fmt: skip
    model.to(device)
    adam = torch.optim.Adam(model.parameters(), lr=settings.lr)
    data = [_tensors(m, device) for m in months]
    known = [bool(np.isfinite(m.r).all()) for m in months]
    rows, sub = _subset_rows(months, data, subsets)
    links, stock = _pricing(months, settings, device, subsets, stock_costs)
    tune_rng = np.random.default_rng(settings.seed + 1)
    weights, log, trained = [], [], False
    for t, month in enumerate(months):
        lo = 0 if settings.window is None else t - settings.window
        if month.date < settings.start or lo < 0:
            continue
        clock = time.perf_counter()
        train = [k for k in range(lo, t) if known[k]]
        source = sub if settings.subset_train == "only" else data
        epochs = settings.warm_epochs if trained else settings.first_epochs
        fit = [k for k in train if source[k] is not None]
        losses = _train(model, adam, source, fit, epochs, settings, rng, links, stock)
        trained = True
        with torch.no_grad():
            w = model.weights(data[t][0]).cpu().numpy().astype(np.float64)
        sized = settings.scale or (settings.softplus and settings.long_only)
        m = float(w.sum()) if sized else 1.0
        if sized and m > 0:
            w = w / m  # hold the mix fully invested; m is the model's sizing
        ret = float(w @ month.r) if known[t] else None
        frame = pl.DataFrame({"date": [month.date] * len(w), "symbol": month.symbols, "weight": w})
        extra = {}
        if subsets is not None:
            keep, picker = rows[t], model
            if settings.subset_train == "finetune":
                picker = copy.deepcopy(model)
                tune = torch.optim.Adam(picker.parameters(), lr=settings.lr)
                fit = [k for k in train if sub[k] is not None]
                _train(picker, tune, sub, fit, settings.finetune_epochs, settings, tune_rng)
            with torch.no_grad():
                ws = picker.weights(data[t][0][keep]).cpu().numpy().astype(np.float64)
            column = np.full(len(w), np.nan)
            column[keep] = ws
            frame = frame.with_columns(pl.Series("subset_weight", column).fill_nan(None))
            extra = {"ret_subset": float(ws @ month.r[keep]) if known[t] and keep else None,
                     "stocks_subset": len(keep)}  # fmt: skip
        weights.append(frame)
        gross = float(np.abs(w).sum())  # 0 only if no stock is held: all cash
        log.append({"date": month.date, "ret": ret, "long": float(w[w > 0].sum()),
                    "short": float(w[w < 0].sum()),
                    "largest": float(np.abs(w).max()) / gross if gross else None,
                    "loss": float(torch.stack(losses).mean()), "m": m,
                    "holdings": float(1 / (w**2).sum()) if settings.long_only and gross else None,
                    "seconds": time.perf_counter() - clock} | extra)  # fmt: skip
    return pl.concat(weights), pl.DataFrame(log), model
