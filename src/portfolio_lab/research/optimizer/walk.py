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
import math
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
        short_hurdle: Extra yearly cost per dollar shorted in the loss, on top of the borrow
            fee: the willingness-to-short knob (0: none; large: shorts must earn that much).
        short_min_easy: Only stocks at least this likely to be easy to borrow (the cost
            table's ``easy_chance``) may be held short, in training and in the walk: the others'
            weights are clamped at 0 or above (``None``: every stock).
        no_shorts: Hold no shorts in the walk (holdings clamped at 0 or above). Training
            still lets weights go negative (else every score can sink below 0 and the gradient
            dies, run16); pair it with a large ``short_hurdle`` so shorts never pay there.
        scaled_attention: Divide attention scores by sqrt(D) (model.Block); not in the paper.
        init: ``variance`` or ``std``: how to read the paper's initialization (model docs).
        schedule: Learning rate within each month's training: ``flat`` (``lr`` every epoch) or
            ``cosine`` (from the month's starting rate down to ``lr_min`` over its epochs).
        lr_min: The cosine schedule's last rate.
        warm_lr: Starting rate for months that continue last month's model (``None``: ``lr``);
            a fresh or first fit starts at ``lr``.
        compile: ``torch.compile`` each model (faster; same results).
        max_return: Training months' returns above this in size are set to 0 (data errors such
            as a $0.001 stock "returning" 94,800% in a month); the test month's are kept.
            ``None``: kept as they are.
        fresh: Retrain from scratch every test month, as the paper does ("roll the training
            sample forward one month and retrain"): new starting weights (seeded by ``seed`` and
            the month) and a new Adam, ``first_epochs`` passes. Default: continue last month's
            model for ``warm_epochs`` passes (faster; ours).
        every: Pick a portfolio only every ``every``-th test month (1: all). Needs ``fresh``,
            since each picked month is then trained on its own; for quick checks.
        partial: Trade only part of the way to the model's weights each month, at a learned
            pace a: holdings = (1 - a) * last month's + a * new weights. In training, last
            month's holdings are taken as the model's weights for the month before (both
            with gradients); in the walk, the real holdings carry forward. Paper loss only.
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
    partial: bool = False
    short_hurdle: float = 0.0
    short_min_easy: float | None = None
    no_shorts: bool = False
    scaled_attention: bool = False
    fresh: bool = False
    every: int = 1
    init: str = "variance"
    schedule: str = "flat"
    lr_min: float = 1e-5
    warm_lr: float | None = None
    compile: bool = False
    max_return: float | None = None


def _tensors(m: Month, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(m.x).to(device), torch.from_numpy(m.r).to(device)


def _data(months: list[Month], settings: Settings, device: str) -> list[tuple]:
    """Each month's (x, r) tensors, returns above ``settings.max_return`` in size set to 0.

    The test month's return is read from ``month.r`` itself, so it stays untouched.
    """
    data = [_tensors(m, device) for m in months]
    if not settings.max_return:
        return data
    return [(x, torch.where(r.abs() > settings.max_return, 0.0, r)) for x, r in data]


def _steps(train: list[int], settings: Settings, rng: np.random.Generator) -> list[list[int]]:
    """One pass's training steps: single months (paper) or blocks of consecutive months."""
    if settings.objective is None:
        return [[int(k)] for k in rng.permutation(train)]
    offset = int(rng.integers(settings.batch)) if len(train) > settings.batch else 0
    cuts = [0, *range(offset or settings.batch, len(train), settings.batch), len(train)]
    blocks = [train[a:b] for a, b in pairwise(cuts) if b - a >= 3]
    return [blocks[i] for i in rng.permutation(len(blocks))]


Pair = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class Pricing:
    """What the cost-aware loss needs (:func:`_pricing`).

    Attributes:
        links: ``costs.links`` of the months.
        stock: Each month's per-stock (trade cost, borrow fee), or ``None`` for flat rates.
        shortable: Each month's per-stock flag: may be held short (``None``: all may).
    """

    links: list[costs.Link | None]
    stock: list[Pair] | None
    shortable: list[torch.Tensor] | None


def _allow(w: torch.Tensor, shortable: torch.Tensor | None) -> torch.Tensor:
    """``w`` with the stocks that may not be shorted clamped at 0 or above."""
    return w if shortable is None else torch.where(shortable, w, torch.relu(w))


def _train(  # noqa: PLR0913 - one training run's pieces
    model: Optimizer,
    adam: torch.optim.Optimizer,
    source: list[Pair | None],
    train: list[int],
    epochs: int,
    settings: Settings,
    rng: np.random.Generator,
    pricing: Pricing | None = None,
    *,
    start_lr: float | None = None,
) -> list[torch.Tensor]:
    """``epochs`` passes over the ``train`` months of ``source``; the last pass's losses.

    Each pass's learning rate follows ``settings.schedule`` from ``start_lr`` (default
    ``settings.lr``).

    With ``pricing`` the loss uses the return net of costs, the weights the month before
    computed by the same model (without a gradient, unless partial trading blends them into
    the holdings); with per-stock costs each stock pays its own rates, and stocks that may not
    be shorted are clamped at 0 or above.
    """
    losses = []
    for epoch in range(epochs):
        for group in adam.param_groups:
            group["lr"] = _lr(settings, start_lr or settings.lr, epoch, epochs)
        losses = []  # kept on the device: reading each one would wait for the GPU
        for step in _steps(train, settings, rng):
            adam.zero_grad()
            if pricing is not None:
                k = step[0]
                mask = pricing.shortable or [None] * len(source)
                w = _allow(model.weights(source[k][0]), mask[k])
                if model.pace_logit is not None and k > 0:  # holdings: part way from before
                    before = _allow(model.weights(source[k - 1][0]), mask[k - 1])
                    w = (1 - model.pace()) * costs.carry(
                        before, pricing.links[k]
                    ) + model.pace() * w
                    w = _allow(w, mask[k])
                else:
                    with torch.no_grad():
                        before = (_allow(model.weights(source[k - 1][0]), mask[k - 1])
                                  if k > 0 else None)  # fmt: skip
                spread, fee, spread_before = settings.spread, settings.short_fee, None
                if pricing.stock is not None:
                    (spread, fee), spread_before = (
                        pricing.stock[k],
                        (pricing.stock[k - 1][0] if k > 0 else None),
                    )
                net = costs.net_return(w, source[k][1], before, pricing.links[k], spread,
                                       fee + settings.short_hurdle, spread_before)  # fmt: skip
                value = (1 - net) ** 2
            elif settings.objective is None:
                value = loss(model, *source[step[0]])
            else:
                r = torch.stack([model.weights(source[k][0]) @ source[k][1] for k in step])
                value = objectives.grade(settings.objective, r, settings.gamma)
            if not torch.isfinite(value):  # skip a broken step rather than poison Adam
                continue
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
    stock_costs: list[tuple[np.ndarray, ...]] | None,
) -> Pricing | None:
    """The cost-aware loss's inputs (``None`` without costs); checks the settings agree.

    ``stock_costs``: each month's per-stock (trade cost, borrow fee[, easy-to-borrow chance])
    from ``costs.align``.
    """
    if settings.subset_train and subsets is None:
        raise ValueError("subset_train needs subsets")
    if settings.every > 1 and not settings.fresh:
        raise ValueError("every > 1 needs fresh")
    shorts = settings.short_hurdle > 0 or settings.short_min_easy is not None or settings.no_shorts
    flat = settings.spread > 0 or settings.short_fee > 0
    if not (flat or shorts or settings.costs or settings.partial):
        return None
    if settings.objective or settings.subset_train:
        raise ValueError("costs need the paper loss on every candidate")
    if (settings.costs or settings.short_min_easy is not None) and stock_costs is None:
        raise ValueError("per-stock costs need stock_costs")
    stock = shortable = None
    if settings.costs:
        stock = [(torch.from_numpy(c[0]).to(device), torch.from_numpy(c[1]).to(device))
                 for c in stock_costs]  # fmt: skip
    if settings.short_min_easy is not None:
        shortable = [torch.from_numpy(c[2] >= settings.short_min_easy).to(device)
                     for c in stock_costs]  # fmt: skip
    return Pricing(costs.links(months, device), stock, shortable)


def _hold(
    model: Optimizer,
    held: tuple[list[str], np.ndarray] | None,
    month: Month,
    w: np.ndarray,
    shortable: np.ndarray | None,
) -> tuple[np.ndarray, tuple[list[str], np.ndarray] | None, float | None]:
    """The month's holdings from the model's weights ``w``.

    Stocks that may not be shorted (``shortable`` False) are clamped at 0 or above, before and
    after partial trading blends last month's real holdings in at the model's pace. Returns
    the holdings, the state to pass next month and the pace (``None`` without partial).
    """
    clamp = (
        (lambda v: v) if shortable is None else (lambda v: np.where(shortable, v, np.maximum(v, 0)))
    )
    w = clamp(w)
    if model.pace_logit is None:
        return w, None, None
    pace = float(model.pace())
    if held is not None:
        before = dict(zip(*held, strict=True))
        w = clamp((1 - pace) * np.array([before.get(s, 0.0) for s in month.symbols]) + pace * w)
    return w, (month.symbols, w), pace


def _lr(settings: Settings, start: float, epoch: int, epochs: int) -> float:
    """The learning rate for pass ``epoch`` (0-based) of ``epochs`` (``settings.schedule``)."""
    if settings.schedule == "flat":
        return start
    if settings.schedule == "cosine":
        done = epoch / max(epochs - 1, 1)
        return settings.lr_min + (start - settings.lr_min) * 0.5 * (1 + math.cos(math.pi * done))
    raise ValueError(f"schedule must be flat or cosine, not {settings.schedule!r}")


def _start_lr(settings: Settings, trained: bool) -> float:
    """The month's starting rate: ``warm_lr`` when continuing last month's model, else ``lr``."""
    warm = trained and not settings.fresh
    return settings.warm_lr if warm and settings.warm_lr else settings.lr


def _new_model(
    settings: Settings, width: int, device: str, seed: int, month: int = 0
) -> tuple[Optimizer, torch.optim.Optimizer]:
    """A freshly initialized model and its Adam (seeded by ``seed``, and ``month`` if any)."""
    torch.manual_seed(seed if month == 0 else seed * 100_003 + month)
    model = Optimizer(settings.model, width, settings.blocks, settings.hidden,
                      settings.long_only, settings.fast_attention, settings.scale,
                      settings.cap, settings.names, settings.softplus,
                      partial=settings.partial,
                      scaled_attention=settings.scaled_attention, init=settings.init)  # fmt: skip
    model.to(device)
    if settings.compile:
        model.compile(dynamic=True)
    return model, torch.optim.Adam(model.parameters(), lr=settings.lr)


def _shortable(pricing: Pricing | None, settings: Settings, t: int, n: int) -> np.ndarray | None:
    """Month ``t``'s may-be-shorted flags for the walk's holdings (``None``: all may)."""
    if settings.no_shorts:
        return np.zeros(n, dtype=bool)
    if pricing is None or pricing.shortable is None:
        return None
    return pricing.shortable[t].cpu().numpy()


def walk(
    months: list[Month],
    settings: Settings,
    device: str = "cpu",
    subsets: dict[date, set[str]] | None = None,
    stock_costs: list[tuple[np.ndarray, ...]] | None = None,
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
    rng = np.random.default_rng(settings.seed)
    model, adam = _new_model(settings, months[0].x.shape[1], device, settings.seed)
    data = _data(months, settings, device)
    known = [bool(np.isfinite(m.r).all()) for m in months]
    rows, sub = _subset_rows(months, data, subsets)
    pricing = _pricing(months, settings, device, subsets, stock_costs)
    tune_rng = np.random.default_rng(settings.seed + 1)
    weights, log, trained, held, tested = [], [], False, None, -1
    for t, month in enumerate(months):
        lo = 0 if settings.window is None else t - settings.window
        if month.date < settings.start or lo < 0:
            continue
        if (tested := tested + 1) % settings.every:
            continue
        clock = time.perf_counter()
        if settings.fresh:  # each month on its own: new weights, Adam and shuffle
            model, adam = _new_model(settings, months[0].x.shape[1], device, settings.seed, t)
            rng = np.random.default_rng(settings.seed * 100_003 + t)
        train = [k for k in range(lo, t) if known[k]]
        source = sub if settings.subset_train == "only" else data
        epochs = settings.warm_epochs if trained and not settings.fresh else settings.first_epochs
        fit = [k for k in train if source[k] is not None]
        losses = _train(model, adam, source, fit, epochs, settings, rng, pricing,
                        start_lr=_start_lr(settings, trained))  # fmt: skip
        trained = True
        with torch.no_grad():
            w = model.weights(data[t][0]).cpu().numpy().astype(np.float64)
        w, held, pace = _hold(model, held, month, w, _shortable(pricing, settings, t, len(w)))
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
                    "loss": float(torch.stack(losses).mean()) if losses else math.nan,
                    "m": m, "pace": pace,
                    "holdings": float(1 / (w**2).sum()) if settings.long_only and gross else None,
                    "seconds": time.perf_counter() - clock} | extra)  # fmt: skip
    return pl.concat(weights), pl.DataFrame(log), model
