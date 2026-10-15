"""Walk-forward training: at each month end, train on earlier months, then pick the portfolio.

At month end t, the months whose next-month returns are already known are the ones before
t. The training months are the ``window`` months before t (all earlier months if ``window``
is ``None``). The first test month trains a fresh model for ``first_epochs`` passes over its
training months; every later month continues from the previous month's model (and Adam's
state) for ``warm_epochs`` passes over its own training months. Each pass visits the training
months in a shuffled order and takes one Adam step per month (paper, Algorithm 2). Then the
model's weights for month t's candidates are the portfolio held over the next month.

With per-stock ``costs`` the loss is (1 - net)^2, net the month's return after each stock's
spread on the dollars traded from the month before's weights (the same model's, without a
gradient) and its borrow fee plus ``short_hurdle`` a year on shorts (research.optimizer.costs).

With ``holdings_input`` (2026-10-15; needs ``costs``) each stock's row also gets its holding
and spread (research.optimizer.holdings): in training, from a book table of each month's
weights (filled in month order the first time a month enters the window, refreshed at every
step, kept from month to month); for the test month, from the walk's real holdings of the
month before after its returns. ``borrow_input`` adds the stock's borrow fee as well.

With ``stop`` each month trains with early stopping (research.optimizer.stopping): a random
block of ``holdout`` consecutive training months is held out, passes run one at a time (the
rate following the schedule over at most ``first_epochs`` or ``warm_epochs`` passes) until
the block's loss, smoothed over ``smooth`` passes, has not improved for ``patience`` passes,
and the model goes back to the best pass; the log records the passes kept, the best smoothed
held-out loss and every pass's held-out loss (``held_curve``).

With ``average`` the walk holds a moving average of the model's parameters instead of its
latest step (research.optimizer.averaging); training is unchanged.
"""

import math
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
import torch

from portfolio_lab.research.optimizer import costs, holdings, stopping
from portfolio_lab.research.optimizer.averaging import Average
from portfolio_lab.research.optimizer.clipping import Clipper
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
        first_epochs: Passes over the training months for the first test month (with
            ``stop``: at most this many).
        warm_epochs: Passes for every later test month, continuing from the last model
            (with ``stop``: at most this many).
        lr: Adam's learning rate.
        seed: Random seed (initial weights and month order).
        start: First test month end.
        fast_attention: bfloat16 fused attention (model.Block).
        clip: Largest gradient size per step: a number, ``"auto"`` (the 95th percentile of
            recent steps, research.optimizer.clipping) or ``None`` (no clipping, the paper).
        costs: A per-stock cost table (``costs.align``) for the loss; the walk gets it aligned
            as ``stock_costs``. Recorded here as its path.
        short_hurdle: Extra yearly cost per dollar shorted in the loss, on top of the borrow
            fee: the willingness-to-short knob (0: none; large: shorts must earn that much).
        schedule: Learning rate within each month's training: ``flat`` (the starting rate every
            epoch) or ``cosine`` (from the month's starting rate down to ``lr_min``).
        lr_min: The cosine schedule's last rate.
        warm_lr: Starting rate for months that continue last month's model (``None``: ``lr``).
        compile: ``torch.compile`` each model (faster; same results).
        max_return: Training months' returns above this in size are set to 0 (data errors such
            as a $0.001 stock "returning" 94,800% in a month); the test month's are kept.
            ``None``: kept as they are.
        holdings_input: Feed each stock's holding and spread to the model (module docs).
        borrow_input: Also feed each stock's borrow fee (needs ``holdings_input``).
        stop: Early stopping each month (module docs).
        patience: Passes without a better smoothed held-out loss before stopping.
        holdout: Consecutive training months held out each month for stopping.
        smooth: Passes the held-out loss is averaged over before the rule reads it.
        average: Decay of the parameter average held each month (0: hold the latest model).
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
    fast_attention: bool = False
    clip: float | str | None = 1.0
    costs: str | None = None
    short_hurdle: float = 0.0
    schedule: str = "flat"
    lr_min: float = 1e-5
    warm_lr: float | None = None
    compile: bool = False
    max_return: float | None = None
    holdings_input: bool = False
    borrow_input: bool = False
    stop: bool = False
    patience: int = 5
    holdout: int = 12
    smooth: int = 3
    average: float = 0.0


Pair = tuple[torch.Tensor, torch.Tensor]


def _data(months: list[Month], settings: Settings, device: str) -> list[Pair]:
    """Each month's (x, r) tensors, returns above ``settings.max_return`` in size set to 0.

    The test month's return is read from ``month.r`` itself, so it stays untouched.
    """
    data = [(torch.from_numpy(m.x).to(device), torch.from_numpy(m.r).to(device)) for m in months]
    if not settings.max_return:
        return data
    return [(x, torch.where(r.abs() > settings.max_return, 0.0, r)) for x, r in data]


@dataclass(frozen=True)
class Pricing:
    """What the cost-aware loss needs (:func:`_pricing`).

    Attributes:
        links: ``costs.links`` of the months.
        stock: Each month's per-stock (trade cost, borrow fee).
        book: The training book table with ``holdings_input`` (module docs), else ``None``.
        borrow: Feed the borrow fee too (``borrow_input``).
    """

    links: list[costs.Link | None]
    stock: list[Pair]
    book: dict[int, torch.Tensor] | None = None
    borrow: bool = False


def _pricing(
    months: list[Month], settings: Settings, device: str,
    stock_costs: list[tuple[np.ndarray, ...]] | None,
) -> Pricing | None:  # fmt: skip
    """The cost-aware loss's inputs (``None`` without costs); checks the settings agree.

    ``stock_costs``: each month's per-stock (trade cost, borrow fee, ...) from ``costs.align``.
    """
    if settings.holdings_input and not settings.costs:
        raise ValueError("holdings_input needs per-stock costs")
    if settings.borrow_input and not settings.holdings_input:
        raise ValueError("borrow_input needs holdings_input")
    if not settings.costs:
        return None
    if stock_costs is None:
        raise ValueError("per-stock costs need stock_costs")
    stock = [(torch.from_numpy(c[0]).to(device), torch.from_numpy(c[1]).to(device))
             for c in stock_costs]  # fmt: skip
    return Pricing(costs.links(months, device), stock, {} if settings.holdings_input else None,
                   settings.borrow_input)  # fmt: skip


def rate(schedule: str, start: float, floor: float, epoch: int, epochs: int) -> float:
    """The learning rate for pass ``epoch`` (0-based) of ``epochs``: ``flat`` or ``cosine``."""
    if schedule == "flat":
        return start
    if schedule == "cosine":
        done = epoch / max(epochs - 1, 1)
        return floor + (start - floor) * 0.5 * (1 + math.cos(math.pi * done))
    raise ValueError(f"schedule must be flat or cosine, not {schedule!r}")


def _net(model: Optimizer, data: list[Pair], pricing: Pricing, settings: Settings, k: int,
         w: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """Month k's return on weights ``w`` net of costs, trading from the month before's weights."""
    before = spread_before = None
    if k > 0:
        with torch.no_grad():
            before = model(holdings.inputs(data, pricing, pricing.book, k - 1))
        spread_before = pricing.stock[k - 1][0]
    spread, fee = pricing.stock[k]
    return costs.net_return(w, data[k][1], before, pricing.links[k], spread,
                            fee + settings.short_hurdle, spread_before)  # fmt: skip


def _priced_loss(model: Optimizer, data: list[Pair], pricing: Pricing, settings: Settings,
                 k: int) -> torch.Tensor:  # fmt: skip
    """Month k's paper loss on its return net of costs (module docs); stores w in the book."""
    w = model(holdings.inputs(data, pricing, pricing.book, k))
    if pricing.book is not None:
        pricing.book[k] = w.detach()
    return (1 - _net(model, data, pricing, settings, k, w)) ** 2


def _pass(  # noqa: PLR0913, PLR0917 - one pass's pieces
    model: Optimizer, adam: torch.optim.Optimizer, data: list[Pair], train: list[int],
    settings: Settings, rng: np.random.Generator, pricing: Pricing | None, lr: float,
    clipper: Clipper, norms: list[torch.Tensor], average: Average | None,
) -> list[torch.Tensor]:  # fmt: skip
    """One pass over the ``train`` months in shuffled order at rate ``lr``; its losses.

    Each step's (gradient size before clipping, limit) goes on ``norms``. A step whose loss
    is not finite is skipped rather than poison Adam.
    """
    for group in adam.param_groups:
        group["lr"] = lr
    losses = []  # kept on the device: reading each one would wait for the GPU
    for k in rng.permutation(train):
        adam.zero_grad()
        value = (loss(model, *data[k]) if pricing is None
                 else _priced_loss(model, data, pricing, settings, int(k)))  # fmt: skip
        if not torch.isfinite(value):
            continue
        value.backward()
        norms.append(clipper(model.parameters()))
        adam.step()
        if average is not None:
            average.update()
        losses.append(value.detach())
    return losses


def _held_loss(model: Optimizer, data: list[Pair], pricing: Pricing | None,
               settings: Settings, months: list[int]) -> float:  # fmt: skip
    """Mean loss over ``months`` (net of costs with ``pricing``); leaves the book alone."""
    with torch.no_grad():
        r = []
        for k in months:
            w = model(holdings.inputs(data, pricing, pricing.book if pricing else None, k))
            r.append(w @ data[k][1] if pricing is None
                     else _net(model, data, pricing, settings, k, w))  # fmt: skip
        r = torch.stack(r)
    return float(((1 - r) ** 2).mean())


def _fit(  # noqa: PLR0913 - one month's training pieces
    model: Optimizer, adam: torch.optim.Optimizer, data: list[Pair], train: list[int],
    epochs: int, settings: Settings, rng: np.random.Generator, pricing: Pricing | None, *,
    start_lr: float, span: range, clipper: Clipper, average: Average | None,
) -> tuple[list, int, float, list[float], list[torch.Tensor]]:  # fmt: skip
    """One month's training: losses, passes, held-out loss, its curve, gradient sizes.

    Returns the kept pass's losses, the passes kept, the best smoothed held-out loss, its
    per-pass curve and every step's (gradient size before clipping, limit) over all the
    passes run.
    ``epochs`` passes (with ``settings.stop``: at most), the rate following
    ``settings.schedule`` from ``start_lr`` over them. With a book (``holdings_input``) the
    ``span`` months not in it get the model's weights first, in month order.
    """
    if pricing is not None:
        holdings.fill(model, data, pricing, span)
    fit, block = (stopping.held_block(train, settings.holdout, rng) if settings.stop
                  else (train, []))  # fmt: skip

    norms: list[torch.Tensor] = []

    def one_pass(p: int) -> list:
        lr = rate(settings.schedule, start_lr, settings.lr_min, p, epochs)
        return _pass(model, adam, data, fit, settings, rng, pricing, lr, clipper, norms,
                     average)  # fmt: skip

    return (*stopping.fit_stopped(model, adam, one_pass,
                                  lambda: _held_loss(model, data, pricing, settings, block),
                                  bool(block), epochs, settings.patience, settings.smooth,
                                  average),
            norms)  # fmt: skip


def _new_model(settings: Settings, width: int, device: str) -> tuple[Optimizer, torch.optim.Adam]:
    """A freshly initialized model and its Adam; the holdings inputs add their columns."""
    torch.manual_seed(settings.seed)
    extra = 2 * settings.holdings_input + settings.borrow_input
    model = Optimizer(settings.model, width + extra, settings.blocks, settings.hidden,
                      settings.fast_attention)  # fmt: skip
    model.to(device)
    if settings.compile:
        model.compile(dynamic=True)
    return model, torch.optim.Adam(model.parameters(), lr=settings.lr)


def walk(
    months: list[Month],
    settings: Settings,
    device: str = "cpu",
    stock_costs: list[tuple[np.ndarray, ...]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, Optimizer]:
    """Train and pick portfolios month by month (module docs).

    Args:
        months: From :func:`.data.months`, ascending.
        settings: The walk's settings.
        device: ``cuda`` or ``cpu``.
        stock_costs: Each month's ``(trade_cost, borrow_fee, ...)`` per candidate
            (``costs.align``), needed when ``settings.costs`` is set.

    Returns:
        ``weights`` (date, symbol, weight) for every test month; ``log`` (date, ret = the
        portfolio's next-month return over the T-bill before costs, ``long`` and ``short`` =
        the sums of the positive and negative weights, ``largest`` = the largest absolute
        weight's share of the sum of absolute weights, mean training loss over the last pass,
        passes kept, best smoothed held-out loss and its per-pass curve with ``stop``, the
        median and largest gradient size over the month's steps before clipping, the share
        of steps clipped and the last step's limit, seconds); and the final model.
    """
    rng = np.random.default_rng(settings.seed)
    clipper = Clipper(settings.clip)
    model, adam = _new_model(settings, months[0].x.shape[1], device)
    average = Average(model, settings.average) if settings.average else None
    data = _data(months, settings, device)
    known = [bool(np.isfinite(m.r).all()) for m in months]
    pricing = _pricing(months, settings, device, stock_costs)
    weights, log, trained, last = [], [], False, None
    for t, month in enumerate(months):
        lo = 0 if settings.window is None else t - settings.window
        if month.date < settings.start or lo < 0:
            continue
        clock = time.perf_counter()
        train = [k for k in range(lo, t) if known[k]]
        epochs = settings.warm_epochs if trained else settings.first_epochs
        start_lr = settings.warm_lr if trained and settings.warm_lr else settings.lr
        losses, passes, held_loss, curve, norms = _fit(model, adam, data, train, epochs,
                                                       settings, rng, pricing,
                                                       start_lr=start_lr,
                                                       span=range(lo, t), clipper=clipper,
                                                       average=average)  # fmt: skip
        grad, limit = torch.stack(norms).T if norms else (torch.zeros(0), torch.zeros(0))
        trained = True
        if average is None:
            w = holdings.pick(model, data, pricing, last, t)
        else:
            with average.swapped():
                w = holdings.pick(model, data, pricing, last, t)
        last = holdings.remember(pricing, w, data, t)
        gross = float(np.abs(w).sum())  # 0 only if no stock is held: all cash
        weights.append(pl.DataFrame({"date": [month.date] * len(w), "symbol": month.symbols,
                                     "weight": w}))  # fmt: skip
        log.append({"date": month.date, "ret": float(w @ month.r) if known[t] else None,
                    "long": float(w[w > 0].sum()), "short": float(w[w < 0].sum()),
                    "largest": float(np.abs(w).max()) / gross if gross else None,
                    "loss": float(torch.stack(losses).mean()) if losses else math.nan,
                    "passes": passes, "held_loss": held_loss, "held_curve": curve,
                    "grad_median": float(grad.median()) if norms else math.nan,
                    "grad_max": float(grad.max()) if norms else math.nan,
                    "clipped": float((grad > limit).float().mean()) if norms else 0.0,
                    "limit": float(limit[-1]) if norms else math.nan,
                    "seconds": time.perf_counter() - clock})  # fmt: skip
    return pl.concat(weights), pl.DataFrame(log), model
