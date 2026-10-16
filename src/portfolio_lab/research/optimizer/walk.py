"""Walk-forward training: at each month end, train on earlier months, then pick the portfolio.

At month end t, the months whose next-month returns are already known are the ones before
t. The training months are the ``window`` months before t (all earlier months if ``window``
is ``None``). The first test month trains a fresh model for ``first_epochs`` passes over its
training months; every later month continues from the previous month's model (and Adam's
state) for ``warm_epochs`` passes over its own training months. Each pass visits the training
months in a shuffled order and takes one Adam step per month (paper, Algorithm 2). Then the
model's weights for month t's candidates are the portfolio held over the next month.

With per-stock ``costs`` the loss is (1 - net)^2, net the month's return after each stock's
spread on the dollars traded from the book held going in (the month before's weights after its
returns) and its borrow fee plus ``short_hurdle`` a year on shorts (research.optimizer.priced).

With ``holdings_input`` (2026-10-15; needs ``costs``) each stock's row also gets its holding
and spread (research.optimizer.holdings): in training, from a book table of each month's
weights (filled in month order the first time a month enters the window, refreshed at every
step, kept from month to month); for the test month, from the walk's real holdings of the
month before after its returns. ``borrow_input`` adds the stock's borrow fee as well.

With a ``ridge`` (run43's overestimate grows as w^2; failed its check, to be removed) the net
return in the loss is cut by sum c_i w_i^2, c_i = ``ridge`` times ``ridge_tilt`` for the
stock's spread bucket (research.optimizer.optimism). Held-out scoring keeps the real return.

With ``stop`` each month trains with early stopping (research.optimizer.stopping): a random
block of ``holdout`` consecutive training months is held out, passes run one at a time (the
rate following the schedule over at most ``first_epochs`` or ``warm_epochs`` passes) until
the block's loss, smoothed over ``smooth`` passes, has not improved for ``patience`` passes,
and the model goes back to the best pass; the log records the passes kept, the best smoothed
held-out loss and every pass's held-out loss (``held_curve``).

With ``cold`` (2026-10-15) every month trains a fresh model instead of continuing last
month's: the seed's same starting weights, a fresh Adam and average, ``first_epochs`` passes
from ``lr``. A warm model has already trained on most of this month's held-out block (the block
moves every month), so its stopping score is not out of sample; a fresh one has not. With
``watch`` the block is held out and scored every pass (``held_curve``) but every pass runs and
the last is kept; with either, ``test_curve`` is the test month's loss after each pass (a
diagnostic: it uses the month's return, which the walk does not know when it picks; with
``watch`` and nothing held out, ``holdout`` 0, it is still logged). A cold month also starts
from an empty book, so the holdings inputs of its training months come from its own model, not
from earlier months' models that trained on them, and its month order and held-out draw come
from random streams of its own (the seed and the month), so a month's result does not depend
on which months ran before it; before each score the held-out months' book entries are rolled
forward, in month order, with the model as it stands.

Held-out design (2026-10-15): ``scatter`` holds out ``holdout`` single months drawn anywhere in
the window instead of one consecutive block; ``holdout_seed`` draws them from their own random
stream, so runs that differ only in it train on the same month order with different months
held out (several such fits are averaged like seeds). ``only`` trains and trades just the
listed test months (with ``cold`` each is independent of the others): a cheap sample of the
walk. The month before a listed one is then not a traded month, so the test month's holdings
inputs come from the model's own weights for it.

``average``: the walk holds a moving average of the model's parameters, not its latest step.

``inherit`` (2026-10-16; needs ``holdings_input``): training steps trade from books the model
did not build (research.optimizer.priced); the noise has its own random stream (the seed, and
with ``cold`` the month), so month order, held-out draws and stock drops are as without it.
"""

import contextlib
import copy
import math
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
import torch

from portfolio_lab.research.optimizer import holdings, optimism, priced, stopping
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
        winsor: Training months' returns are pulled in to their month's ``winsor`` and
            1 - ``winsor`` quantiles (0.01: the 1st and 99th percentiles; 0: off). The paper
            does not clip; our returns carry data errors and lottery outcomes that a book many
            times its capital bets on, and blow training up (bench1). Everything that is not
            a training step stays real: held-out and test scores, trades, optimism readings.
        cost_scale: Costs in the training loss are charged at this multiple (held-out scoring
            and the walk's own measures use the real costs): in-sample returns run about twice
            what the model earns out of sample, so at 1x costs look half as large as they are.
        measure_optimism: Log in-sample and out-of-sample return per dollar by spread, borrow
            fee, position size and volatility bucket each month (research.optimizer.optimism;
            needs ``costs``).
        vol_column: The inputs' column of the ranked volatility trait for those buckets.
        holdings_input / borrow_input: Feed each stock's holding and spread / also its borrow fee.
        ridge: Charge per unit of w^2 in the training loss (module docs; 0: none; ``costs``).
        ridge_tilt: Multipliers of ``ridge`` by spread bucket, cheap to dear (``None``: 1).
        stop: Early stopping each month (module docs).
        patience: Passes without a better smoothed held-out loss before stopping.
        holdout: Consecutive training months held out each month for stopping.
        smooth: Passes the held-out loss is averaged over before the rule reads it.
        average: Decay of the parameter average held each month (0: hold the latest model).
        cold: Train a fresh model every month (module docs).
        watch: Hold out and score the block every pass without stopping (module docs).
        scatter: Hold out single months drawn anywhere in the window, not a block.
        holdout_seed: Seed of the held-out draw's own random stream (``None``: ``seed``'s).
        only: Train and trade only these test month ends (``None``: every month).
        real_book: Training months the walk actually traded are held as traded, not as the
            model's own weights (research.optimizer.priced; needs ``holdings_input``).
        dropout: Dropout share in the model's blocks while training (``model`` docs).
        stock_drop: Share of the month's stocks each training step hides from the loss, drawn
            afresh (returns 0, the rest x 1 / (1 - share)); the model sees all; costs still paid.
        inherit: Average L1 size, as a share of the book, of the displacement each training
            step's book going in gets (module docs; 0: off). Each step draws its size from 0
            to twice this.
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
    cost_scale: float = 1.0
    measure_optimism: bool = False
    vol_column: int | None = None
    schedule: str = "flat"
    lr_min: float = 1e-5
    warm_lr: float | None = None
    compile: bool = False
    max_return: float | None = None
    winsor: float = 0.0
    holdings_input: bool = False
    borrow_input: bool = False
    ridge: float = 0.0
    ridge_tilt: tuple[float, ...] | None = None
    stop: bool = False
    patience: int = 5
    holdout: int = 12
    smooth: int = 3
    average: float = 0.0
    cold: bool = False
    watch: bool = False
    scatter: bool = False
    holdout_seed: int | None = None
    only: tuple[date, ...] | None = None
    real_book: bool = False
    dropout: float = 0.0
    stock_drop: float = 0.0
    inherit: float = 0.0


Pair = tuple[torch.Tensor, torch.Tensor]


def _data(months: list[Month], settings: Settings, device: str) -> tuple[list[Pair], list[Pair]]:
    """Each month's (x, r) tensors for training, and the same with real returns.

    Training returns above ``settings.max_return`` in size are set to 0, then pulled in to
    the month's ``settings.winsor`` quantiles (months whose returns are not all known are
    left as they are). The real pairs share the inputs and score the test month.
    """
    raw = [(torch.from_numpy(m.x).to(device), torch.from_numpy(m.r).to(device)) for m in months]
    data = raw
    if settings.max_return:
        data = [(x, torch.where(r.abs() > settings.max_return, 0.0, r)) for x, r in data]
    if settings.winsor:
        q, known = settings.winsor, [len(r) and bool(torch.isfinite(r).all()) for _, r in data]
        data = [(x, r.clamp(torch.quantile(r, q), torch.quantile(r, 1 - q))) if ok else (x, r)
                for (x, r), ok in zip(data, known, strict=True)]  # fmt: skip
    return data, raw


def rate(schedule: str, start: float, floor: float, epoch: int, epochs: int) -> float:
    """The learning rate for pass ``epoch`` (0-based) of ``epochs``: ``flat`` or ``cosine``."""
    if schedule == "flat":
        return start
    if schedule == "cosine":
        done = epoch / max(epochs - 1, 1)
        return floor + (start - floor) * 0.5 * (1 + math.cos(math.pi * done))
    raise ValueError(f"schedule must be flat or cosine, not {schedule!r}")


def _pass(  # noqa: PLR0913, PLR0917 - one pass's pieces
    model: Optimizer, adam: torch.optim.Optimizer, data: list[Pair], train: list[int],
    settings: Settings, rng: np.random.Generator, pricing: priced.Pricing | None, lr: float,
    clipper: Clipper, norms: list[torch.Tensor], average: Average | None,
    noise: np.random.Generator | None = None,
) -> list[torch.Tensor]:  # fmt: skip
    """One pass over the ``train`` months in shuffled order at rate ``lr``; its losses.

    Each step's (gradient size before clipping, limit) goes on ``norms``. A step whose loss
    is not finite is skipped rather than poison Adam. ``noise``: the inherited books' stream.
    """
    for group in adam.param_groups:
        group["lr"] = lr
    losses = []  # kept on the device: reading each one would wait for the GPU
    model.train()  # dropout on; everything else scores the whole model
    for k in rng.permutation(train):
        adam.zero_grad()
        x, r = data[k]
        seen = data  # with stock_drop: this step's month with the hidden stocks' returns at 0
        if settings.stock_drop:
            kept = torch.from_numpy(rng.random(len(r)) >= settings.stock_drop).to(r)
            r = r * kept / (1 - settings.stock_drop)
            seen = [*data[:k], (x, r), *data[k + 1 :]]
        value = (loss(model, x, r) if pricing is None
                 else priced.loss(model, seen, pricing, settings, int(k), noise))  # fmt: skip
        if not torch.isfinite(value):
            continue
        value.backward()
        norms.append(clipper(model.parameters()))
        adam.step()
        if average is not None:
            average.update()
        losses.append(value.detach())
    model.eval()
    return losses


def _held_loss(model: Optimizer, data: list[Pair], pricing: priced.Pricing | None,
               settings: Settings, months: list[int],
               keep: list | None = None) -> float:  # fmt: skip
    """Mean loss over ``months`` (net of costs with ``pricing``; their returns into ``keep``)."""
    with torch.no_grad():
        r = []
        for k in months:
            w = model(holdings.inputs(data, pricing, pricing.held() if pricing else None, k))
            r.append(w @ data[k][1] if pricing is None
                     else priced.net(model, data, pricing, settings, k, w))  # fmt: skip
        r = torch.stack(r)
    if keep is not None:
        keep[:] = r.tolist()
    return float(((1 - r) ** 2).mean())


def _roll(model: Optimizer, data: list[Pair], pricing: priced.Pricing | None,
          months: list[int]) -> None:  # fmt: skip
    """With a book, ``months``' entries become the model's weights, in month order."""
    if pricing is None or pricing.book is None:
        return
    with torch.no_grad():
        for k in sorted(months):
            pricing.book[k] = model(holdings.inputs(data, pricing, pricing.held(), k))


def _fit(  # noqa: PLR0913 - one month's training pieces
    model: Optimizer, adam: torch.optim.Optimizer, data: list[Pair], train: list[int],
    epochs: int, settings: Settings, rng: np.random.Generator, pricing: priced.Pricing | None, *,
    start_lr: float, span: range, clipper: Clipper, average: Average | None, test: int,
    hold_rng: np.random.Generator, raw: list[Pair], noise: np.random.Generator | None = None,
) -> tuple[list, int, float, list[float], list[torch.Tensor], list[float], dict]:  # fmt: skip
    """One month's training.

    Returns the kept pass's losses, passes kept, best smoothed held-out loss, its per-pass
    curve, every step's (gradient size before clipping, limit), the test month ``test``'s loss
    after each pass (``test`` < 0: none) and for the log the passes rolled back
    (:class:`.stopping.Guard`) and the held-out months' returns at the last score (month order;
    ``watch``: the kept model's). ``epochs`` passes (``settings.stop``: at most), the rate
    following ``settings.schedule`` from ``start_lr``. With a book (``holdings_input``) the
    ``span`` months not in it get the model's weights first, in month order. The held-out months
    are drawn from ``hold_rng``; they and the test month are scored on their real returns
    (``raw``), training alone sees ``data``'s clipped ones. ``noise``: the inherited books' stream.
    """
    if pricing is not None:
        holdings.fill(model, data, pricing, span)
    fit, block = (stopping.held_months(train, settings.holdout, hold_rng, settings.scatter)
                  if settings.stop or settings.watch else (train, []))  # fmt: skip
    tests: list[float] = []
    held: list[float] = []

    def score() -> float:
        _roll(model, raw, pricing, block)
        if test >= 0:
            tests.append(_held_loss(model, raw, pricing, settings, [test]))
        return _held_loss(model, raw, pricing, settings, block, held)

    norms: list[torch.Tensor] = []
    guard = stopping.Guard(model, adam, average)

    def one_pass(p: int) -> list:
        lr = rate(settings.schedule, start_lr, settings.lr_min, p, epochs) * guard.scale
        losses = _pass(model, adam, data, fit, settings, rng, pricing, lr, clipper, norms,
                       average, noise)  # fmt: skip
        if not block and settings.watch and test >= 0:  # nothing held out: still log the test
            tests.append(_held_loss(model, raw, pricing, settings, [test]))
        return losses

    return (*stopping.fit_stopped(model, adam, lambda p: guard(one_pass, p), score,
                                  bool(block), epochs, settings.patience, settings.smooth,
                                  average, keep_last=settings.watch),
            norms, tests, {"rollbacks": guard.rollbacks, "held_returns": held})  # fmt: skip


def _new_model(settings: Settings, width: int, device: str) -> tuple[Optimizer, torch.optim.Adam]:
    """A freshly initialized model and its Adam; the holdings inputs add their columns."""
    torch.manual_seed(settings.seed)
    extra = 2 * settings.holdings_input + settings.borrow_input
    model = Optimizer(settings.model, width + extra, settings.blocks, settings.hidden,
                      settings.fast_attention, settings.dropout)  # fmt: skip
    model.to(device)
    if settings.compile:
        model.compile(dynamic=True)
    return model, torch.optim.Adam(model.parameters(), lr=settings.lr)


def _streams(settings: Settings, day: date) -> tuple[np.random.Generator, ...]:
    """A cold month's own (month order, held-out draw, inherited books) random streams."""
    rng = np.random.default_rng([settings.seed, day.toordinal()])
    noise = np.random.default_rng([settings.seed, day.toordinal(), 1])
    if settings.holdout_seed is None:
        return rng, rng, noise
    return rng, np.random.default_rng([settings.holdout_seed, day.toordinal()]), noise


def _walk_streams(settings: Settings) -> tuple[np.random.Generator, ...]:
    """The walk's (month order, held-out draw, inherited books) random streams: the seed's."""
    rng = np.random.default_rng(settings.seed)
    hold = rng if settings.holdout_seed is None else np.random.default_rng(settings.holdout_seed)
    return rng, hold, np.random.default_rng([settings.seed, 1])


def _traded(model: Optimizer, average: Average | None) -> Optimizer:
    """The model as it traded: with an ``average``, the averaged parameters put in for good."""
    if average is not None:
        average.install()
    return model


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
        of steps clipped and the last step's limit, seconds); and the final model (with
        ``average``, its averaged parameters: the model that traded the last month).
    """
    rng, hold_rng, noise = _walk_streams(settings)
    clipper = Clipper(settings.clip)
    if settings.stop and settings.watch:
        raise ValueError("stop and watch are alternatives")
    model, adam = _new_model(settings, months[0].x.shape[1], device)
    start_state = copy.deepcopy(model.state_dict()) if settings.cold else None
    average = Average(model, settings.average) if settings.average else None
    data, raw = _data(months, settings, device)
    known = [bool(np.isfinite(m.r).all()) for m in months]
    pricing = priced.pricing(months, settings, device, stock_costs)
    if settings.measure_optimism and pricing is None:
        raise ValueError("measure_optimism needs per-stock costs")
    cuts = optimism.edges(pricing.stock) if settings.measure_optimism else None
    only = None if settings.only is None else set(settings.only)
    weights, log, trained, last, traded = [], [], False, None, -1
    for t, month in enumerate(months):
        lo = 0 if settings.window is None else t - settings.window
        if month.date < settings.start or lo < 0 or (only is not None and month.date not in only):
            continue
        clock = time.perf_counter()
        train = [k for k in range(lo, t) if known[k]]
        if settings.cold and trained:  # same starting weights, fresh Adam and average
            model.load_state_dict(start_state)
            adam = torch.optim.Adam(model.parameters(), lr=settings.lr)
            average = Average(model, settings.average) if settings.average else None
        if settings.cold:  # the month's own random streams; its model fills the book
            rng, hold_rng, noise = _streams(settings, month.date)
            _ = pricing is not None and pricing.book is not None and pricing.book.clear()
        warm = trained and not settings.cold
        epochs = settings.warm_epochs if warm else settings.first_epochs
        start_lr = settings.warm_lr if warm and settings.warm_lr else settings.lr
        watched = t if known[t] and (settings.stop or settings.watch) else -1
        losses, passes, held_loss, curve, norms, tests, diag = _fit(
            model, adam, data, train, epochs, settings, rng, pricing, start_lr=start_lr,
            span=range(lo, t), clipper=clipper, average=average, test=watched,
            hold_rng=hold_rng, raw=raw, noise=noise,
        )  # fmt: skip
        grad, limit = torch.stack(norms).T if norms else (torch.zeros(0), torch.zeros(0))
        if not trained:
            clipper.arm()  # "auto" calibrates on the steps after the cold start
        trained = True
        extra = {}
        with average.swapped() if average is not None else contextlib.nullcontext():
            if traded != t - 1 and pricing is not None and pricing.book is not None:
                last = pricing.book.get(t - 1)  # last month not traded: the model's own weights
            w = holdings.pick(model, raw, pricing, last, t)
            if cuts is not None and known[t]:  # the traded model, before the book takes w
                vol = settings.vol_column
                extra = {"opt_in": optimism.measure(model, raw, pricing, train, cuts, vol),
                         "opt_out": optimism.test_month(w, raw, pricing, t, cuts,
                                                        vol)}  # fmt: skip
        last, traded = holdings.remember(pricing, w, raw, t), t
        gross = float(np.abs(w).sum())  # 0 only if no stock is held: all cash
        weights.append(pl.DataFrame({"date": [month.date] * len(w), "symbol": month.symbols,
                                     "weight": w}))  # fmt: skip
        log.append({"date": month.date, "ret": float(w @ month.r) if known[t] else None,
                    "long": float(w[w > 0].sum()), "short": float(w[w < 0].sum()),
                    "largest": float(np.abs(w).max()) / gross if gross else None,
                    "loss": float(torch.stack(losses).mean()) if losses else math.nan,
                    "passes": passes, "held_loss": held_loss, "held_curve": curve,
                    "test_curve": tests,
                    "grad_median": float(grad.median()) if norms else math.nan,
                    "grad_max": float(grad.max()) if norms else math.nan,
                    "clipped": float((grad > limit).float().mean()) if norms else 0.0,
                    "limit": float(limit[-1]) if norms else math.nan,
                    "seconds": time.perf_counter() - clock} | diag | extra)  # fmt: skip
    return pl.concat(weights), pl.DataFrame(log), _traded(model, average)
