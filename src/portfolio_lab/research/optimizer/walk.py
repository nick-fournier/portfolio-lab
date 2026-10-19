"""Walk-forward training: at each month end, train on earlier months, then pick the portfolio.

At month end t the training months are the ``window`` months before t (all earlier months if
``window`` is ``None``), whose next-month returns are known. Each pass visits them in a
shuffled order and takes one Adam step per month (paper, Algorithm 2), the rate following
``schedule``. Then the model's weights for month t's candidates are held over the next month.

With ``cold`` (2026-10-15) every month trains a fresh model: the seed's same starting weights,
a fresh Adam and average, ``first_epochs`` passes from ``lr``, an empty book table (so its
training months' holdings inputs come from its own model) and random streams of its own (the
seed and the month) for the month order and the held-out draw, so a month's result does not
depend on which months ran before it. Without ``cold`` (warm continuation): the first test
month trains a fresh model for ``first_epochs`` passes, and every later month continues from
the previous month's model (and Adam's state) for ``warm_epochs`` passes from ``warm_lr``
(``None``: ``lr``) over its own training months.

With ``watch`` (research.optimizer.stopping) ``holdout`` training months are held out each
month and scored after every pass (``held_curve``; ``held_returns``: their returns at the last
score); every pass runs and the last is kept, unless ``keep_best`` (2026-10-19): then the
score is minus the held-out ranking (of the parameter average when there is one), the pass
with the best smoothed score is kept (``best_pass``) and training stops after ``keep_patience``
passes without a new best. ``scatter`` holds out single months drawn anywhere in the window
instead of one block; ``holdout_seed`` draws them from their own stream. Before each score the
held-out months' book entries are rolled forward with the model as it stands. ``test_curve``
is the test month's loss after each pass. A warm model has already trained on most of this
month's held-out months (they move every month), so its score is not out of sample.

With per-stock ``costs`` the loss is (1 - net)^2, net the month's return after each stock's
spread on the dollars traded from the book held going in (the month before's weights after its
returns) and its borrow fee on shorts (research.optimizer.priced), less the account's
size-dependent charges when any realism term is on (research.optimizer.realism; with
``trade_margin`` each step also pays the traded month's penalty).

With ``holdings_input`` (2026-10-15; needs ``costs``) each stock's row also gets its holding
and spread (research.optimizer.holdings): in training, from a book table of each month's
weights (cash, an all-zero book, the first time a month enters the window; the model's weights
from its first training step on, refreshed at every step; holdings.fill); for the test month,
from the walk's real holdings of the month before after its returns. A training month whose
month before is not in the table or not yet visited holds cash going in (priced.held_going_in).
``borrow_input`` adds the stock's borrow fee as well. When the month before was not traded
(the walk's first month, or with ``only`` a month after one not listed) the test month starts
from ``start_book``: nothing (``cash``), the market (realism.start_book), or with ``guess`` the
model's own book-table entry for that month, as its training left it.

``winsor`` pulls training returns in to each month's quantiles and ``stock_drop`` hides a share
of each step's stocks from the loss; held-out and test months are scored on real returns.
``average``: a moving average of the model's parameters is what trades. ``trade_gross``
(2026-10-18) sizes the picked book (realism.trade_size); ``loss_gross``: Settings.
"""

import contextlib
import copy
import math
import time
from dataclasses import dataclass, fields
from datetime import date

import numpy as np
import polars as pl
import torch

from portfolio_lab.research.optimizer import holdings, priced, realism, stopping, walkbits
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
        first_epochs: Passes over the training months for a fresh model (every month with
            ``cold``; the first test month without it).
        warm_epochs: Passes for every later test month, continuing from the last model
            (without ``cold``).
        lr: Adam's learning rate.
        seed: Random seed (initial weights and month order).
        start: First test month end.
        fast_attention: bfloat16 fused attention (model.Block).
        clip: Largest gradient size per step: a number, ``"auto"`` (clipping) or ``None``.
        costs: Path of the per-stock cost table (``costs.align``) the walk gets as ``stock_costs``.
        schedule: Rate within a month's training: ``flat`` or ``cosine`` (down to ``lr_min``).
        lr_min: The cosine schedule's last rate.
        warm_lr: Starting rate for months that continue last month's model (``None``: ``lr``).
        compile: ``torch.compile`` each model (faster; same results).
        winsor: Training months' returns are pulled in to their month's ``winsor`` and
            1 - ``winsor`` quantiles (0.01: the 1st and 99th percentiles; 0: off, the paper).
            Everything that is not a training step stays real: held-out and test scores, trades.
        holdings_input / borrow_input: Feed each stock's holding and spread / also its borrow fee.
        holdout: Training months held out each month with ``watch``.
        average: Decay of the parameter average held each month (0: hold the latest model).
        cold: Train a fresh model every month (module docs).
        watch: Hold out and score ``holdout`` months every pass; every pass runs (module docs).
        keep_best / keep_patience: With ``watch``: score the held-out months by ranking, keep
            the best pass and stop after this many passes without a new best (stopping.Keeper).
        scatter / holdout_seed: Draw single months anywhere in the window; the draw's own seed.
        holdout_every: Hold out every this-many-th month end for good (0: draw them instead).
        only: Train and trade only these test month ends (``None``: every month).
        stock_drop: Share of each step's stocks hidden from the loss (returns 0, rest scaled up).
        rebate / margin_rate / cash_rate / margin_cushion / cushion_rate / impact / equity: The
            account's size-dependent costs in the loss (research.optimizer.realism; need
            ``costs`` and the walk's ``market``): lost short rebate on/off; margin rate a year
            (0: off); idle cash's rate a year (``None``: the T-bill); maintenance requirement
            allowed before it is charged, as a share of equity (0: off), and the charge a month
            per dollar above it (also for ``gross_limit`` and ``invest_floor``); impact
            constant K (0: off); account size in dollars (for impact).
        gross_limit: Gross (sum |w|) allowed before it is charged like the cushion (0: off).
        invest_floor: Longs below this are charged like the cushion (0: off; long priority).
        max_weight / short_cap / max_names: The box (2026-10-19): each name's dollars above
            the cap; short dollars above the cap (0: long-only; ``None``: off); the effective
            number of names above the cap (realism.penalty); all charged like the cushion.
        start_book: What a traded month starts from when nothing was traded the month before:
            ``guess`` (the model's own weights), ``cash`` or ``market`` (realism.start_book).
        trade_margin: Each training step also pays the traded month's penalty (realism).
        trade_gross / trade_collateral: The traded book's size at trade time: gross (sum |w|)
            ``trade_gross`` (0: off), or less if its maintenance requirement would pass
            ``trade_collateral`` x equity (0: no cap); realism.trade_size (needs ``market``).
        price_gross: Margin interest, idle cash and impact in the loss are priced on the book
            scaled to this gross, per dollar of its gross (0: at its own size); realism.charges.
        holding_column / borrow_flag / spread_rank / holding_rank: The extra columns'
            encodings (2026-10-18; holdings.with_holdings): the holding column on/off; the borrow
            fee as a centred hard-to-borrow flag; the spread, or the holding, as its rank.
        loss_gross / risk_aversion / size_anchor: The loss is taken on the book sized to
            ``loss_gross`` (= ``trade_gross``, capped by ``trade_collateral``) as expected return
            less ``risk_aversion`` / 2 times its square, plus ``size_anchor`` times the squared
            log of the model's own gross over ``loss_gross`` (priced; 0: the paper's loss).
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
    schedule: str = "flat"
    lr_min: float = 1e-5
    warm_lr: float | None = None
    compile: bool = False
    winsor: float = 0.0
    holdings_input: bool = False
    borrow_input: bool = False
    holdout: int = 12
    average: float = 0.0
    cold: bool = False
    watch: bool = False
    scatter: bool = False
    holdout_seed: int | None = None
    only: tuple[date, ...] | None = None
    stock_drop: float = 0.0
    rebate: bool = False
    margin_rate: float = 0.0
    cash_rate: float | None = None
    margin_cushion: float = 0.0
    cushion_rate: float = 0.10
    gross_limit: float = 0.0
    invest_floor: float = 0.0
    trade_margin: bool = False
    start_book: str = "guess"
    impact: float = 0.0
    equity: float = 100_000.0
    trade_gross: float = 0.0
    trade_collateral: float = 0.5
    price_gross: float = 0.0
    holding_column: bool = True
    borrow_flag: bool = False
    spread_rank: bool = False
    holding_rank: bool = False
    loss_gross: float = 0.0
    risk_aversion: float = 1.0
    size_anchor: float = 0.0
    keep_best: bool = False
    keep_patience: int = 4
    holdout_every: int = 0
    max_weight: float = 0.0
    short_cap: float | None = None
    max_names: int = 0


#: Settings removed in the cleanup of 2026-10-18, each with the value that left it off: an
#: older run's settings.json still loads (:func:`settings_from`) when it left them off.
REMOVED = {"short_hurdle": 0.0, "cost_scale": 1.0, "max_return": None, "ridge": 0.0,
           "ridge_tilt": None, "stop": False, "patience": 5, "smooth": 3, "real_book": False,
           "dropout": 0.0, "inherit": 0.0, "margin_spread": 0.0, "cushion_curve": 0.0,
           "margin_input": False, "size_bound": False, "gross_cap": 1.25,
           "margin_cap": 0.5}  # fmt: skip
#: Removed settings that only tuned another removed one: with that one off they did nothing
#: (``smooth`` also set how the logged held-out loss was smoothed; the weights never saw it).
TUNES = {"ridge_tilt": "ridge", "patience": "stop", "smooth": "stop", "gross_cap": "size_bound",
         "margin_cap": "size_bound"}  # fmt: skip
#: Removed settings that only fed the log, dropped whatever their value.
LOG_ONLY = ("measure_optimism", "vol_column")
#: What ``plab optimizer walk`` adds to a run's settings.json beside the settings.
EXTRAS = ("inputs", "end", "floors")


def settings_from(meta: dict) -> Settings:
    """A run's :class:`Settings` from its settings.json (as ``plab optimizer walk`` writes it).

    Lists come back as tuples, ``start`` and ``only`` as dates. Dropped: the file's extras
    (:data:`EXTRAS`), log-only settings (:data:`LOG_ONLY`), a removed setting (:data:`REMOVED`)
    at its off value or tuning one that is (:data:`TUNES`). Raises ValueError for a removed
    setting that was on (the recipe cannot be rebuilt) or a key that is not a setting at all.
    """
    names = {f.name for f in fields(Settings)}

    def off(key: str) -> bool:
        return meta.get(key, REMOVED[key]) == REMOVED[key]

    kw = {}
    for key, value in meta.items():
        if key in names:
            kw[key] = tuple(value) if isinstance(value, list) else value
        elif key in REMOVED:
            if not (off(key) or (key in TUNES and off(TUNES[key]))):
                raise ValueError(f"{key}={value!r} was removed (cleanup 2026-10-18); this run "
                                 f"cannot be rebuilt")  # fmt: skip
        elif key not in LOG_ONLY + EXTRAS:
            raise ValueError(f"{key!r} is not a walk setting (a typo, or removed before "
                             f"2026-10-18)")  # fmt: skip
    if isinstance(kw.get("start"), str):
        kw["start"] = date.fromisoformat(kw["start"])
    if kw.get("only"):
        kw["only"] = tuple(d if isinstance(d, date) else date.fromisoformat(d) for d in kw["only"])
    return Settings(**kw)


Pair = tuple[torch.Tensor, torch.Tensor]


def _data(months: list[Month], settings: Settings, device: str) -> tuple[list[Pair], list[Pair]]:
    """Each month's (x, r) tensors for training, and the same with real returns.

    Training returns are pulled in to the month's ``settings.winsor`` quantiles (months whose
    returns are not all known are left as they are). The real pairs share the inputs and score
    the held-out and test months.
    """
    raw = [(torch.from_numpy(m.x).to(device), torch.from_numpy(m.r).to(device)) for m in months]
    data = raw
    if settings.winsor:
        q, known = settings.winsor, [len(r) and bool(torch.isfinite(r).all()) for _, r in raw]
        data = [(x, r.clamp(torch.quantile(r, q), torch.quantile(r, 1 - q))) if ok else (x, r)
                for (x, r), ok in zip(raw, known, strict=True)]  # fmt: skip
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
    trade: tuple | None = None,
) -> list[torch.Tensor]:  # fmt: skip
    """One pass over the ``train`` months in shuffled order at rate ``lr``; its losses.

    Each step's (gradient size before clipping, limit) goes on ``norms``; a step whose loss is
    not finite is skipped. ``trade``: (traded month, its real held book or ``None``): with
    ``trade_margin`` each step also pays that month's penalty (research.optimizer.realism).
    """
    for group in adam.param_groups:
        group["lr"] = lr
    losses = []  # kept on the device: reading each one would wait for the GPU
    for k in rng.permutation(train):
        adam.zero_grad()
        x, r = data[k]
        seen = data  # with stock_drop: this step's month with the hidden stocks' returns at 0
        if settings.stock_drop:
            kept = torch.from_numpy(rng.random(len(r)) >= settings.stock_drop).to(r)
            r = r * kept / (1 - settings.stock_drop)
            seen = [*data[:k], (x, r), *data[k + 1 :]]
        value = (loss(model, x, r) if pricing is None
                 else priced.loss(model, seen, pricing, settings, int(k)))  # fmt: skip
        if settings.trade_margin and pricing is not None and trade is not None:
            value = value + priced.trade_penalty(model, data, pricing, settings, *trade)
        if not torch.isfinite(value):
            continue
        value.backward()
        norms.append(clipper(model.parameters()))
        adam.step()
        if average is not None:
            average.update()
        losses.append(value.detach())
    return losses


def _held_loss(model: Optimizer, data: list[Pair], pricing: priced.Pricing | None,
               settings: Settings, months: list[int],
               keep: list | None = None) -> float:  # fmt: skip
    """Mean loss over ``months`` (net of costs with ``pricing``; their returns into ``keep``)."""
    with torch.no_grad():
        r = []
        for k in months:
            if pricing is None:
                r.append(realism.book(model, holdings.inputs(data, None, None, k), None, k)
                         @ data[k][1])  # fmt: skip
            else:
                w = priced.book_for(model, data, pricing, settings, k)
                r.append(priced.net(model, data, pricing, settings, k, w))
        r = torch.stack(r)
    if keep is not None:
        keep[:] = r.tolist()
    if pricing is not None and settings.loss_gross:
        return float(priced.objective(r, settings).mean())
    return float(((1 - r) ** 2).mean())


def _roll(model: Optimizer, data: list[Pair], pricing: priced.Pricing | None, settings: Settings,
          months: list[int]) -> None:  # fmt: skip
    """With a book, ``months``' entries become the model's weights, in month order."""
    if pricing is None or pricing.book is None:
        return
    with torch.no_grad():
        for k in sorted(months):
            pricing.book[k] = priced.book_for(model, data, pricing, settings, k)


def _fit(  # noqa: PLR0913 - one month's training pieces
    model: Optimizer, adam: torch.optim.Optimizer, data: list[Pair], train: list[int],
    epochs: int, settings: Settings, rng: np.random.Generator, pricing: priced.Pricing | None, *,
    start_lr: float, span: range, clipper: Clipper, average: Average | None, test: int,
    hold_rng: np.random.Generator, raw: list[Pair], trade: tuple | None = None,
) -> tuple[list, int, float, list[float], list[torch.Tensor], list[float], dict]:  # fmt: skip
    """One month's training.

    Returns the last pass's losses, passes run, best smoothed held-out score, its per-pass
    curve (research.optimizer.stopping; with ``keep_best`` minus the held-out ranking), every
    step's (gradient size before clipping, limit), the test month ``test``'s loss after each
    pass (``test`` < 0: none) and for the log the passes rolled back (:class:`.stopping.Guard`),
    the held-out months' returns at the last score (month order) and the pass kept (``None``
    without ``keep_best``). Up to ``epochs`` passes, the rate following ``settings.schedule``
    from ``start_lr``. With a book (``holdings_input``) the ``span`` months not in it get cash
    first (holdings.fill). The held-out months (``watch``) are drawn from ``hold_rng``; they
    and the test month are scored on their real returns (``raw``), training alone sees
    ``data``'s clipped ones. ``trade``: the traded month's penalty pieces (:func:`_pass`).
    """
    if pricing is not None:
        holdings.fill(data, pricing, span)
    fit, block = (stopping.held_months(train, settings.holdout, hold_rng, settings.scatter,
                                       settings.holdout_every)
                  if settings.watch else (train, []))  # fmt: skip
    tests: list[float] = []
    held: list[float] = []

    def score() -> float:
        _roll(model, raw, pricing, settings, block)
        if test >= 0:
            tests.append(_held_loss(model, raw, pricing, settings, [test]))
        value = _held_loss(model, raw, pricing, settings, block, held)
        if not settings.keep_best:
            return value
        with average.swapped() if average is not None else contextlib.nullcontext():
            return -priced.held_ranking(model, raw, pricing, settings, block)

    norms: list[torch.Tensor] = []
    guard = stopping.Guard(model, adam, average, pricing.book if pricing is not None else None)
    keeper = (stopping.Keeper(model, adam, average, settings.keep_patience)
              if settings.keep_best else None)  # fmt: skip

    def one_pass(p: int) -> list:
        lr = rate(settings.schedule, start_lr, settings.lr_min, p, epochs) * guard.scale
        losses = _pass(model, adam, data, fit, settings, rng, pricing, lr, clipper, norms,
                       average, trade)  # fmt: skip
        if not block and settings.watch and test >= 0:  # nothing held out: still log the test
            tests.append(_held_loss(model, raw, pricing, settings, [test]))
        return losses

    return (*stopping.fit_watched(lambda p: guard(one_pass, p), score, bool(block), epochs, keeper),
            norms, tests, {"rollbacks": guard.rollbacks, "held_returns": held,
                           "best_pass": None if keeper is None else keeper.best_pass})  # fmt: skip


def _new_model(settings: Settings, width: int, device: str) -> tuple[Optimizer, torch.optim.Adam]:
    """A freshly initialized model and its Adam; the holdings inputs add their columns."""
    torch.manual_seed(settings.seed)
    extra = settings.holdings_input * (1 + settings.holding_column) + settings.borrow_input
    model = Optimizer(settings.model, width + extra, settings.blocks, settings.hidden,
                      settings.fast_attention)  # fmt: skip
    model.to(device)
    if settings.compile:
        model.compile(dynamic=True)
    return model, torch.optim.Adam(model.parameters(), lr=settings.lr)


def _streams(settings: Settings, day: date) -> tuple[np.random.Generator, np.random.Generator]:
    """A cold month's own (month order, held-out draw) random streams."""
    rng = np.random.default_rng([settings.seed, day.toordinal()])
    if settings.holdout_seed is None:
        return rng, rng
    return rng, np.random.default_rng([settings.holdout_seed, day.toordinal()])


def walk(
    months: list[Month],
    settings: Settings,
    device: str = "cpu",
    stock_costs: list[tuple[np.ndarray, ...]] | None = None,
    market: list | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, Optimizer]:
    """Train and pick portfolios month by month (module docs).

    Args:
        months: From :func:`.data.months`, ascending.
        settings: The walk's settings.
        device: ``cuda`` or ``cpu``.
        stock_costs: Each month's ``(trade_cost, borrow_fee, ...)`` per candidate
            (``costs.align``), needed when ``settings.costs`` is set.
        market: Each month's ``realism.Market`` (``realism.align``), needed with any realism
            term.

    Returns:
        ``weights`` (date, symbol, weight) for every test month; ``log`` (date, ret = the
        portfolio's next-month return over the T-bill before costs, ``long`` and ``short`` =
        the sums of the positive and negative weights, ``largest`` = the largest absolute
        weight's share of gross, mean training loss over the last pass, passes run, best
        smoothed held-out score and its per-pass curve with ``watch`` (minus the held-out
        ranking with ``keep_best``, and ``best_pass`` the pass kept), the median and largest
        gradient size over the month's steps before clipping, the share of steps clipped and
        the last step's limit, seconds, ``model_gross`` and ``scale``, realism.trade_size); and
        the final model (with ``average``, its averaged parameters: the one that traded last).
    """
    if settings.keep_best and not settings.watch:
        raise ValueError("keep_best needs watch: held-out months to score each pass")
    rng, hold_rng = walkbits.walk_streams(settings)
    clipper = Clipper(settings.clip)
    model, adam = _new_model(settings, months[0].x.shape[1], device)
    start_state = copy.deepcopy(model.state_dict()) if settings.cold else None
    average = Average(model, settings.average) if settings.average else None
    data, raw = _data(months, settings, device)
    known = [bool(np.isfinite(m.r).all()) for m in months]
    pricing = priced.pricing(months, settings, device, stock_costs, market)
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
        if settings.cold:  # the month's own random streams; an empty book, filled with cash
            rng, hold_rng = _streams(settings, month.date)
            _ = pricing is not None and pricing.book is not None and pricing.book.clear()
        warm = trained and not settings.cold
        epochs = settings.warm_epochs if warm else settings.first_epochs
        start_lr = settings.warm_lr if warm and settings.warm_lr else settings.lr
        watched = t if known[t] and settings.watch else -1
        last, traded = walkbits.starting(settings, pricing, months, data, t, last, traded)
        losses, passes, held_loss, curve, norms, tests, diag = _fit(
            model, adam, data, train, epochs, settings, rng, pricing, start_lr=start_lr,
            span=range(lo, t), clipper=clipper, average=average, test=watched,
            hold_rng=hold_rng, raw=raw, trade=(t, last if traded == t - 1 else None),
        )  # fmt: skip
        grad, limit = torch.stack(norms).T if norms else (torch.zeros(0), torch.zeros(0))
        if not trained:
            clipper.arm()  # "auto" calibrates on the steps after the cold start
        trained = True
        with average.swapped() if average is not None else contextlib.nullcontext():
            if traded != t - 1 and pricing is not None and pricing.book is not None:
                last = pricing.book.get(t - 1)  # last month not traded: the model's own weights
            w = holdings.pick(model, raw, pricing, last, t)
        w, size = realism.trade_size(w, market[t] if settings.trade_gross else None, settings)
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
                    "seconds": time.perf_counter() - clock} | diag | size)  # fmt: skip
    return pl.concat(weights), pl.DataFrame(log), walkbits.traded(model, average)
