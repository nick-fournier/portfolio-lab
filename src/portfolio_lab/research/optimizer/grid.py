"""Quick training grid: how each setup trains on a few windows, epoch by epoch (2026-10-13).

A full walk trains ~275 months; most questions about training (does it stay stable, which
learning rate or schedule, how many epochs, does a speedup change the result) show up within
one fit. For each window end t, a setup trains a fresh model on the ``window`` months before t
with the paper's loss, one Adam step per month, months shuffled each pass, except that the last
``validation`` of those months are held out. After every epoch it records:

- ``train_loss``: mean (1 - w.r)^2 over the training months in that pass.
- ``val_loss``: the same on the held-out months (no gradient). Its lowest epoch is where more
  training starts to fit noise; ``val_sharpe`` is the held-out months' monthly Sharpe.
- ``test_ret``: month t's return (the month after the window, never trained on), scaled to
  2x gross so setups compare.
- ``row_max``: the largest row size inside the network on month t (blow-ups show here first).
- ``gross``: month t's total absolute weight; ``seconds``: the epoch's training time.
- ``val_long_sharpe``: the held-out months' monthly Sharpe of the long side alone (positive
  weights scaled to 1); ``val_short_share``: shorts' share of gross on those months;
  ``test_long_ret``: month t's long side alone at 1x.
- ``names``: month t's effective names held, 1 / sum of squared weights (long side at 1x);
  ``top_weight``: its largest weight.

With ``costs`` the loss uses the return net of each stock's spread on turnover (from the
model's weights the month before) and borrow fee, plus ``short_hurdle`` a year per dollar
short (``costs.net_return``); a large hurdle is long-only as an exact penalty. Training
months' returns above ``max_return`` in size are set to 0 (data errors), and a step whose loss
is not finite is skipped.

Carried holdings (with ``costs``): with ``carry`` the spread is charged on the dollars traded
from h, the book stored the last time month k - 1 was trained (or scored), after its stocks'
returns (w (1 + r); stocks that left are sold, new ones start at 0), instead of from the
current model's weights on month k - 1. With ``holdings_input`` each stock's row also gets
two inputs: h on this month's stocks, scaled to N x h / sum |h| (traits' scale), and
log(half-spread / 0.5%) clipped to [-3, 3] (keeps the level, so a costly month shows). The
book table starts from one in-order pass through the training months and is refreshed at
every step. With ``costs``, each epoch also scores the held-out months rolled forward in
order from the last training month's book (each month trades from the one before, drifted):
``val_net_sharpe`` and ``val_carry_sharpe`` (net and gross, annualized), and at 2x gross a
year ``val_net_ret``, ``val_gross_ret``, ``val_spread`` and ``val_borrow``, plus
``val_turnover`` (dollars traded a year per dollar of gross).

Stopping check (``inner``, 2026-10-15): a random block of consecutive training months is held
out of training and scored each epoch exactly like the validation months (rolled forward from
the month before it), as ``inner_net_loss`` and the other ``inner_*`` measures; with
``val_net_loss`` (the paper loss net of costs on the validation months, which come after the
training months) it shows whether a block inside the window tracks months the model has
truly never seen, epoch by epoch.

Training loss L of a best-scaled portfolio equals 1 / (1 + S^2), S its monthly Sharpe, so a
low training loss is an in-sample Sharpe, not a better model.
"""

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
import torch

from portfolio_lab.research.optimizer import costs as cost_model
from portfolio_lab.research.optimizer.data import Month
from portfolio_lab.research.optimizer.holdings import carried, with_holdings
from portfolio_lab.research.optimizer.model import Optimizer
from portfolio_lab.research.optimizer.stopping import held_block
from portfolio_lab.research.optimizer.walk import rate


@dataclass(frozen=True)
class Config:
    """One setup in the grid.

    Args:
        name: Label in the output.
        blocks: Transformer blocks.
        lr: Adam's learning rate (the start of the schedule).
        schedule: ``flat`` or ``cosine`` (decays from ``lr`` to ``lr_min`` over the epochs).
        lr_min: Cosine schedule's last learning rate.
        epochs: Passes over the training months.
        fast_attention: 16-bit fused attention.
        tf32: TensorFloat-32 for the other float32 matrix products.
        compile: ``torch.compile`` the model.
        stocks: Use a fixed random subset of this many stocks per month (0: all); for
            stability checks only.
        seed: Starting weights, month order and stock subset.
        clip: Largest total gradient size per step (``None``: none, the paper).
        hidden: Feed-forward hidden width.
        costs: Per-stock costs in the loss (module docs; needs the grid's ``stock_costs``).
        short_hurdle: Extra yearly cost per dollar short in the loss.
        max_return: Training returns above this in size set to 0 (``None``: kept).
        carry: Charge spreads from the stored, drifted book (module docs).
        holdings_input: Feed each stock's holding and spread to the model (module docs).
        borrow_input: Also feed each stock's log borrow fee (needs ``holdings_input``).
        inner: Hold out a random block of this many consecutive training months, never
            trained on, scored like the validation months (``inner_*``; needs ``costs``).
        weight_decay: Decoupled weight decay (AdamW): each step shrinks every parameter by
            ``lr x weight_decay`` of itself (0: plain Adam, the paper).
    """

    name: str
    blocks: int = 2
    lr: float = 1e-3
    schedule: str = "flat"
    lr_min: float = 1e-5
    epochs: int = 30
    fast_attention: bool = False
    tf32: bool = False
    compile: bool = False
    stocks: int = 0
    seed: int = 0
    clip: float | None = None
    hidden: int = 256
    costs: bool = False
    short_hurdle: float = 0.0
    max_return: float | None = None
    carry: bool = False
    holdings_input: bool = False
    borrow_input: bool = False
    inner: int = 0
    weight_decay: float = 0.0


def lr_at(config: Config, epoch: int) -> float:
    """The learning rate for ``epoch`` (0-based) under the config's schedule."""
    return rate(config.schedule, config.lr, config.lr_min, epoch, config.epochs)


def _subset(m: Month, stocks: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if not stocks or stocks >= len(m.symbols):
        return m.x, m.r
    rows = np.sort(rng.choice(len(m.symbols), stocks, replace=False))
    return m.x[rows], m.r[rows]


def _loss(w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return (1 - w @ r) ** 2


def _long(w: torch.Tensor) -> torch.Tensor:
    """The long side alone, scaled to $1 (zeros if nothing is long)."""
    pos = torch.relu(w)
    return pos / pos.sum().clamp(min=1e-12)


def _prepare(
    months: list[Month], end: int, config: Config, window: int, device: str,
    stock_costs: list | None, rng: np.random.Generator,
) -> tuple[list, list | None, list | None]:  # fmt: skip
    """The window's (x, r) tensors (training returns cleaned), cost links and per-stock costs."""
    span = months[end - window : end + 1]
    tensors = [tuple(torch.from_numpy(a).to(device) for a in _subset(m, config.stocks, rng))
               for m in span]  # fmt: skip
    if config.max_return:
        tensors[:window] = [(x, torch.where(r.abs() > config.max_return, 0.0, r))
                            for x, r in tensors[:window]]  # fmt: skip
    if not (config.costs or config.short_hurdle > 0):
        return tensors, None, None
    if config.stocks:
        raise ValueError("costs need every stock (stocks=0)")
    stock = None
    if config.costs:
        if stock_costs is None:
            raise ValueError("costs need stock_costs")
        stock = [(torch.from_numpy(c[0]).to(device), torch.from_numpy(c[1]).to(device))
                 for c in stock_costs[end - window : end + 1]]  # fmt: skip
    return tensors, cost_model.links(span, device), stock


def _measure(run, features, model: Optimizer, tensors: list, held: list[int],
             test_known: bool) -> dict:  # fmt: skip
    """The epoch's held-out and test-month measures (module docs); ``run(k)``: month k's weights."""
    test = len(tensors) - 1
    with torch.no_grad():
        ws = [run(k) for k in held]
        rs = [tensors[k][1] for k in held]
        val = torch.stack([w @ r for w, r in zip(ws, rs, strict=True)])
        val_long = torch.stack([_long(w) @ r for w, r in zip(ws, rs, strict=True)])
        r_t = tensors[test][1]
        y, row_max = features(test), 0.0
        for block in model.blocks:
            y = block(y)
            row_max = max(row_max, float(y.norm(dim=1).max()))
        w = run(test)
        gross = float(w.abs().sum())
        ok = test_known and gross > 0
        held_t = _long(w)
    sharpe = lambda v: float(v.mean() / v.std()) if len(held) > 1 else None  # noqa: E731
    return {"val_loss": float(((1 - val) ** 2).mean()), "val_sharpe": sharpe(val),
            "val_long_sharpe": sharpe(val_long),
            "val_short_share": float(torch.stack([torch.relu(-v).sum() / v.abs().sum()
                                                  for v in ws]).mean()),
            "test_ret": float(w @ r_t) / gross * 2 if ok else None,
            "test_long_ret": float(_long(w) @ r_t) if ok else None,
            "names": float(1 / (held_t**2).sum()), "top_weight": float(held_t.max()),
            "row_max": row_max, "gross": gross}  # fmt: skip


def _new_model(config: Config, width: int, device: str) -> Optimizer:
    """A fresh transformer for ``config``."""
    return Optimizer("transformer", width, config.blocks, config.hidden,
                     config.fast_attention).to(device)  # fmt: skip


def _epoch(model: Optimizer, adam, step_loss, order: list[int], clip: float | None) -> list:
    """One pass: one Adam step per month in ``order``; returns the losses of the steps taken.

    A step whose loss is not finite is skipped rather than poison Adam.
    """
    losses = []
    for k in order:
        adam.zero_grad()
        value = step_loss(int(k))
        if not torch.isfinite(value):
            continue
        value.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        adam.step()
        losses.append(value.detach())
    return losses


def _check(config: Config) -> None:
    if (config.carry or config.holdings_input or config.inner) and not config.costs:
        raise ValueError("carry, holdings_input and inner need costs")
    if config.borrow_input and not config.holdings_input:
        raise ValueError("borrow_input needs holdings_input")


def _before(config: Config, run, book: dict, tensors: list, k: int) -> torch.Tensor | None:
    """The book month k trades from: carried (stored, drifted) or the model's on month k - 1."""
    with torch.no_grad():
        if config.carry:
            return book[k - 1] * (1 + tensors[k - 1][1]) if k - 1 in book else None
        return run(k - 1) if k > 0 else None


def _fill(book: dict, run, months: int) -> None:
    """The book table's start: one in-order pass, each month's book from the one before."""
    with torch.no_grad():
        for k in range(months):
            book[k] = run(k)


def fit(
    months: list[Month], end: int, config: Config, window: int = 60, validation: int = 12,
    device: str = "cuda", stock_costs: list | None = None,
) -> list[dict]:  # fmt: skip
    """Train one fresh model for ``config`` on the window before ``months[end]`` (module docs).

    Returns one row per epoch.
    """
    torch.set_float32_matmul_precision("high" if config.tf32 else "highest")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    _check(config)
    table = config.carry or config.holdings_input
    tensors, links, stock = _prepare(months, end, config, window, device, stock_costs, rng)
    known = [bool(torch.isfinite(r).all()) for _, r in tensors]
    train = [k for k in range(window - validation) if known[k]]
    held = [k for k in range(window - validation, window) if known[k]]
    train, block = held_block(train, config.inner, rng) if config.inner else (train, [])
    width = tensors[0][0].shape[1] + 2 * config.holdings_input + config.borrow_input
    model = _new_model(config, width, device)
    scores = torch.compile(model, dynamic=True) if config.compile else model
    book: dict[int, torch.Tensor] = {}  # month -> its stored weights (carried holdings)

    def features(k: int) -> torch.Tensor:
        if not config.holdings_input:
            return tensors[k][0]
        return with_holdings(tensors, links, stock, book, k, config.borrow_input)

    def run(k: int) -> torch.Tensor:
        return scores(features(k))

    adam = (
        torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        if config.weight_decay
        else torch.optim.Adam(model.parameters(), lr=config.lr)
    )

    def net_of(k: int, w: torch.Tensor) -> torch.Tensor:
        """Month k's return net of spreads and borrow, trading from last month's book."""
        before = _before(config, run, book, tensors, k)
        spread, fee, spread_before = 0.0, 0.0, None
        if stock is not None:
            (spread, fee), spread_before = stock[k], stock[k - 1][0] if k > 0 else None
        return cost_model.net_return(w, tensors[k][1], before, links[k], spread,
                                     fee + config.short_hurdle, spread_before)  # fmt: skip

    def step_loss(k: int) -> torch.Tensor:
        w = run(k)
        if table:
            book[k] = w.detach()
        if links is None:
            return _loss(w, tensors[k][1])
        return (1 - net_of(k, w)) ** 2

    if table:
        _fill(book, run, window - validation)
    rows = []
    for epoch in range(config.epochs):
        for group in adam.param_groups:
            group["lr"] = lr_at(config, epoch)
        clock = time.perf_counter()
        losses = _epoch(model, adam, step_loss, rng.permutation(train), config.clip)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - clock
        scored = carried(
            run, tensors, held, links, stock, book if table else None, config.short_hurdle
        )
        if block:
            inner = carried(run, tensors, block, links, stock, book if table else None,
                            config.short_hurdle)  # fmt: skip
            scored |= {"inner" + key[3:]: value for key, value in inner.items()}
        rows.append({"config": config.name, "end": months[end].date, "epoch": epoch + 1,
                     "lr": lr_at(config, epoch),
                     "train_loss": float(torch.stack(losses).mean()) if losses else math.nan,
                     **scored, **_measure(run, features, model, tensors, held, known[window]),
                     "seconds": seconds})  # fmt: skip
        if not math.isfinite(rows[-1]["train_loss"]):
            break
    return rows


def grid(
    months: list[Month], ends: list[int], configs: list[Config], window: int = 60,
    validation: int = 12, device: str = "cuda", stock_costs: list | None = None,
) -> pl.DataFrame:  # fmt: skip
    """Every config on every window end; one row per config, window and epoch."""
    rows = []
    for config in configs:
        for end in ends:
            rows += fit(months, end, config, window, validation, device, stock_costs)
    return pl.DataFrame(rows)


def summary(curves: pl.DataFrame) -> pl.DataFrame:
    """Per config, averaged over windows: the best-validation epoch and what it gave.

    ``best_epoch``: the epoch of lowest validation loss (median over windows); ``val_loss``,
    ``train_loss`` and ``test_ret`` at that epoch; ``test_ret_last`` at the last epoch;
    ``blew_up``: windows whose training loss went non-finite or above 1,000.
    """
    best = (
        curves.filter(pl.col("val_loss").is_finite())
        .sort("val_loss")
        .group_by("config", "end", maintain_order=True)
        .first()
    )
    order = curves.select("config").unique(maintain_order=True).with_row_index("order")
    last = curves.sort("epoch").group_by("config", "end").last()
    blew = curves.group_by("config", "end").agg(
        (~pl.col("train_loss").is_finite() | (pl.col("train_loss") > 1e3)).any().alias("blew")
    )
    return (
        best.join(
            last.select("config", "end", pl.col("test_ret").alias("test_ret_last")),
            on=["config", "end"],
        )
        .join(blew, on=["config", "end"])
        .group_by("config", maintain_order=True)
        .agg(
            pl.col("epoch").median().alias("best_epoch"),
            pl.col("val_loss").mean(),
            pl.col("train_loss").mean(),
            pl.col("test_ret").mean(),
            pl.col("test_ret_last").mean(),
            pl.col("blew").sum().alias("blew_up"),
        )
        .join(
            curves.group_by("config").agg(pl.col("seconds").median().alias("s_per_epoch")),
            on="config",
        )
        .join(order, on="config")
        .sort("order")
        .drop("order")
    )


def configs_from(spec: list[dict]) -> list[Config]:
    """Configs from a list of dicts (a JSON file); unknown keys are an error."""
    known = set(asdict(Config("x")))
    for item in spec:
        extra = set(item) - known
        if extra:
            raise ValueError(f"unknown config keys {sorted(extra)} in {item}")
    return [Config(**item) for item in spec]
