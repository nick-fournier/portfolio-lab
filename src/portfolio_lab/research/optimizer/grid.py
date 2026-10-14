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

With ``long_only`` the weights are the softmax of the scores (positive, summing to 1: fully
invested, no shorts, no borrowing) and the loss is computed on them, so training sees only
such portfolios (model.Optimizer ``long_only``).

``size_cost`` instead keeps the paper's weights (w = Y lam) and charges that much per month per
dollar the total weight is away from 1, kappa |sum w - 1|, inside the net return: with a large
kappa (above any month's gain from more or less size) the portfolio is fully invested as an
exact penalty, not a clamp. With a large ``short_hurdle`` it is long-only at size 1.

With ``costs`` the loss uses the return net of each stock's spread on turnover (from the
model's weights the month before) and borrow fee, plus ``short_hurdle`` a year per dollar
short (``costs.net_return``); a large hurdle is long-only as an exact penalty. Training
months' returns above ``max_return`` in size are set to 0 (data errors), and a step whose loss
is not finite is skipped.

``objective="growth"`` replaces the paper's pricing loss with an investor's: maximize the
average of U(1 + R_net) over the training months, U = log (Kelly) or, with ``gamma`` above 1,
((1 + R)^(1 - gamma) - 1) / (1 - gamma). R_net is the month's return over the T-bill net of
spreads on the dollars traded from h, last month's weights after their prices moved (cash earns
the T-bill, 0 here), minus ``short_hurdle`` a year per dollar short and ``over_cost`` a month per
dollar invested above the account: both exact penalties, so the book is long-only and at most
fully invested. The model starts in cash (lam = 0) so log(1 + R) is defined from the first step,
and its weights are the scores divided by the month's number of stocks, w = Y lam / N (the
paper's linear portfolio carries the same 1 / N): without it one Adam step from cash moves the
total weight by ~N times the step and the first months lose more than everything.
Each epoch also records ``val_growth`` (held-out months, before costs) and ``val_net_growth``
(after costs), both as growth a year, exp(12 x mean log(1 + R)) - 1, plus ``val_cost`` (a
year) and ``cash`` (1 - sum w on month t).

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
from portfolio_lab.research.optimizer.model import Optimizer


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
        init: ``std`` or ``variance`` reading of the paper's initialization (model docs).
        fast_attention: 16-bit fused attention.
        tf32: TensorFloat-32 for the other float32 matrix products.
        compile: ``torch.compile`` the model.
        stocks: Use a fixed random subset of this many stocks per month (0: all); for
            stability checks only.
        seed: Starting weights, month order and stock subset.
        clip: Largest total gradient size per step (``None``: none, the paper).
        scaled_attention: Divide attention scores by sqrt(D) (model.Block).
        hidden: Feed-forward hidden width.
        costs: Per-stock costs in the loss (module docs; needs the grid's ``stock_costs``).
        short_hurdle: Extra yearly cost per dollar short in the loss.
        max_return: Training returns above this in size set to 0 (``None``: kept).
        long_only: Weights = softmax of the scores (module docs).
        size_cost: Monthly cost per dollar the total weight is away from 1 (module docs).
        objective: ``paper`` (1 - R)^2 or ``growth`` (module docs).
        gamma: Caution for ``growth`` (1 = log, Kelly).
        over_cost: Monthly cost per dollar invested above the account (``growth``).
    """

    name: str
    blocks: int = 2
    lr: float = 1e-3
    schedule: str = "flat"
    lr_min: float = 1e-5
    epochs: int = 30
    init: str = "std"
    fast_attention: bool = False
    tf32: bool = False
    compile: bool = False
    stocks: int = 0
    seed: int = 0
    clip: float | None = None
    scaled_attention: bool = False
    hidden: int = 256
    costs: bool = False
    short_hurdle: float = 0.0
    max_return: float | None = None
    long_only: bool = False
    size_cost: float = 0.0
    objective: str = "paper"
    gamma: float = 1.0
    over_cost: float = 0.0


def lr_at(config: Config, epoch: int) -> float:
    """The learning rate for ``epoch`` (0-based) under the config's schedule."""
    if config.schedule == "flat":
        return config.lr
    if config.schedule == "cosine":
        done = epoch / max(config.epochs - 1, 1)
        return config.lr_min + (config.lr - config.lr_min) * 0.5 * (1 + math.cos(math.pi * done))
    raise ValueError(f"schedule must be flat or cosine, not {config.schedule!r}")


def _subset(m: Month, stocks: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if not stocks or stocks >= len(m.symbols):
        return m.x, m.r
    rows = np.sort(rng.choice(len(m.symbols), stocks, replace=False))
    return m.x[rows], m.r[rows]


def _loss(w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return (1 - w @ r) ** 2


def _utility(g: torch.Tensor, gamma: float) -> torch.Tensor:
    """Growth score of wealth multiple ``g``: log at gamma 1, else constant relative risk aversion.

    A month losing everything or more (``g`` <= 0) scores non-finite and its step is skipped.
    """
    return torch.log(g) if gamma == 1 else (g ** (1 - gamma) - 1) / (1 - gamma)


def _drift(w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Weights after a month of returns ``r`` (cash earns 0 over the T-bill)."""
    return w * (1 + r) / (1 + w @ r)


def _growth(net: torch.Tensor) -> float:
    """Monthly returns as growth a year, exp(12 x mean log(1 + R)) - 1."""
    return float(torch.expm1(12 * torch.log1p(net).mean()))


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
    priced = config.costs or config.short_hurdle > 0 or config.size_cost > 0
    if not (priced or config.objective == "growth"):
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


def _measure(run, model: Optimizer, tensors: list, held: list[int], test_known: bool) -> dict:
    """The epoch's held-out and test-month measures (module docs)."""
    with torch.no_grad():
        ws = [run(tensors[k][0]) for k in held]
        rs = [tensors[k][1] for k in held]
        val = torch.stack([w @ r for w, r in zip(ws, rs, strict=True)])
        val_long = torch.stack([_long(w) @ r for w, r in zip(ws, rs, strict=True)])
        x_t, r_t = tensors[-1]
        y, row_max = x_t, 0.0
        for block in model.blocks:
            y = block(y)
            row_max = max(row_max, float(y.norm(dim=1).max()))
        w = run(x_t)
        gross = float(w.abs().sum())
        ok = test_known and gross > 0
        held_t = _long(w)
        growth = _growth(val) if bool((val > -1).all()) else None
    sharpe = lambda v: float(v.mean() / v.std()) if len(held) > 1 else None  # noqa: E731
    return {"val_loss": float(((1 - val) ** 2).mean()), "val_sharpe": sharpe(val),
            "val_long_sharpe": sharpe(val_long),
            "val_short_share": float(torch.stack([torch.relu(-v).sum() / v.abs().sum()
                                                  for v in ws]).mean()),
            "test_ret": float(w @ r_t) / gross * 2 if ok else None,
            "test_long_ret": float(_long(w) @ r_t) if ok else None,
            "names": float(1 / (held_t**2).sum()), "top_weight": float(held_t.max()),
            "val_growth": growth, "cash": float(1 - w.sum()),
            "row_max": row_max, "gross": gross}  # fmt: skip


def _new_model(config: Config, width: int, device: str, cash: bool) -> Optimizer:
    """A fresh transformer for ``config``; with ``cash`` it starts holding nothing (lam = 0)."""
    model = Optimizer("transformer", width, config.blocks, config.hidden,
                      fast_attention=config.fast_attention, init=config.init,
                      scaled_attention=config.scaled_attention).to(device)  # fmt: skip
    if cash:
        with torch.no_grad():
            model.lam.zero_()
    return model


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
    if config.objective not in {"paper", "growth"}:
        raise ValueError(f"objective must be paper or growth, not {config.objective!r}")
    growth = config.objective == "growth"
    tensors, links, stock = _prepare(months, end, config, window, device, stock_costs, rng)
    known = [bool(torch.isfinite(r).all()) for _, r in tensors]
    train = [k for k in range(window - validation) if known[k]]
    held = [k for k in range(window - validation, window) if known[k]]
    model = _new_model(config, tensors[0][0].shape[1], device, cash=growth)
    scores = torch.compile(model, dynamic=True) if config.compile else model

    def run(x: torch.Tensor) -> torch.Tensor:
        if config.long_only:
            return torch.softmax(scores(x), dim=0)
        return scores(x) / len(x) if growth else scores(x)

    adam = torch.optim.Adam(model.parameters(), lr=config.lr)

    def net_of(k: int, w: torch.Tensor) -> torch.Tensor:
        """Month k's return net of spreads and penalties, trading from last month's book."""
        with torch.no_grad():
            before = run(tensors[k - 1][0]) if k > 0 else None
            if growth and before is not None:
                before = _drift(before, tensors[k - 1][1])
        spread, fee, spread_before = 0.0, 0.0, None
        if stock is not None:
            (spread, fee), spread_before = stock[k], stock[k - 1][0] if k > 0 else None
        net = cost_model.net_return(w, tensors[k][1], before, links[k], spread,
                                    fee + config.short_hurdle, spread_before)  # fmt: skip
        return net - config.over_cost * torch.relu(w.sum() - 1)

    def step_loss(k: int) -> torch.Tensor:
        w = run(tensors[k][0])
        if links is None:
            return _loss(w, tensors[k][1])
        if growth:
            return -_utility(1 + net_of(k, w), config.gamma)
        return (1 - net_of(k, w) + config.size_cost * (w.sum() - 1).abs()) ** 2

    def held_net() -> dict:
        """Held-out months after costs (growth objective only)."""
        if not (growth and held):
            return {}
        with torch.no_grad():
            ws = [run(tensors[k][0]) for k in held]
            net = torch.stack([net_of(k, w) for k, w in zip(held, ws, strict=True)])
            gross = torch.stack([w @ tensors[k][1] for k, w in zip(held, ws, strict=True)])
        ok = bool((net > -1).all())
        return {"val_net_growth": _growth(net) if ok else None,
                "val_cost": float(12 * (gross - net).mean())}  # fmt: skip

    rows = []
    for epoch in range(config.epochs):
        for group in adam.param_groups:
            group["lr"] = lr_at(config, epoch)
        clock = time.perf_counter()
        losses = _epoch(model, adam, step_loss, rng.permutation(train), config.clip)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - clock
        rows.append({"config": config.name, "end": months[end].date, "epoch": epoch + 1,
                     "lr": lr_at(config, epoch),
                     "train_loss": float(torch.stack(losses).mean()) if losses else math.nan,
                     **_measure(run, model, tensors, held, known[window]), **held_net(),
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
