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

Training loss L of a best-scaled portfolio equals 1 / (1 + S^2), S its monthly Sharpe, so a
low training loss is an in-sample Sharpe, not a better model.
"""

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
import torch

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


def fit(
    months: list[Month], end: int, config: Config, window: int = 60, validation: int = 12,
    device: str = "cuda",
) -> list[dict]:  # fmt: skip
    """Train one fresh model for ``config`` on the window before ``months[end]`` (module docs).

    Returns one row per epoch.
    """
    torch.set_float32_matmul_precision("high" if config.tf32 else "highest")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    tensors = [
        tuple(torch.from_numpy(a).to(device) for a in _subset(m, config.stocks, rng))
        for m in months[end - window : end + 1]
    ]
    known = [bool(torch.isfinite(r).all()) for _, r in tensors]
    train = [k for k in range(window - validation) if known[k]]
    held = [k for k in range(window - validation, window) if known[k]]
    x_t, r_t = tensors[window]
    model = Optimizer("transformer", x_t.shape[1], config.blocks, config.hidden,
                      fast_attention=config.fast_attention, init=config.init,
                      scaled_attention=config.scaled_attention).to(device)  # fmt: skip
    run = torch.compile(model, dynamic=True) if config.compile else model
    adam = torch.optim.Adam(model.parameters(), lr=config.lr)
    rows = []
    for epoch in range(config.epochs):
        for group in adam.param_groups:
            group["lr"] = lr_at(config, epoch)
        clock = time.perf_counter()
        losses = []
        for k in rng.permutation(train):
            adam.zero_grad()
            value = _loss(run(tensors[k][0]), tensors[k][1])
            value.backward()
            if config.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            adam.step()
            losses.append(value.detach())
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - clock
        with torch.no_grad():
            val = torch.stack([run(tensors[k][0]) @ tensors[k][1] for k in held])
            y, row_max = x_t, 0.0
            for block in model.blocks:
                y = block(y)
                row_max = max(row_max, float(y.norm(dim=1).max()))
            w = run(x_t)
            gross = float(w.abs().sum())
            test = float(w @ r_t) / gross * 2 if known[window] and gross > 0 else None
        rows.append({"config": config.name, "end": months[end].date, "epoch": epoch + 1,
                     "lr": lr_at(config, epoch),
                     "train_loss": float(torch.stack(losses).mean()),
                     "val_loss": float(((1 - val) ** 2).mean()),
                     "val_sharpe": float(val.mean() / val.std()) if len(held) > 1 else None,
                     "test_ret": test, "row_max": row_max, "gross": gross,
                     "seconds": seconds})  # fmt: skip
        if not math.isfinite(rows[-1]["train_loss"]):
            break
    return rows


def grid(
    months: list[Month], ends: list[int], configs: list[Config], window: int = 60,
    validation: int = 12, device: str = "cuda",
) -> pl.DataFrame:  # fmt: skip
    """Every config on every window end; one row per config, window and epoch."""
    rows = []
    for config in configs:
        for end in ends:
            rows += fit(months, end, config, window, validation, device)
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
    )


def configs_from(spec: list[dict]) -> list[Config]:
    """Configs from a list of dicts (a JSON file); unknown keys are an error."""
    known = set(asdict(Config("x")))
    for item in spec:
        extra = set(item) - known
        if extra:
            raise ValueError(f"unknown config keys {sorted(extra)} in {item}")
    return [Config(**item) for item in spec]
