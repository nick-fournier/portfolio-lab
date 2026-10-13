"""Trading costs inside the training loss (2026-10-13).

The paper's loss, (1 - w.r)^2, treats every stock as free to trade and to short, so the model
leans on small stocks whose spreads and borrow fees would eat the edge. Here the month's
return in the loss is net of two flat costs:

    net = w.r - spread * turnover - short_fee / 12 * (sum of the short weights' sizes)

``turnover`` is the dollars traded to get from the model's own weights the month before (same
stocks matched by symbol; stocks that left are sold, new ones bought from nothing) to this
month's: sum |w - w_before| plus the size of what was sold. Both costs scale with the weights,
so the paper's free size still sets the leverage. They are proportional to |w|, not w^2, so
marginal positions go to exactly zero rather than shrinking a little.

Per-stock costs (:func:`align`, from a table of ``trade_cost`` per dollar traded and
``borrow_fee`` per dollar shorted a year for each stock-month) replace the flat rates: each
stock pays its own, and a stock sold because it left the candidates pays last month's.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import polars as pl
import torch

from portfolio_lab.research.optimizer.data import Month


@dataclass(frozen=True)
class Link:
    """How one month's stocks line up with the month before's.

    Attributes:
        prev: For each of this month's stocks, its row the month before (-1: new).
        gone: Rows of the month before whose stocks are not in this month.
    """

    prev: torch.Tensor
    gone: torch.Tensor


def links(months: list[Month], device: str = "cpu") -> list[Link | None]:
    """One :class:`Link` per month (``None`` for the first: nothing before it)."""
    out: list[Link | None] = [None]
    for before, now in pairwise(months):
        row = {symbol: i for i, symbol in enumerate(before.symbols)}
        here = set(now.symbols)
        prev = [row.get(symbol, -1) for symbol in now.symbols]
        gone = [i for i, symbol in enumerate(before.symbols) if symbol not in here]
        as_rows = {"dtype": torch.long, "device": device}
        out.append(Link(torch.tensor(prev, **as_rows), torch.tensor(gone, **as_rows)))
    return out


Cost = float | torch.Tensor


def align(months: list[Month], table: pl.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each month's ``(trade_cost, borrow_fee)`` in its candidates' row order.

    A stock missing from ``table`` pays the month's median trade cost and a borrow fee of 1
    (100% a year: treated as not lendable).
    """
    by_date = table.partition_by("date", as_dict=True)
    out = []
    for m in months:
        part = by_date.get((m.date,))
        rows = pl.DataFrame({"symbol": m.symbols})
        if part is not None:
            rows = rows.join(part.select("symbol", "trade_cost", "borrow_fee"), on="symbol",
                             how="left", maintain_order="left")  # fmt: skip
        blank = np.full(len(rows), np.nan)
        trade = rows["trade_cost"].to_numpy() if "trade_cost" in rows.columns else blank
        fee = rows["borrow_fee"].to_numpy() if "borrow_fee" in rows.columns else blank
        known = np.isfinite(trade)
        trade = np.where(known, trade, np.median(trade[known]) if known.any() else 0.005)
        fee = np.where(np.isfinite(fee), fee, 1.0)
        out.append((trade.astype(np.float32), fee.astype(np.float32)))
    return out


def carry(w_before: torch.Tensor, link: Link) -> torch.Tensor:
    """Last month's weights on this month's stocks, in this month's row order (new: 0)."""
    return torch.where(link.prev >= 0, w_before[link.prev.clamp(min=0)], 0.0)


def net_return(
    w: torch.Tensor,
    r: torch.Tensor,
    w_before: torch.Tensor | None,
    link: Link | None,
    spread: Cost,
    short_fee: Cost,
    spread_before: Cost | None = None,
) -> torch.Tensor:
    """The month's return net of costs (module docs); ``w_before`` ``None``: from cash.

    ``spread`` and ``short_fee`` are one rate for every stock or one per stock this month;
    ``spread_before`` (default ``spread``) is last month's, charged on the stocks that left.
    """
    before = spread if spread_before is None else spread_before
    if w_before is None or link is None:
        held, sold = torch.zeros_like(w), w.new_zeros(())
    else:
        held = carry(w_before, link)
        gone = before[link.gone] if isinstance(before, torch.Tensor) else before
        sold = (gone * w_before[link.gone].abs()).sum()
    traded = (spread * (w - held).abs()).sum() + sold
    return w @ r - traded - (short_fee / 12 * torch.relu(-w)).sum()
