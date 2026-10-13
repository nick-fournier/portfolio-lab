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
"""

from dataclasses import dataclass
from itertools import pairwise

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


def net_return(
    w: torch.Tensor,
    r: torch.Tensor,
    w_before: torch.Tensor | None,
    link: Link | None,
    spread: float,
    short_fee: float,
) -> torch.Tensor:
    """The month's return net of costs (module docs); ``w_before`` ``None``: from cash."""
    if w_before is None or link is None:
        held, sold = torch.zeros_like(w), w.new_zeros(())
    else:
        held = torch.where(link.prev >= 0, w_before[link.prev.clamp(min=0)], 0.0)
        sold = w_before[link.gone].abs().sum()
    turnover = (w - held).abs().sum() + sold
    return w @ r - spread * turnover - short_fee / 12 * torch.relu(-w).sum()
