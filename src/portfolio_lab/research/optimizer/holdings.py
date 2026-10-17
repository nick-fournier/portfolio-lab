"""Carried holdings for the optimizer grid (2026-10-14): the book a month trades from, as an input.

The grid's ``carry`` and ``holdings_input`` settings (research.optimizer.grid module docs) keep a
table of each month's book. :func:`holding` turns last month's book into this month's holding
column, :func:`with_holdings` appends it, the spread and (``borrow``) the borrow fee to the
traits, and :func:`carried`
scores the held-out months trading month to month from the carried book. The walk's
``holdings_input`` (research.optimizer.walk) uses :func:`inputs`, :func:`fill`, :func:`pick`
and :func:`remember` with its pricing (``links``, ``stock`` and ``book``).
"""

import numpy as np
import torch

from portfolio_lab.research.optimizer import costs as cost_model
from portfolio_lab.research.optimizer import realism

#: Spread level the spread input is measured from (log(half-spread / this)).
SPREAD_REF = 0.005
#: Borrow fee a year the borrow input is measured from (log(fee / this)).
FEE_REF = 0.03


def holding(before: torch.Tensor | None, r: torch.Tensor, link, n: int) -> torch.Tensor:
    """Last month's book after its returns, on this month's ``n`` stocks, scaled to N h / sum |h|.

    ``before`` ``None`` (nothing held): zeros.
    """
    if before is None:
        return r.new_zeros(n)
    h = cost_model.carry(before * (1 + r), link)
    return h * n / h.abs().sum().clamp(min=1e-12)


def with_holdings(tensors: list, links: list, stock: list, book: dict, k: int,
                  borrow: bool = False) -> torch.Tensor:  # fmt: skip
    """Month k's traits plus holding, log spread and (``borrow``) log borrow fee columns.

    The holding is :func:`holding`; the logs are clipped to [-3, 3].
    """
    x = tensors[k][0]
    h = holding(book.get(k - 1), tensors[k - 1][1], links[k], len(x)) if k > 0 else \
        x.new_zeros(len(x))  # fmt: skip
    cols = [x, h[:, None], torch.log(stock[k][0] / SPREAD_REF).clamp(-3, 3)[:, None]]
    if borrow:
        cols.append(torch.log(stock[k][1] / FEE_REF).clamp(-3, 3)[:, None])
    return torch.cat(cols, dim=1)


def carried(run, tensors: list, held: list[int], links: list, stock: list | None,
             book: dict | None, hurdle: float) -> dict:  # fmt: skip
    """Held-out months rolled forward in order from the last training month's book (grid docs).

    ``book`` (carried holdings, else ``None``) supplies the starting book and gets each held-out
    month's; without it the start is the model's weights on the last training month. Also
    ``val_net_loss``: the mean paper loss on the return net of costs at the model's size.
    """
    if stock is None or not held:
        return {}
    parts, losses = [], []
    with torch.no_grad():
        if book is not None:
            prev = book.get(held[0] - 1)
        else:
            prev = run(held[0] - 1) if held[0] > 0 else None
        for k in held:
            w, r = run(k), tensors[k][1]
            before = None if prev is None else prev * (1 + tensors[k - 1][1])
            (spread, fee), gross = stock[k], w @ r
            spread_cost = gross - cost_model.net_return(w, r, before, links[k], spread, 0.0,
                                                        stock[k - 1][0])  # fmt: skip
            kept = w.new_zeros(len(w)) if before is None else cost_model.carry(before, links[k])
            sold = 0.0 if before is None else before[links[k].gone].abs().sum()
            borrow = ((fee + hurdle) / 12 * torch.relu(-w)).sum()
            losses.append((1 - (gross - spread_cost - borrow)) ** 2)
            parts.append(torch.stack([gross, spread_cost, borrow,
                                      (w - kept).abs().sum() + sold]) / w.abs().sum())  # fmt: skip
            if book is not None:
                book[k] = w
            prev = w
    g, sp, bo, tr = torch.stack(parts).T
    net = g - sp - bo
    sharpe = lambda v: float(v.mean() / v.std() * 12**0.5) if len(v) > 1 else None  # noqa: E731
    return {"val_net_sharpe": sharpe(net), "val_carry_sharpe": sharpe(g),
            "val_net_ret": float(24 * net.mean()), "val_gross_ret": float(24 * g.mean()),
            "val_spread": float(24 * sp.mean()), "val_borrow": float(24 * bo.mean()),
            "val_turnover": float(12 * tr.mean()),
            "val_net_loss": float(torch.stack(losses).mean())}  # fmt: skip


def inputs(source: list, pricing, book: dict | None, k: int) -> torch.Tensor:
    """Month k's model inputs: its traits, plus holding and spread columns with a ``book``."""
    if pricing is None or book is None:
        return source[k][0]
    return with_holdings(source, pricing.links, pricing.stock, book, k, pricing.borrow)


def fill(weights, source: list, pricing, span: range) -> None:
    """Months in ``span`` not yet in the pricing's book get ``weights``' output, in month order."""
    if pricing.book is None:
        return
    with torch.no_grad():
        for k in span:
            if k not in pricing.book:
                x = inputs(source, pricing, pricing.held(), k)
                pricing.book[k] = realism.book(weights, x, pricing, k)


def pick(weights, data: list, pricing, last: torch.Tensor | None, t: int) -> np.ndarray:
    """The test month's weights; with a book, its inputs carry the real holdings ``last``."""
    with torch.no_grad():
        real = None if pricing is None or pricing.book is None else \
            ({} if last is None else {t - 1: last})  # fmt: skip
        w = realism.book(weights, inputs(data, pricing, real, t), pricing, t)
        return w.cpu().numpy().astype(np.float64)


def remember(pricing, w: np.ndarray, data: list, t: int) -> torch.Tensor | None:
    """Store the test month's real holdings in the book (and the traded record, if kept)."""
    if pricing is None or pricing.book is None:
        return None
    pricing.book[t] = torch.from_numpy(w).to(data[t][0])
    if pricing.traded is not None:
        pricing.traded[t] = pricing.book[t]
    return pricing.book[t]
