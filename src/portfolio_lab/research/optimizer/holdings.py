"""Holdings as inputs (2026-10-14): the book a month trades from, fed to the model.

With ``holdings_input`` (research.optimizer.walk) the walk keeps a table of each month's book.
:func:`holding` turns last month's book into this month's holding column and
:func:`with_holdings` appends it, the spread and (``borrow``) the borrow fee to the traits. The
walk uses :func:`inputs`, :func:`fill`, :func:`pick` and :func:`remember` with its pricing
(``links``, ``stock`` and ``book``; research.optimizer.priced).
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


def with_holdings(tensors: list, links: list, stock: list, book, k: int,
                  borrow: bool = False) -> torch.Tensor:  # fmt: skip
    """Month k's traits plus holding, log spread and (``borrow``) log borrow fee columns.

    ``book``: anything with ``get(month)``. The holding is :func:`holding`; the logs are
    clipped to [-3, 3].
    """
    x = tensors[k][0]
    h = holding(book.get(k - 1), tensors[k - 1][1], links[k], len(x)) if k > 0 else \
        x.new_zeros(len(x))  # fmt: skip
    cols = [x, h[:, None], torch.log(stock[k][0] / SPREAD_REF).clamp(-3, 3)[:, None]]
    if borrow:
        cols.append(torch.log(stock[k][1] / FEE_REF).clamp(-3, 3)[:, None])
    return torch.cat(cols, dim=1)


def inputs(source: list, pricing, book, k: int) -> torch.Tensor:
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
    """Store the test month's real holdings in the book; they are what the next month holds."""
    if pricing is None or pricing.book is None:
        return None
    pricing.book[t] = torch.from_numpy(w).to(data[t][0])
    return pricing.book[t]
