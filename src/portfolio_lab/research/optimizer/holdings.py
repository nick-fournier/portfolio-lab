"""Carried holdings for the optimizer grid (2026-10-14): the book a month trades from, as an input.

The grid's ``carry`` and ``holdings_input`` settings (research.optimizer.grid module docs) keep a
table of each month's book. :func:`holding` turns last month's book into this month's holding
column, :func:`with_holdings` appends it and the spread to the traits, and :func:`carried`
scores the held-out months trading month to month from the carried book.
"""

import torch

from portfolio_lab.research.optimizer import costs as cost_model

#: Spread level the spread input is measured from (log(half-spread / this)).
SPREAD_REF = 0.005


def holding(before: torch.Tensor | None, r: torch.Tensor, link, n: int) -> torch.Tensor:
    """Last month's book after its returns, on this month's ``n`` stocks, scaled to N h / sum |h|.

    ``before`` ``None`` (nothing held): zeros.
    """
    if before is None:
        return r.new_zeros(n)
    h = cost_model.carry(before * (1 + r), link)
    return h * n / h.abs().sum().clamp(min=1e-12)


def with_holdings(tensors: list, links: list, stock: list, book: dict, k: int) -> torch.Tensor:
    """Month k's traits plus each stock's holding (:func:`holding`) and log spread columns."""
    x = tensors[k][0]
    h = holding(book.get(k - 1), tensors[k - 1][1], links[k], len(x)) if k > 0 else \
        x.new_zeros(len(x))  # fmt: skip
    s = torch.log(stock[k][0] / SPREAD_REF).clamp(-3, 3)
    return torch.cat([x, h[:, None], s[:, None]], dim=1)


def carried(run, tensors: list, held: list[int], links: list, stock: list | None,
             book: dict | None, hurdle: float) -> dict:  # fmt: skip
    """Held-out months rolled forward in order from the last training month's book (grid docs).

    ``book`` (carried holdings, else ``None``) supplies the starting book and gets each held-out
    month's; without it the start is the model's weights on the last training month.
    """
    if stock is None or not held:
        return {}
    parts = []
    with torch.no_grad():
        prev = run(held[0] - 1) if book is None else book.get(held[0] - 1)
        for k in held:
            w, r = run(k), tensors[k][1]
            before = None if prev is None else prev * (1 + tensors[k - 1][1])
            (spread, fee), gross = stock[k], w @ r
            spread_cost = gross - cost_model.net_return(w, r, before, links[k], spread, 0.0,
                                                        stock[k - 1][0])  # fmt: skip
            kept = w.new_zeros(len(w)) if before is None else cost_model.carry(before, links[k])
            sold = 0.0 if before is None else before[links[k].gone].abs().sum()
            parts.append(torch.stack([gross, spread_cost,
                                      ((fee + hurdle) / 12 * torch.relu(-w)).sum(),
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
            "val_turnover": float(12 * tr.mean())}  # fmt: skip
