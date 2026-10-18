"""The walk's cost-aware loss (research.optimizer.walk): what it needs and how a month is priced.

Each training month k is traded from the book held going into it: month k-1's weights after
k-1's returns (they grow with their stocks), from the book table (``holdings_input``) or, without
one, the current model's own weights for k-1. Holdings are tracked, not recomputed (2026-10-16):
the book table keeps each month's weights as last written (a training step, a held-out roll or
the walk's real trade), and the month after trades from them.
"""

from dataclasses import dataclass

import numpy as np
import torch

from portfolio_lab.research.optimizer import costs, holdings, realism
from portfolio_lab.research.optimizer.data import Month

Pair = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class Pricing:
    """What the cost-aware loss needs (:func:`pricing`).

    Attributes:
        links: ``costs.links`` of the months.
        stock: Each month's per-stock (trade cost, borrow fee).
        book: The training book table with ``holdings_input`` (walk docs), else ``None``.
        borrow: Feed the borrow fee too (``borrow_input``).
        market: Each month's :class:`.realism.Market` with any realism term on, else ``None``.
    """

    links: list[costs.Link | None]
    stock: list[Pair]
    book: dict[int, torch.Tensor] | None = None
    borrow: bool = False
    market: list | None = None

    def held(self) -> "Held | None":
        """The book a month's holdings come from: the book table, read-only."""
        return None if self.book is None else Held(self.book)


class Held:
    """A read-only view of the book table."""

    def __init__(self, book: dict[int, torch.Tensor]):
        self.book = book

    def get(self, k: int) -> torch.Tensor | None:
        """Month k's weights in the book table (``None``: not in it)."""
        return self.book.get(k)


def pricing(months: list[Month], settings, device: str,
            stock_costs: list[tuple[np.ndarray, ...]] | None,
            market: list | None = None) -> Pricing | None:  # fmt: skip
    """The cost-aware loss's inputs (``None`` without costs); checks the settings agree.

    ``stock_costs``: each month's per-stock (trade cost, borrow fee, ...) from ``costs.align``;
    ``market``: each month's :class:`.realism.Market` (``realism.align``), needed when any
    realism term is on.
    """
    if settings.holdings_input and not settings.costs:
        raise ValueError("holdings_input needs per-stock costs")
    if settings.borrow_input and not settings.holdings_input:
        raise ValueError("borrow_input needs holdings_input")
    if realism.on(settings) and not settings.costs:
        raise ValueError("the realism terms need costs")
    if realism.on(settings) and market is None:
        raise ValueError("the realism terms need market data (realism.align)")
    if settings.trade_margin and not (settings.margin_cushion or settings.gross_limit
                                      or settings.invest_floor):  # fmt: skip
        raise ValueError("trade_margin needs margin_cushion, gross_limit or invest_floor")
    if settings.start_book != "guess" and not settings.holdings_input:
        raise ValueError("start_book needs holdings_input")
    if settings.start_book == "market" and not realism.on(settings):
        raise ValueError("a market start needs a realism term (its market data)")
    if not settings.costs:
        return None
    if stock_costs is None:
        raise ValueError("per-stock costs need stock_costs")
    stock = [(torch.from_numpy(c[0]).to(device), torch.from_numpy(c[1]).to(device))
             for c in stock_costs]  # fmt: skip
    moved = None if not realism.on(settings) else [
        realism.Market(m.price.to(device), m.vol.to(device), m.adv.to(device), m.tbill,
                       None if m.mv is None else m.mv.to(device))
        for m in market
    ]  # fmt: skip
    return Pricing(costs.links(months, device), stock, {} if settings.holdings_input else None,
                   settings.borrow_input, moved)  # fmt: skip


def held_going_in(model, data: list[Pair], pricing: Pricing, k: int,
                  held=None) -> torch.Tensor | None:  # fmt: skip
    """The book month k trades from: month k-1's weights after its returns (no gradient).

    From ``held`` (anything with ``get(month)``; default the pricing's book table) when it has
    month k-1, else the model's own weights for it; ``None`` (cash) for the first month.
    """
    if k == 0:
        return None
    with torch.no_grad():
        held = pricing.held() if held is None else held
        before = held.get(k - 1) if held is not None else None
        if before is None:
            before = realism.book(model, holdings.inputs(data, pricing, held, k - 1),
                                  pricing, k - 1)  # fmt: skip
        return before * (1 + data[k - 1][1])


def net(model, data: list[Pair], pricing: Pricing, settings, k: int, w: torch.Tensor,
        held=None) -> torch.Tensor:  # fmt: skip
    """Month k's return on ``w`` net of costs, traded from the book going in.

    ``held``: where that book comes from (:func:`held_going_in`).
    """
    before = held_going_in(model, data, pricing, k, held)
    spread_before = None if k == 0 else pricing.stock[k - 1][0]
    spread, fee = pricing.stock[k]
    value = costs.net_return(w, data[k][1], before, pricing.links[k], spread, fee, spread_before)
    if pricing.market is not None:
        kept = torch.zeros_like(w) if before is None else costs.carry(before, pricing.links[k])
        value = value - realism.charges(w, kept, pricing.market[k], settings, settings.equity)
    return value


def trade_penalty(model, data: list[Pair], pricing: Pricing, settings, t: int,
                  real: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """The traded month t's penalty on the model's book for it (realism ``penalty`` and docs).

    Its holdings input is the real book traded the month before (``real``), else the book's.
    """
    held = {t - 1: real} if real is not None else pricing.held()
    w = realism.book(model, holdings.inputs(data, pricing, held, t), pricing, t)
    return realism.penalty(w, pricing.market[t], settings)


def loss(model, data: list[Pair], pricing: Pricing, settings, k: int) -> torch.Tensor:
    """Month k's paper loss on its return net of costs (walk docs); stores w in the book."""
    held = pricing.held()
    w = realism.book(model, holdings.inputs(data, pricing, held, k), pricing, k)
    if pricing.book is not None:
        pricing.book[k] = w.detach()
    return (1 - net(model, data, pricing, settings, k, w, held)) ** 2
