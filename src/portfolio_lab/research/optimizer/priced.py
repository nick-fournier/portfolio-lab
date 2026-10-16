"""The walk's cost-aware loss (research.optimizer.walk): what it needs and how a month is priced.

Each training month k is traded from the book held going into it: month k-1's weights after
k-1's returns (they grow with their stocks), from the book table (``holdings_input``) or, without
one, the current model's own weights for k-1. Holdings are tracked, not recomputed (2026-10-16):
with ``real_book`` a month the walk actually traded is held as traded (``Pricing.traded``, kept
for the whole walk, including across cold months), otherwise as the book table has it.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from portfolio_lab.research.optimizer import costs, holdings, optimism
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
        ridge: Each month's per-stock charge per unit of w^2 (``ridge``), else ``None``.
        traded: With ``real_book``, the weights the walk actually traded, by month (else
            ``None``); never cleared.
        speed: Append the unscaled book held going in, for the model's trade speed.
    """

    links: list[costs.Link | None]
    stock: list[Pair]
    book: dict[int, torch.Tensor] | None = None
    borrow: bool = False
    ridge: list[torch.Tensor] | None = None
    traded: dict[int, torch.Tensor] | None = field(default=None)
    speed: bool = False

    def held(self) -> "Held | None":
        """The book a month's holdings come from: traded months as traded, else the table."""
        return None if self.book is None else Held(self.book, self.traded)


class Held:
    """A read-only view of the book table with the real traded months laid over it."""

    def __init__(self, book: dict[int, torch.Tensor], traded: dict[int, torch.Tensor] | None):
        self.book, self.traded = book, traded or {}

    def get(self, k: int) -> torch.Tensor | None:
        """Month k's weights: as traded if the walk traded it, else the book table's."""
        return self.traded.get(k, self.book.get(k))


def pricing(months: list[Month], settings, device: str,
            stock_costs: list[tuple[np.ndarray, ...]] | None) -> Pricing | None:  # fmt: skip
    """The cost-aware loss's inputs (``None`` without costs); checks the settings agree.

    ``stock_costs``: each month's per-stock (trade cost, borrow fee, ...) from ``costs.align``.
    """
    if settings.holdings_input and not settings.costs:
        raise ValueError("holdings_input needs per-stock costs")
    if settings.borrow_input and not settings.holdings_input:
        raise ValueError("borrow_input needs holdings_input")
    if (settings.real_book or settings.trade_speed) and not settings.holdings_input:
        raise ValueError("real_book and trade_speed need holdings_input")
    if not settings.costs:
        return None
    if stock_costs is None:
        raise ValueError("per-stock costs need stock_costs")
    stock = [(torch.from_numpy(c[0]).to(device), torch.from_numpy(c[1]).to(device))
             for c in stock_costs]  # fmt: skip
    ridge = None
    if settings.ridge:
        tilt = torch.tensor(settings.ridge_tilt or (1.0,) * optimism.BUCKETS, device=device)
        if len(tilt) != optimism.BUCKETS:
            raise ValueError(f"ridge_tilt needs {optimism.BUCKETS} values")
        cuts = optimism.edges(stock)[0]
        ridge = [settings.ridge * tilt[torch.bucketize(s, cuts)] for s, _ in stock]
    return Pricing(costs.links(months, device), stock, {} if settings.holdings_input else None,
                   settings.borrow_input, ridge, {} if settings.real_book else None,
                   settings.trade_speed)  # fmt: skip


def held_going_in(model, data: list[Pair], pricing: Pricing, k: int) -> torch.Tensor | None:
    """The book month k trades from: month k-1's weights after its returns (no gradient).

    From the book (traded months as traded) when it has month k-1, else the model's own
    weights for it; ``None`` (cash) for the first month.
    """
    if k == 0:
        return None
    with torch.no_grad():
        held = pricing.held()
        before = held.get(k - 1) if held is not None else None
        if before is None:
            before = model(holdings.inputs(data, pricing, held, k - 1))
        return before * (1 + data[k - 1][1])


def net(model, data: list[Pair], pricing: Pricing, settings, k: int, w: torch.Tensor,
        scale: float = 1.0) -> torch.Tensor:  # fmt: skip
    """Month k's return on ``w`` net of costs times ``scale``, traded from the book going in."""
    before = held_going_in(model, data, pricing, k)
    spread_before = None if k == 0 else pricing.stock[k - 1][0] * scale
    spread, fee = pricing.stock[k]
    return costs.net_return(w, data[k][1], before, pricing.links[k], spread * scale,
                            (fee + settings.short_hurdle) * scale, spread_before)  # fmt: skip


def loss(model, data: list[Pair], pricing: Pricing, settings, k: int) -> torch.Tensor:
    """Month k's paper loss on its return net of costs (walk docs); stores w in the book."""
    w = model(holdings.inputs(data, pricing, pricing.held(), k))
    if pricing.book is not None:
        pricing.book[k] = w.detach()
    value = net(model, data, pricing, settings, k, w, settings.cost_scale)
    if pricing.ridge is not None:
        value = value - (pricing.ridge[k] * w * w).sum()
    return (1 - value) ** 2
