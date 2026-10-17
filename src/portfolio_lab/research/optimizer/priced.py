"""The walk's cost-aware loss (research.optimizer.walk): what it needs and how a month is priced.

Each training month k is traded from the book held going into it: month k-1's weights after
k-1's returns (they grow with their stocks), from the book table (``holdings_input``) or, without
one, the current model's own weights for k-1. Holdings are tracked, not recomputed (2026-10-16):
with ``real_book`` a month the walk actually traded is held as traded (``Pricing.traded``, kept
for the whole walk, including across cold months), otherwise as the book table has it.

Inherited books (``inherit``, 2026-10-16): live, the book going into a month was built by last
month's fit, a different model that disagrees with this month's on 20-29% of the book with no
new information (bench9: two seeds' books on the same month). In training the book is always
the model's own, so it never learns what to do with a book it would not have built. With
``inherit`` each training step instead trades from its book displaced by noise (:func:`displaced`),
of a size drawn afresh from 0 to twice ``inherit``; the same displaced book is the holding input
and the base the spread is charged on, so the objective alone decides how much of it is worth
changing. Held-out and test months keep the real book.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from portfolio_lab.research.optimizer import costs, holdings, optimism, realism
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
        market: Each month's :class:`.realism.Market` with any realism term on, else ``None``.
        bound: (gross cap, margin cap) with ``size_bound``, else ``None`` (realism ``book``).
    """

    links: list[costs.Link | None]
    stock: list[Pair]
    book: dict[int, torch.Tensor] | None = None
    borrow: bool = False
    ridge: list[torch.Tensor] | None = None
    traded: dict[int, torch.Tensor] | None = field(default=None)
    market: list | None = None
    bound: tuple[float, float] | None = None

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
    if settings.real_book and not settings.holdings_input:
        raise ValueError("real_book needs holdings_input")
    if settings.inherit and not settings.holdings_input:
        raise ValueError("inherit needs holdings_input")
    if realism.on(settings) and not settings.costs:
        raise ValueError("the realism terms need costs")
    if realism.on(settings) and market is None:
        raise ValueError("the realism terms need market data (realism.align)")
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
    bound = (settings.gross_cap, settings.margin_cap) if settings.size_bound else None
    moved = None if not realism.on(settings) else [
        realism.Market(m.price.to(device), m.vol.to(device), m.adv.to(device), m.tbill)
        for m in market
    ]  # fmt: skip
    return Pricing(costs.links(months, device), stock, {} if settings.holdings_input else None,
                   settings.borrow_input, ridge, {} if settings.real_book else None,
                   moved, bound)  # fmt: skip


def displaced(book: torch.Tensor, size: float, rng: np.random.Generator) -> torch.Tensor:
    """``book`` plus Gaussian noise on every stock, its L1 size ``size`` times the book's.

    Each stock's noise has the same spread (the book's mean absolute weight times ``size``
    times sqrt(pi / 2)), so sum |noise| / sum |book| is ``size`` on average.
    """
    noise = torch.from_numpy(rng.standard_normal(len(book))).to(book)
    return book + noise * (size * math.sqrt(math.pi / 2) * book.abs().mean())


def held_going_in(model, data: list[Pair], pricing: Pricing, k: int,
                  held=None) -> torch.Tensor | None:  # fmt: skip
    """The book month k trades from: month k-1's weights after its returns (no gradient).

    From ``held`` (anything with ``get(month)``; default the pricing's book, traded months as
    traded) when it has month k-1, else the model's own weights for it; ``None`` (cash) for
    the first month.
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
        scale: float = 1.0, held=None) -> torch.Tensor:  # fmt: skip
    """Month k's return on ``w`` net of costs times ``scale``, traded from the book going in.

    ``held``: where that book comes from (:func:`held_going_in`).
    """
    before = held_going_in(model, data, pricing, k, held)
    spread_before = None if k == 0 else pricing.stock[k - 1][0] * scale
    spread, fee = pricing.stock[k]
    value = costs.net_return(w, data[k][1], before, pricing.links[k], spread * scale,
                             (fee + settings.short_hurdle) * scale, spread_before)  # fmt: skip
    if pricing.market is not None:
        kept = torch.zeros_like(w) if before is None else costs.carry(before, pricing.links[k])
        value = value - scale * realism.charges(w, kept, pricing.market[k], settings,
                                                settings.equity)  # fmt: skip
    return value


def trade_penalty(model, data: list[Pair], pricing: Pricing, settings, t: int,
                  real: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """The traded month t's margin and gross penalty on the model's book for it (realism docs).

    Its holdings input is the real book traded the month before (``real``), else the book's.
    """
    held = {t - 1: real} if real is not None else pricing.held()
    w = realism.book(model, holdings.inputs(data, pricing, held, t), pricing, t)
    return realism.penalty(w, pricing.market[t], settings)


def loss(model, data: list[Pair], pricing: Pricing, settings, k: int,
         noise: np.random.Generator | None = None) -> torch.Tensor:  # fmt: skip
    """Month k's paper loss on its return net of costs (walk docs); stores w in the book.

    With ``settings.inherit`` and a ``noise`` stream, a stored book going in is displaced first
    (module docs): one draw serves both the holding input and the cost.
    """
    held = pricing.held()
    before = held.get(k - 1) if held is not None and k > 0 else None
    if settings.inherit and noise is not None and before is not None:
        size = noise.uniform(0, 2 * settings.inherit)
        held = {k - 1: displaced(before, size, noise)}
    w = realism.book(model, holdings.inputs(data, pricing, held, k), pricing, k)
    if pricing.book is not None:
        pricing.book[k] = w.detach()
    value = net(model, data, pricing, settings, k, w, settings.cost_scale, held)
    if pricing.ridge is not None:
        value = value - (pricing.ridge[k] * w * w).sum()
    return (1 - value) ** 2
