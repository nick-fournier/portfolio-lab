"""The walk's cost-aware loss (research.optimizer.walk): what it needs and how a month is priced.

Each training month k is traded from the book held going into it: month k-1's weights after
k-1's returns (they grow with their stocks), from the book table (``holdings_input``) or, without
one, the current model's own weights for k-1. Holdings are tracked, not recomputed (2026-10-16):
the book table keeps each month's weights as last written (a training step, a held-out roll or
the walk's real trade), and the month after trades from them. A month whose month before is not
in the table holds cash going in, as its holding input (research.optimizer.holdings) says, and
its trades are priced from cash: the window's first month after a cold start, whose table is
cleared and filled from that month on (:func:`held_going_in`). The fill is cash too (an all-zero
book, holdings.fill; 2026-10-18), so a month whose month before has not yet been visited in
training also holds, and trades from, cash.
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
        holding_column / borrow_flag / spread_rank / holding_rank: How the extra columns are
            encoded
            (holdings.with_holdings).
    """

    links: list[costs.Link | None]
    stock: list[Pair]
    book: dict[int, torch.Tensor] | None = None
    borrow: bool = False
    market: list | None = None
    holding_column: bool = True
    borrow_flag: bool = False
    spread_rank: bool = False
    holding_rank: bool = False

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


def _check_sizes(settings, market: list | None) -> None:
    """Checks the size settings and the extra columns' encodings agree with the rest.

    The trade-time size and the size the loss prices at need costs and market data; the
    encodings need the columns they shape.
    """
    shaped = not settings.holding_column or settings.spread_rank or settings.holding_rank
    if shaped and not settings.holdings_input:
        raise ValueError("holding_column, holding_rank and spread_rank shape the holdings_input"
                         " columns")  # fmt: skip
    if settings.holding_rank and not settings.holding_column:
        raise ValueError("holding_rank needs the holding column")
    if settings.borrow_flag and not settings.borrow_input:
        raise ValueError("borrow_flag shapes the borrow_input column")
    if settings.trade_gross < 0 or settings.trade_collateral < 0 or settings.price_gross < 0:
        raise ValueError("trade_gross, trade_collateral and price_gross cannot be negative")
    if settings.loss_gross < 0 or settings.risk_aversion <= 0:
        raise ValueError("loss_gross cannot be negative and risk_aversion must be positive")
    if settings.loss_gross and settings.loss_gross != settings.trade_gross:
        raise ValueError("loss_gross needs trade_gross set to the same size (the loss sizes the "
                         "book as the trade will)")  # fmt: skip
    if settings.loss_gross and settings.price_gross:
        raise ValueError("loss_gross already prices the charges at its size; drop price_gross")
    if settings.price_gross and market is None:
        raise ValueError("price_gross needs a realism term (it reprices margin, cash and impact)")
    if settings.trade_gross and not settings.costs:
        raise ValueError("trade_gross needs costs (and market data for the collateral)")
    if settings.trade_gross and market is None:
        raise ValueError("trade_gross needs market data (realism.align) for the collateral")


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
    _check_sizes(settings, market)
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
                   settings.borrow_input, moved, settings.holding_column, settings.borrow_flag,
                   settings.spread_rank, settings.holding_rank)  # fmt: skip


def held_going_in(model, data: list[Pair], pricing: Pricing, k: int,
                  held=None) -> torch.Tensor | None:  # fmt: skip
    """The book month k trades from: month k-1's weights after its returns (no gradient).

    ``None`` is cash. From ``held`` (anything with ``get(month)``; default the pricing's book
    table): its month k-1, or cash when it lacks that month, the same test month k's holding
    input makes (``holdings.with_holdings``: zeros), so a month trades from the book it is shown.
    Only the window's first month, ``lo``, can lack it: the walk fills the table from ``lo`` on,
    and a cold month (or a warm walk's first) starts from an empty table, so ``lo - 1`` is not in
    it; a later warm month keeps ``lo - 1`` from the months before and trades from it. The fill
    is cash (all zeros, holdings.fill), so until training visits k-1 the book is cash here too.
    Until 2026-10-18 a month the table lacked was priced from the model's own weights for k-1,
    against its cash holding input. Without a table: the model's own weights for month k-1. The
    first month (k 0) starts in cash.
    """
    if k == 0:
        return None
    with torch.no_grad():
        held = pricing.held() if held is None else held
        if held is None:  # no book table: the model's own weights for month k-1
            before = realism.book(model, holdings.inputs(data, pricing, None, k - 1),
                                  pricing, k - 1)  # fmt: skip
        else:
            before = held.get(k - 1)
            if before is None:  # not in the table: cash, as the holding input says
                return None
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


def book_for(
    model, data: list[Pair], pricing: Pricing, settings, k: int, held=None
) -> torch.Tensor:
    """Month k's book as the loss sees it: the model's, sized to ``loss_gross`` when set.

    ``held``: where the holdings input comes from (default the pricing's book table).
    """
    held = pricing.held() if held is None else held
    w = realism.book(model, holdings.inputs(data, pricing, held, k), pricing, k)
    if settings.loss_gross:
        w = realism.sized(w, pricing.market[k], settings.loss_gross, settings.trade_collateral)
    return w


def objective(r: torch.Tensor, settings) -> torch.Tensor:
    """The loss on a month's net return ``r`` (2026-10-18).

    At the model's own size (``loss_gross`` 0) the paper's ``(1 - r)^2``: free size makes it a
    maximum-Sharpe book. At a set size, ``-r + risk_aversion / 2 * r^2``: expected return less
    a chosen aversion to its variance (the paper's loss is this with aversion 1, up to a
    constant, but at ~1x that aversion leaves the variance term almost no weight).
    """
    if settings.loss_gross:
        return -r + 0.5 * settings.risk_aversion * r * r
    return (1 - r) ** 2


def loss(model, data: list[Pair], pricing: Pricing, settings, k: int) -> torch.Tensor:
    """Month k's loss on its return net of costs (walk docs); stores its book in the table."""
    held = pricing.held()
    w = book_for(model, data, pricing, settings, k, held)
    if pricing.book is not None:
        pricing.book[k] = w.detach()  # with loss_gross the sized book: what next month holds
    return objective(net(model, data, pricing, settings, k, w, held), settings)
