"""A retail account's costs that depend on the book's size (2026-10-17): the loss's realism terms.

The loss measures the book in dollars per $1 of equity (w), so the account's own limits can be
priced in it and the model left to choose its size. Each term is a monthly charge on the
return in the loss (research.optimizer.priced), off unless its setting is on:

* lost short rebate (``rebate``): a retail broker keeps the interest on short-sale proceeds,
  so each short dollar forgoes the month's T-bill: ``tbill * sum relu(-w)``.
* margin interest: longs beyond equity are borrowed at the broker's rate; against the T-bill
  the loss already assumes, that costs the excess over it. ``margin_rate`` (a year): the
  broker's rate, ``relu(margin_rate / 12 - tbill) * relu(sum relu(w) - 1)``.
* cash rate (``cash_rate``, a year; ``None``: the T-bill): equity not spent on longs earns the
  broker's cash rate, so it forgoes ``relu(tbill - cash_rate / 12) * relu(1 - sum relu(w))``.
* maintenance margin (``margin_cushion``): the collateral the broker requires, per dollar held,
  from Alpaca's price tables (longs: 100% under $2.50, 50% under $6, else 30%; shorts: under $5
  the larger of $2.50 a share and 100%, else the larger of $5 a share and 30%). Per-share
  requirements are per dollar too (``2.5 / price``), so no account size is needed. Requirement
  above ``margin_cushion`` x equity, ``over``, is charged ``cushion_rate * over`` a month.
  ``gross_limit``: gross (sum |w|) above it is charged the same way (2026-10-17: the user's
  1.25x). ``invest_floor`` (the long-priority toggle): longs below it (idle cash; at Alpaca short
  proceeds cannot fund longs) are charged the same way (2026-10-17: the user wants the account
  fully invested).
* the traded month (``trade_margin``, walk): the penalties above only shape the training
  months' books; the traded month's book, built after training, was never checked (curve1). Its
  penalty (:func:`penalty`: margin cushion, gross limit and, when set, the investment floor)
  needs only today's prices and positions, no return, so each training step adds it, and the
  model learns to keep the book it will trade inside.
* market impact (``impact``, the constant K): trading t dollars per $1 of equity in a stock
  with daily volatility s and daily dollar volume V costs ``K * s * sqrt(t * equity / V) * t``
  (square-root law), on the dollars traded from the book held going in.

``price_gross`` (2026-10-18): margin interest, idle cash and impact are priced as if the book
were scaled to that gross (the traded size) and charged per dollar of the book's gross
(:func:`charges`), so a free-size book in training pays the traded account's costs per dollar.

``loss_gross`` (2026-10-18): the loss is taken on the book scaled to that gross at the month's
prices, capped by ``trade_collateral`` (:func:`sized`, the same rule the trade applies), with an
explicit risk aversion (research.optimizer.priced). The model then learns the direction of the
book it will trade, and every charge is priced at the real size.

The book is the model's output as it is (:func:`book`): any size, limited only by these charges.
Only the traded book can be sized at trade time (``trade_gross``, 2026-10-18; :func:`trade_size`):
scaled to that gross, or less if its maintenance requirement would pass ``trade_collateral`` x
equity. It is a clamp on the final holdings alone: training, the loss and held-out scores never
see it.

:func:`align` reads what they need from the cost table (columns ``prc``, ``rvol_21d``, ``adv``
and ``tbill``, the holding month's T-bill, monthly).
"""

from collections.abc import Collection
from dataclasses import dataclass, replace

import numpy as np
import polars as pl
import torch

from portfolio_lab.research.optimizer.data import Month

#: Columns :func:`align` needs in the cost table.
COLUMNS = ("prc", "rvol_21d", "adv", "tbill")
#: The Alpaca account profile (2026-10-17; docs.alpaca.markets margin-and-short-selling, Alpaca
#: high-yield cash): what the walk's ``--profile alpaca`` sets (the flags not given) and the
#: scorer charges. Easy-to-borrow shorts cost nothing (the cost table's borrow fee), hard-to-
#: borrow ones are barred.
ALPACA = {"rebate": True, "margin_rate": 0.065, "cash_rate": 0.0356, "margin_cushion": 0.5,
          "gross_limit": 1.25, "cushion_rate": 0.10, "impact": 1.0,
          "equity": 100_000.0}  # fmt: skip
#: Account profiles by name (the walk's ``--profile``; ``none``: no profile).
PROFILES = {"alpaca": ALPACA}


@dataclass(frozen=True)
class Market:
    """One month's per-stock price, daily volatility and daily dollar volume, and its T-bill."""

    price: torch.Tensor
    vol: torch.Tensor
    adv: torch.Tensor
    tbill: float
    mv: torch.Tensor | None = None  # market value, when the cost table has it (market start)


#: Stocks in the market starting book (``start_book`` "market"): the largest by market value.
MARKET_NAMES = 500


def start_book(kind: str, market: Market | None, n: int) -> torch.Tensor | None:
    """The book a walk's first traded month starts from when nothing was traded the month before.

    ``guess``: ``None`` (the model's own weights for that month, as before); ``cash``: nothing
    held; ``market``: $1 long per $1 of equity in the :data:`MARKET_NAMES` largest stocks by
    market value, weighted by it (close to holding the S&P 500). ``n``: that month's stocks.
    """
    if kind == "guess":
        return None
    if kind == "cash":
        return torch.zeros(n)
    if kind != "market":
        raise ValueError(f"start_book must be guess, cash or market, not {kind!r}")
    if market is None or market.mv is None:
        raise ValueError("a market start needs market_value in the cost table")
    mv = market.mv.clone()
    if len(mv) > MARKET_NAMES:
        mv = torch.where(mv >= torch.topk(mv, MARKET_NAMES).values[-1], mv, 0.0)
    return mv / mv.sum()


def with_profile(settings, name: str, keep: Collection[str] = ()):
    """``settings`` with an account profile's realism terms, except the ones in ``keep``.

    Args:
        settings: A walk's settings.
        name: A profile in :data:`PROFILES`, or ``none`` (or ``""``): no profile, ``settings``
            as it is.
        keep: Terms given explicitly (the walk's flags; 0 too): they stay as ``settings`` has
            them, and the profile fills only the others.

    Raises:
        ValueError: ``name`` is not a profile.
    """
    if name in ("", "none"):
        return settings
    if name not in PROFILES:
        raise ValueError(f"profile must be none or one of {sorted(PROFILES)}, not {name!r}")
    return replace(settings, **{k: v for k, v in PROFILES[name].items() if k not in keep})


def on(settings) -> bool:
    """Whether any realism term is switched on."""
    return bool(settings.rebate or settings.margin_rate or settings.cash_rate is not None
                or settings.margin_cushion or settings.gross_limit or settings.invest_floor
                or settings.impact or settings.max_weight
                or settings.short_cap is not None)  # fmt: skip


def align(months: list[Month], table: pl.DataFrame, device: str = "cpu") -> list[Market]:
    """Each month's :class:`Market` in its candidates' row order.

    A stock missing a value gets the month's median (a missing T-bill: 0).
    """
    missing = [c for c in COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"the cost table needs {missing} for the realism terms")
    by_date = table.partition_by("date", as_dict=True)
    out = []
    for m in months:
        rows = pl.DataFrame({"symbol": m.symbols})
        part = by_date.get((m.date,))
        if part is not None:
            rows = rows.join(part.select("symbol", *COLUMNS), on="symbol", how="left",
                             maintain_order="left")  # fmt: skip
        cols, blank = [], np.full(len(rows), np.nan)
        for c in ("prc", "rvol_21d", "adv"):
            v = rows[c].to_numpy().astype(np.float64) if c in rows.columns else blank
            ok = np.isfinite(v) & (v > 0)
            v = np.where(ok, v, np.median(v[ok]) if ok.any() else 1.0)
            cols.append(torch.tensor(v, dtype=torch.float32, device=device))
        tb = part["tbill"].drop_nulls() if part is not None else pl.Series([], dtype=pl.Float64)
        mv = None
        if part is not None and "market_value" in part.columns:
            v = rows.join(part.select("symbol", "market_value"), on="symbol", how="left",
                          maintain_order="left")["market_value"]  # fmt: skip
            mv = torch.tensor(v.fill_null(0).to_numpy(), dtype=torch.float32, device=device)
        out.append(Market(*cols, float(tb[0]) if len(tb) else 0.0, mv))
    return out


def collateral_rates(price: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Alpaca's maintenance requirement per $1 held long and per $1 short, by price (docs)."""
    long_rate = torch.where(price < 2.5, 1.0, torch.where(price < 6.0, 0.5, 0.3))
    short_rate = torch.where(price < 5.0, torch.clamp(2.5 / price, min=1.0),
                             torch.clamp(5.0 / price, min=0.3))  # fmt: skip
    return long_rate, short_rate


def requirement(w: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
    """Alpaca's maintenance requirement, per $1 of equity, of the book ``w`` (module docs)."""
    long_rate, short_rate = collateral_rates(price)
    return (long_rate * torch.relu(w)).sum() + (short_rate * torch.relu(-w)).sum()


def book(model, x: torch.Tensor, pricing=None, k: int | None = None) -> torch.Tensor:
    """Month k's book: the model's output on inputs ``x``, as it is (module docs).

    Every book the walk forms, in training, scoring and trading, is formed here, so a rule on
    its size has one place to go; ``pricing`` and ``k`` (the month) are not used today. The
    traded book's clamp at trade time (:func:`trade_size`) comes after it, outside the loss.
    """
    return model(x)


def sized(w: torch.Tensor, market: Market, gross: float, collateral: float) -> torch.Tensor:
    """``w`` scaled to ``gross`` (sum |w|), or less if its requirement would pass ``collateral``.

    The torch twin of :func:`trade_size` for the loss (``loss_gross``, 2026-10-18): the factor
    ``min(gross / sum |w|, collateral / requirement(w))`` is a function of ``w``, so the model
    learns the direction of a book that is always traded at this size. ``collateral`` 0: no
    cap. An all-cash book stays cash.
    """
    total = w.abs().sum()
    scale = gross / total.clamp(min=1e-12)
    if collateral:
        scale = torch.minimum(scale, collateral / requirement(w, market.price).clamp(min=1e-12))
    return w * scale


def trade_size(w: np.ndarray, market: Market | None, settings) -> tuple[np.ndarray, dict]:
    """The traded book ``w`` scaled by ``settings.trade_gross`` (module docs) and its log entry.

    The factor is ``min(trade_gross / sum |w|, trade_collateral / requirement(w))``, the
    requirement per $1 of equity at the month's prices (``market``); ``trade_collateral`` 0: no
    collateral cap. Both move with the book's size, so the scaled book's gross is
    ``trade_gross`` or its requirement ``trade_collateral``, whichever comes first, and its
    direction is the model's. ``trade_gross`` 0 (off) or an all-cash book (which stays cash): ``w``
    as it is, factor 1.

    Returns:
        The book traded and ``{"model_gross": sum |w| before scaling, "scale": the factor}``.

    Raises:
        ValueError: ``trade_gross`` is on without ``market``.
    """
    gross = float(np.abs(w).sum())
    if not settings.trade_gross or not gross:
        return w, {"model_gross": gross, "scale": 1.0}
    if market is None:
        raise ValueError("trade_gross needs market data (realism.align) for the collateral")
    scale = settings.trade_gross / gross
    if settings.trade_collateral:
        need = float(requirement(torch.from_numpy(w), market.price.cpu().double()))
        scale = min(scale, settings.trade_collateral / need)
    return w * scale, {"model_gross": gross, "scale": scale}


def penalty(w: torch.Tensor, market: Market, settings) -> torch.Tensor:
    """The limit charges on ``w`` (docs); no return.

    Margin cushion, gross limit, investment floor, and (2026-10-19, the box) ``max_weight``,
    each name's dollars above the cap, and ``short_cap``, short dollars above it (0: long-only;
    ``None``: off); all at ``cushion_rate``.
    """
    total = w.new_zeros(())
    for over in ((torch.relu(requirement(w, market.price) - settings.margin_cushion)
                  if settings.margin_cushion else None),
                 (torch.relu(w.abs().sum() - settings.gross_limit)
                  if settings.gross_limit else None),
                 (torch.relu(settings.invest_floor - torch.relu(w).sum())
                  if settings.invest_floor else None),
                 (torch.relu(w.abs() - settings.max_weight).sum()
                  if settings.max_weight else None),
                 (torch.relu(torch.relu(-w).sum() - settings.short_cap)
                  if settings.short_cap is not None else None)):  # fmt: skip
        if over is not None:
            total = total + settings.cushion_rate * over
    return total


def _sized(w: torch.Tensor, held: torch.Tensor, market: Market, settings,
           equity: float) -> torch.Tensor:  # fmt: skip
    """Margin interest, idle cash and impact on the book ``w`` traded from ``held`` (docs)."""
    total = w.new_zeros(())
    longs = torch.relu(w).sum()
    if settings.margin_rate:
        total = total + max(settings.margin_rate / 12 - market.tbill, 0.0) * torch.relu(longs - 1)
    if settings.cash_rate is not None:
        total = total + max(market.tbill - settings.cash_rate / 12, 0.0) * torch.relu(1 - longs)
    if settings.impact:
        t = (w - held).abs()
        total = total + settings.impact * (market.vol * torch.sqrt(t * equity / market.adv + 1e-12)
                                           * t).sum()  # fmt: skip
    return total


def charges(w: torch.Tensor, held: torch.Tensor, market: Market, settings,
            equity: float) -> torch.Tensor:  # fmt: skip
    """The month's realism charges on the book ``w`` traded from ``held`` (module docs).

    With ``price_gross`` (2026-10-18) the charges that do not grow in step with the book's size
    (margin interest, idle cash, impact) are priced on the book and its held book scaled by the
    same factor to gross ``price_gross``, the size the account trades (``trade_gross``), and
    charged per dollar of the book's own gross: ``sized(c w, c held) / c`` with
    ``c = price_gross / sum |w|``. Every charge then scales with the book, as its return does,
    so the model's own size stays free (the paper's loss) while the account's costs are those
    of the traded size. The lost rebate is per dollar already; the limit penalties stay on ``w``.
    """
    if settings.price_gross:
        total = w.new_zeros(())
        if settings.rebate:
            total = total + market.tbill * torch.relu(-w).sum()
        c = settings.price_gross / w.abs().sum().clamp(min=1e-12)
        return total + penalty(w, market, settings) + _sized(c * w, c * held, market, settings,
                                                             equity) / c  # fmt: skip
    total = w.new_zeros(())  # at the book's own size: the terms in their original order
    if settings.rebate:
        total = total + market.tbill * torch.relu(-w).sum()
    longs = torch.relu(w).sum()
    if settings.margin_rate:
        total = total + max(settings.margin_rate / 12 - market.tbill, 0.0) * torch.relu(longs - 1)
    if settings.cash_rate is not None:
        total = total + max(market.tbill - settings.cash_rate / 12, 0.0) * torch.relu(1 - longs)
    total = total + penalty(w, market, settings)
    if settings.impact:
        t = (w - held).abs()
        total = total + settings.impact * (market.vol * torch.sqrt(t * equity / market.adv + 1e-12)
                                           * t).sum()  # fmt: skip
    return total
