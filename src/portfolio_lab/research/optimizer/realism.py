"""A retail account's costs that depend on the book's size (2026-10-17): the loss's realism terms.

The loss measures the book in dollars per $1 of equity (w), so the account's own limits can be
priced in it and the model left to choose its size. Each term is a monthly charge on the
return in the loss (research.optimizer.priced), off unless its setting is on:

* lost short rebate (``rebate``): a retail broker keeps the interest on short-sale proceeds,
  so each short dollar forgoes the month's T-bill: ``tbill * sum relu(-w)``.
* margin interest: longs beyond equity are borrowed at the broker's rate; against the T-bill
  the loss already assumes, that costs the excess over it. ``margin_rate`` (a year): the
  broker's rate, ``relu(margin_rate / 12 - tbill) * relu(sum relu(w) - 1)``; else
  ``margin_spread`` (a year): a fixed spread over the T-bill.
* cash rate (``cash_rate``, a year; ``None``: the T-bill): equity not spent on longs earns the
  broker's cash rate, so it forgoes ``relu(tbill - cash_rate / 12) * relu(1 - sum relu(w))``.
* maintenance margin (``margin_cushion``): the collateral the broker requires, per dollar held,
  from Alpaca's price tables (longs: 100% under $2.50, 50% under $6, else 30%; shorts: under $5
  the larger of $2.50 a share and 100%, else the larger of $5 a share and 30%). Per-share
  requirements are per dollar too (``2.5 / price``), so no account size is needed. Requirement
  above ``margin_cushion`` x equity is charged ``cushion_rate`` a month per dollar.
* market impact (``impact``, the constant K): trading t dollars per $1 of equity in a stock
  with daily volatility s and daily dollar volume V costs ``K * s * sqrt(t * equity / V) * t``
  (square-root law), on the dollars traded from the book held going in.

:func:`align` reads what they need from the cost table (columns ``prc``, ``rvol_21d``, ``adv``
and ``tbill``, the holding month's T-bill, monthly).
"""

from dataclasses import dataclass, replace

import numpy as np
import polars as pl
import torch

from portfolio_lab.research.optimizer.data import Month

#: Columns :func:`align` needs in the cost table.
COLUMNS = ("prc", "rvol_21d", "adv", "tbill")
#: The Alpaca account profile (2026-10-17; docs.alpaca.markets margin-and-short-selling, Alpaca
#: high-yield cash): what the walk's ``--profile alpaca`` sets and the scorer charges. Easy-to-
#: borrow shorts cost nothing (the cost table's borrow fee), hard-to-borrow ones are barred.
ALPACA = {"rebate": True, "margin_rate": 0.065, "cash_rate": 0.0356, "margin_cushion": 0.6,
          "cushion_rate": 0.10, "impact": 1.0, "equity": 100_000.0}  # fmt: skip


@dataclass(frozen=True)
class Market:
    """One month's per-stock price, daily volatility and daily dollar volume, and its T-bill."""

    price: torch.Tensor
    vol: torch.Tensor
    adv: torch.Tensor
    tbill: float


def with_profile(settings, name: str):
    """``settings`` with an account profile's realism terms (``alpaca``)."""
    profiles = {"alpaca": ALPACA}
    if name not in profiles:
        raise ValueError(f"profile must be one of {sorted(profiles)}, not {name!r}")
    return replace(settings, **profiles[name])


def on(settings) -> bool:
    """Whether any realism term is switched on."""
    return bool(settings.rebate or settings.margin_spread or settings.margin_rate
                or settings.cash_rate is not None or settings.margin_cushion
                or settings.impact)  # fmt: skip


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
        out.append(Market(*cols, float(tb[0]) if len(tb) else 0.0))
    return out


def requirement(w: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
    """Alpaca's maintenance requirement, per $1 of equity, of the book ``w`` (module docs)."""
    long_rate = torch.where(price < 2.5, 1.0, torch.where(price < 6.0, 0.5, 0.3))
    short_rate = torch.where(price < 5.0, torch.clamp(2.5 / price, min=1.0),
                             torch.clamp(5.0 / price, min=0.3))  # fmt: skip
    return (long_rate * torch.relu(w)).sum() + (short_rate * torch.relu(-w)).sum()


def charges(w: torch.Tensor, held: torch.Tensor, market: Market, settings,
            equity: float) -> torch.Tensor:  # fmt: skip
    """The month's realism charges on the book ``w`` traded from ``held`` (module docs)."""
    total = w.new_zeros(())
    if settings.rebate:
        total = total + market.tbill * torch.relu(-w).sum()
    longs = torch.relu(w).sum()
    if settings.margin_rate:
        total = total + max(settings.margin_rate / 12 - market.tbill, 0.0) * torch.relu(longs - 1)
    elif settings.margin_spread:
        total = total + settings.margin_spread / 12 * torch.relu(longs - 1)
    if settings.cash_rate is not None:
        total = total + max(market.tbill - settings.cash_rate / 12, 0.0) * torch.relu(1 - longs)
    if settings.margin_cushion:
        over = torch.relu(requirement(w, market.price) - settings.margin_cushion)
        total = total + settings.cushion_rate * over
    if settings.impact:
        t = (w - held).abs()
        total = total + settings.impact * (market.vol * torch.sqrt(t * equity / market.adv + 1e-12)
                                           * t).sum()  # fmt: skip
    return total
