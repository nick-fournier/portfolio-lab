"""Taxes on a backtest in a taxable account: tax lots, yearly bills and after-tax growth.

A run's trade log (``trades.parquet``, written by the engine) lists, per event, each
holding's value before and after it plus the dividends (or, for cash, interest) earned
since its previous row, all relative to a starting NAV of 1. :func:`after_tax` replays it
with real tax lots:

- Every purchase is a lot (date, cost). Sales pick lots in the chosen order and realize
  short-term gains (held a year or less) or long-term ones. Reinvested dividends are new
  lots at their value, since they were taxed when paid.
- Each calendar year nets short- against long-term results as the IRS does; a net loss
  offsets up to $3,000 of ordinary income and the rest carries forward. Dividends and
  interest are taxed as received.
- The bill is paid at the first event on or after April 15 by selling a slice of every
  holding, so the after-tax portfolio holds the same weights as the backtest, just less
  of it.

At the end the portfolio is optionally sold, paying tax on every remaining gain.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import polars as pl

#: A lot held longer than this many days is long-term.
LONG_TERM_DAYS = 365
#: Trade-log key for cash (taxed interest, no lots).
CASH = "_cash"
#: Day of the year a year's tax is due.
DUE = (4, 15)

LotOrder = Literal["tax", "fifo"]


@dataclass(frozen=True)
class TaxRates:
    """Combined federal + state marginal rates.

    Args:
        short: Short-term gains and interest (ordinary income).
        long: Long-term gains.
        dividends: Dividends (qualified).
        loss_offset: Net capital loss deductible against ordinary income each year, in dollars.
    """

    short: float = 0.37
    long: float = 0.28
    dividends: float = 0.28
    loss_offset: float = 3000.0


class Lots:
    """Tax lots per holding: ``[purchase day ordinal, cost, current value]``.

    Args:
        rates: Rates used to rank lots when selling in tax order.
        order: ``"tax"`` sells the lots costing the least tax per dollar first (losses,
            then small long-term gains, ...); ``"fifo"`` sells the oldest first.
    """

    def __init__(self, rates: TaxRates | None = None, order: LotOrder = "tax"):
        self.rates = rates or TaxRates()
        self.order = order
        self.lots: dict = {}

    def value(self, key) -> float:
        """Current value of a holding."""
        return sum(lot[2] for lot in self.lots.get(key, ()))

    def revalue(self, key, value: float) -> None:
        """Scale a holding's lots to a new total value (its price moved)."""
        lots = self.lots.get(key)
        if not lots:
            return
        old = sum(lot[2] for lot in lots)
        if old <= 0:
            return
        for lot in lots:
            lot[2] *= value / old

    def buy(self, key, day: int, amount: float) -> None:
        """Add a lot bought on ``day`` for ``amount``."""
        if amount > 0:
            self.lots.setdefault(key, []).append([day, amount, amount])

    def _ranked(self, key, day: int) -> list:
        lots = self.lots.get(key, [])
        if self.order == "fifo":
            return sorted(lots, key=lambda lot: lot[0])

        def tax_per_dollar(lot: list) -> float:
            short = day - lot[0] <= LONG_TERM_DAYS
            gain = (lot[2] - lot[1]) / lot[2] if lot[2] > 0 else 0.0
            return gain * (self.rates.short if short else self.rates.long)

        return sorted(lots, key=tax_per_dollar)

    def sell(self, key, day: int, amount: float) -> tuple[float, float]:
        """Sell ``amount`` of a holding; return the realized (short-term, long-term) gains."""
        short = long = 0.0
        left = amount
        for lot in self._ranked(key, day):
            if left <= 1e-15:
                break
            part = min(1.0, left / lot[2]) if lot[2] > 0 else 1.0
            gain = part * (lot[2] - lot[1])
            if day - lot[0] <= LONG_TERM_DAYS:
                short += gain
            else:
                long += gain
            left -= part * lot[2]
            lot[1] *= 1 - part
            lot[2] *= 1 - part
        self.lots[key] = [lot for lot in self.lots.get(key, []) if lot[2] > 1e-15]
        if not self.lots[key]:
            del self.lots[key]
        return short, long

    def close(self, key, day: int) -> tuple[float, float]:
        """Sell a holding entirely."""
        return self.sell(key, day, self.value(key) + 1.0)

    def free_of_short_gains(self, key, day: int) -> float:
        """Value that can be sold without realizing a short-term gain."""
        return sum(
            lot[2]
            for lot in self.lots.get(key, ())
            if lot[2] <= lot[1] or day - lot[0] > LONG_TERM_DAYS
        )

    def unrealized(self, day: int) -> tuple[float, float]:
        """Unrealized (short-term, long-term) gains across every holding."""
        short = long = 0.0
        for lots in self.lots.values():
            for lot in lots:
                if day - lot[0] <= LONG_TERM_DAYS:
                    short += lot[2] - lot[1]
                else:
                    long += lot[2] - lot[1]
        return short, long


@dataclass
class _Year:
    """One tax year's realized results, in dollars."""

    short: float = 0.0
    long: float = 0.0
    dividends: float = 0.0
    interest: float = 0.0


@dataclass
class _Carry:
    """Capital losses carried into later years (both <= 0)."""

    short: float = 0.0
    long: float = 0.0


def year_tax(year: _Year, carry: _Carry, rates: TaxRates) -> tuple[float, _Carry]:
    """The year's tax (negative is a saving on other income) and the losses carried on."""
    short, long = year.short + carry.short, year.long + carry.long
    net = short + long
    if short < 0 < long or long < 0 < short:  # one side's loss offsets the other's gain
        short, long = (0.0, net) if (short < 0) == (net >= 0) else (net, 0.0)
    tax = max(short, 0) * rates.short + max(long, 0) * rates.long
    tax += year.dividends * rates.dividends + year.interest * rates.short
    loss_short, loss_long = -min(short, 0.0), -min(long, 0.0)
    used = min(loss_short + loss_long, rates.loss_offset)
    tax -= used * rates.short
    from_short = min(loss_short, used)
    return tax, _Carry(-(loss_short - from_short), -(loss_long - (used - from_short)))


@dataclass
class AfterTax:
    """Result of :func:`after_tax`, in dollars.

    Attributes:
        growth: date, before (pre-tax value) and after (after-tax value) per trading day.
        taxes: Total tax paid (net of savings), including any final sale.
        short_gains: Short-term gains realized (net of short-term losses), over the run.
        long_gains: Long-term gains realized, likewise.
        dividends: Dividends and interest received.
        unrealized: Gains still untaxed at the end (zero after a final sale).
        by_year: year, tax per tax year.
    """

    growth: pl.DataFrame
    taxes: float
    short_gains: float
    long_gains: float
    dividends: float
    unrealized: float
    by_year: list[tuple[int, float]] = field(default_factory=list)


#: One trade-log event: date, keys, and per key the value before, after and income.
Event = tuple[date, list[str], np.ndarray, np.ndarray, np.ndarray]


def events(trades: pl.DataFrame) -> list[Event]:
    """The trade log grouped into events, ready for repeated :func:`after_tax` calls."""
    groups = trades.sort("seq").partition_by("seq", maintain_order=True)
    columns = ("before", "after", "income")
    return [(g["date"][0], g["key"].to_list(), *(g[c].to_numpy() for c in columns)) for g in groups]


class _Replay:
    """State of :func:`after_tax` while it walks the trade log."""

    def __init__(self, rates: TaxRates, start: float, order: LotOrder):
        self.rates = rates
        self.lots = Lots(rates, order)
        self.scale = start  # after-tax dollars per unit of pre-tax NAV
        self.year, self.carry = _Year(), _Carry()
        self.current_year: int | None = None
        self.due: tuple[date, float] | None = None  # (due date, amount)
        self.paid: list[tuple[date, float]] = []  # (date, scale after paying)
        self.by_year: list[tuple[int, float]] = []
        self.short = self.long = self.income = 0.0
        self.day: date | None = None
        #: (year, month) -> [short, long, dividends, interest] realized, in dollars, then
        #: the latest unrealized (short, long) gains as shares of the portfolio.
        self.months: dict[tuple[int, int], list[float]] = {}

    def _month(self) -> list[float]:
        key = (self.day.year, self.day.month)
        return self.months.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def realize(self, gains: tuple[float, float]) -> None:
        self.year.short += gains[0]
        self.year.long += gains[1]
        self.short += gains[0]
        self.long += gains[1]
        month = self._month()
        month[0] += gains[0]
        month[1] += gains[1]

    def close_year(self) -> float:
        """Tax the finished year; return its bill."""
        tax, self.carry = year_tax(self.year, self.carry, self.rates)
        self.by_year.append((self.current_year, tax))
        self.year = _Year()
        return tax

    def event(self, event: Event) -> None:
        """Apply one event: income and price moves, any tax due, then the trades."""
        day, keys, before, after, income = event
        self.day = day
        ordinal = day.toordinal()
        if self.current_year is not None and day.year != self.current_year:
            owed = self.close_year() + (self.due[1] if self.due else 0.0)
            self.due = (date(day.year, *DUE), owed)
        self.current_year = day.year
        for key, b, inc in zip(keys, before, income, strict=True):
            received = max(inc, 0.0) * self.scale
            self.income += received
            if key == CASH:
                self.year.interest += received
                self._month()[3] += received
                continue
            self.year.dividends += received
            self._month()[2] += received
            self.lots.revalue(key, max(b - max(inc, 0.0), 0.0) * self.scale)
            self.lots.buy(key, ordinal, received)
        # Pay only at full snapshots (cash present), where every holding is then trimmed.
        if self.due is not None and day >= self.due[0] and CASH in keys:
            self.pay(day, before.sum() * self.scale, self.due[1])
            self.due = None
        for key, a in zip(keys, after, strict=True):
            gap = a * self.scale - self.lots.value(key) if key != CASH else 0.0
            if gap > 1e-9 * self.scale:
                self.lots.buy(key, ordinal, gap)
            elif gap < -1e-9 * self.scale:
                self.realize(self.lots.sell(key, ordinal, -gap))
        if CASH in keys:  # a full snapshot: note how much of the portfolio is untaxed gain
            value = after.sum() * self.scale
            if value > 0:
                month = self._month()
                month[4:6] = [g / value for g in self.lots.unrealized(ordinal)]

    def pay(self, day: date, value: float, tax: float) -> None:
        """Pay ``tax`` out of a portfolio worth ``value``, shrinking every holding alike."""
        if value > 0:
            self.scale *= max(value - tax, 0.0) / value
        self.paid.append((day, self.scale))

    def finish(self, day: date, nav: float, sell: bool) -> float:
        """Settle the final year (selling everything if ``sell``); return untaxed gains."""
        ordinal = day.toordinal()
        unrealized = sum(self.lots.unrealized(ordinal))
        if sell:
            for key in list(self.lots.lots):
                self.realize(self.lots.close(key, ordinal))
            unrealized = 0.0
        owed = self.close_year() + (self.due[1] if self.due else 0.0)
        self.pay(day, nav * self.scale, owed)
        return unrealized


def after_tax(
    trades: pl.DataFrame | list[Event],
    nav: pl.DataFrame,
    rates: TaxRates,
    start: float = 100_000.0,
    order: LotOrder = "tax",
    sell_at_end: bool = False,
) -> AfterTax:
    """Replay a run's trade log in a taxable account.

    Args:
        trades: The run's trade log (seq, date, key, before, after, income), or its
            :func:`events`.
        nav: The run's daily date and nav (pre-tax, starting at 1).
        rates: Tax rates.
        start: Starting amount in dollars (matters only for the $3,000 loss offset).
        order: Which lots to sell first (see :class:`Lots`).
        sell_at_end: Sell everything on the last day and pay the tax due.
    """
    replay = _Replay(rates, start, order)
    for event in events(trades) if isinstance(trades, pl.DataFrame) else trades:
        replay.event(event)
    unrealized = 0.0
    if replay.current_year is not None:
        unrealized = replay.finish(nav["date"][-1], float(nav["nav"][-1]), sell_at_end)
    return AfterTax(
        growth=_growth(nav, start, replay.paid),
        taxes=sum(t for _, t in replay.by_year),
        short_gains=replay.short,
        long_gains=replay.long,
        dividends=replay.income,
        unrealized=unrealized,
        by_year=replay.by_year,
    )


def _growth(nav: pl.DataFrame, start: float, paid: list[tuple[date, float]]) -> pl.DataFrame:
    """Daily pre- and after-tax values: the after-tax scale steps down at each payment."""
    dates = nav["date"].to_numpy().astype("datetime64[D]")
    scale = np.full(len(dates), start)
    for day, after in paid:  # payments are in date order; each holds until the next
        scale[dates >= np.datetime64(day)] = after
    values = nav["nav"].to_numpy()
    return pl.DataFrame({"date": nav["date"], "before": values * start, "after": values * scale})


def monthly_gains(
    trades: pl.DataFrame, nav: pl.DataFrame
) -> tuple[pl.DataFrame, tuple[float, float]]:
    """What a run realizes each month with no tax paid: the inputs of :func:`approx_after_tax`.

    Rate-free: lots are sold in tax order (losses, then long-term gains, then short-term),
    which barely depends on the rates.

    Returns:
        Per month: date (last session), nav, short, long, dividends, interest (in pre-tax
        NAV units), open_short and open_long (unrealized gains as shares of the portfolio);
        and the (short, long) gains still unrealized on the last day.
    """
    replay = _Replay(TaxRates(0.0, 0.0, 0.0, 0.0), 1.0, "tax")
    replay.lots = Lots(TaxRates(), "tax")
    for event in events(trades):
        replay.event(event)
    unrealized = replay.lots.unrealized(nav["date"][-1].toordinal())
    flows = pl.DataFrame(
        [(y, m, *v) for (y, m), v in replay.months.items()],
        schema=[
            "year",
            "month",
            "short",
            "long",
            "dividends",
            "interest",
            "open_short",
            "open_long",
        ],
        orient="row",
    )
    ends = (
        nav.with_columns(
            pl.col("date").dt.year().alias("year"), pl.col("date").dt.month().alias("month")
        )
        .group_by("year", "month", maintain_order=True)
        .last()
    )
    months = ends.join(flows, on=["year", "month"], how="left").fill_null(0.0).sort("date")
    columns = ("date", "nav", "short", "long", "dividends", "interest", "open_short", "open_long")
    return months.select(columns), unrealized


def approx_after_tax(
    months: pl.DataFrame,
    unrealized: tuple[float, float],
    rates: TaxRates,
    start: float = 100_000.0,
) -> list[tuple[date, float]]:
    """Fast after-tax growth from :func:`monthly_gains`, selling everything at the end.

    Each month's gains and income are scaled by the after-tax portfolio's size; a year's
    tax (netted as in :func:`year_tax`) is paid at the end of the next April, and the final
    year's, with every remaining gain, on the last day. Selling to pay the tax realizes the
    portfolio's average share of untaxed gain. The page's JavaScript mirrors this function.

    Returns:
        (date, after-tax dollars per unit of pre-tax NAV) from each payment on, as
        :func:`_growth` takes them.
    """
    scale, carry, year, owed = start, _Carry(), _Year(), 0.0
    paid: list[tuple[date, float]] = []
    current = None
    rows = months.iter_rows(named=True)
    for row in rows:
        day = row["date"]
        if current is not None and day.year != current:
            tax, carry = year_tax(year, carry, rates)
            owed += tax
            year = _Year()
        current = day.year
        year.short += row["short"] * scale
        year.long += row["long"] * scale
        year.dividends += row["dividends"] * scale
        year.interest += row["interest"] * scale
        if day.month == DUE[0] and owed:
            value = row["nav"] * scale
            year.short += owed * row["open_short"]
            year.long += owed * row["open_long"]
            scale *= max(value - owed, 0.0) / value
            paid.append((day, scale))
            owed = 0.0
    last = months.row(-1, named=True)
    year.short += unrealized[0] * scale
    year.long += unrealized[1] * scale
    tax, _ = year_tax(year, carry, rates)
    value = last["nav"] * scale
    scale *= max(value - tax - owed, 0.0) / value
    paid.append((last["date"], scale))
    return paid
