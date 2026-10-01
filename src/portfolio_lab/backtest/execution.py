"""How target weights become trades, and the trade log that taxes are computed from.

:class:`Execution` makes the portfolio sticky, both off by default:

- **Skip small changes** (``band``): a holding the strategy keeps, whose target differs from
  its current weight by less than ``band``, is left alone. New names and dropped names
  always trade, so the portfolio never collects leftovers.
- **Defer short-term gains** (``defer_short_gains``): a sale that would realize a gain on
  shares held a year or less is postponed until they turn long-term; losses and long-term
  shares are sold first.

Names left alone keep their weight; the rest share what is left of the target's total in
proportion to their targets.

:class:`TradeLog` records, at each trade, each month end and each forced sale, every
holding's value before and after plus the dividends earned since its previous row
(see ``backtest.tax``).
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from portfolio_lab.backtest.tax import CASH, Lots

#: Daily implied dividend yields outside this range are splits or bad data, not dividends.
DIVIDEND_RANGE = (-0.05, 0.25)


@dataclass(frozen=True)
class Execution:
    """Stickiness settings (see module docs).

    Args:
        band: Smallest weight change worth trading (0.02 = 2 percentage points).
        defer_short_gains: Hold shares with short-term gains until they turn long-term.
    """

    band: float = 0.0
    defer_short_gains: bool = False

    @property
    def active(self) -> bool:
        """Whether anything differs from trading straight to the targets."""
        return self.band > 0 or self.defer_short_gains

    def describe(self) -> str:
        """Short label, e.g. ``skip < 2%, defer short-term gains``."""
        parts = [f"skip < {self.band:.0%}"] if self.band > 0 else []
        if self.defer_short_gains:
            parts.append("defer short-term gains")
        return ", ".join(parts)


def adjust(
    target: np.ndarray,
    held: np.ndarray,
    execution: Execution,
    lots: Lots | None = None,
    day: int = 0,
    nav: float = 1.0,
) -> np.ndarray:
    """Weights to trade to: ``target`` with small changes skipped and short gains deferred.

    Args:
        target: The strategy's target weights.
        held: Current (drifted) weights.
        execution: Stickiness settings.
        lots: Pre-tax lots in NAV units, keyed by symbol index (needed to defer gains).
        day: Today's date ordinal.
        nav: Portfolio value, converting lot values to weights.
    """
    new = target.copy()
    fixed = np.zeros(len(target), dtype=bool)
    if execution.band > 0:
        small = (np.abs(target - held) < execution.band) & (held > 0) & (target > 0)
        new[small] = held[small]
        fixed |= small
    if execution.defer_short_gains and lots is not None:
        for j in np.flatnonzero(new < held - 1e-12):
            floor = held[j] - lots.free_of_short_gains(int(j), day) / nav
            if new[j] < floor:
                new[j] = floor
                fixed[j] = True
    loose = ~fixed
    wanted = target[loose].sum()
    if wanted > 0:
        room = max(target.sum() - new[fixed].sum(), 0.0)
        new[loose] = target[loose] * room / wanted
    return new


class TradeLog:
    """Records holdings and dividends for after-tax replays (see module docs).

    Args:
        symbols: Panel symbols, indexed like the holdings vector.
    """

    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.income = np.zeros(len(symbols))
        self.interest = 0.0
        self.rows: list[tuple] = []
        self.seq = 0

    def accrue(self, holdings: np.ndarray, cash: float, ret_cc: np.ndarray,
               close: np.ndarray, prev_close: np.ndarray, rf_daily: float) -> None:  # fmt: skip
        """Add a day's dividends on last night's holdings (implied by total vs price return)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = (1 + ret_cc) * prev_close / close - 1
        lo, hi = DIVIDEND_RANGE
        ok = np.isfinite(implied) & (implied > lo) & (implied < hi) & (holdings > 0)
        self.income[ok] += holdings[ok] * implied[ok]
        self.interest += cash * rf_daily

    def record(
        self,
        day: date,
        before: np.ndarray,
        after: np.ndarray,
        cash_before: float | None = None,
        cash_after: float | None = None,
    ) -> None:
        """One event: holdings before and after (and cash, for full snapshots)."""
        for j in np.flatnonzero((before > 0) | (after > 0)):
            self.rows.append((self.seq, day, self.symbols[j], float(before[j]),
                              float(after[j]), float(self.income[j])))  # fmt: skip
            self.income[j] = 0.0
        if cash_before is not None:
            self.rows.append((self.seq, day, CASH, cash_before, cash_after, self.interest))
            self.interest = 0.0
        self.seq += 1

    def frame(self) -> pl.DataFrame:
        """The log: seq, date, key, before, after, income."""
        schema = {"seq": pl.Int64, "date": pl.Date, "key": pl.String, "before": pl.Float64,
                  "after": pl.Float64, "income": pl.Float64}  # fmt: skip
        return pl.DataFrame(self.rows, schema=schema, orient="row")
