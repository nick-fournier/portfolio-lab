"""The optimizer's data: one row per stock-month, then per-month arrays.

:func:`build` joins the derived monthly table (every stock characteristic) with what the
models and the backtest need at each month end: whether the stock may be held under
production's rules, its trailing dollar volume (to pick the most liquid candidates), and
next month's return over the T-bill (the same definition as the Forecaster's target:
missing bars earn zero, a stock that fell to OTC takes the delisting return).

:func:`months` turns that table into what one training step sees: for each month end, the
candidates (the ``pool`` most liquid stocks eligible under production's rules, among those
missing at most a third of the characteristics, as in the paper), their
characteristics ranked across the candidates into [-0.5, 0.5] (missing = 0, the middle), and
their next-month excess returns.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.scoreboard import forward_returns

#: Monthly-table columns that are not characteristics.
NOT_INPUTS = ("date", "symbol", "sic2")


@dataclass(frozen=True)
class Month:
    """One month end's candidates.

    Attributes:
        date: The month end (decision date).
        symbols: The candidates, in row order.
        x: Characteristics ranked across the candidates, (candidates x inputs), in [-0.5, 0.5].
        r: Next month's return over the T-bill, (candidates,); NaN in the latest month.
    """

    date: date
    symbols: list[str]
    x: np.ndarray
    r: np.ndarray


def inputs(table: pl.DataFrame) -> list[str]:
    """The characteristic columns of a :func:`build` table, in table order."""
    extra = {"eligible", "adv", "actual"}
    return [c for c in table.columns if c not in NOT_INPUTS and c not in extra]


def build(root: Path, panel: Panel | None = None) -> pl.DataFrame:
    """Stock-months of the derived monthly table with eligibility, liquidity and the target.

    Args:
        root: The data hive.
        panel: Prices under production's eligibility rules (default: loaded from ``root``).

    Returns:
        The monthly table's columns plus ``eligible`` (production rules), ``adv`` (trailing
        median dollar volume) and ``actual`` (next month's return minus the T-bill; null in
        the latest month), for month ends the panel has.
    """
    panel = panel or Panel.load(root)
    monthly = pl.read_parquet(DataPaths(root).features)
    ends = sorted(d for d in monthly["date"].unique().to_list() if d in panel.date_index)
    # The table's last row may be a month in progress (today's derive): no full month after
    # the one before it, so that month's target stays unknown too.
    known = len(ends) - 1 - (ends[-1] == panel.dates[-1])
    frames = []
    for k, day in enumerate(ends):
        i = panel.date_index[day]
        actual = np.full(len(panel.symbols), np.nan)
        if k < known:
            rf = DataView(panel, i).risk_free() / 12
            actual = forward_returns(panel, i, panel.date_index[ends[k + 1]] - i) - rf
        frames.append(pl.DataFrame({
            "date": [day] * len(panel.symbols), "symbol": panel.symbols,
            "eligible": panel.eligible[i], "adv": panel.field("adv")[i].astype(np.float64),
            "actual": actual.astype(np.float64),
        }))  # fmt: skip
    priced = pl.concat(frames).with_columns(pl.col("adv", "actual").fill_nan(None))
    return monthly.join(priced, on=["date", "symbol"], how="inner").sort("date", "symbol")


def _ranked(values: np.ndarray) -> np.ndarray:
    """Each column ranked across rows into [-0.5, 0.5]; missing values go to 0."""
    out = np.zeros_like(values, dtype=np.float32)
    for j in range(values.shape[1]):
        ok = np.isfinite(values[:, j])
        n = int(ok.sum())
        if n > 1:  # ranks 1..n, ties averaged, mapped to (-0.5, 0.5)
            out[ok, j] = (rankdata(values[ok, j]) - 0.5) / n - 0.5
    return out


def months(
    table: pl.DataFrame, pool: int, columns: list[str] | None = None, max_missing: float = 1 / 3
) -> list[Month]:
    """Per-month candidates and their ranked characteristics (module docs).

    Args:
        table: From :func:`build`.
        pool: Candidates per month: the most liquid stocks eligible under production's rules.
        columns: Characteristics to use (default: all of :func:`inputs`).
        max_missing: Stocks missing more than this share of the characteristics are left out
            before the most liquid are picked (the paper's filter, sec. 4.1).

    Returns:
        One :class:`Month` per month end, ascending.
    """
    columns = columns or inputs(table)
    out = []
    missing = pl.sum_horizontal(
        pl.col(c).cast(pl.Float64).is_nan().fill_null(True) for c in columns
    )
    usable = table.filter(
        pl.col("eligible") & pl.col("adv").is_not_null() & (missing <= max_missing * len(columns))
    )
    for (day,), month in usable.group_by("date", maintain_order=True):
        g = month.sort("adv", descending=True).head(pool)
        x = g.select(pl.col(c).cast(pl.Float64) for c in columns).to_numpy()
        r = g["actual"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        out.append(Month(day, g["symbol"].to_list(), _ranked(x), r.astype(np.float32)))
    return sorted(out, key=lambda m: m.date)
