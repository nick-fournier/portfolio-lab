"""The signal scoreboard: how well did each signal's ranking predict the returns that followed?

Each signal declares its ``horizon``: 1 session (a day), 5 (a week), 21 (a month) or 63
(a quarter). At the
end of every period of that length from the start date, the signal scores a pool of stocks
using only data up to that close, and the scores are compared with each stock's return
over the next ``horizon`` sessions, so samples don't overlap. Per date:

- ``ic``: the rank information coefficient, the correlation between the ranking and the
  realized return ranking (Spearman). Zero means no skill; 0.02 to 0.05 is typical of
  signals that are useful in practice, because it compounds over many stocks and months.
- ``top`` / ``bottom``: the average forward return of the top and bottom fifth by score.

This measures the prediction itself, separately from portfolio construction and costs.
Forward returns follow the backtest's rules: a stock acquired mid-window earns nothing
after its last bar, and one that fell to OTC takes the delisting return.
"""

from collections.abc import Callable, Sequence
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from portfolio_lab.core.calendar import Frequency, rebalance_dates
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import Panel

HORIZON = 21
#: Evaluation frequency for each supported horizon (sessions).
FREQUENCY: dict[int, Frequency] = {1: "D", 5: "W", 21: "M", 63: "Q"}
#: Fewest scored stocks for a date to count.
MIN_NAMES = 20
#: Candidate pools: production's, the 500 most liquid eligible stocks.
POOLS: dict[str, Callable[[DataView], list[str]]] = {
    "top500": lambda view: view.top_liquid(500),
}
SCHEMA = {
    "signal": pl.String,
    "pool": pl.String,
    "horizon": pl.Int64,
    "date": pl.Date,
    "n": pl.Int64,
    "ic": pl.Float64,
    "top": pl.Float64,
    "bottom": pl.Float64,
}


def forward_returns(
    panel: Panel, index: int, horizon: int = HORIZON, delisting_return: float = -0.30
) -> np.ndarray:
    """Every symbol's compounded close-to-close return over the next ``horizon`` sessions.

    Missing bars count as zero return (after an acquisition the money sits in cash). A
    stock that fell to OTC and never trades after the window starts takes
    ``delisting_return`` on top.
    """
    end = index + horizon
    rets = np.nan_to_num(panel.field("ret_cc")[index + 1 : end + 1])
    growth = np.prod(1 + rets, axis=0)
    delisted = panel.fell_to_otc & (panel.last_bar >= index) & (panel.last_bar < end)
    return np.where(delisted, growth * (1 + delisting_return), growth) - 1


def _rank_ic(scores: np.ndarray, returns: np.ndarray) -> float:
    """Spearman correlation (ties get average ranks)."""
    a, b = pd.Series(scores).rank(), pd.Series(returns).rank()
    return float(a.corr(b))


def score_date(scores: dict[str, float], forward: np.ndarray, panel: Panel) -> dict | None:
    """IC and top/bottom-fifth returns for one date's scores.

    ``None`` if too few names, or if every score is the same (no ranking to judge).
    """
    scores = {s: v for s, v in scores.items() if np.isfinite(v)}
    if len(scores) < MIN_NAMES:
        return None
    x = np.array(list(scores.values()))
    if np.ptp(x) == 0:
        return None
    y = forward[[panel.symbol_index[s] for s in scores]]
    # Quintiles by score quantile, so tied scores (e.g. F-scores) land in the same group.
    hi, lo = np.quantile(x, 0.8), np.quantile(x, 0.2)
    top, bottom = y[x >= hi], y[x <= lo]
    return {
        "n": len(x),
        "ic": _rank_ic(x, y),
        "top": float(top.mean()),
        "bottom": float(bottom.mean()),
    }


def evaluate(
    signal: object,
    label: str,
    panel: Panel,
    start: date,
    pools: Sequence[str] = tuple(POOLS),
    delisting_return: float = -0.30,
) -> pl.DataFrame:
    """Score ``signal`` at the end of every period from ``start`` with a complete forward window.

    Args:
        signal: An object with ``score(view, symbols) -> {symbol: score}`` and optionally a
            ``horizon`` in sessions (default :data:`HORIZON`; one of :data:`FREQUENCY`).
        label: Name stored in the ``signal`` column (e.g. ``forecast (model=historical_mean)``).
        panel: The data.
        start: First evaluation date.
        pools: Keys of :data:`POOLS` to evaluate.
        delisting_return: Return applied to stocks that fell to OTC in the window.

    Returns:
        One row per (pool, date) in :data:`SCHEMA`.

    Raises:
        ValueError: For an unsupported horizon.
    """
    horizon = int(getattr(signal, "horizon", HORIZON))
    if horizon not in FREQUENCY:
        raise ValueError(f"horizon {horizon} not supported; use one of {sorted(FREQUENCY)}")
    last = len(panel.dates) - 1 - horizon
    days = [d for d in panel.dates[: last + 1] if d >= start]
    rows = []
    for day in rebalance_dates(days, FREQUENCY[horizon]):
        i = panel.date_index[day]
        view = DataView(panel, i)
        forward = forward_returns(panel, i, horizon, delisting_return)
        for pool in pools:
            result = score_date(signal.score(view, POOLS[pool](view)), forward, panel)
            if result:
                rows.append(
                    {"signal": label, "pool": pool, "horizon": horizon, "date": day, **result}
                )
    return pl.DataFrame(rows, schema=SCHEMA)


def summarize(scores: pl.DataFrame) -> pl.DataFrame:
    """One row per (signal, pool, horizon) with the headline statistics.

    Columns: periods, names (average scored), mean_ic, ic_t (mean IC over its standard
    error; above about 2 is unlikely to be luck), hit (share of periods with positive IC),
    spread (top minus bottom fifth, annualized), top and bottom (each annualized).
    """
    per_year = 252 / pl.col("horizon").first()
    return (
        scores.group_by("signal", "pool", "horizon")
        .agg(
            pl.len().alias("periods"),
            pl.col("n").mean().alias("names"),
            pl.col("ic").mean().alias("mean_ic"),
            (pl.col("ic").mean() / pl.col("ic").std() * pl.len().sqrt()).alias("ic_t"),
            (pl.col("ic") > 0).mean().alias("hit"),
            ((pl.col("top") - pl.col("bottom")).mean() * per_year).alias("spread"),
            ((1 + pl.col("top")).product() ** (per_year / pl.len()) - 1).alias("top"),
            ((1 + pl.col("bottom")).product() ** (per_year / pl.len()) - 1).alias("bottom"),
        )
        .sort(["pool", "horizon", "mean_ic"], descending=[True, False, True])
    )
