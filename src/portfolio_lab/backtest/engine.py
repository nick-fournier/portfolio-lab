"""Walk-forward backtest engine.

Timing, per session ``t`` after the start:

1. **Decide** at the close of ``t-1`` (a rebalance date): the strategy sees a
   :class:`~portfolio_lab.research.dataview.DataView` bounded at ``t-1``.
2. **Overnight**: holdings earn the close-to-open return of ``t``.
3. **Trade** at the open of ``t`` to the target weights; costs are charged on the
   turnover measured against the *drifted* weights actually held.
4. **Intraday**: holdings earn the open-to-close return of ``t``; cash earns the daily
   risk-free rate.

Between rebalances weights drift with returns. A held name with no bar earns zero; after
``max_missing_days`` consecutive missing bars it is liquidated at its last value (halts,
delistings).
"""

import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date

import numpy as np
import polars as pl

from portfolio_lab.backtest.costs import CostModel
from portfolio_lab.backtest.metrics import compute
from portfolio_lab.backtest.results import RunResult
from portfolio_lab.core.calendar import rebalance_dates
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import Panel
from portfolio_lab.strategies.base import Strategy, Weights

log = logging.getLogger(__name__)

CAVEATS = (
    "Survivorship bias: the universe is today's listed stocks; delisted companies are missing.",
    "Universe classification (ETF, warrant, preferred, ...) is as of the latest snapshot.",
    "Costs are a deterministic model (half-spread + volume-based impact), not real fills.",
)
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class BacktestConfig:
    """Settings for one backtest run.

    Args:
        start: First session (the initial decision is made at its close).
        end: Last session.
        costs: Trading cost model.
        max_weight: Largest weight allowed in any single name.
        benchmark: Symbol whose returns are reported alongside the strategy.
        max_missing_days: Consecutive missing bars before a held name is liquidated.
    """

    start: date
    end: date
    costs: CostModel = field(default_factory=CostModel)
    max_weight: float = 1.0
    benchmark: str = "SPY"
    max_missing_days: int = 5


def validate_weights(
    weights: Weights, view: DataView, panel: Panel, max_weight: float
) -> np.ndarray:
    """Check target weights and return them as a vector aligned to ``panel.symbols``.

    Allowed names are the stocks eligible at the decision date plus price-only symbols
    (benchmark ETFs) that have a price then.

    Raises:
        ValueError: On unknown or ineligible names, non-finite or negative weights, a
            weight above ``max_weight``, or weights summing to more than 1.
    """
    closes = view.close(list(weights))
    allowed = set(view.eligible()) | {
        s for s in weights if s not in panel.universe and np.isfinite(closes.get(s, np.nan))
    }
    problems = [f"{s} not allowed at {view.asof}" for s in weights if s not in allowed]
    for symbol, w in weights.items():
        if not np.isfinite(w) or w < -_TOLERANCE or w > max_weight + _TOLERANCE:
            problems.append(f"{symbol} weight {w} outside [0, {max_weight}]")
    total = sum(weights.values())
    if total > 1 + _TOLERANCE:
        problems.append(f"weights sum to {total:.6f} > 1")
    if problems:
        raise ValueError("; ".join(problems[:5]))
    vector = np.zeros(len(panel.symbols))
    for symbol, w in weights.items():
        vector[panel.symbol_index[symbol]] = max(w, 0.0)
    return vector


def _params(strategy: Strategy) -> dict:
    """Strategy parameters for the run's metadata."""
    if is_dataclass(strategy):
        return {k: v for k, v in asdict(strategy).items() if k not in ("name", "schedule")}
    return {}


def _git_sha() -> str | None:
    """Short commit hash of the running code: ``GIT_SHA`` (set in the image) or git."""
    if sha := os.environ.get("GIT_SHA"):
        return sha[:7]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class _Portfolio:
    """Simulation state: dollar holdings per name, cash, and missing-bar counters.

    Values are relative to a starting NAV of 1. Each session runs :meth:`overnight`, then
    optionally :meth:`trade`, then :meth:`intraday` and :meth:`liquidate_stale`.
    """

    def __init__(self, n_symbols: int, costs: CostModel):
        self.holdings = np.zeros(n_symbols)
        self.missing = np.zeros(n_symbols, dtype=int)
        self.cash = 1.0
        self.costs = costs

    @property
    def nav(self) -> float:
        """Current portfolio value."""
        return float(self.holdings.sum() + self.cash)

    def overnight(self, ret_co: np.ndarray) -> None:
        """Apply close-to-open returns; names without an open earn zero."""
        self.holdings = self.holdings * (1 + np.where(np.isfinite(ret_co), ret_co, 0.0))

    def trade(self, target: np.ndarray, adv: np.ndarray) -> tuple[float, float]:
        """Rebalance to ``target`` weights at the open; return (turnover, cost fraction)."""
        nav_open = self.nav
        trades = target - self.holdings / nav_open
        turnover = float(np.abs(trades).sum())
        cost = self.costs.cost(trades, adv, nav_open)
        nav_after = nav_open * (1 - cost)
        self.holdings = target * nav_after
        self.cash = nav_after * (1 - target.sum())
        return turnover, cost

    def intraday(self, ret_cc: np.ndarray, ret_co: np.ndarray, rf_daily: float) -> None:
        """Apply open-to-close returns and the day's risk-free rate on cash."""
        co = np.where(np.isfinite(ret_co), ret_co, 0.0)
        oc = np.where(np.isfinite(ret_cc), (1 + np.nan_to_num(ret_cc)) / (1 + co) - 1, 0.0)
        self.holdings = self.holdings * (1 + oc)
        self.cash *= 1 + rf_daily
        self.missing = np.where((self.holdings > 0) & ~np.isfinite(ret_cc), self.missing + 1, 0)

    def liquidate_stale(self, max_missing_days: int) -> np.ndarray:
        """Move names missing bars for ``max_missing_days`` sessions to cash; return them."""
        stale = self.missing >= max_missing_days
        if stale.any():
            self.cash += float(self.holdings[stale].sum())
            self.holdings[stale] = 0.0
            self.missing[stale] = 0
        return np.flatnonzero(stale)


def _simulate(
    strategy: Strategy, panel: Panel, config: BacktestConfig, first: int, last: int
) -> tuple[list[tuple], list[tuple], int]:
    """Run the day loop; return daily rows, target-weight rows and forced liquidations."""
    rebalance = {
        panel.date_index[d]
        for d in rebalance_dates(panel.dates[first : last + 1], strategy.schedule)
    }
    decisions = ({first} | rebalance) - {last}
    ret_cc, ret_co, adv = panel.field("ret_cc"), panel.field("ret_co"), panel.field("adv")
    book = _Portfolio(len(panel.symbols), config.costs)
    prev_nav, pending = 1.0, None
    daily, weight_rows, liquidations = [], [], 0

    for i in range(first, last + 1):
        if i > first:
            book.overnight(ret_co[i])
            turnover = cost = 0.0
            if pending is not None:
                turnover, cost = book.trade(pending, adv[i - 1])
                pending = None
            book.intraday(ret_cc[i], ret_co[i], panel.rf_daily[i])
            stale = book.liquidate_stale(config.max_missing_days)
            if stale.size:
                liquidations += stale.size
                names = [panel.symbols[j] for j in stale]
                log.debug("liquidating %s on %s after missing bars", names, panel.dates[i])
            nav = book.nav
            held = int((book.holdings > 0).sum())
            daily.append(
                (panel.dates[i], nav, nav / prev_nav - 1, turnover, cost, book.cash / nav, held)
            )
            prev_nav = nav

        if i in decisions:
            view = DataView(panel, i)
            weights = strategy.target_weights(view)
            pending = validate_weights(weights, view, panel, config.max_weight)
            weight_rows += [(view.asof, s, float(w)) for s, w in sorted(weights.items()) if w > 0]
    return daily, weight_rows, liquidations


def run(strategy: Strategy, panel: Panel, config: BacktestConfig) -> RunResult:
    """Simulate ``strategy`` over ``panel`` and return the run's results.

    Args:
        strategy: The strategy to test.
        panel: Data covering the backtest period (plus any lookback history before it).
        config: Dates, costs and constraints.
    """
    window = [i for i, d in enumerate(panel.dates) if config.start <= d <= config.end]
    if len(window) < 2:
        raise ValueError(f"need at least two sessions between {config.start} and {config.end}")
    first, last = window[0], window[-1]
    daily, weight_rows, liquidations = _simulate(strategy, panel, config, first, last)

    columns = ["date", "nav", "ret", "turnover", "cost", "cash", "holdings"]
    daily_df = pl.DataFrame(daily, schema=columns, orient="row")
    days = slice(first + 1, last + 1)
    bench_col = panel.symbol_index.get(config.benchmark)
    bench = np.nan_to_num(panel.field("ret_cc")[days, bench_col]) if bench_col is not None else None
    if bench is not None:
        daily_df = daily_df.with_columns(pl.Series("benchmark_ret", bench))
    weights_df = pl.DataFrame(weight_rows, schema=["date", "symbol", "weight"], orient="row")

    metrics = compute(
        daily_df["ret"].to_numpy(),
        panel.rf_daily[days],
        bench,
        daily_df["turnover"].to_numpy(),
        daily_df["holdings"].to_numpy(),
    )
    metrics["forced_liquidations"] = float(liquidations)
    meta = {
        "strategy": strategy.name,
        "params": _params(strategy),
        "schedule": strategy.schedule,
        "start": panel.dates[first],
        "end": panel.dates[last],
        "benchmark": config.benchmark,
        "costs": asdict(config.costs),
        "max_weight": config.max_weight,
        "universe_size": len(panel.universe),
        "data_max_date": panel.dates[-1],
        "git_sha": _git_sha(),
        "caveats": list(CAVEATS),
    }
    return RunResult(meta=meta, metrics=metrics, daily=daily_df, weights=weights_df)
