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

import inspect
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import date

import numpy as np
import polars as pl

from portfolio_lab.backtest.costs import CostModel
from portfolio_lab.backtest.execution import Execution, TradeLog, adjust
from portfolio_lab.backtest.metrics import compute
from portfolio_lab.backtest.results import RunResult
from portfolio_lab.backtest.tax import Lots
from portfolio_lab.core.calendar import rebalance_dates
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import Panel
from portfolio_lab.strategies import explain
from portfolio_lab.strategies.base import Strategy, Weights

log = logging.getLogger(__name__)

CAVEATS = (
    "Delisted companies come from Tiingo's ticker list; names Alpaca has no bars for "
    "(mostly ones that only ever traded OTC) are still missing.",
    "A stock that fell to OTC and never traded on an exchange again is sold at its last "
    "exchange price plus the delisting return; acquisitions exit at the last price.",
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
        max_missing_days: Consecutive sessions without a trade before a held name is sold.
        delisting_return: Return applied when liquidating a name that fell to OTC and has
            no later bars (it kept trading OTC, usually far lower; Shumway, 1997, finds
            about -30%). Use -1.0 for a total loss or 0.0 to exit at the last price.
        execution: How targets become trades: skip small changes, defer short-term gains.
    """

    start: date
    end: date
    costs: CostModel = field(default_factory=CostModel)
    max_weight: float = 1.0
    benchmark: str = "SPY"
    max_missing_days: int = 5
    delisting_return: float = -0.30
    execution: Execution = field(default_factory=Execution)


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
    if not is_dataclass(strategy):
        return {}
    skip = {"name", "schedule", "description", "signal", "construct"}
    return {
        f.name: getattr(strategy, f.name)
        for f in fields(strategy)
        if f.name not in skip and f.metadata.get("param", True)
    }


_DOC_SECTIONS = ("Args:", "Returns:", "Raises:", "Attributes:", "Example")


def describe(strategy: Strategy, summary_only: bool = False) -> str:
    """Plain-text description for the dashboard.

    Uses a ``description`` attribute if set, else the class docstring's summary line and
    extended description (everything before its ``Args:``-style sections).

    Args:
        strategy: The strategy.
        summary_only: Return just the first paragraph (the one-line summary).
    """
    cls = type(strategy)
    doc = getattr(strategy, "description", None) or inspect.getdoc(cls) or ""
    if is_dataclass(cls) and doc.startswith(f"{cls.__name__}("):
        doc = ""  # the signature dataclasses generate when there is no docstring
    prose = []
    for paragraph in doc.split("\n\n"):
        if paragraph.lstrip().startswith(_DOC_SECTIONS):
            break
        prose.append(" ".join(paragraph.split()))
        if summary_only:
            break
    return " ".join(prose).replace("``", "")


def label(strategy: Strategy) -> str:
    """Short display name: the strategy's title, else its name and non-default parameters.

    Includes the rebalance schedule when it differs from the strategy's default.
    """
    if title := getattr(strategy, "title", None):
        return title
    if not is_dataclass(strategy):
        return strategy.name
    defaults = {f.name: f.default for f in fields(strategy)}
    changed = {k: v for k, v in _params(strategy).items() if defaults.get(k) != v}
    if "schedule" in defaults and strategy.schedule != defaults["schedule"]:
        changed["schedule"] = strategy.schedule
    return strategy.name + (
        f" ({', '.join(f'{k}={v}' for k, v in changed.items())})" if changed else ""
    )


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

    def __init__(self, n_symbols: int, costs: CostModel, execution: Execution | None = None):
        self.holdings = np.zeros(n_symbols)
        self.missing = np.zeros(n_symbols, dtype=int)
        self.cash = 1.0
        self.costs = costs
        self.execution = execution or Execution()
        self.lots = Lots() if self.execution.defer_short_gains else None

    @property
    def nav(self) -> float:
        """Current portfolio value."""
        return float(self.holdings.sum() + self.cash)

    def overnight(self, ret_co: np.ndarray) -> None:
        """Apply close-to-open returns; names without an open earn zero."""
        self.holdings = self.holdings * (1 + np.where(np.isfinite(ret_co), ret_co, 0.0))

    def trade(self, target: np.ndarray, adv: np.ndarray, day: int = 0) -> tuple[float, float]:
        """Rebalance to ``target`` weights at the open; return (turnover, cost fraction).

        With stickiness (:class:`Execution`), the target is first adjusted; ``day`` (a date
        ordinal) dates the tax lots it tracks.
        """
        nav_open = self.nav
        held = self.holdings / nav_open
        if self.lots is not None:
            for j in list(self.lots.lots):
                self.lots.revalue(j, self.holdings[j])
        if self.execution.active:
            target = adjust(target, held, self.execution, self.lots, day, nav_open)
        trades = target - held
        turnover = float(np.abs(trades).sum())
        cost = self.costs.cost(trades, adv, nav_open)
        nav_after = nav_open * (1 - cost)
        if self.lots is not None:
            for j in np.flatnonzero(np.abs(trades) > 1e-12):
                if trades[j] > 0:
                    self.lots.buy(int(j), day, trades[j] * nav_open)
                else:
                    self.lots.sell(int(j), day, -trades[j] * nav_open)
        self.holdings = target * nav_after
        self.cash = nav_after * (1 - target.sum())
        return turnover, cost

    def intraday(
        self, ret_cc: np.ndarray, ret_co: np.ndarray, rf_daily: float, traded: np.ndarray
    ) -> None:
        """Apply open-to-close returns and the day's risk-free rate on cash.

        Held names that did not trade (no bar, or a zero-volume one) count a missing day.
        """
        co = np.where(np.isfinite(ret_co), ret_co, 0.0)
        oc = np.where(np.isfinite(ret_cc), (1 + np.nan_to_num(ret_cc)) / (1 + co) - 1, 0.0)
        self.holdings = self.holdings * (1 + oc)
        self.cash *= 1 + rf_daily
        self.missing = np.where((self.holdings > 0) & ~traded, self.missing + 1, 0)

    def liquidate_stale(self, max_missing_days: int, exit_ret: np.ndarray) -> np.ndarray:
        """Move names missing bars for ``max_missing_days`` sessions to cash; return them.

        Args:
            max_missing_days: Consecutive missing bars that trigger a sale.
            exit_ret: Per-name return applied to the sale (the delisting return, or 0).
        """
        stale = self.missing >= max_missing_days
        if stale.any():
            if self.lots is not None:
                for j in np.flatnonzero(stale):
                    self.lots.lots.pop(int(j), None)
            self.cash += float((self.holdings[stale] * (1 + exit_ret[stale])).sum())
            self.holdings[stale] = 0.0
            self.missing[stale] = 0
        return np.flatnonzero(stale)


def _simulate(
    strategy: Strategy,
    panel: Panel,
    config: BacktestConfig,
    first: int,
    last: int,
    trades: TradeLog,
) -> tuple[list[tuple], list[tuple], tuple[int, int]]:
    """Run the day loop; return daily rows, target-weight rows and liquidation counts.

    ``trades`` records every trade, forced sale and month end, for after-tax replays.
    """
    rebalance = {
        panel.date_index[d]
        for d in rebalance_dates(panel.dates[first : last + 1], strategy.schedule)
    }
    decisions = ({first} | rebalance) - {last}
    ret_cc, ret_co, adv = panel.field("ret_cc"), panel.field("ret_co"), panel.field("adv")
    close = panel.field("close")
    book = _Portfolio(len(panel.symbols), config.costs, config.execution)
    prev_nav, pending = 1.0, None
    daily, weight_rows, liquidations, delistings = [], [], 0, 0

    for i in range(first, last + 1):
        if i > first:
            day = panel.dates[i]
            trades.accrue(book.holdings, book.cash, ret_cc[i], close[i], close[i - 1],
                          panel.rf_daily[i])  # fmt: skip
            book.overnight(ret_co[i])
            turnover = cost = 0.0
            if pending is not None:
                before, cash_before = book.holdings.copy(), book.cash
                turnover, cost = book.trade(pending, adv[i - 1], day.toordinal())
                trades.record(day, before, book.holdings, cash_before, book.cash)
                pending = None
            book.intraday(ret_cc[i], ret_co[i], panel.rf_daily[i], panel.traded[i])
            delisted = panel.fell_to_otc & (i > panel.last_bar)
            exit_ret = np.where(delisted, config.delisting_return, 0.0)
            values = book.holdings.copy()
            stale = book.liquidate_stale(config.max_missing_days, exit_ret)
            if stale.size:
                sold = np.zeros_like(values)
                sold[stale] = values[stale] * (1 + exit_ret[stale])
                trades.record(day, sold, np.zeros_like(values))
                liquidations += stale.size
                delistings += int(delisted[stale].sum())
                names = [panel.symbols[j] for j in stale]
                log.debug("liquidating %s on %s after missing bars", names, panel.dates[i])
            nav = book.nav
            held = int((book.holdings > 0).sum())
            daily.append(
                (panel.dates[i], nav, nav / prev_nav - 1, turnover, cost, book.cash / nav, held)
            )
            prev_nav = nav
            if i == last or panel.dates[i + 1].month != day.month:
                trades.record(day, book.holdings, book.holdings, book.cash, book.cash)

        if i in decisions:
            view = DataView(panel, i)
            weights = strategy.target_weights(view)
            pending = validate_weights(weights, view, panel, config.max_weight)
            weight_rows += [(view.asof, s, float(w)) for s, w in sorted(weights.items()) if w > 0]
    return daily, weight_rows, (liquidations, delistings)


#: Holdings shown in a run's worked example.
EXAMPLE_ROWS = 5


def _explanation(strategy: Strategy, config: BacktestConfig) -> dict[str, str]:
    """The run's plain-language explanation (``strategies.explain``), with execution."""
    if callable(getattr(strategy, "explain", None)):
        text = dict(strategy.explain())
    else:  # e.g. an ad-hoc Composed strategy: fall back to its description
        text = {"summary": describe(strategy, summary_only=True), "signal": describe(strategy)}
    text["execution"] = explain.execution(
        strategy.schedule, config.costs.half_spread_bps, config.costs.notional,
        config.max_missing_days, config.delisting_return,
    )  # fmt: skip
    return text


def _example(strategy: Strategy, weights: pl.DataFrame) -> dict | None:
    """The latest rebalance's largest holdings with the signal values behind them."""
    if weights.is_empty():
        return None
    latest = weights.filter(pl.col("date") == weights["date"].max())
    top = latest.sort("weight", "symbol", descending=[True, False]).head(EXAMPLE_ROWS)
    signals = getattr(strategy, "last_signals", None) or {}
    columns = dict(getattr(strategy, "example_columns", {}))
    return {
        "date": latest["date"][0],
        "holdings": latest.height,
        "columns": columns,
        "rows": [
            {"symbol": s, "weight": w, **{c: signals.get(s, {}).get(c) for c in columns}}
            for s, w in top.select("symbol", "weight").iter_rows()
        ],
    }


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
    trades = TradeLog(panel.symbols)
    try:
        daily, weight_rows, (liquidations, delistings) = _simulate(
            strategy, panel, config, first, last, trades
        )
    finally:
        if callable(close := getattr(strategy, "close", None)):
            close()  # e.g. worker pools

    columns = ["date", "nav", "ret", "turnover", "cost", "cash", "holdings"]
    daily_df = pl.DataFrame(daily, schema=columns, orient="row")
    days = slice(first + 1, last + 1)
    bench_col = panel.symbol_index.get(panel.resolve(config.benchmark))
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
    explanation = _explanation(strategy, config)
    if any(s != n for s, n in panel.names.items()):  # the hive: sids -> tickers
        signals = getattr(strategy, "last_signals", None)
        if signals:
            strategy.last_signals = {panel.names.get(s, s): v for s, v in signals.items()}
        weights_df = weights_df.with_columns(pl.col("symbol").replace(panel.names))
        trades.symbols = [panel.names.get(s, s) for s in trades.symbols]
    metrics["forced_liquidations"] = float(liquidations)
    metrics["otc_delistings"] = float(delistings)
    meta = {
        "strategy": strategy.name,
        "summary": explanation["summary"],
        "explain": explanation,
        "example": _example(strategy, weights_df),
        "description": describe(strategy),
        "label": label(strategy)
        + (f" [{config.execution.describe()}]" if config.execution.active else ""),
        "params": _params(strategy),
        "schedule": strategy.schedule,
        "start": panel.dates[first],
        "end": panel.dates[last],
        "benchmark": config.benchmark,
        "costs": asdict(config.costs),
        "max_weight": config.max_weight,
        "delisting_return": config.delisting_return,
        **({"execution": asdict(config.execution)} if config.execution.active else {}),
        "universe_size": len(panel.universe),
        "delisted_in_universe": panel.n_delisted,
        "data_max_date": panel.dates[-1],
        "git_sha": _git_sha(),
        "caveats": list(CAVEATS),
    }
    if callable(diagnostics := getattr(strategy, "diagnostics", None)):
        meta["diagnostics"] = diagnostics()
    return RunResult(
        meta=meta, metrics=metrics, daily=daily_df, weights=weights_df, trades=trades.frame()
    )
