"""Job entry points shared by the CLI and the scheduler.

Three kinds, in the order the scheduler runs them: **fetch** (each source's downloads,
kept in the ingest store, ``Settings.for_ingest``), **conform and derive** (the hive:
``data.conform`` then ``data.derived``), and **model** (backtests, the paper account,
the scoreboard), which read the hive through ``Panel.load``. Each task opens its own
HTTP clients, runs one unit of work end to end, and returns a summary dict. They are
idempotent: re-running after a crash or restart is safe.
"""

import logging
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from portfolio_lab.backtest.costs import CostModel
from portfolio_lab.backtest.engine import BacktestConfig, run
from portfolio_lab.backtest.engine import label as strategy_label
from portfolio_lab.backtest.results import list_runs, load_run, prune_runs, save_run
from portfolio_lab.core.calendar import last_complete_session
from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import read_status, write_parquet_atomic, write_status
from portfolio_lab.data import ids as ids_
from portfolio_lab.data import quality, reader
from portfolio_lab.data.conform import alpaca, edgar, fred, nasdaq, tiingo
from portfolio_lab.data.derived import daily, fundamentals, monthly
from portfolio_lab.data.ingest.delisted import ingest_delisted
from portfolio_lab.data.ingest.fundamentals import ingest_fundamentals
from portfolio_lab.data.ingest.macro import ingest_macro
from portfolio_lab.data.ingest.prices import update_prices, verify_prices
from portfolio_lab.data.ingest.rates import ingest_rates
from portfolio_lab.data.ingest.universe import current_symbols, ingest_universe
from portfolio_lab.data.sources.alpaca import make_client
from portfolio_lab.data.sources.tiingo import fetch_fund_history
from portfolio_lab.research.conditions import caution_dial, conditional_ic
from portfolio_lab.research.context import STOCK_FEATURES
from portfolio_lab.research.features import FEATURES
from portfolio_lab.research.forecaster import nine, report
from portfolio_lab.research.funds import FUNDS, compare
from portfolio_lab.research.panel import EligibilityRules, Panel
from portfolio_lab.research.scoreboard import HORIZON, evaluate, summarize
from portfolio_lab.signals import base as signals
from portfolio_lab.strategies.base import create
from portfolio_lab.trading import paper
from portfolio_lab.trading.broker import PaperBroker

log = logging.getLogger(__name__)

#: The production strategy, "Piotroski health" (class 1): meanvar on the healthiest
#: (continuous F-score) of the most liquid stocks, monthly, with weight limits tapering near
#: the edges of both lists instead of hard cutoffs (``strategies.meanvar.soft``), weighted for
#: the highest expected long-run growth (the Kelly objective, ``strategies.meanvar.optimize``),
#: holding the minimum-variance mix in bear markets and equal weights in rebounds (see
#: ``strategies.meanvar`` and ``research.regimes``). Kelly replaced max-Sharpe on 2026-10-06.
PRODUCTION: tuple[str, dict[str, Any]] = (
    "meanvar",
    {"health_rank_pool": 400, "soften": "taper", "bear_defense": True, "rebound": "equal",
     "objective": "kelly", "title": "Piotroski health"},
)  # fmt: skip
#: "Forecaster" (class 2): the same pool, Kelly and bear switch as :data:`PRODUCTION`, with
#: the nine-term forecasts (Grinold's form) as expected returns, the price/factor/residual
#: covariance and no rebound switch (``strategies.meanvar.nine``). 27.4% a year 2004-2026 on
#: the hive, against production's 26.2%.
CLASS_2: tuple[str, dict[str, Any]] = (
    "meanvar",
    {"health_rank_pool": 400, "soften": "taper", "bear_defense": True, "objective": "kelly",
     "expected": "nine", "covariance": "thirds", "title": "Forecaster"},
)  # fmt: skip
#: The production models, drawn on the Overview and Compare charts against SPY and the best
#: funds (every other strategy is listed in their tables but starts hidden on the chart).
PRODUCTION_MODELS: tuple[tuple[str, dict[str, Any]], ...] = (PRODUCTION, CLASS_2)
#: Backtests the scheduler refreshes weekly so the dashboard always shows current baselines.
SCHEDULED_BACKTESTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("buy_hold", {}),  # SPY
    # Control for meanvar: the same candidates (100 most liquid), equally weighted.
    ("equal_weight", {"top_n": 100}),
    ("momentum", {}),  # the 20 strongest of the 100 most liquid
    ("meanvar", {}),
    # The original design: a Piotroski quality filter, alone and in front of meanvar.
    ("piotroski", {"pool": 100}),
    ("meanvar", {"min_fscore": 7}),
    # The first baseline: meanvar on the healthiest 27% by continuous F-score.
    ("meanvar", {"healthy_share": 0.27}),
    PRODUCTION,
    CLASS_2,
)
#: Signals the weekly scoreboard evaluates: mean-variance's forecasts and classic anomalies.
SCOREBOARD_SIGNALS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("forecast", {"model": "arima320_price"}),
    ("forecast", {"model": "historical_mean"}),
    ("momentum", {}),
    ("reversal", {}),
    ("low_vol", {}),
    ("fscore", {}),
    # Every feature of the monthly panel, predicting the next month and the next quarter.
    # Next month only: production rebalances monthly, so that is the horizon that matters.
    *(("feature", {"column": c}) for c in FEATURES),
    *(("feature", {"column": c}) for c in STOCK_FEATURES),
)
#: Worker processes for model fits: orange's four fast A76 cores (more workers land on the
#: slow A55 cores and measured slower).
FORECAST_WORKERS = 4
#: SEC fair-access limit is 10 requests/second; stay well under it.
EDGAR_REQUESTS_PER_MINUTE = 300
#: Runs kept per configuration; older ones are deleted after each scheduled refresh.
RUNS_KEPT_PER_CONFIG = 3


def run_label(meta: dict[str, Any]) -> str:
    """A saved run's display name: its configuration's current title, else its saved label.

    Configurations are matched by their untitled label, which spells out every non-default
    parameter (a title is not one), so renaming a model renames its earlier runs too.
    """
    titled = {}
    for name, params in PRODUCTION_MODELS:
        untitled = {k: v for k, v in params.items() if k != "title"}
        titled[strategy_label(create(name, **untitled))] = strategy_label(create(name, **params))
    try:
        saved = strategy_label(create(meta["strategy"], **meta.get("params", {})))
    except (KeyError, TypeError, ValueError):  # a retired strategy or parameter
        saved = None
    return titled.get(saved) or meta.get("label") or meta["strategy"]


def public_client(settings: Settings) -> RateLimitedClient:
    """Return a client for unauthenticated sources (NASDAQ Trader, FRED)."""
    return RateLimitedClient(headers={"User-Agent": settings.edgar_user_agent})


def ingest_universe_task(settings: Settings) -> dict:
    """Snapshot the symbol directory for today."""
    settings = settings.for_ingest()
    with public_client(settings) as client:
        return ingest_universe(settings, client, date.today())


def ingest_prices_task(settings: Settings, full: bool = False) -> dict:
    """Update daily prices for the current universe.

    Raises:
        RuntimeError: If no universe has been ingested yet.
    """
    settings = settings.for_ingest()
    paths = DataPaths(settings.data_dir)
    symbols = current_symbols(paths)
    if not symbols:
        raise RuntimeError("no universe stored yet; ingest the universe first")
    with make_client(settings) as client:
        return update_prices(settings, client, symbols, paths.prices_daily, "prices", full=full)


def ingest_benchmarks_task(settings: Settings, full: bool = False) -> dict:
    """Update daily prices for the benchmark ETFs."""
    settings = settings.for_ingest()
    dataset = DataPaths(settings.data_dir).prices_benchmarks
    with make_client(settings) as client:
        return update_prices(
            settings, client, settings.benchmark_symbols, dataset, "benchmarks", full=full
        )


def ingest_rates_task(settings: Settings) -> dict:
    """Refresh the risk-free rate history."""
    settings = settings.for_ingest()
    with public_client(settings) as client:
        return ingest_rates(settings, client)


def daily_ingest_task(settings: Settings, full: bool = False) -> dict:
    """Run universe, prices, benchmarks and rates in order; return their summaries."""
    return {
        "universe": ingest_universe_task(settings),
        "prices": ingest_prices_task(settings, full),
        "benchmarks": ingest_benchmarks_task(settings, full),
        "rates": ingest_rates_task(settings),
    }


#: FRED's API allows 120 requests a minute.
FRED_REQUESTS_PER_MINUTE = 100


def macro_task(settings: Settings) -> dict:
    """Refresh the FRED market and economic context series."""
    with RateLimitedClient(max_per_minute=FRED_REQUESTS_PER_MINUTE) as client:
        return ingest_macro(settings.for_ingest(), client)


def delisted_task(settings: Settings) -> dict:
    """Find stocks delisted since the history start and backfill their prices."""
    settings = settings.for_ingest()
    with public_client(settings) as public, make_client(settings) as alpaca:
        return ingest_delisted(settings, public, alpaca)


def fundamentals_task(settings: Settings, force: bool = False) -> dict:
    """Refresh SEC EDGAR's bulk facts and company profiles for the universe."""
    settings = settings.for_ingest()
    symbols = current_symbols(DataPaths(settings.data_dir))
    headers = {"User-Agent": settings.edgar_user_agent}
    with RateLimitedClient(headers=headers, max_per_minute=EDGAR_REQUESTS_PER_MINUTE) as client:
        return ingest_fundamentals(settings, client, symbols, force=force)


def conform_task(settings: Settings) -> dict:
    """Rewrite every fetched source into the hive's conformed tables (``data.conform``)."""
    root, store = settings.data_dir, settings.for_ingest().data_dir
    out = {m.SOURCE: m.build(root, store) for m in (alpaca, nasdaq, tiingo, fred)}
    zip_path = DataPaths(store).edgar_bulk
    if zip_path.exists():
        out[edgar.SOURCE] = edgar.build(root, zip_path)
    write_status(root, "conform", out)
    return out


#: Research's wider universe; production uses ``EligibilityRules()``.
RESEARCH_RULES = EligibilityRules(min_price=1.0, min_dollar_volume=1e5)


def derive_task(settings: Settings) -> dict:
    """Rebuild the derived tables, then the nine-term forecasts on them.

    The monthly table covers :data:`RESEARCH_RULES`' wider universe, the one the
    forecaster is fitted on; strategies still choose among their own eligible stocks.
    """
    root = settings.data_dir
    out = {"daily": daily.build(root), "fundamentals": fundamentals.build(root)}
    panel = Panel.load(root, rules=RESEARCH_RULES)
    out["monthly"] = monthly.build(root, panel=panel)
    paths = DataPaths(root)
    data = nine.table(panel, pl.read_parquet(paths.features))
    forecasts = nine.walk(data)
    write_parquet_atomic(forecasts, paths.forecaster / nine.FILE)
    write_parquet_atomic(nine.slopes(data), paths.forecaster / nine.SLOPES)
    out["forecasts"] = {"months": forecasts["date"].n_unique(), "latest": forecasts["date"].max()}
    del data
    report.publish(paths.forecaster, panel, pl.read_parquet(paths.features))
    out["quality"] = quality.check(root, "derive")  # raises on a broken invariant
    write_status(root, "derive", out)
    return out


def verify_task(settings: Settings, sample: int = 50) -> dict:
    """Spot-check stored returns against a fresh fetch, repairing drift."""
    settings = settings.for_ingest()
    with make_client(settings) as client:
        return verify_prices(settings, client, DataPaths(settings.data_dir).prices_daily, sample)


def _attach_runtime(obj: Any, settings: Settings) -> Any:
    """Give model-fitting strategies and signals their cache and worker processes."""
    if hasattr(obj, "cache_dir"):
        obj.cache_dir = DataPaths(settings.data_dir).forecast_cache
    if hasattr(obj, "forecaster_dir"):
        obj.forecaster_dir = DataPaths(settings.data_dir).forecaster
    if hasattr(obj, "workers"):
        obj.workers = FORECAST_WORKERS
    return obj


def first_start(panel: Panel) -> date:
    """The first session with a year of history before it: where backtests start."""
    return panel.dates[min(EligibilityRules().min_history, len(panel.dates) - 1)]


def backtest_task(  # noqa: PLR0913 - the CLI's options, one each
    settings: Settings,
    strategy: str,
    start: date | None = None,
    end: date | None = None,
    params: dict[str, Any] | None = None,
    notional: float = 100_000,
    max_weight: float = 1.0,
    delisting_return: float = BacktestConfig.delisting_return,
    *,
    panel: Panel | None = None,
) -> tuple[str, dict[str, float]]:
    """Run one backtest on the stored data and save it.

    Args:
        settings: Application settings.
        strategy: Registered strategy name.
        start: First session (default: a year into the data, ``first_start``, or the
            strategy's ``first_decision`` when later).
        end: Last session (default: the latest).
        params: Strategy parameters.
        notional: Portfolio size in dollars, for costs.
        max_weight: Largest weight in any one name.
        delisting_return: Return on exit for stocks that fell to OTC.
        panel: An already loaded panel (jobs running several backtests share one).

    Returns:
        The run id and its metrics.
    """
    strat = _attach_runtime(create(strategy, **(params or {})), settings)
    panel = panel or Panel.load(settings.data_dir, end=end)
    if start is None:
        start = first_start(panel)
        if (first := getattr(strat, "first_decision", lambda: None)()) and first > start:
            start = first
    config = BacktestConfig(
        start=start,
        end=end or panel.dates[-1],
        costs=CostModel(notional=notional),
        max_weight=max_weight,
        delisting_return=delisting_return,
    )
    result = run(strat, panel, config)
    run_id = save_run(result, settings.data_dir)
    log.info("backtest %s saved as %s", strategy, run_id)
    return run_id, result.metrics


def paper_backtest_task(settings: Settings) -> dict:
    """Re-run the paper strategy's backtest through the latest session, then prune old runs.

    The Paper page draws this run beside the paper account, so it has to be as current as
    the account: the weekly baselines would leave it up to a week behind.
    """
    name, params = PAPER_STRATEGY
    run_id, metrics = backtest_task(settings, name, params=params)
    pruned = prune_runs(settings.data_dir, keep=RUNS_KEPT_PER_CONFIG)
    return {"run": run_id, "cagr": metrics.get("cagr"), "pruned": len(pruned)}


def scheduled_backtests_task(settings: Settings) -> dict:
    """Re-run the baseline backtests through the latest data, then prune old runs."""
    panel = Panel.load(settings.data_dir)
    runs = {
        f"{name} {params}".strip(): backtest_task(settings, name, params=params, panel=panel)[0]
        for name, params in SCHEDULED_BACKTESTS
    }
    pruned = prune_runs(settings.data_dir, keep=RUNS_KEPT_PER_CONFIG)
    log.info("pruned %d old runs", len(pruned))
    return {"runs": runs, "pruned": pruned}


def signal_label(name: str, params: dict[str, Any]) -> str:
    """Display name for a signal configuration, e.g. ``forecast (model=historical_mean)``.

    Features are named by their column (the horizon shows in the scoreboard's sections).
    """
    if name == "feature":
        return params["column"]
    return name + (f" ({', '.join(f'{k}={v}' for k, v in params.items())})" if params else "")


def scoreboard_task(
    settings: Settings, only: tuple[tuple[str, dict[str, Any]], ...] = SCOREBOARD_SIGNALS
) -> dict:
    """Score every signal's monthly rankings against the returns that followed.

    Rewrites ``results/scoreboard.parquet``; with ``only`` a subset, other signals' stored
    rows are kept.
    """
    paths = DataPaths(settings.data_dir)
    panel = Panel.load(settings.data_dir)
    frames = []
    for name, params in only:
        label = signal_label(name, params)
        signal = _attach_runtime(signals.create(name, **params), settings)
        try:
            frames.append(evaluate(signal, label, panel, first_start(panel)))
        finally:
            if callable(close := getattr(signal, "close", None)):
                close()
        log.info("scoreboard: %s scored", label)
    scores = pl.concat(frames)
    if paths.scoreboard.exists():
        kept = pl.read_parquet(paths.scoreboard).filter(
            ~pl.col("signal").is_in(scores["signal"].unique().to_list())
            & ~pl.col("signal").str.starts_with("model: ")  # the retired beat-the-median models
        )
        if "horizon" not in kept.columns:  # stored before horizons existed: all monthly
            kept = kept.with_columns(pl.lit(HORIZON, pl.Int64).alias("horizon"))
        scores = pl.concat([kept.select(scores.columns), scores])
    write_parquet_atomic(scores.sort("signal", "pool", "date"), paths.scoreboard)
    table = summarize(scores)
    status = {"signals": table["signal"].n_unique(), "periods": int(table["periods"].max())}
    write_status(settings.data_dir, "scoreboard", status)
    return {"summary": table.to_dicts()}


def context_task(settings: Settings) -> dict:
    """Measure trait payoffs and forward market risk by prevailing conditions."""
    paths = DataPaths(settings.data_dir)
    env = pl.read_parquet(paths.environment)
    by_condition = conditional_ic(pl.read_parquet(paths.scoreboard), env)
    write_parquet_atomic(by_condition, paths.context_conditions)
    panel = Panel.load(settings.data_dir)
    market = np.nan_to_num(panel.field("ret_cc")[:, panel.market])
    dial = caution_dial(env, panel.dates, market, panel.date_index)
    write_parquet_atomic(dial, paths.context_dial)
    status = {"conditions_rows": by_condition.height, "dial_rows": dial.height,
              "latest": env["date"].max()}  # fmt: skip
    write_status(settings.data_dir, "context", status)
    return status


#: Mutual funds' history is fetched from this date (Tiingo); everything else is in the hive.
FUND_HISTORY_START = date(1998, 1, 2)


def fund_returns(settings: Settings) -> dict[str, pl.DataFrame]:
    """Daily returns (date, ret) of every fund in ``research.funds.FUNDS``.

    Exchange-traded funds (and BRK.B) come from the hive's prices; open-end mutual funds,
    which no exchange quotes, from Tiingo since :data:`FUND_HISTORY_START`.
    """
    root = settings.data_dir
    traded = [f.symbol for f in FUNDS if f.source != "tiingo"]
    today = [last_complete_session()] * len(traded)
    found = ids_.lookup(ids_.Ids.load(DataPaths(root).ids),
                        pl.DataFrame({"ticker": traded, "date": today}))  # fmt: skip
    sids = dict(found.drop_nulls("sid").select("ticker", "sid").rows())
    prices = reader.read(root, "prices", ["ret_cc"], sids=list(sids.values()))
    out = {t: prices.filter(pl.col("sid") == sid).select("date", pl.col("ret_cc").alias("ret"))
           for t, sid in sids.items()}  # fmt: skip
    mutual = [f.symbol for f in FUNDS if f.source == "tiingo"]
    if settings.tiingo_api_key is None:
        log.warning("make_vs_buy: TIINGO_API_KEY not set; leaving out %d mutual funds", len(mutual))
        return out
    token = settings.tiingo_api_key.get_secret_value()
    with RateLimitedClient(max_per_minute=60) as client:
        for symbol in mutual:
            history = fetch_fund_history(client, symbol, token, FUND_HISTORY_START).sort("date")
            out[symbol] = history.select(
                "date", (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1).alias("ret")
            )
    return out


def make_vs_buy_task(settings: Settings) -> dict:
    """Compare funds anyone can buy with our strategies' latest runs, over all history."""
    paths = DataPaths(settings.data_dir)
    names = {f.symbol: (f.name, f.category) for f in FUNDS}
    series = {s: (*names[s], frame) for s, frame in fund_returns(settings).items() if frame.height}
    for stored in list_runs(settings.data_dir, latest_only=True):
        meta = stored["meta"]
        name = run_label(meta)
        if meta["strategy"] == "buy_hold" or f"ours: {name}" in series:
            continue  # SPY is already a fund; runs come newest first
        daily = load_run(settings.data_dir, meta["run_id"]).daily.select("date", "ret")
        series[f"ours: {name}"] = (name, "ours", daily)
    rates = reader.read(settings.data_dir, "series").filter(pl.col("series") == "DTB3")
    rates = rates.select("date", pl.col("value").alias("rate"))
    start = min(frame["date"].min() for _, _, frame in series.values())
    production = f"ours: {strategy_label(create(PRODUCTION[0], **PRODUCTION[1]))}"
    production = production if production in series else None
    summary, growth = compare(series, rates, start, ours=production)
    write_parquet_atomic(summary, paths.make_vs_buy / "summary.parquet")
    write_parquet_atomic(growth, paths.make_vs_buy / "growth.parquet")
    common = summary.filter(pl.col("period") == "common")
    status = {"series": summary["key"].n_unique(), "common_start": common["start"].min()}
    write_status(settings.data_dir, "make_vs_buy", status)
    return status


#: The strategy the paper account follows: production (switch to :data:`CLASS_2` here).
PAPER_STRATEGY: tuple[str, dict[str, Any]] = PRODUCTION


def paper_task(settings: Settings, dry_run: bool = False) -> dict:
    """Record the paper account and rebalance it at month ends (``trading.paper``).

    Refuses to run while the last quality check found a broken invariant (``data.quality``).
    """
    last = read_status(settings.data_dir, "quality") or {}
    if last.get("hard"):
        raise quality.QualityError("not trading on data that failed: " + "; ".join(last["hard"]))
    name, params = PAPER_STRATEGY
    strategy = _attach_runtime(create(name, **params), settings)
    broker = PaperBroker(settings)
    try:
        return paper.run(settings.data_dir, strategy, broker,
                         refresh=lambda: derive_task(settings), dry_run=dry_run,
                         trading_dir=settings.trading_dir)  # fmt: skip
    finally:
        broker.close()
        if callable(close := getattr(strategy, "close", None)):
            close()
