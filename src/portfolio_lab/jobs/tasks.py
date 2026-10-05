"""Job entry points shared by the CLI and the scheduler.

Each task opens its own HTTP clients, runs one unit of work end to end, and returns a
summary dict. They are idempotent: re-running after a crash or restart is safe.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from portfolio_lab.backtest.costs import CostModel
from portfolio_lab.backtest.engine import BacktestConfig, run
from portfolio_lab.backtest.results import list_runs, load_run, prune_runs, save_run
from portfolio_lab.core.calendar import last_complete_session
from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic, write_status
from portfolio_lab.data.ingest.delisted import ingest_delisted
from portfolio_lab.data.ingest.fundamentals import ingest_fundamentals
from portfolio_lab.data.ingest.funds import ingest_funds
from portfolio_lab.data.ingest.macro import ingest_macro
from portfolio_lab.data.ingest.prices import update_prices, verify_prices
from portfolio_lab.data.ingest.rates import ingest_rates
from portfolio_lab.data.ingest.universe import current_symbols, ingest_universe
from portfolio_lab.data.sources.alpaca import make_client
from portfolio_lab.data.sources.edgar import annual
from portfolio_lab.data.sources.tiingo import fetch_fund_history
from portfolio_lab.research.conditions import caution_dial, conditional_ic
from portfolio_lab.research.context import STOCK_FEATURES, environment, sensitivities, tailwinds
from portfolio_lab.research.features import FEATURES, build_features
from portfolio_lab.research.fundamentals import filing_states
from portfolio_lab.research.funds import FUNDS, compare
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.piotroski import build_fscores, fscores_by_symbol
from portfolio_lab.research.scoreboard import HORIZON, evaluate, summarize
from portfolio_lab.signals import base as signals
from portfolio_lab.strategies.base import create
from portfolio_lab.trading import paper
from portfolio_lab.trading.broker import PaperBroker

log = logging.getLogger(__name__)

#: The production strategy: meanvar on the healthiest (continuous F-score) of the most liquid
#: stocks, monthly, with weight limits tapering near the edges of both lists instead of hard
#: cutoffs (``strategies.meanvar.soft``), holding the minimum-variance mix in bear markets and
#: equal weights in rebounds (see ``strategies.meanvar`` and ``research.regimes``).
PRODUCTION: tuple[str, dict[str, Any]] = (
    "meanvar",
    {"health_rank_pool": 400, "soften": "taper", "bear_defense": True, "rebound": "equal"},
)
#: Backtests the scheduler refreshes weekly so the dashboard always shows current baselines.
SCHEDULED_BACKTESTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("buy_hold", {}),  # SPY
    # Control for meanvar: the same candidates (100 most liquid), equally weighted.
    ("equal_weight", {"top_n": 100}),
    ("momentum", {}),  # the 20 strongest of the 100 most liquid
    ("meanvar", {"model": "ar1_logret"}),
    # The original design: a Piotroski quality filter, alone and in front of meanvar.
    ("piotroski", {"pool": 100}),
    ("meanvar", {"min_fscore": 7}),
    # The first baseline: meanvar on the healthiest 27% by continuous F-score.
    ("meanvar", {"healthy_share": 0.27}),
    PRODUCTION,
)
#: Signals the weekly scoreboard evaluates: mean-variance's forecasts and classic anomalies.
SCOREBOARD_SIGNALS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("forecast", {"model": "ar1_logret"}),
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
#: Start date for scheduled backtests (one year after data starts, for lookbacks).
SCHEDULED_START = date(2017, 1, 3)
#: Runs kept per configuration; older ones are deleted after each scheduled refresh.
RUNS_KEPT_PER_CONFIG = 3


def public_client(settings: Settings) -> RateLimitedClient:
    """Return a client for unauthenticated sources (NASDAQ Trader, FRED)."""
    return RateLimitedClient(headers={"User-Agent": settings.edgar_user_agent})


def ingest_universe_task(settings: Settings) -> dict:
    """Snapshot the symbol directory for today."""
    with public_client(settings) as client:
        return ingest_universe(settings, client, date.today())


def ingest_prices_task(settings: Settings, full: bool = False) -> dict:
    """Update daily prices for the current universe.

    Raises:
        RuntimeError: If no universe has been ingested yet.
    """
    paths = DataPaths(settings.data_dir)
    symbols = current_symbols(paths)
    if not symbols:
        raise RuntimeError("no universe stored yet; ingest the universe first")
    with make_client(settings) as client:
        return update_prices(settings, client, symbols, paths.prices_daily, "prices", full=full)


def ingest_benchmarks_task(settings: Settings, full: bool = False) -> dict:
    """Update daily prices for the benchmark ETFs."""
    dataset = DataPaths(settings.data_dir).prices_benchmarks
    with make_client(settings) as client:
        return update_prices(
            settings, client, settings.benchmark_symbols, dataset, "benchmarks", full=full
        )


def ingest_rates_task(settings: Settings) -> dict:
    """Refresh the risk-free rate history."""
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
        return ingest_macro(settings, client)


def delisted_task(settings: Settings) -> dict:
    """Find stocks delisted since the history start and backfill their prices."""
    with public_client(settings) as public, make_client(settings) as alpaca:
        return ingest_delisted(settings, public, alpaca)


def fundamentals_task(settings: Settings, force: bool = False) -> dict:
    """Refresh SEC fundamentals for the universe and recompute point-in-time F-scores."""
    paths = DataPaths(settings.data_dir)
    symbols = current_symbols(paths)
    headers = {"User-Agent": settings.edgar_user_agent}
    with RateLimitedClient(headers=headers, max_per_minute=EDGAR_REQUESTS_PER_MINUTE) as client:
        summary = ingest_fundamentals(settings, client, symbols, force=force)
    facts = pl.read_parquet(paths.fundamentals_facts)
    tickers = pl.read_parquet(paths.fundamentals_tickers)
    scores = fscores_by_symbol(build_fscores(annual(facts)), tickers)
    write_parquet_atomic(scores, paths.fscores)
    status = {
        "filings_scored": scores.height,
        "symbols": scores["symbol"].n_unique(),
        "with_8_signals": scores.filter(pl.col("n_signals") >= 8).height,
        "latest_filing": scores["filed"].max(),
    }
    write_status(settings.data_dir, "fscores", status)
    log.info("fscores: %d filings for %d symbols", status["filings_scored"], status["symbols"])
    states = filing_states(facts)
    write_parquet_atomic(states, paths.fundamentals_states)
    log.info("fundamentals: %d filing states", states.height)
    return {"fundamentals": summary, "fscores": status, "states": states.height}


def features_task(settings: Settings) -> dict:
    """Rebuild the monthly point-in-time feature panel and market environment.

    Fundamentals, profiles and prices give the stock features; with FRED context series
    ingested, the environment table is written and each stock's factor sensitivities and
    tailwinds are added (see ``research.context``).
    """
    paths = DataPaths(settings.data_dir)
    companies = paths.fundamentals_companies
    panel = Panel.load(settings.data_dir)
    features = build_features(
        panel,
        pl.read_parquet(paths.fundamentals_states),
        pl.read_parquet(paths.fundamentals_tickers),
        pl.read_parquet(companies) if companies.exists() else None,
        pl.read_parquet(paths.fscores) if paths.fscores.exists() else None,
    )
    if paths.macro.exists():
        observations = pl.read_parquet(paths.macro)
        dates = features["date"].unique().sort().to_list()
        env = environment(observations, dates, features)
        write_parquet_atomic(env, paths.environment)
        stock_context = tailwinds(sensitivities(panel, observations, dates), env)
        features = features.join(stock_context, on=["date", "symbol"], how="left")
    write_parquet_atomic(features, paths.features)
    covered = features.select(pl.col("earnings_yield").is_not_null().mean()).item()
    status = {
        "rows": features.height,
        "months": features["date"].n_unique(),
        "latest": features["date"].max(),
        "with_fundamentals": round(float(covered), 3),
    }
    write_status(settings.data_dir, "features", status)
    return status


def verify_task(settings: Settings, sample: int = 50) -> dict:
    """Spot-check stored returns against a fresh fetch, repairing drift."""
    with make_client(settings) as client:
        return verify_prices(settings, client, DataPaths(settings.data_dir).prices_daily, sample)


def _attach_runtime(obj: Any, settings: Settings) -> Any:
    """Give model-fitting strategies and signals their cache and worker processes."""
    if hasattr(obj, "cache_dir"):
        obj.cache_dir = DataPaths(settings.data_dir).forecast_cache
    if hasattr(obj, "workers"):
        obj.workers = FORECAST_WORKERS
    return obj


def backtest_task(
    settings: Settings,
    strategy: str,
    start: date,
    end: date | None = None,
    params: dict[str, Any] | None = None,
    notional: float = 100_000,
    max_weight: float = 1.0,
    delisting_return: float = BacktestConfig.delisting_return,
) -> tuple[str, dict[str, float]]:
    """Run one backtest on the stored data and save it.

    Returns:
        The run id and its metrics.
    """
    strat = _attach_runtime(create(strategy, **(params or {})), settings)
    panel = Panel.load(settings.data_dir, end=end)
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


def scheduled_backtests_task(settings: Settings) -> dict:
    """Re-run the baseline backtests through the latest data, then prune old runs."""
    runs = {
        f"{name} {params}".strip(): backtest_task(settings, name, SCHEDULED_START, params=params)[0]
        for name, params in SCHEDULED_BACKTESTS
    }
    pruned = prune_runs(settings.data_dir, keep=RUNS_KEPT_PER_CONFIG)
    log.info("pruned %d old runs", len(pruned))
    return {"runs": runs, "pruned": pruned}


def signal_label(name: str, params: dict[str, Any]) -> str:
    """Display name for a signal configuration, e.g. ``forecast (model=ar1_logret)``.

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
            frames.append(evaluate(signal, label, panel, SCHEDULED_START))
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
    market = np.nan_to_num(panel.field("ret_cc")[:, panel.symbol_index["SPY"]])
    dial = caution_dial(env, panel.dates, market, panel.date_index)
    write_parquet_atomic(dial, paths.context_dial)
    status = {"conditions_rows": by_condition.height, "dial_rows": dial.height,
              "latest": env["date"].max()}  # fmt: skip
    write_status(settings.data_dir, "context", status)
    return status


def make_vs_buy_task(settings: Settings) -> dict:
    """Refresh fund prices and compare them with our strategies' latest runs."""
    paths = DataPaths(settings.data_dir)
    with make_client(settings) as alpaca, RateLimitedClient(max_per_minute=60) as tiingo:
        ingest_funds(settings, alpaca, tiingo, last_complete_session())
    prices = pl.read_parquet(paths.fund_prices)
    series = {
        f.symbol: (f.name, f.category, prices.filter(pl.col("symbol") == f.symbol))
        for f in FUNDS
        if prices.filter(pl.col("symbol") == f.symbol).height
    }
    for stored in list_runs(settings.data_dir, latest_only=True):
        meta = stored["meta"]
        if meta["strategy"] == "buy_hold":
            continue  # SPY is already in the comparison as a fund
        label = meta.get("label") or meta["strategy"]
        daily = load_run(settings.data_dir, meta["run_id"]).daily.select("date", "ret")
        series[f"ours: {label}"] = (label, "ours", daily)
    rates = pl.read_parquet(paths.rates) if paths.rates.exists() else None
    summary, growth = compare(series, rates, SCHEDULED_START)
    write_parquet_atomic(summary, paths.make_vs_buy / "summary.parquet")
    write_parquet_atomic(growth, paths.make_vs_buy / "growth.parquet")
    common = summary.filter(pl.col("period") == "common")
    status = {"series": summary["key"].n_unique(), "common_start": common["start"].min()}
    write_status(settings.data_dir, "make_vs_buy", status)
    return status


#: Start of the long make-vs-buy comparison (the Sharadar history's first full year).
HISTORY_COMPARE_START = date(1999, 1, 4)


def make_vs_buy_history_task(settings: Settings, publish: Path | None = None) -> dict:
    """Make vs buy over the Sharadar history: funds since launch vs our strategies since 1999.

    Fund prices (distribution-adjusted) come from Tiingo, so nothing published below is
    Sharadar data; ours are backtests on the Sharadar history in ``settings.data_dir``:
    meanvar, meanvar behind F-score >= 7, meanvar behind the continuous F-score (healthiest
    27%) and :data:`PRODUCTION`, all from :data:`HISTORY_COMPARE_START`.

    With ``publish`` (the main data directory), the summary and the growth of $1 are also
    written there (``make_vs_buy/history_summary.parquet``, ``history_growth.parquet``) for
    the Compare page: fund series from Tiingo and our derived results, which the Sharadar
    license lets us keep.
    """
    paths = DataPaths(settings.data_dir)
    if not settings.tiingo_api_key:
        raise RuntimeError("TIINGO_API_KEY is needed for fund prices")
    token = settings.tiingo_api_key.get_secret_value()
    with RateLimitedClient(max_per_minute=30) as tiingo:
        frames = [
            fetch_fund_history(tiingo, f.symbol.replace(".", "-"), token, HISTORY_COMPARE_START)
            .with_columns(pl.lit(f.symbol).alias("symbol"))
            for f in FUNDS
        ]  # fmt: skip
    prices = (
        pl.concat(frames)
        .sort("symbol", "date")
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("ret")
        )
    )
    write_parquet_atomic(prices, paths.fund_prices)
    start = HISTORY_COMPARE_START
    ours = {
        "meanvar": backtest_task(settings, "meanvar", start)[0],
        "meanvar + F-score >= 7": backtest_task(settings, "meanvar", start,
                                                params={"min_fscore": 7})[0],
        "meanvar + continuous F-score (healthiest 27%)": backtest_task(
            settings, "meanvar", start, params={"healthy_share": 0.27})[0],
        "production (tapered healthiest of the most liquid + bear defense)": backtest_task(
            settings, PRODUCTION[0], start, params=PRODUCTION[1])[0],
    }  # fmt: skip
    series = {
        f.symbol: (f.name, f.category, prices.filter(pl.col("symbol") == f.symbol))
        for f in FUNDS
        if prices.filter(pl.col("symbol") == f.symbol).height
    }
    production = None
    for name, run_id in ours.items():  # named like the live runs, so the pages merge them
        stored = load_run(settings.data_dir, run_id)
        label = stored.meta.get("label") or name
        series[f"ours: {label}"] = (label, "ours", stored.daily.select("date", "ret"))
        if name.startswith("production"):
            production = f"ours: {label}"
    rates = pl.read_parquet(paths.rates) if paths.rates.exists() else None
    summary, growth = compare(series, rates, start, ours=production)
    write_parquet_atomic(summary, paths.make_vs_buy / "summary.parquet")
    write_parquet_atomic(growth, paths.make_vs_buy / "growth.parquet")
    if publish is not None:
        folder = DataPaths(publish).make_vs_buy
        write_parquet_atomic(summary, folder / "history_summary.parquet")
        write_parquet_atomic(growth, folder / "history_growth.parquet")
    return {"runs": ours, "series": len(series)}


#: The strategy the paper account follows: production.
PAPER_STRATEGY: tuple[str, dict[str, Any]] = PRODUCTION


def paper_task(settings: Settings, dry_run: bool = False) -> dict:
    """Record the paper account and rebalance it at month ends (``trading.paper``)."""
    name, params = PAPER_STRATEGY
    strategy = _attach_runtime(create(name, **params), settings)
    broker = PaperBroker(settings)
    try:
        return paper.run(settings.data_dir, strategy, broker,
                         refresh=lambda: features_task(settings), dry_run=dry_run,
                         trading_dir=settings.trading_dir)  # fmt: skip
    finally:
        broker.close()
        if callable(close := getattr(strategy, "close", None)):
            close()
