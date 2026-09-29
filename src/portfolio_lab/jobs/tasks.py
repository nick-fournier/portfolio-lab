"""Job entry points shared by the CLI and the scheduler.

Each task opens its own HTTP clients, runs one unit of work end to end, and returns a
summary dict. They are idempotent: re-running after a crash or restart is safe.
"""

import logging
from datetime import date
from typing import Any

import polars as pl

from portfolio_lab.backtest.costs import CostModel
from portfolio_lab.backtest.engine import BacktestConfig, run
from portfolio_lab.backtest.results import prune_runs, save_run
from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic, write_status
from portfolio_lab.data.ingest.delisted import ingest_delisted
from portfolio_lab.data.ingest.fundamentals import ingest_fundamentals
from portfolio_lab.data.ingest.prices import update_prices, verify_prices
from portfolio_lab.data.ingest.rates import ingest_rates
from portfolio_lab.data.ingest.universe import current_symbols, ingest_universe
from portfolio_lab.data.sources.alpaca import make_client
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.piotroski import build_fscores, fscores_by_symbol
from portfolio_lab.research.scoreboard import evaluate, summarize
from portfolio_lab.signals import base as signals
from portfolio_lab.strategies.base import create

log = logging.getLogger(__name__)

#: Backtests the scheduler refreshes weekly so the dashboard always shows current baselines.
SCHEDULED_BACKTESTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("buy_hold", {}),
    ("equal_weight", {}),
    # Control for meanvar: the same candidates (100 most liquid), equally weighted.
    ("equal_weight", {"top_n": 100}),
    # Momentum on meanvar's candidates, and on every eligible stock.
    ("momentum", {}),
    ("momentum", {"pool": None}),
    ("meanvar", {"model": "ar1_logret"}),
    ("meanvar", {"model": "arima320_price"}),
    ("meanvar", {"model": "historical_mean"}),
    # The original design: a Piotroski quality filter, alone and in front of meanvar.
    ("piotroski", {}),
    ("piotroski", {"pool": 100}),
    ("meanvar", {"min_fscore": 7}),
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
    # Short horizons: lead-lag networks, against each stock's own short-term reversal.
    ("reversal", {"lookback": 1, "horizon": 1}),
    ("leadlag", {"horizon": 1}),
    ("leadlag", {"horizon": 1, "mode": "market"}),
    ("reversal", {"lookback": 5, "horizon": 5}),
    ("leadlag", {"horizon": 5}),
    ("leadlag", {"horizon": 5, "mode": "market"}),
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
    scores = fscores_by_symbol(build_fscores(facts), tickers)
    write_parquet_atomic(scores, paths.fscores)
    status = {
        "filings_scored": scores.height,
        "symbols": scores["symbol"].n_unique(),
        "with_8_signals": scores.filter(pl.col("n_signals") >= 8).height,
        "latest_filing": scores["filed"].max(),
    }
    write_status(settings.data_dir, "fscores", status)
    log.info("fscores: %d filings for %d symbols", status["filings_scored"], status["symbols"])
    return {"fundamentals": summary, "fscores": status}


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
    """Display name for a signal configuration, e.g. ``forecast (model=ar1_logret)``."""
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
        )
        scores = pl.concat([kept, scores])
    write_parquet_atomic(scores.sort("signal", "pool", "date"), paths.scoreboard)
    table = summarize(scores)
    status = {"signals": table["signal"].n_unique(), "periods": int(table["periods"].max())}
    write_status(settings.data_dir, "scoreboard", status)
    return {"summary": table.to_dicts()}
