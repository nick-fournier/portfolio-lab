"""Job entry points shared by the CLI and the scheduler.

Each task opens its own HTTP clients, runs one unit of work end to end, and returns a
summary dict. They are idempotent: re-running after a crash or restart is safe.
"""

import logging
from datetime import date
from typing import Any

from portfolio_lab.backtest.costs import CostModel
from portfolio_lab.backtest.engine import BacktestConfig, run
from portfolio_lab.backtest.results import save_run
from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data.ingest.prices import update_prices, verify_prices
from portfolio_lab.data.ingest.rates import ingest_rates
from portfolio_lab.data.ingest.universe import current_symbols, ingest_universe
from portfolio_lab.data.sources.alpaca import make_client
from portfolio_lab.research.panel import Panel
from portfolio_lab.strategies.base import create

log = logging.getLogger(__name__)

#: Backtests the scheduler refreshes weekly so the dashboard always shows current baselines.
SCHEDULED_BACKTESTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("buy_hold", {}),
    ("equal_weight", {}),
)
#: Start date for scheduled backtests (one year after data starts, for lookbacks).
SCHEDULED_START = date(2017, 1, 3)


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


def verify_task(settings: Settings, sample: int = 50) -> dict:
    """Spot-check stored returns against a fresh fetch, repairing drift."""
    with make_client(settings) as client:
        return verify_prices(settings, client, DataPaths(settings.data_dir).prices_daily, sample)


def backtest_task(
    settings: Settings,
    strategy: str,
    start: date,
    end: date | None = None,
    params: dict[str, Any] | None = None,
    notional: float = 100_000,
    max_weight: float = 1.0,
) -> tuple[str, dict[str, float]]:
    """Run one backtest on the stored data and save it.

    Returns:
        The run id and its metrics.
    """
    strat = create(strategy, **(params or {}))
    panel = Panel.load(settings.data_dir, end=end)
    config = BacktestConfig(
        start=start,
        end=end or panel.dates[-1],
        costs=CostModel(notional=notional),
        max_weight=max_weight,
    )
    result = run(strat, panel, config)
    run_id = save_run(result, settings.data_dir)
    log.info("backtest %s saved as %s", strategy, run_id)
    return run_id, result.metrics


def scheduled_backtests_task(settings: Settings) -> dict:
    """Re-run the baseline backtests through the latest data."""
    return {
        name: backtest_task(settings, name, SCHEDULED_START, params=params)[0]
        for name, params in SCHEDULED_BACKTESTS
    }
