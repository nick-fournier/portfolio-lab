"""Command-line entry point (``plab``).

Commands stay thin: they parse arguments and delegate to the ``data``, ``backtest``,
``web`` and ``jobs`` packages, so the same logic is reachable from tests and the scheduler.
"""

import json
from datetime import date, datetime
from importlib.metadata import version
from typing import Annotated, Any

import typer

from portfolio_lab.backtest.costs import CostModel
from portfolio_lab.backtest.engine import BacktestConfig, run
from portfolio_lab.backtest.results import save_run
from portfolio_lab.core.config import get_settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.log import setup_logging
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data.ingest.prices import update_prices, verify_prices
from portfolio_lab.data.ingest.rates import ingest_rates
from portfolio_lab.data.ingest.universe import current_symbols, ingest_universe
from portfolio_lab.data.sources.alpaca import make_client
from portfolio_lab.research.panel import Panel
from portfolio_lab.strategies.base import create

app = typer.Typer(help="Portfolio lab: ingest data, run backtests, serve the dashboard.")
ingest_app = typer.Typer(help="Fetch and store market data.")
app.add_typer(ingest_app, name="ingest")

Full = Annotated[bool, typer.Option("--full", help="Re-fetch full history instead of updating.")]


def _print_version(value: bool) -> None:
    """Print the installed package version and exit when ``--version`` is passed."""
    if value:
        typer.echo(version("portfolio-lab"))
        raise typer.Exit


@app.callback()
def main(
    _version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_print_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Portfolio lab command-line interface."""
    setup_logging(get_settings().log_level)


def _public_client() -> RateLimitedClient:
    """Return a client for unauthenticated public sources (NASDAQ Trader, FRED)."""
    return RateLimitedClient(headers={"User-Agent": get_settings().edgar_user_agent})


@ingest_app.command("universe")
def ingest_universe_cmd() -> None:
    """Snapshot the NASDAQ Trader symbol directory and update the symbol table."""
    with _public_client() as client:
        ingest_universe(get_settings(), client, date.today())


@ingest_app.command("prices")
def ingest_prices_cmd(full: Full = False) -> None:
    """Update daily prices for the current universe (run ``ingest universe`` first)."""
    settings = get_settings()
    paths = DataPaths(settings.data_dir)
    symbols = current_symbols(paths)
    if not symbols:
        raise typer.BadParameter("no universe stored yet; run `plab ingest universe` first")
    with make_client(settings) as client:
        update_prices(settings, client, symbols, paths.prices_daily, "prices", full=full)


@ingest_app.command("benchmarks")
def ingest_benchmarks_cmd(full: Full = False) -> None:
    """Update daily prices for the benchmark ETFs (SPY, QQQ, IWM)."""
    settings = get_settings()
    paths = DataPaths(settings.data_dir)
    with make_client(settings) as client:
        update_prices(
            settings,
            client,
            settings.benchmark_symbols,
            paths.prices_benchmarks,
            "benchmarks",
            full,
        )


@ingest_app.command("rates")
def ingest_rates_cmd() -> None:
    """Refresh the 3-month T-bill rate history from FRED."""
    with _public_client() as client:
        ingest_rates(get_settings(), client)


@ingest_app.command("verify")
def verify_prices_cmd(
    sample: Annotated[int, typer.Option(help="Number of symbols to check.")] = 50,
) -> None:
    """Check stored returns against a fresh fetch for a random sample; repair mismatches."""
    settings = get_settings()
    with make_client(settings) as client:
        verify_prices(settings, client, DataPaths(settings.data_dir).prices_daily, sample)


@ingest_app.command("all")
def ingest_all_cmd(full: Full = False) -> None:
    """Run universe, prices, benchmarks and rates in order."""
    ingest_universe_cmd()
    ingest_prices_cmd(full)
    ingest_benchmarks_cmd(full)
    ingest_rates_cmd()


def _parse_params(pairs: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` pairs; values are JSON when possible (numbers, null), else strings."""
    params = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise typer.BadParameter(f"expected key=value, got {pair!r}")
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw
    return params


Day = Annotated[datetime, typer.Option(formats=["%Y-%m-%d"], help="Date (YYYY-MM-DD).")]


@app.command("backtest")
def backtest_cmd(
    strategy: Annotated[str, typer.Argument(help="Registered strategy name.")],
    start: Day,
    end: Annotated[
        datetime | None, typer.Option(formats=["%Y-%m-%d"], help="Last date (default: latest).")
    ] = None,
    param: Annotated[
        list[str] | None, typer.Option(help="Strategy parameter as key=value; repeatable.")
    ] = None,
    notional: Annotated[
        float, typer.Option(help="Portfolio size in dollars, for costs.")
    ] = 100_000,
    max_weight: Annotated[float, typer.Option(help="Largest weight in any one name.")] = 1.0,
) -> None:
    """Run a strategy through the walk-forward backtest and save the run."""
    settings = get_settings()
    strat = create(strategy, **_parse_params(param or []))
    panel = Panel.load(settings.data_dir, end=end.date() if end else None)
    config = BacktestConfig(
        start=start.date(),
        end=end.date() if end else panel.dates[-1],
        costs=CostModel(notional=notional),
        max_weight=max_weight,
    )
    result = run(strat, panel, config)
    run_id = save_run(result, settings.data_dir)
    typer.echo(f"run {run_id}")
    for key, value in result.metrics.items():
        typer.echo(f"  {key:>20}: {value:,.4f}")
