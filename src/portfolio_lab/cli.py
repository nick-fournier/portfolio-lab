"""Command-line entry point (``plab``).

Commands stay thin: they parse arguments and delegate to ``jobs.tasks`` (shared with the
scheduler), the web app factory, or the scheduler loop.
"""

import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import typer

from portfolio_lab.core.config import get_settings
from portfolio_lab.core.log import setup_logging
from portfolio_lab.data.ingest import sharadar
from portfolio_lab.jobs import probes, tasks

app = typer.Typer(help="Portfolio lab: ingest data, run backtests, serve the dashboard.")
ingest_app = typer.Typer(help="Fetch and store market data.")
app.add_typer(ingest_app, name="ingest")

Full = Annotated[bool, typer.Option("--full", help="Re-fetch full history instead of updating.")]
Day = Annotated[datetime, typer.Option(formats=["%Y-%m-%d"], help="Date (YYYY-MM-DD).")]


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


@ingest_app.command("universe")
def ingest_universe_cmd() -> None:
    """Snapshot the NASDAQ Trader symbol directory and update the symbol table."""
    tasks.ingest_universe_task(get_settings())


@ingest_app.command("prices")
def ingest_prices_cmd(full: Full = False) -> None:
    """Update daily prices for the current universe (run ``ingest universe`` first)."""
    tasks.ingest_prices_task(get_settings(), full)


@ingest_app.command("benchmarks")
def ingest_benchmarks_cmd(full: Full = False) -> None:
    """Update daily prices for the benchmark ETFs (SPY, QQQ, IWM)."""
    tasks.ingest_benchmarks_task(get_settings(), full)


@ingest_app.command("rates")
def ingest_rates_cmd() -> None:
    """Refresh the 3-month T-bill rate history from FRED."""
    tasks.ingest_rates_task(get_settings())


@ingest_app.command("verify")
def verify_prices_cmd(
    sample: Annotated[int, typer.Option(help="Number of symbols to check.")] = 50,
) -> None:
    """Check stored returns against a fresh fetch for a random sample; repair mismatches."""
    tasks.verify_task(get_settings(), sample)


@ingest_app.command("fundamentals")
def ingest_fundamentals_cmd(
    force: Annotated[
        bool, typer.Option(help="Re-parse even if the bulk file is unchanged.")
    ] = False,
) -> None:
    """Refresh SEC fundamentals for the universe and recompute Piotroski F-scores."""
    tasks.fundamentals_task(get_settings(), force)


@ingest_app.command("macro")
def ingest_macro_cmd() -> None:
    """Refresh the FRED market and economic context series (needs FRED_API_KEY)."""
    typer.echo(tasks.macro_task(get_settings()))


@ingest_app.command("delisted")
def ingest_delisted_cmd() -> None:
    """Add stocks delisted since 2016 (Tiingo list) and backfill their prices."""
    tasks.delisted_task(get_settings())


@ingest_app.command("all")
def ingest_all_cmd(full: Full = False) -> None:
    """Run universe, prices, benchmarks and rates in order."""
    tasks.daily_ingest_task(get_settings(), full)


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


@app.command("ingest-sharadar")
def ingest_sharadar_cmd(
    out: Annotated[Path, typer.Option(help="New data directory to write.")],
    raw: Annotated[
        Path | None, typer.Option(help="Folder with the bulk zips (default: raw/sharadar).")
    ] = None,
) -> None:
    """Build a separate data directory from Sharadar's full-history bulk files."""
    settings = get_settings()
    raw = raw or settings.data_dir / "raw" / "sharadar"
    typer.echo(sharadar.build(raw, out, settings.data_dir))


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
    delisting_return: Annotated[
        float, typer.Option(help="Return on exit for stocks that fell to OTC (-1 = total loss).")
    ] = -0.30,
) -> None:
    """Run a strategy through the walk-forward backtest and save the run."""
    run_id, metrics = tasks.backtest_task(
        get_settings(),
        strategy,
        start.date(),
        end.date() if end else None,
        _parse_params(param or []),
        notional,
        max_weight,
        delisting_return,
    )
    typer.echo(f"run {run_id}")
    for key, value in metrics.items():
        typer.echo(f"  {key:>20}: {value:,.4f}")


@app.command("features")
def features_cmd() -> None:
    """Rebuild the monthly point-in-time feature panel."""
    typer.echo(tasks.features_task(get_settings()))


@app.command("scoreboard")
def scoreboard_cmd(
    signal: Annotated[
        list[str] | None, typer.Option(help="Only these signals (by name); repeatable.")
    ] = None,
) -> None:
    """Score each signal's monthly rankings against the returns that followed."""
    only = tuple((n, p) for n, p in tasks.SCOREBOARD_SIGNALS if not signal or n in signal)
    if not only:
        raise typer.BadParameter(f"no scoreboard signal named {signal}")
    result = tasks.scoreboard_task(get_settings(), only)
    for row in result["summary"]:
        typer.echo(
            f"{row['pool']:>6}  {row['signal']:<36} IC {row['mean_ic']:+.3f} "
            f"(t {row['ic_t']:+.1f}, hit {row['hit']:.0%})  spread {row['spread']:+.1%}"
        )


@app.command("context")
def context_cmd() -> None:
    """Measure trait payoffs and forward market risk by prevailing market conditions."""
    typer.echo(tasks.context_task(get_settings()))


@app.command("models")
def models_cmd() -> None:
    """Train and evaluate the relaxed models walk-forward (results under results/models)."""
    typer.echo(tasks.models_task(get_settings()))


@app.command("make-vs-buy")
def make_vs_buy_cmd() -> None:
    """Compare our strategies with funds anyone can buy (needs TIINGO_API_KEY for mutual funds)."""
    typer.echo(tasks.make_vs_buy_task(get_settings()))


@app.command("forecasts")
def forecasts_cmd() -> None:
    """Measure direction, magnitude and hurdle forecasts at 1-12 month horizons."""
    typer.echo(probes.forecasts_task(get_settings()))


@app.command("bakeoff")
def bakeoff_cmd() -> None:
    """Compare learners (trees, MLP, their average) on the 1-month return rank."""
    for row in probes.bakeoff_task(get_settings())["learners"]:
        typer.echo(row)


@app.command("pick-test")
def pick_test_cmd() -> None:
    """Model variants' top picks vs the 100 most liquid, SPY and meanvar each month."""
    for row in probes.pick_test_task(get_settings())["summary"]:
        typer.echo(row)


@app.command("health")
def health_cmd() -> None:
    """Train the trash-risk score walk-forward and compare it with the F-score filter."""
    for pool, rows in probes.health_task(get_settings()).items():
        typer.echo(f"pool {pool}")
        for row in rows:
            typer.echo(f"  {row}")


@app.command("fscore-models")
def fscore_models_cmd() -> None:
    """Build the continuous and learned F-scores walk-forward."""
    typer.echo(probes.fscore_models_task(get_settings()))


@app.command("filter-backtests")
def filter_backtests_cmd(
    score: Annotated[
        list[str] | None, typer.Option(help="Only these scores (no reference runs); repeatable.")
    ] = None,
) -> None:
    """Backtest meanvar behind each F-score model at each cut (after ``plab fscore-models``)."""
    typer.echo(probes.filter_backtests_task(get_settings(), tuple(score or ())))


@app.command("extended-scores")
def extended_scores_cmd() -> None:
    """Backtest meanvar behind the continuous F-score with extra health metrics, and PCA."""
    typer.echo(probes.extended_scores_task(get_settings()))


@app.command("make-vs-buy-history")
def make_vs_buy_history_cmd(
    raw: Annotated[Path, typer.Option(help="Folder with the Sharadar bulk zips.")],
) -> None:
    """Compare funds (since launch) with our strategies since 1999 on the Sharadar history."""
    typer.echo(tasks.make_vs_buy_history_task(get_settings(), raw))


@app.command("model-portfolio")
def model_portfolio_cmd() -> None:
    """Refresh 1-month forecast scores, then backtest the model portfolios and controls."""
    settings = get_settings()
    typer.echo(probes.forecast_scores_task(settings))
    typer.echo(probes.model_backtests_task(settings))


@app.command("serve")
def serve_cmd(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8100,
) -> None:
    """Serve the read-only dashboard."""
    import uvicorn  # noqa: PLC0415 - only needed for this command

    from portfolio_lab.web.app import create_app  # noqa: PLC0415 - keeps other commands fast

    uvicorn.run(create_app(get_settings().data_dir), host=host, port=port, proxy_headers=True)


@app.command("schedule")
def schedule_cmd(
    once: Annotated[bool, typer.Option(help="Run due jobs once and exit.")] = False,
) -> None:
    """Run ingest and maintenance jobs whenever they are due (long-running)."""
    from portfolio_lab.jobs.scheduler import run_forever, run_once  # noqa: PLC0415

    settings = get_settings()
    if once:
        typer.echo(", ".join(run_once(settings)) or "nothing due")
    else:
        run_forever(settings)
