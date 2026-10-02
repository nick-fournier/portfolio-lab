"""Command-line entry point (``plab``).

Commands stay thin: they parse arguments and delegate to ``jobs.tasks`` (shared with the
scheduler), the web app factory, or the scheduler loop.
"""

import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import httpx
import polars as pl
import typer

from portfolio_lab.backtest.results import load_run
from portfolio_lab.core.config import get_settings
from portfolio_lab.core.log import setup_logging
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.data.ingest import sharadar
from portfolio_lab.jobs import tasks, taxes
from portfolio_lab.research import forecasting, history, regimes, sources
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.scorecard import scorecard
from portfolio_lab.strategies.base import create

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


@app.command("make-vs-buy-history")
def make_vs_buy_history_cmd(
    publish: Annotated[
        Path | None, typer.Option(help="Main data directory to publish the results to.")
    ] = None,
) -> None:
    """Compare funds (since launch) with our strategies since 1999 on the Sharadar history."""
    typer.echo(tasks.make_vs_buy_history_task(get_settings(), publish))


@app.command("tax-runs")
def tax_runs_cmd(
    publish: Annotated[
        Path | None, typer.Option(help="Main data directory to publish the results to.")
    ] = None,
) -> None:
    """Backtest production at each level of stickiness, plus SPY, for the tax calculator."""
    for key, run_id in taxes.tax_runs_task(get_settings(), publish).items():
        typer.echo(f"{key:>14}: {run_id}")


@app.command("sources")
def sources_cmd(
    live: Annotated[Path, typer.Option(help="Live data directory (Alpaca + SEC).")],
    sharadar: Annotated[Path, typer.Option(help="Sharadar data directory.")],
    scratch: Annotated[Path, typer.Option(help="Folder for the mixed data directories.")],
    start: Day = datetime(2017, 1, 3),
    swap: Annotated[bool, typer.Option(help="Also run the four-way swap test.")] = True,
) -> None:
    """Split production's live-vs-Sharadar gap: prices vs fundamentals, inputs, timing."""
    settings = get_settings()
    feats = {n: pl.read_parquet(d / "features" / "monthly.parquet") for n, d in
             (("live", live), ("sharadar", sharadar))}  # fmt: skip
    out = sources.compare_inputs(feats["live"], feats["sharadar"], start.date())
    with pl.Config(tbl_rows=50, float_precision=3, tbl_width_chars=200):
        typer.echo(
            f"Sharadar's top-{sources.POOL} pool found in live data: {out['pool_in_live']:.1%}"
        )
        typer.echo(out["inputs"])
        overlap = out["overlap"]
        by_year = overlap.group_by(pl.col("date").dt.year().alias("year")).agg(
            pl.col("overlap").mean()).sort("year")  # fmt: skip
        typer.echo(f"Healthiest-{sources.TOP} overlap: {overlap['overlap'].mean():.1%}")
        typer.echo(by_year)
        typer.echo(f"Filing timing (live minus Sharadar days since filing): {out['timing']}")
        if swap:

            def factory():
                return tasks._attach_runtime(create(*tasks.PRODUCTION[:1], **tasks.PRODUCTION[1]),
                                             settings)  # fmt: skip

            dirs = {"live": live, "sharadar": sharadar}
            typer.echo(sources.swap_test(factory, dirs, scratch, start.date()))


@app.command("forecast-study")
def forecast_study_cmd(
    model: Annotated[list[str] | None, typer.Option(help="Models to run (default: all).")] = None,
    drop: Annotated[
        list[str] | None, typer.Option(help="Input groups to leave out; repeatable.")
    ] = None,
    importance: Annotated[
        bool, typer.Option(help="Also measure each input group's importance.")
    ] = False,
) -> None:
    """Can next month's stock returns be forecast? Walk-forward study (research.forecasting)."""
    settings = get_settings()
    paths = DataPaths(settings.data_dir)
    panel = Panel.load(settings.data_dir)
    env = pl.read_parquet(paths.environment) if paths.environment.exists() else None
    data = forecasting.load(panel, pl.read_parquet(paths.features), env)
    del panel
    out = settings.data_dir / "results" / "forecast_study"
    out.mkdir(parents=True, exist_ok=True)
    tag = "".join(f"-no_{g}" for g in drop or ())
    rows = []
    for name in model or forecasting.MODELS:
        scores: list | None = [] if importance else None
        forecasts = forecasting.walk_forward(data, name, tuple(drop or ()), scores)
        months = forecasting.grade_months(forecasts)
        write_parquet_atomic(months, out / f"{name}{tag}.parquet")
        if scores:
            table = pl.DataFrame(scores)
            write_parquet_atomic(table, out / f"{name}{tag}.importance.parquet")
            with pl.Config(tbl_rows=20, float_precision=4):
                typer.echo(f"{name}: IC drop when each group is shuffled (mean over years)")
                typer.echo(table.group_by("group").agg(pl.col("drop").mean(),
                           (pl.col("drop") > 0).mean().alias("years_helped"))
                           .sort("drop", descending=True))  # fmt: skip
        summary = forecasting.summarize(months)
        for part in ("all", "first_half", "second_half"):
            rows.append({"model": name + tag, "period": part, **summary[part]})
        rows.append({"model": name + tag, "period": "years IC > 0",
                     "ic": summary["years_ic_positive"]})  # fmt: skip
    with pl.Config(tbl_rows=60, tbl_cols=20, float_precision=3, tbl_width_chars=200):
        typer.echo(pl.DataFrame(rows))


@app.command("scorecard")
def scorecard_cmd(
    run: Annotated[list[str], typer.Option(help="name=run_id; repeatable.")],
    reference: Annotated[str, typer.Option(help="Name of the run others are compared with.")],
) -> None:
    """Compare backtest runs across eras, halves and against a reference run."""
    settings = get_settings()
    named = dict(item.split("=", 1) for item in run)
    loaded = {n: load_run(settings.data_dir, r) for n, r in named.items()}
    table = scorecard(
        {n: r.daily.select("date", "ret") for n, r in loaded.items()},
        {n: r.metrics.get("turnover_annual") for n, r in loaded.items()},
        reference,
    )
    with pl.Config(tbl_rows=50, tbl_cols=30, float_precision=3, tbl_width_chars=250):
        typer.echo(table)


@app.command("regimes")
def regimes_cmd() -> None:
    """Event study: what followed fragile, bear and rebound signals (``research.regimes``)."""
    settings = get_settings()
    paths = DataPaths(settings.data_dir)
    frame = regimes.signals(Panel.load(settings.data_dir), pl.read_parquet(paths.environment))
    write_parquet_atomic(frame, settings.data_dir / "results" / "regimes.parquet")
    with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=3, tbl_width_chars=250,
                   fmt_str_lengths=400):  # fmt: skip
        typer.echo(regimes.summarize(frame))


@app.command("oracle")
def oracle_cmd(
    months: Annotated[int, typer.Option(help="Crash-warning window in months.")],
    start: Day,
) -> None:
    """Research only: healthy meanvar with a perfect crash warning (hindsight), its ceiling."""
    settings = get_settings()
    dates = regimes.crash_ahead(Panel.load(settings.data_dir), months)
    params = {"healthy_share": 0.27, "defend_dates": dates}
    run_id, metrics = tasks.backtest_task(settings, "meanvar", start.date(), params=params)
    typer.echo(f"run {run_id}: {len(dates)} defended month ends, cagr {metrics['cagr']:.4f}")


@app.command("history")
def history_cmd(
    french: Annotated[Path, typer.Option(help="Folder with Kenneth French's daily zips.")],
) -> None:
    """Regime signals since 1926 and a momentum proxy with the bear/rebound switches."""
    texts = {
        s: httpx.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": s},
                     timeout=60).text
        for s in history.FRED_SERIES
    }  # fmt: skip
    daily = history.load_french(french)
    frame = history.signals(daily, history.load_fred(texts))
    out = get_settings().data_dir / "results" / "history"
    write_parquet_atomic(frame, out / "signals.parquet")
    proxy = history.momentum_proxy(daily, frame)
    write_parquet_atomic(proxy, out / "momentum_proxy.parquet")
    with pl.Config(tbl_rows=60, float_precision=3, tbl_width_chars=200):
        typer.echo(history.summarize(frame))
        typer.echo(history.summarize(frame, by_era=True))
        typer.echo(proxy)
    episodes = history.bear_episodes(frame)
    typer.echo(
        f"{len(episodes)} bear episodes: " + ", ".join(f"{a:%Y-%m}..{b:%Y-%m}" for a, b in episodes)
    )


@app.command("paper")
def paper_cmd(
    dry_run: Annotated[
        bool, typer.Option(help="Show the orders a rebalance would place now; place none.")
    ] = False,
) -> None:
    """Record the paper account and rebalance it at month ends (Alpaca paper, dummy money)."""
    result = tasks.paper_task(get_settings(), dry_run=dry_run)
    if not dry_run:
        typer.echo(result)
        return
    typer.echo(f"session {result['session']}, equity ${result['equity']:,.0f}")
    typer.echo(f"targets: {', '.join(f'{s} {w:.1%}' for s, w in result['targets'].items())}")
    typer.echo(f"close: {', '.join(result['closes']) or 'none'}")
    for o in result["orders"]:
        size = f"${o.notional:,.2f}" if o.notional is not None else f"{o.qty:g} shares"
        typer.echo(f"  {o.side:4} {o.symbol:6} {size}")


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
