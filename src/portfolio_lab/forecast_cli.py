"""``plab forecast ...``: the next-month forecaster (``forecasters``).

The nightly derive step runs both (``jobs.tasks.derive_task``); these are for running them
by hand: ``linear`` fits the forecaster month by month and ``publish`` writes the Forecasts
page's summary.
"""

import polars as pl
import typer

from portfolio_lab.core.config import get_settings
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.forecasters import linear, report
from portfolio_lab.research.panel import Panel

forecast_app = typer.Typer(help="Next-month stock forecaster.")


@forecast_app.command("linear")
def linear_cmd() -> None:
    """Fit the forecaster month by month and save its forecasts (forecasters.linear)."""
    from portfolio_lab.jobs.tasks import RESEARCH_RULES  # noqa: PLC0415 - avoids a cycle

    data_dir = get_settings().data_dir
    paths = DataPaths(data_dir)
    data = linear.table(Panel.load(data_dir, rules=RESEARCH_RULES), pl.read_parquet(paths.features))
    forecasts = linear.walk(data)
    write_parquet_atomic(forecasts, paths.forecaster / linear.FILE)
    typer.echo(f"{forecasts.height:,} forecasts, {forecasts['date'].n_unique()} months "
               f"({forecasts['date'].min()} to {forecasts['date'].max()})")  # fmt: skip


@forecast_app.command("publish")
def publish_cmd() -> None:
    """Write the Forecasts page's summary (forecasters.report)."""
    data_dir = get_settings().data_dir
    paths = DataPaths(data_dir)
    path = report.publish(paths.forecaster, Panel.load(data_dir), pl.read_parquet(paths.features))
    typer.echo(path)
