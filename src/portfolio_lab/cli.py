"""Command-line entry point (``plab``).

Commands stay thin: they parse arguments and delegate to the ``data``, ``backtest``,
``web`` and ``jobs`` packages, so the same logic is reachable from tests and the scheduler.
"""

from importlib.metadata import version
from typing import Annotated

import typer

app = typer.Typer(help="Portfolio lab: ingest data, run backtests, serve the dashboard.")


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
