"""``plab optimizer ...``: build the optimizer's data, walk a model forward, backtest its weights.

Research commands (class 3, optimizer track). ``data`` and ``backtest`` read the data hive;
``walk`` needs only the table ``data`` wrote and torch (``uv sync --extra nets``), and goes on
the GPU host.
"""

import json
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

from portfolio_lab.core.config import get_settings

optimizer_app = typer.Typer(help="Class 3 optimizer track (research).")


def _sharpe(monthly: np.ndarray) -> float:
    return float(monthly.mean() / monthly.std() * 12**0.5) if len(monthly) > 1 else float("nan")


@optimizer_app.command("data")
def data_cmd(out: Annotated[Path, typer.Option(help="Parquet file to write.")]) -> None:
    """Write the stock-month table: characteristics, eligibility, liquidity, target."""
    from portfolio_lab.research.optimizer import data  # noqa: PLC0415

    table = data.build(get_settings().data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out)
    typer.echo(f"{table.height:,} rows, {table['date'].n_unique()} months, "
               f"{len(data.inputs(table))} characteristics -> {out}")  # fmt: skip


@optimizer_app.command("walk")
def walk_cmd(  # noqa: PLR0913, PLR0917 - one option per setting
    table: Annotated[Path, typer.Option(help="The table from `plab optimizer data`.")],
    out: Annotated[Path, typer.Option(help="Folder for weights, log, model and settings.")],
    model: Annotated[str, typer.Option(help="linear, mlp or transformer.")],
    pool: Annotated[int, typer.Option(help="0: the paper's universe; N: N most liquid.")] = 0,
    blocks: Annotated[int, typer.Option()] = 1,
    hidden: Annotated[int, typer.Option()] = 256,
    window: Annotated[int, typer.Option(help="Training months (0: all earlier months).")] = 60,
    first_epochs: Annotated[int, typer.Option()] = 100,
    warm_epochs: Annotated[int, typer.Option()] = 5,
    lr: Annotated[float, typer.Option()] = 1e-3,
    seed: Annotated[int, typer.Option()] = 0,
    start: Annotated[str, typer.Option(help="First test month end.")] = "2003-12-31",
    device: Annotated[str, typer.Option()] = "cuda",
    long_only: Annotated[bool, typer.Option(help="Softmax weights (not the paper's).")] = False,
) -> None:
    """Walk one model forward and save its monthly portfolios."""
    import torch  # noqa: PLC0415

    from portfolio_lab.research.optimizer import data, walk  # noqa: PLC0415

    settings = walk.Settings(model, blocks, hidden, window or None, first_epochs, warm_epochs,
                             lr, seed, date.fromisoformat(start), long_only)  # fmt: skip
    months = data.months(pl.read_parquet(table), pool or None)
    weights, log, fitted = walk.walk(months, settings, device)
    out.mkdir(parents=True, exist_ok=True)
    weights.write_parquet(out / "weights.parquet")
    log.write_parquet(out / "log.parquet")
    torch.save(fitted.state_dict(), out / "model.pt")
    meta = asdict(replace(settings, start=settings.start.isoformat())) | {"pool": pool}
    (out / "settings.json").write_text(json.dumps(meta, indent=2))
    graded = log.filter(pl.col("ret").is_not_null())
    r, d = graded["ret"].to_numpy(), graded["date"].to_list()
    half = next((k for k, x in enumerate(d) if x >= date(2015, 12, 31)), len(d))
    typer.echo(f"{model}: {len(r)} months, Sharpe (monthly returns over T-bill, annualized) "
               f"{_sharpe(r):.2f} (to 2015 {_sharpe(r[:half]):.2f}, "
               f"after {_sharpe(r[half:]):.2f}); "
               f"median long {graded['long'].median():.2f}, short {graded['short'].median():.2f}, "
               f"largest {graded['largest'].median():.1%} of gross; "
               f"{log['seconds'].sum() / 60:.1f} min")  # fmt: skip


@optimizer_app.command("backtest")
def backtest_cmd(
    weights: Annotated[Path, typer.Option(help="weights.parquet from `plab optimizer walk`.")],
    start: Annotated[str, typer.Option()] = "2004-01-30",
    min_weight: Annotated[float, typer.Option(help="Smallest weight held.")] = 0.0,
) -> None:
    """Hold a walk's weights in the backtest engine (costs included) and report."""
    from portfolio_lab.backtest.costs import CostModel  # noqa: PLC0415
    from portfolio_lab.backtest.engine import BacktestConfig, run  # noqa: PLC0415
    from portfolio_lab.research.optimizer.replay import Replay  # noqa: PLC0415
    from portfolio_lab.research.panel import Panel  # noqa: PLC0415

    panel = Panel.load(get_settings().data_dir)
    strategy = Replay(pl.read_parquet(weights), min_weight)
    config = BacktestConfig(start=date.fromisoformat(start), end=panel.dates[-1],
                            costs=CostModel(notional=100_000))  # fmt: skip
    res = run(strategy, panel, config)
    daily = res.daily.select("date", "ret", "turnover", "cost")
    daily.write_parquet(weights.with_name("daily.parquet"))
    r, d = daily["ret"].to_numpy(), daily["date"].to_list()
    half = next((k for k, x in enumerate(d) if x >= date(2016, 1, 1)), len(d))
    years = len(r) / 252

    def cagr(x: np.ndarray) -> float:
        return float(np.prod(1 + x) ** (252 / len(x)) - 1)

    m = res.metrics
    typer.echo(f"CAGR {m['cagr']:.2%}  Sharpe {m['sharpe']:.2f}  max DD {m['max_drawdown']:.0%}  "
               f"2004-15 {cagr(r[:half]):.1%}  2016-26 {cagr(r[half:]):.1%}  "
               f"turnover/yr {daily['turnover'].sum() / years:.1f}x  "
               f"cost {daily['cost'].sum() / years:.2%}/yr")  # fmt: skip
