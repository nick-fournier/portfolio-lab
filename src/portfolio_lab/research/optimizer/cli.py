"""``plab optimizer ...``: build the optimizer's data, walk a model forward, backtest its weights.

Research commands (class 3, optimizer track). ``data`` and ``backtest`` read the data hive;
``walk`` needs only the table ``data`` wrote and torch (``uv sync --extra nets``), and goes on
the GPU host.
"""

import json
import pickle
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


def _months(table: Path, pool: int | None) -> list:
    """``data.months`` of a table, cached next to it (ranking every month takes a while)."""
    from portfolio_lab.research.optimizer import data  # noqa: PLC0415

    cache = table.with_name(f"{table.stem}.months-{pool or 'paper'}.pkl")
    if cache.exists() and cache.stat().st_mtime > table.stat().st_mtime:
        return pickle.loads(cache.read_bytes())
    months = data.months(pl.read_parquet(table), pool)
    cache.write_bytes(pickle.dumps(months))
    return months


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

    from portfolio_lab.research.optimizer import walk  # noqa: PLC0415

    settings = walk.Settings(model, blocks, hidden, window or None, first_epochs, warm_epochs,
                             lr, seed, date.fromisoformat(start), long_only)  # fmt: skip
    months = _months(table, pool or None)
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


def _halves(dates: list[date], r: np.ndarray) -> str:
    half = next((k for k, x in enumerate(dates) if x >= date(2015, 12, 31)), len(dates))
    return f"{_sharpe(r):.2f} (to 2015 {_sharpe(r[:half]):.2f}, after {_sharpe(r[half:]):.2f})"


@optimizer_app.command("average")
def average_cmd(runs: Annotated[list[Path], typer.Argument(help="Walk folders (one per seed).")]):
    """Average several seeds' portfolios (the paper's 10-seed average) and report Sharpe.

    The averaged weights' monthly return is the average of the seeds' returns.
    """
    logs = [pl.read_parquet(r / "log.parquet").select("date", pl.col("ret").alias(r.name))
            for r in runs]  # fmt: skip
    joined = logs[0]
    for log in logs[1:]:
        joined = joined.join(log, on="date")
    joined = joined.drop_nulls().sort("date")
    names = [r.name for r in runs]
    each = [_sharpe(joined[n].to_numpy()) for n in names]
    mean = joined.select(pl.mean_horizontal(names)).to_series().to_numpy()
    typer.echo(f"{len(runs)} seeds, each: min {min(each):.2f} median {np.median(each):.2f} "
               f"max {max(each):.2f}; "
               f"average portfolio: {_halves(joined['date'].to_list(), mean)}")  # fmt: skip


@optimizer_app.command("ridge")
def ridge_cmd(
    table: Annotated[Path, typer.Option(help="The table from `plab optimizer data`.")],
    start: Annotated[str, typer.Option(help="First test month end.")] = "2003-12-31",
) -> None:
    """The paper's linear benchmark: ridge in closed form, penalty picked two ways."""
    from portfolio_lab.research.optimizer import data, ridge  # noqa: PLC0415

    months = data.months(pl.read_parquet(table))
    first = next(k for k, m in enumerate(months) if m.date >= date.fromisoformat(start))
    out = ridge.walk(months, start_index=first)
    best = ridge.GRID.index(out["best_z"])
    typer.echo("Sharpe by penalty: " + ", ".join(
        f"{z:.0e} {s:.2f}" for z, s in zip(ridge.GRID, out["sharpe"], strict=True)))  # fmt: skip
    typer.echo(f"best penalty {out['best_z']:.0e} (looks ahead, as the paper's BSV): "
               f"{_halves(out['dates'], out['returns'][:, best])}")  # fmt: skip
    typer.echo(f"leave-one-out each month (no look-ahead): {_halves(out['dates'], out['loo'])}; "
               f"median penalty {np.median(out['loo_z']):.0e}")  # fmt: skip


@optimizer_app.command("signal")
def signal_cmd(
    table: Annotated[Path, typer.Option(help="The table from `plab optimizer data`.")],
    forecasts: Annotated[Path, typer.Option(help="Parquet with date, symbol, forecast.")],
    start: Annotated[str, typer.Option(help="First month end.")] = "2003-12-31",
) -> None:
    """Sharpe of a forecast traded the same long-short way, on the same candidates.

    Each month: the candidates that have a forecast, weights = the forecast's rank across
    them in [-0.5, 0.5] (a characteristic-managed portfolio, like each input's factor).
    """
    from portfolio_lab.research.optimizer import data  # noqa: PLC0415

    f = pl.read_parquet(forecasts).select("date", "symbol", "forecast")
    months = [m for m in data.months(pl.read_parquet(table))
              if m.date >= date.fromisoformat(start) and np.isfinite(m.r).all()]  # fmt: skip
    dates, rets, counts = [], [], []
    for m in months:
        got = dict(f.filter(pl.col("date") == m.date).select("symbol", "forecast").iter_rows())
        keep = [k for k, s in enumerate(m.symbols) if s in got]
        if len(keep) < 100:
            continue
        w = data._ranked(np.array([[got[m.symbols[k]]] for k in keep]))[:, 0]
        dates.append(m.date)
        rets.append(float(w @ m.r[keep]))
        counts.append(len(keep))
    typer.echo(f"{len(dates)} months, median {int(np.median(counts))} stocks with a forecast: "
               f"Sharpe {_halves(dates, np.array(rets))}")  # fmt: skip
