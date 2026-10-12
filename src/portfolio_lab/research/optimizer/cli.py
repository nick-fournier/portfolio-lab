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


def _months(
    table: Path, pool: int | None, inputs: str = "all", min_price: float = 0, min_adv: float = 0
) -> list:
    """``data.months`` of a table, cached next to it (ranking every month takes a while)."""
    from portfolio_lab.research.optimizer import data  # noqa: PLC0415

    floor = f"-p{min_price:g}-a{min_adv:g}" if min_price or min_adv else ""
    cache = table.with_name(f"{table.stem}.months-{pool or 'paper'}-{inputs}{floor}.pkl")
    if cache.exists() and cache.stat().st_mtime > table.stat().st_mtime:
        return pickle.loads(cache.read_bytes())
    frame = pl.read_parquet(table)
    months = data.months(frame, pool, data.inputs(frame, inputs), min_price=min_price,
                         min_adv=min_adv)  # fmt: skip
    cache.write_bytes(pickle.dumps(months))
    return months


def _walk_one(  # noqa: PLR0913, PLR0917 - one walk's pieces
    table: Path,
    out: Path,
    settings,
    pool: int,
    device: str,
    threads: int,
    precision: str,
    inputs: str,
    floor: tuple[float, float] = (0.0, 0.0),
) -> str:
    """One walk: train, save weights, log, model and settings; return a summary line."""
    import torch  # noqa: PLC0415

    from portfolio_lab.research.optimizer import walk  # noqa: PLC0415

    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision(precision)
    weights, log, fitted = walk.walk(_months(table, pool or None, inputs, *floor), settings, device)
    out.mkdir(parents=True, exist_ok=True)
    weights.write_parquet(out / "weights.parquet")
    log.write_parquet(out / "log.parquet")
    torch.save(fitted.state_dict(), out / "model.pt")
    meta = asdict(replace(settings, start=settings.start.isoformat())) | {
        "pool": pool,
        "inputs": inputs,
    }
    (out / "settings.json").write_text(json.dumps(meta, indent=2))
    graded = log.filter(pl.col("ret").is_not_null())
    return (f"{settings.model} seed {settings.seed}: {graded.height} months, Sharpe "
            f"{_halves(graded['date'].to_list(), graded['ret'].to_numpy())}; "
            f"median long {graded['long'].median():.2f}, short {graded['short'].median():.2f}, "
            f"largest {graded['largest'].median():.1%} of gross; "
            f"{log['seconds'].sum() / 60:.1f} min")  # fmt: skip


def _die_with_parent() -> None:
    """Make a seed worker exit if the walk command that started it is killed (Linux)."""
    import ctypes  # noqa: PLC0415
    import signal  # noqa: PLC0415

    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _seeds(text: str) -> list[int]:
    """``"3"`` -> [3]; ``"0-9"`` -> [0, ..., 9]; ``"1,4,7"`` -> [1, 4, 7]."""
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


@optimizer_app.command("walk")
def walk_cmd(  # noqa: PLR0913, PLR0917 - one option per setting
    table: Annotated[Path, typer.Option(help="The table from `plab optimizer data`.")],
    out: Annotated[Path, typer.Option(help="Folder for the walk (one subfolder per seed).")],
    model: Annotated[str, typer.Option(help="linear, mlp or transformer.")],
    pool: Annotated[int, typer.Option(help="0: the paper's universe; N: N most liquid.")] = 0,
    blocks: Annotated[int, typer.Option()] = 1,
    hidden: Annotated[int, typer.Option()] = 256,
    window: Annotated[int, typer.Option(help="Training months (0: all earlier months).")] = 60,
    first_epochs: Annotated[int, typer.Option()] = 100,
    warm_epochs: Annotated[int, typer.Option()] = 5,
    lr: Annotated[float, typer.Option()] = 1e-3,
    seeds: Annotated[str, typer.Option(help='Seeds: "0", "0-9" or "1,4,7".')] = "0",
    jobs: Annotated[int, typer.Option(help="Seeds trained at once (processes on the GPU).")] = 4,
    start: Annotated[str, typer.Option(help="First test month end.")] = "2003-12-31",
    device: Annotated[str, typer.Option()] = "cuda",
    long_only: Annotated[bool, typer.Option(help="Softmax weights (not the paper's).")] = False,
    scale: Annotated[bool, typer.Option(help="Long-only with a learned overall size m.")] = False,
    cap: Annotated[float, typer.Option(help="Long-only's largest weight (0: no cap).")] = 0.0,
    names: Annotated[float, typer.Option(help="Long-only: effective names held (0: off).")] = 0.0,
    fast_attention: Annotated[
        bool, typer.Option(help="bfloat16 fused attention (~5x faster transformer).")
    ] = False,
    clip: Annotated[float, typer.Option(help="Gradient clipping (0: none).")] = 1.0,
    inputs: Annotated[str, typer.Option(help="Characteristics: all, ours, jkp or paper.")] = "all",
    min_price: Annotated[float, typer.Option(help="Leave out stocks below this price.")] = 0.0,
    min_adv: Annotated[float, typer.Option(help="...or below this daily dollar volume.")] = 0.0,
    precision: Annotated[
        str, typer.Option(help="float32 matrix products: highest, or high (TF32, faster).")
    ] = "highest",
) -> None:
    """Walk a model forward, one run per seed, and save each seed's monthly portfolios.

    Seeds run ``jobs`` at a time in separate processes: one walk takes small steps (one month
    each) and leaves the GPU mostly idle, so several share it. With more than one seed the
    seeds' portfolios are also averaged (the paper's 10-seed average).
    """
    import multiprocessing  # noqa: PLC0415
    import os  # noqa: PLC0415
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from portfolio_lab.research.optimizer import walk  # noqa: PLC0415

    picked = _seeds(seeds)
    _months(table, pool or None, inputs, min_price, min_adv)  # build the cache once first
    runs = [out / f"seed-{s}" for s in picked]
    threads = max(1, (os.cpu_count() or 1) // min(jobs, len(picked)))
    with ProcessPoolExecutor(min(jobs, len(picked)), initializer=_die_with_parent,
                             mp_context=multiprocessing.get_context("spawn")) as pool_:  # fmt: skip
        futures = []
        for s, run in zip(picked, runs, strict=True):
            settings = walk.Settings(
                model, blocks, hidden, window or None, first_epochs, warm_epochs, lr, s,
                date.fromisoformat(start), long_only, fast_attention, clip or None, scale,
                cap or None, names or None,
            )  # fmt: skip
            futures.append(
                pool_.submit(
                    _walk_one,
                    table,
                    run,
                    settings,
                    pool,
                    device,
                    threads,
                    precision,
                    inputs,
                    (min_price, min_adv),
                )
            )
        for f in futures:
            typer.echo(f.result())
    if len(runs) > 1:
        typer.echo(_average(runs))


@optimizer_app.command("backtest")
def backtest_cmd(
    weights: Annotated[Path, typer.Option(help="weights.parquet from `plab optimizer walk`.")],
    start: Annotated[str, typer.Option()] = "2004-01-30",
    min_weight: Annotated[float, typer.Option(help="Smallest weight held.")] = 0.0,
    min_price: Annotated[float, typer.Option(help="Eligible: price at least this.")] = 5.0,
    min_adv: Annotated[float, typer.Option(help="...and daily dollar volume at least.")] = 1e6,
    min_history: Annotated[int, typer.Option(help="...and this many sessions listed.")] = 252,
) -> None:
    """Hold a walk's weights in the backtest engine (costs included) and report.

    Names must be eligible at each decision (defaults: the strategies' rules); the rest of
    the month's weights are scaled back up to sum to 1.
    """
    from portfolio_lab.backtest.costs import CostModel  # noqa: PLC0415
    from portfolio_lab.backtest.engine import BacktestConfig, run  # noqa: PLC0415
    from portfolio_lab.research.optimizer.replay import Replay  # noqa: PLC0415
    from portfolio_lab.research.panel import EligibilityRules, Panel  # noqa: PLC0415

    rules = EligibilityRules(min_price=min_price, min_dollar_volume=min_adv,
                             min_history=min_history)  # fmt: skip
    panel = Panel.load(get_settings().data_dir, rules=rules)
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
    """Average several seeds' portfolios (the paper's 10-seed average) and report Sharpe."""
    typer.echo(_average(runs))


def _average(runs: list[Path]) -> str:
    """The seeds' Sharpe spread and their averaged portfolio's Sharpe.

    Each seed's portfolio is first scaled to $1 of gross exposure (longs plus shorts) each
    month, so a seed whose weights grew huge cannot swamp the others; the averaged
    portfolio's monthly return is then the average of the scaled returns.
    """
    logs = [pl.read_parquet(r / "log.parquet").select(
                "date", (pl.col("ret") / (pl.col("long") - pl.col("short"))).alias(r.name))
            for r in runs]  # fmt: skip
    joined = logs[0]
    for log in logs[1:]:
        joined = joined.join(log, on="date")
    joined = joined.drop_nulls().sort("date")
    names = [r.name for r in runs]
    each = [_sharpe(joined[n].to_numpy()) for n in names]
    mean = joined.select(pl.mean_horizontal(names)).to_series().to_numpy()
    return (f"{len(runs)} seeds, each: min {min(each):.2f} median {np.median(each):.2f} "
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
