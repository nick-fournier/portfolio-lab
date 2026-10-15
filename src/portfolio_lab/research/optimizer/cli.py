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


def _months(table: Path, inputs: str = "all") -> list:
    """``data.months`` of a table, cached next to it (ranking every month takes a while)."""
    from portfolio_lab.research.optimizer import data  # noqa: PLC0415

    cache = table.with_name(f"{table.stem}.months-paper-{inputs}.pkl")
    if cache.exists() and cache.stat().st_mtime > table.stat().st_mtime:
        return pickle.loads(cache.read_bytes())
    frame = pl.read_parquet(table)
    months = data.months(frame, data.inputs(frame, inputs))
    cache.write_bytes(pickle.dumps(months))
    return months


def _walk_one(table: Path, out: Path, settings, device: str, threads: int, precision: str,
              inputs: str, end: date | None = None) -> str:  # fmt: skip
    """One walk (test months up to ``end``, if any): save weights, log, model and settings."""
    import torch  # noqa: PLC0415

    from portfolio_lab.research.optimizer import walk  # noqa: PLC0415

    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision(precision)
    months = [m for m in _months(table, inputs) if end is None or m.date <= end]
    stock = None
    if settings.costs:
        from portfolio_lab.research.optimizer import costs  # noqa: PLC0415

        stock = costs.align(months, pl.read_parquet(settings.costs))
    weights, log, fitted = walk.walk(months, settings, device, stock)
    out.mkdir(parents=True, exist_ok=True)
    weights.write_parquet(out / "weights.parquet")
    log.write_parquet(out / "log.parquet")
    torch.save(fitted.state_dict(), out / "model.pt")
    meta = asdict(replace(settings, start=settings.start.isoformat())) | {
        "inputs": inputs,
        "end": end.isoformat() if end else None,
    }
    (out / "settings.json").write_text(json.dumps(meta, indent=2))
    graded = log.filter(pl.col("ret").is_not_null())
    return (f"{settings.model} seed {settings.seed}: {graded.height} months, Sharpe "
            f"{_halves(graded['date'].to_list(), graded['ret'].to_numpy())}; "
            f"median long {graded['long'].median():.2f}, short {graded['short'].median():.2f}, "
            f"largest {graded['largest'].median() or 0:.1%} of gross; "
            f"{(log['long'] - log['short'] == 0).sum()} months all cash; "
            f"{log['seconds'].sum() / 60:.1f} min")  # fmt: skip


def _die_with_parent() -> None:
    """Make a seed worker exit if the walk command that started it is killed (Linux)."""
    import ctypes  # noqa: PLC0415
    import signal  # noqa: PLC0415

    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _clip(text: str) -> float | str | None:
    """``"auto"``, or a number (0: none)."""
    if text == "auto":
        return text
    return float(text) or None


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
    blocks: Annotated[int, typer.Option()] = 1,
    hidden: Annotated[int, typer.Option()] = 256,
    window: Annotated[int, typer.Option(help="Training months (0: all earlier months).")] = 60,
    first_epochs: Annotated[int, typer.Option()] = 100,
    warm_epochs: Annotated[int, typer.Option()] = 5,
    lr: Annotated[float, typer.Option()] = 1e-3,
    seeds: Annotated[str, typer.Option(help='Seeds: "0", "0-9" or "1,4,7".')] = "0",
    jobs: Annotated[int, typer.Option(help="Seeds trained at once (processes on the GPU).")] = 4,
    start: Annotated[str, typer.Option(help="First test month end.")] = "2003-12-31",
    end: Annotated[str, typer.Option(help="Last test month end (empty: the latest).")] = "",
    device: Annotated[str, typer.Option()] = "cuda",
    fast_attention: Annotated[
        bool, typer.Option(help="bfloat16 fused attention (~5x faster transformer).")
    ] = False,
    clip: Annotated[str, typer.Option(help="Gradient clipping: a limit, auto, or 0 (none).")] = "1",
    costs: Annotated[
        str, typer.Option(help="Loss: per-stock cost table (date, symbol, trade_cost, borrow_fee).")
    ] = "",
    short_hurdle: Annotated[
        float, typer.Option(help="Loss: extra yearly cost per dollar shorted (willingness).")
    ] = 0.0,
    schedule: Annotated[
        str, typer.Option(help="Learning rate per month: flat or cosine.")
    ] = "flat",
    lr_min: Annotated[float, typer.Option(help="Cosine schedule's last rate.")] = 1e-5,
    warm_lr: Annotated[
        float, typer.Option(help="Starting rate when continuing last month's model (0: lr).")
    ] = 0.0,
    compile_model: Annotated[
        bool, typer.Option("--compile", help="torch.compile the model (faster).")
    ] = False,
    max_return: Annotated[
        float, typer.Option(help="Training returns above this in size set to 0 (0: off).")
    ] = 0.0,
    holdings_input: Annotated[
        bool, typer.Option(help="Feed each stock's holding and spread to the model (--costs).")
    ] = False,
    stop: Annotated[
        bool, typer.Option(help="Early stopping each month on a held-out block (walk docs).")
    ] = False,
    patience: Annotated[int, typer.Option(help="Passes without improvement (--stop).")] = 5,
    holdout: Annotated[int, typer.Option(help="Months held out each month (--stop).")] = 12,
    smooth: Annotated[int, typer.Option(help="Passes the held-out loss is averaged over.")] = 3,
    inputs: Annotated[str, typer.Option(help="Characteristics: all, ours, jkp or paper.")] = "all",
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
    last = date.fromisoformat(end) if end else None
    _months(table, inputs)  # build the cache once first
    runs = [out / f"seed-{s}" for s in picked]
    threads = max(1, (os.cpu_count() or 1) // min(jobs, len(picked)))
    with ProcessPoolExecutor(min(jobs, len(picked)), initializer=_die_with_parent,
                             mp_context=multiprocessing.get_context("spawn")) as pool_:  # fmt: skip
        futures = []
        for s, run in zip(picked, runs, strict=True):
            settings = walk.Settings(
                model, blocks, hidden, window or None, first_epochs, warm_epochs, lr, s,
                date.fromisoformat(start), fast_attention=fast_attention, clip=_clip(clip),
                costs=costs or None, short_hurdle=short_hurdle, schedule=schedule,
                lr_min=lr_min, warm_lr=warm_lr or None, compile=compile_model,
                max_return=max_return or None, holdings_input=holdings_input, stop=stop,
                patience=patience, holdout=holdout, smooth=smooth,
            )  # fmt: skip
            futures.append(pool_.submit(_walk_one, table, run, settings, device, threads,
                                        precision, inputs, last))  # fmt: skip
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
    """Average the seeds' portfolios and report Sharpe; save the average next to the seeds.

    Each seed's portfolio is first scaled to $1 of gross exposure (longs plus shorts) each
    month, so a seed whose weights grew huge cannot swamp the others. A seed's month is a dud,
    and left out of that month's average, when its training loss or return is not finite or it
    holds nothing. The averaged portfolio (date, symbol, weight; $1 gross per seed averaged) is
    written to ``weights.parquet`` in the seeds' parent folder; its monthly return is the
    average of the good seeds' scaled returns.
    """
    good = (pl.col("loss").is_finite() & pl.col("ret").is_finite()
            & ((pl.col("long") - pl.col("short")) > 0))  # fmt: skip
    logs = [pl.read_parquet(r / "log.parquet")
            .select("date", pl.when(good).then(pl.col("ret") / (pl.col("long") - pl.col("short")))
                    .alias(r.name))
            for r in runs]  # fmt: skip
    joined = logs[0]
    for log in logs[1:]:
        joined = joined.join(log, on="date", how="full", coalesce=True)
    names = [r.name for r in runs]
    duds = int(joined.select(pl.sum_horizontal(pl.col(n).is_null() for n in names)).sum().item())
    joined = joined.filter(pl.any_horizontal(pl.col(n).is_not_null() for n in names)).sort("date")
    each = [_sharpe(joined[n].drop_nulls().to_numpy()) for n in names]
    mean = joined.select(pl.mean_horizontal(names)).to_series().to_numpy()
    frames = []
    for r, log in zip(runs, logs, strict=True):
        keep = log.filter(pl.col(r.name).is_not_null()).select("date")
        w = pl.read_parquet(r / "weights.parquet").select("date", "symbol", "weight")
        w = w.join(keep, on="date").with_columns(
            pl.col("weight") / pl.col("weight").abs().sum().over("date")
        )
        frames.append(w.with_columns(pl.lit(r.name).alias("seed")))
    stacked = pl.concat(frames)
    count = stacked.group_by("date").agg(pl.col("seed").n_unique().alias("n"))
    average = (stacked.group_by("date", "symbol").agg(pl.col("weight").sum())
               .join(count, on="date").select("date", "symbol", pl.col("weight") / pl.col("n"))
               .sort("date", "symbol"))  # fmt: skip
    average.write_parquet(runs[0].parent / "weights.parquet")
    return (f"{len(runs)} seeds, each: min {min(each):.2f} median {np.median(each):.2f} "
            f"max {max(each):.2f}; {duds} dud seed-months left out; "
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


@optimizer_app.command("grid")
def grid_cmd(  # noqa: PLR0913, PLR0917 - one option per setting
    table: Annotated[Path, typer.Option(help="The table from `plab optimizer data`.")],
    configs: Annotated[Path, typer.Option(help="JSON list of setups (research.optimizer.grid).")],
    out: Annotated[Path, typer.Option(help="Parquet file for the per-epoch curves.")],
    ends: Annotated[str, typer.Option(help="Window ends, comma-separated month-end dates.")],
    inputs: Annotated[
        str, typer.Option(help="Characteristics: all, ours, jkp or paper.")
    ] = "paper",
    window: Annotated[int, typer.Option(help="Months before each end.")] = 60,
    validation: Annotated[int, typer.Option(help="Last months of each window held out.")] = 12,
    device: Annotated[str, typer.Option()] = "cuda",
    costs: Annotated[
        str, typer.Option(help="Per-stock cost table for setups with costs (costs.align).")
    ] = "",
) -> None:
    """Train each setup fresh on a few windows and record every epoch (research.optimizer.grid)."""
    from portfolio_lab.research.optimizer import grid  # noqa: PLC0415

    months = _months(table, inputs)
    index = {m.date: i for i, m in enumerate(months)}
    picked = [index[date.fromisoformat(d.strip())] for d in ends.split(",")]
    setups = grid.configs_from(json.loads(configs.read_text()))
    stock = None
    if costs:
        from portfolio_lab.research.optimizer import costs as cost_model  # noqa: PLC0415

        stock = cost_model.align(months, pl.read_parquet(costs))
    curves = grid.grid(months, picked, setups, window, validation, device, stock)
    curves.write_parquet(out)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        typer.echo(grid.summary(curves))
