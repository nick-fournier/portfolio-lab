"""``plab optimizer ...``: build the optimizer's data, walk a model forward, score baselines.

Research commands (class 3, optimizer track). ``data`` reads the data hive; ``walk`` needs the
table ``data`` wrote and torch (``uv sync --extra nets``), and goes on the GPU host. ``ridge``
and ``signal`` score the paper's linear benchmark and an outside forecast on the same months a
walk trains on. With ``--no-blank-checks`` all three read the hive's security master on every
call (``_months``).
"""

import hashlib
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
from portfolio_lab.research.optimizer import seedavg, universe
from portfolio_lab.research.optimizer.universe import NO_FLOORS, Floors

optimizer_app = typer.Typer(help="Class 3 optimizer track (research).")


@optimizer_app.command("data")
def data_cmd(out: Annotated[Path, typer.Option(help="Parquet file to write.")]) -> None:
    """Write the stock-month table: characteristics, eligibility, liquidity, target."""
    from portfolio_lab.research.optimizer import data  # noqa: PLC0415

    table = data.build(get_settings().data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out)
    typer.echo(f"{table.height:,} rows, {table['date'].n_unique()} months, "
               f"{len(data.inputs(table))} characteristics -> {out}")  # fmt: skip


def months_cache(table: Path, inputs: str, floors: Floors, drop: frozenset[str]) -> Path:
    """Where :func:`_months` caches a table's months, named by everything that shapes them.

    The name shows the input set and the floors and ends in a short hash of those, the drop
    list (the table's blank checks, :func:`_shells`) and the source of the code that builds the
    months (``data`` and ``universe``): a change to any of them builds a fresh cache instead of
    reading a stale one.
    """
    from portfolio_lab.research.optimizer import data  # noqa: PLC0415

    floors = (float(floors[0]), float(floors[1]), bool(floors[2]))
    key = hashlib.sha256()
    for part in (Path(data.__file__).read_bytes(), Path(universe.__file__).read_bytes(),
                 repr((inputs, floors, sorted(drop))).encode()):  # fmt: skip
        key.update(part + b"\0")
    tag = "" if floors == NO_FLOORS else f"-p{floors[0]:g}-v{floors[1]:g}{'-nb' * floors[2]}"
    return table.with_name(f"{table.stem}.months-{inputs}{tag}-{key.hexdigest()[:12]}.pkl")


def _shells(table: Path) -> frozenset[str]:
    """The blank-check shells (``universe.blank_checks``) among the table's symbols.

    Only those can change its months, so a shell newly listed in the hive does not change the
    cache's key.
    """
    data_dir = get_settings().data_dir
    try:
        shells = universe.blank_checks(data_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"--no-blank-checks reads the security master under {data_dir}; "
                                f"set PORTFOLIO_DATA_DIR (e.g. hive-class3)") from e  # fmt: skip
    symbols = pl.scan_parquet(table).select(pl.col("symbol").unique()).collect()["symbol"]
    return shells & frozenset(symbols.to_list())


def _months(table: Path, inputs: str = "all", floors: Floors = NO_FLOORS) -> list:
    """``data.months`` of a table, cached next to it (ranking every month takes a while).

    With no blank checks (``floors[2]``) every call, a cache hit too, reads the hive's security
    master (the drop list is part of the cache's key), so ``PORTFOLIO_DATA_DIR`` must point at
    the hive (the class 3 runs: hive-class3).
    """
    from portfolio_lab.research.optimizer import data  # noqa: PLC0415

    drop = _shells(table) if floors[2] else frozenset()
    cache = months_cache(table, inputs, floors, drop)
    if cache.exists() and cache.stat().st_mtime > table.stat().st_mtime:
        return pickle.loads(cache.read_bytes())
    frame = pl.read_parquet(table)
    months = data.months(frame, data.inputs(frame, inputs), min_price=floors[0],
                         min_adv=floors[1], drop=drop)  # fmt: skip
    cache.write_bytes(pickle.dumps(months))
    return months


def _walk_one(  # noqa: PLR0913, PLR0917 - one walk's pieces
    table: Path, out: Path, settings, device: str, threads: int, precision: str,
    inputs: str, end: date | None = None, floors: Floors = NO_FLOORS,
) -> str:  # fmt: skip
    """One walk (test months up to ``end``, if any): save weights, log, model and settings."""
    import torch  # noqa: PLC0415

    from portfolio_lab.research.optimizer import walk  # noqa: PLC0415

    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision(precision)
    months = [m for m in _months(table, inputs, floors) if end is None or m.date <= end]
    stock, market = None, None
    if settings.costs:
        from portfolio_lab.research.optimizer import costs, realism  # noqa: PLC0415

        table = pl.read_parquet(settings.costs)
        stock = costs.align(months, table)
        need = realism.on(settings) or settings.trade_gross  # trade_gross: prices for collateral
        market = realism.align(months, table) if need else None
    weights, log, fitted = walk.walk(months, settings, device, stock, market)
    out.mkdir(parents=True, exist_ok=True)
    weights.write_parquet(out / "weights.parquet")
    log.write_parquet(out / "log.parquet")
    torch.save(fitted.state_dict(), out / "model.pt")
    only = settings.only and tuple(d.isoformat() for d in settings.only)
    meta = asdict(replace(settings, start=settings.start.isoformat(), only=only)) | {
        "inputs": inputs,
        "end": end.isoformat() if end else None,
        "floors": list(floors),
    }
    (out / "settings.json").write_text(json.dumps(meta, indent=2))
    graded = log.filter(pl.col("ret").is_not_null())
    return (f"{settings.model} seed {settings.seed}: {graded.height} months, Sharpe "
            f"{seedavg.halves(graded['date'].to_list(), graded['ret'].to_numpy())}; "
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


def _need_costs(costs: str, asked: dict[str, bool]) -> None:
    """Refuse the loss options that work only with the per-stock cost table when it is missing."""
    on = [flag for flag, value in asked.items() if value]
    if on and not costs:
        raise typer.BadParameter(f"--costs (the per-stock cost table) is needed for "
                                 f"{', '.join(on)}")  # fmt: skip


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
    first_epochs: Annotated[int, typer.Option(help="Passes for a fresh model.")] = 100,
    warm_epochs: Annotated[
        int, typer.Option(help="Passes continuing last month's model (without --cold).")
    ] = 5,
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
    winsor: Annotated[
        float, typer.Option(help="Training returns pulled in to each month's q, 1-q quantiles.")
    ] = 0.0,
    stock_drop: Annotated[
        float, typer.Option(help="Share of each month's stocks hidden from each step's loss.")
    ] = 0.0,
    holdings_input: Annotated[
        bool, typer.Option(help="Feed each stock's holding and spread to the model (--costs).")
    ] = False,
    borrow_input: Annotated[
        bool, typer.Option(help="Also feed each stock's borrow fee (--holdings-input).")
    ] = False,
    holding_column: Annotated[
        bool,
        typer.Option(help="--holdings-input: include the holding column (no: spread/borrow only)."),
    ] = True,
    borrow_flag: Annotated[
        bool,
        typer.Option(help="--borrow-input as a centred hard-to-borrow flag, not the log fee."),
    ] = False,
    spread_rank: Annotated[
        bool, typer.Option(help="--holdings-input: the spread as its rank in [-0.5, 0.5].")
    ] = False,
    holding_rank: Annotated[
        bool, typer.Option(help="--holdings-input: the holding as its rank in [-0.5, 0.5].")
    ] = False,
    holdout: Annotated[int, typer.Option(help="Months held out each month (--watch).")] = 12,
    average: Annotated[
        float, typer.Option(help="Hold a moving average of the model (decay per step; 0: off).")
    ] = 0.0,
    cold: Annotated[bool, typer.Option(help="Fresh model every month (walk docs).")] = False,
    watch: Annotated[
        bool, typer.Option(help="Score held-out months every pass; all passes run (walk docs).")
    ] = False,
    scatter: Annotated[
        bool, typer.Option(help="Hold out single months anywhere, not a block (walk docs).")
    ] = False,
    holdout_seed: Annotated[
        int, typer.Option(help="Seed of the held-out draw (-1: the run's seed).")
    ] = -1,
    only: Annotated[
        str, typer.Option(help="Train and trade only these month ends: 2020-03-31,2021-06-30.")
    ] = "",
    inputs: Annotated[str, typer.Option(help="Characteristics: all, ours, jkp or paper.")] = "all",
    precision: Annotated[
        str, typer.Option(help="float32 matrix products: highest, or high (TF32, faster).")
    ] = "highest",
    min_price: Annotated[float, typer.Option(help="Universe floor: month-end price.")] = 0.0,
    min_adv: Annotated[
        float, typer.Option(help="Universe floor: trailing median daily dollar volume.")
    ] = 0.0,
    no_blank_checks: Annotated[
        bool, typer.Option(help="Leave blank-check shells (SIC 6770) out of the universe.")
    ] = False,
    rebate: Annotated[
        bool | None, typer.Option(help="Loss: short dollars forgo the T-bill (retail; --costs).")
    ] = None,
    margin_cushion: Annotated[
        float | None, typer.Option(help="Loss: maintenance requirement allowed, share of equity.")
    ] = None,
    cushion_rate: Annotated[
        float | None,
        typer.Option(help="Loss: charge a month per $ above the cushion, limit or floor (0.10)."),
    ] = None,
    gross_limit: Annotated[
        float | None,
        typer.Option(help="Loss: gross allowed before it is charged like the cushion."),
    ] = None,
    start_book: Annotated[
        str, typer.Option(help="First traded month starts from: guess, cash or market.")
    ] = "guess",
    invest_floor: Annotated[
        float, typer.Option(help="Loss: longs below this share of equity charged (idle cash).")
    ] = 0.0,
    trade_margin: Annotated[
        bool, typer.Option(help="Loss: also charge the traded month's margin, gross and floor.")
    ] = False,
    impact: Annotated[
        float | None, typer.Option(help="Loss: square-root impact constant K.")
    ] = None,
    equity: Annotated[
        float | None, typer.Option(help="Account size in dollars (impact; 100,000).")
    ] = None,
    margin_rate: Annotated[
        float | None, typer.Option(help="Loss: the broker's margin rate a year (minus the T-bill).")
    ] = None,
    cash_rate: Annotated[
        float | None,
        typer.Option(help="Loss: idle cash's rate a year (-1: the T-bill, the default)."),
    ] = None,
    profile: Annotated[
        str,
        typer.Option(help="Account profile filling the realism flags not given: alpaca or none."),
    ] = "none",
    trade_gross: Annotated[
        float,
        typer.Option(
            help="Trade each seed's book scaled to this gross, sum |w| (0: the model's own"
            " size; the seed average is not rescaled)."
        ),
    ] = 0.0,
    trade_collateral: Annotated[
        float,
        typer.Option(
            help="With --trade-gross only: at most this collateral, share of equity (0: no cap)."
        ),
    ] = 0.5,
    loss_gross: Annotated[
        float,
        typer.Option(
            help="Loss on the book sized to this gross (= --trade-gross; 0: the model's own size)."
        ),
    ] = 0.0,
    risk_aversion: Annotated[
        float, typer.Option(help="With --loss-gross: expected return less this / 2 x its square.")
    ] = 1.0,
    size_anchor: Annotated[
        float,
        typer.Option(
            help="With --loss-gross: this x log(model's gross / loss gross)^2 in the loss."
        ),
    ] = 0.0,
    keep_best: Annotated[
        bool,
        typer.Option(
            help="With --watch: score the held-out months by ranking each pass, keep the best"
            " pass and stop after --keep-patience passes without a new best."
        ),
    ] = False,
    keep_patience: Annotated[
        int, typer.Option(help="With --keep-best: passes without a new best before stopping.")
    ] = 4,
    holdout_every: Annotated[
        int, typer.Option(help="With --watch: hold out every K-th month end for good (0: draw).")
    ] = 0,
    price_gross: Annotated[
        float,
        typer.Option(
            help="Loss: price margin, idle cash and impact on the book scaled to this gross,"
            " per dollar of its gross (0: at the book's own size; usually --trade-gross)."
        ),
    ] = 0.0,
) -> None:
    """Walk a model forward, one run per seed, and save each seed's monthly portfolios.

    Seeds run ``jobs`` at a time in separate processes: one walk takes small steps (one month
    each) and leaves the GPU mostly idle, so several share it. With more than one seed the
    seeds' portfolios are also averaged (the paper's 10-seed average). The loss options that
    need the per-stock cost table are refused without ``--costs``, and so is ``--trade-gross``
    (its collateral needs the table's prices). A realism flag given wins (0 too); one left out
    takes the profile's value (``--profile``), else the setting's default (``walk.Settings``:
    off, the T-bill for cash, a 0.10 charge, $100,000).
    """
    _need_costs(costs, {
        "--holdings-input": holdings_input, "--borrow-input": borrow_input,
        "--profile": profile not in ("", "none"), "--rebate": bool(rebate),
        "--margin-rate": bool(margin_rate), "--cash-rate": cash_rate is not None and cash_rate >= 0,
        "--margin-cushion": bool(margin_cushion), "--gross-limit": bool(gross_limit),
        "--invest-floor": bool(invest_floor), "--impact": bool(impact),
        "--trade-margin": trade_margin, "--start-book": start_book != "guess",
        "--trade-gross": bool(trade_gross), "--price-gross": bool(price_gross),
        "--loss-gross": bool(loss_gross),
    })  # fmt: skip
    import multiprocessing  # noqa: PLC0415
    import os  # noqa: PLC0415
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from portfolio_lab.research.optimizer import realism, walk  # noqa: PLC0415

    given = {k: v for k, v in {
        "rebate": rebate, "margin_rate": margin_rate, "margin_cushion": margin_cushion,
        "cushion_rate": cushion_rate, "gross_limit": gross_limit, "impact": impact,
        "equity": equity,
    }.items() if v is not None}  # fmt: skip
    if cash_rate is not None:
        given["cash_rate"] = None if cash_rate < 0 else cash_rate
    picked = _seeds(seeds)
    settings = walk.Settings(
        model, blocks, hidden, window or None, first_epochs, warm_epochs, lr, picked[0],
        date.fromisoformat(start), fast_attention=fast_attention, clip=_clip(clip),
        costs=costs or None, schedule=schedule, lr_min=lr_min,
        warm_lr=warm_lr or None, compile=compile_model, winsor=winsor,
        holdings_input=holdings_input, borrow_input=borrow_input, holdout=holdout,
        holding_column=holding_column, borrow_flag=borrow_flag, spread_rank=spread_rank,
        holding_rank=holding_rank,
        average=average, cold=cold, watch=watch, scatter=scatter,
        holdout_seed=None if holdout_seed < 0 else holdout_seed,
        only=tuple(date.fromisoformat(d) for d in only.split(",")) if only else None,
        stock_drop=stock_drop, invest_floor=invest_floor, trade_margin=trade_margin,
        start_book=start_book, trade_gross=trade_gross, trade_collateral=trade_collateral,
        price_gross=price_gross, loss_gross=loss_gross, risk_aversion=risk_aversion,
        size_anchor=size_anchor, keep_best=keep_best, keep_patience=keep_patience,
        holdout_every=holdout_every,
        **given,
    )  # fmt: skip
    try:
        settings = realism.with_profile(settings, profile, keep=given)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="'--profile'") from e
    last = date.fromisoformat(end) if end else None
    floors = (min_price, min_adv, no_blank_checks)
    _months(table, inputs, floors)  # build the cache once first
    runs = [out / f"seed-{s}" for s in picked]
    threads = max(1, (os.cpu_count() or 1) // min(jobs, len(picked)))
    with ProcessPoolExecutor(min(jobs, len(picked)), initializer=_die_with_parent,
                             mp_context=multiprocessing.get_context("spawn")) as pool_:  # fmt: skip
        futures = [pool_.submit(_walk_one, table, run, replace(settings, seed=s), device, threads,
                                precision, inputs, last, floors)
                   for s, run in zip(picked, runs, strict=True)]  # fmt: skip
        for f in futures:
            typer.echo(f.result())
    if len(runs) > 1:
        typer.echo(seedavg.average(runs))


@optimizer_app.command("average")
def average_cmd(runs: Annotated[list[Path], typer.Argument(help="Walk folders (one per seed).")]):
    """Average several seeds' portfolios (the paper's 10-seed average) and report Sharpe."""
    typer.echo(seedavg.average(runs))


@optimizer_app.command("ridge")
def ridge_cmd(
    table: Annotated[Path, typer.Option(help="The table from `plab optimizer data`.")],
    start: Annotated[str, typer.Option(help="First test month end.")] = "2003-12-31",
    inputs: Annotated[
        str, typer.Option(help="Characteristics: all, ours, jkp or paper.")
    ] = "paper",
    min_price: Annotated[float, typer.Option(help="Universe floor: month-end price.")] = 0.0,
    min_adv: Annotated[
        float, typer.Option(help="Universe floor: trailing median daily dollar volume.")
    ] = 0.0,
    no_blank_checks: Annotated[
        bool, typer.Option(help="Leave blank-check shells (SIC 6770) out of the universe.")
    ] = False,
) -> None:
    """The paper's linear benchmark: ridge in closed form, penalty picked two ways.

    On the months a walk with the same inputs and universe floors trains on (its cache).
    """
    from portfolio_lab.research.optimizer import ridge  # noqa: PLC0415

    months = _months(table, inputs, (min_price, min_adv, no_blank_checks))
    first = next(k for k, m in enumerate(months) if m.date >= date.fromisoformat(start))
    out = ridge.walk(months, start_index=first)
    best = ridge.GRID.index(out["best_z"])
    typer.echo("Sharpe by penalty: " + ", ".join(
        f"{z:.0e} {s:.2f}" for z, s in zip(ridge.GRID, out["sharpe"], strict=True)))  # fmt: skip
    typer.echo(f"best penalty {out['best_z']:.0e} (looks ahead, as the paper's BSV): "
               f"{seedavg.halves(out['dates'], out['returns'][:, best])}")  # fmt: skip
    loo = seedavg.halves(out["dates"], out["loo"])
    typer.echo(f"leave-one-out each month (no look-ahead): {loo}; "
               f"median penalty {np.median(out['loo_z']):.0e}")  # fmt: skip


@optimizer_app.command("signal")
def signal_cmd(
    table: Annotated[Path, typer.Option(help="The table from `plab optimizer data`.")],
    forecasts: Annotated[Path, typer.Option(help="Parquet with date, symbol, forecast.")],
    start: Annotated[str, typer.Option(help="First month end.")] = "2003-12-31",
    inputs: Annotated[
        str, typer.Option(help="Characteristics: all, ours, jkp or paper.")
    ] = "paper",
    min_price: Annotated[float, typer.Option(help="Universe floor: month-end price.")] = 0.0,
    min_adv: Annotated[
        float, typer.Option(help="Universe floor: trailing median daily dollar volume.")
    ] = 0.0,
    no_blank_checks: Annotated[
        bool, typer.Option(help="Leave blank-check shells (SIC 6770) out of the universe.")
    ] = False,
) -> None:
    """Sharpe of a forecast traded the same long-short way, on the same candidates.

    The candidates are a walk's with the same inputs and universe floors (its cache). Each
    month: the candidates that have a forecast, weights = the forecast's rank across them in
    [-0.5, 0.5] (a characteristic-managed portfolio, like each input's factor).
    """
    from portfolio_lab.research.optimizer import data  # noqa: PLC0415

    f = pl.read_parquet(forecasts).select("date", "symbol", "forecast")
    months = [m for m in _months(table, inputs, (min_price, min_adv, no_blank_checks))
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
               f"Sharpe {seedavg.halves(dates, np.array(rets))}")  # fmt: skip
