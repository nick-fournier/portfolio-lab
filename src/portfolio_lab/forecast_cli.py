"""``plab forecast ...``: the next-month forecaster (``research.forecaster``).

Run in order on a Sharadar data directory: ``inputs`` builds the extra stock inputs,
``dataset`` assembles the stock and market tables, ``run`` walks forward (resuming where
an earlier run stopped) and ``grade`` prints the grades.
"""

import json
from datetime import date
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from portfolio_lab.core.config import get_settings
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.research.characteristics import build as characteristics
from portfolio_lab.research.forecaster import dataset, grade, walk
from portfolio_lab.research.forecaster.conditioned import CondNetsPart, pick
from portfolio_lab.research.forecaster.nets import NetsPart
from portfolio_lab.research.forecaster.trees import TreesPart
from portfolio_lab.research.panel import Panel

forecast_app = typer.Typer(help="Next-month stock forecaster (research).")


@forecast_app.command("inputs")
def inputs_cmd(
    raw: Annotated[
        Path | None, typer.Option(help="Sharadar bulk zips (default: <data dir>/raw).")
    ] = None,
) -> None:
    """Build the extra stock inputs (research.characteristics) into the features folder."""
    data_dir = get_settings().data_dir
    frame = characteristics.build(data_dir, raw or data_dir / "raw")
    write_parquet_atomic(frame, DataPaths(data_dir).characteristics)
    typer.echo(f"{frame.height:,} stock-months, {len(characteristics.COLUMNS)} inputs")


@forecast_app.command("dataset")
def dataset_cmd() -> None:
    """Assemble the forecaster's stock and market tables (research.forecaster.dataset)."""
    data_dir = get_settings().data_dir
    paths = DataPaths(data_dir)
    panel = Panel.load(data_dir)
    features = pl.read_parquet(paths.features)
    env = pl.read_parquet(paths.environment)
    stocks = dataset.stocks(panel, features, pl.read_parquet(paths.characteristics))
    market = dataset.market(panel, env, features)
    paths.forecaster.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(stocks, paths.forecaster / "stocks.parquet")
    write_parquet_atomic(market, paths.forecaster / "market.parquet")
    typer.echo(f"{stocks.height:,} stock-months, {stocks['date'].n_unique()} months "
               f"({stocks['date'].min()} to {stocks['date'].max()})")  # fmt: skip


@forecast_app.command("run")
def run_cmd(  # noqa: PLR0913, PLR0917 - one option per CLI flag
    start: Annotated[str, typer.Option(help="First month to forecast (YYYY-MM).")] = (
        f"{walk.FIRST_FORECAST:%Y-%m}"
    ),
    device: Annotated[str, typer.Option(help="Where the trees are fitted: cpu or cuda.")] = "cpu",
    threads: Annotated[int, typer.Option(help="CPU threads for the trees.")] = 6,
    fresh: Annotated[bool, typer.Option(help="Discard earlier forecasts and start over.")] = False,
    components: Annotated[
        int | None,
        typer.Option(help="Fix part 1's component count (default: k-fold; 0 = trees alone)."),
    ] = None,
    sample: Annotated[float, typer.Option(help="Share of rows and columns each tree draws.")] = 1.0,
    seed: Annotated[int, typer.Option(help="Random seed for the trees.")] = 0,
    model: Annotated[str, typer.Option(help="Part 2: trees or nets.")] = "trees",
    yearly_trees: Annotated[bool, typer.Option(help="Trees: refit each January only.")] = False,
    arch: Annotated[str, typer.Option(help="Cond nets: stock, bilinear or film.")] = "bilinear",
    exposures: Annotated[int, typer.Option(help="Cond nets: exposures per stock.")] = 4,
    market_penalty: Annotated[float, typer.Option(help="Cond nets: market-side L2.")] = 1e-3,
    stock_only: Annotated[bool, typer.Option(help="Nets or trees: stock inputs only.")] = False,
    yearly_fresh: Annotated[bool, typer.Option(help="Nets: fresh weights once a year.")] = False,
    dispersion: Annotated[
        bool, typer.Option(help="Add last month's return dispersion to the trees' inputs.")
    ] = False,
    tag: Annotated[str, typer.Option(help="Variant name, saved as forecasts-<tag>.")] = "",
) -> None:
    """Walk forward: refit every month on all earlier months and forecast it."""
    folder = DataPaths(get_settings().data_dir).forecaster
    path = folder / _name("forecasts", tag, "parquet")
    done = pl.read_parquet(path) if path.exists() and not fresh else None
    year, month = (int(p) for p in start.split("-"))
    if model == "cond":
        part2 = CondNetsPart(device, arch=arch, exposures=exposures,
                             market_penalty=market_penalty, seed=seed)  # fmt: skip
    elif model == "nets":
        part2 = NetsPart(device, seed=seed, dispersion=dispersion, market=not stock_only,
                         yearly_fresh=yearly_fresh)  # fmt: skip
    else:
        part2 = TreesPart(device, threads, sample, seed, dispersion=dispersion,
                          yearly=yearly_trees, stock_only=stock_only)  # fmt: skip

    def save(frame: pl.DataFrame) -> None:
        nonlocal done
        done = frame if done is None else pl.concat([done, frame], how="vertical_relaxed")
        write_parquet_atomic(done, path)

    walk.run(
        pl.read_parquet(folder / "stocks.parquet"), pl.read_parquet(folder / "market.parquet"),
        start=date(year, month, 1), components=components, part2=part2,
        skip=set(done["date"].unique().to_list()) if done is not None else None, save=save,
    )  # fmt: skip
    typer.echo(f"forecasts: {path}")


@forecast_app.command("pick")
def pick_cmd(
    tags: Annotated[list[str], typer.Argument(help="Runs to choose between, month by month.")],
    out: Annotated[str, typer.Option(help="Tag for the picked forecasts.")] = "picked",
) -> None:
    """Choose each month's run by the nets' held-out error at that month's fit."""
    folder = DataPaths(get_settings().data_dir).forecaster
    runs = [pl.read_parquet(folder / _name("forecasts", t, "parquet")) for t in tags]
    picked = pick(runs)
    write_parquet_atomic(picked.drop("picked"), folder / _name("forecasts", out, "parquet"))
    share = picked.group_by("date").agg(pl.col("picked").first())["picked"].value_counts()
    typer.echo(f"months picked per run ({', '.join(tags)}): {share.sort('picked').rows()}")


@forecast_app.command("grade")
def grade_cmd(
    since: Annotated[int, typer.Option(help="First year graded.")] = walk.FIRST_GRADED,
    tag: Annotated[str, typer.Option(help="Variant to grade (see run --tag).")] = "",
) -> None:
    """Grade the forecasts on unseen months (research.forecaster.grade)."""
    folder = DataPaths(get_settings().data_dir).forecaster
    combined = walk.combine(pl.read_parquet(folder / _name("forecasts", tag, "parquet")))
    graded = combined.filter(pl.col("date").dt.year() >= since)
    out = grade.report(graded)
    out["strength_by_year"] = (
        graded.group_by(pl.col("date").dt.year().alias("year"))
        .agg(pl.col("strength").mean())
        .sort("year")
        .to_dicts()
    )
    (folder / _name("grades", tag, "json")).write_text(json.dumps(out, indent=1, default=str))
    sizes = " / ".join(f"{v:.3f}" for v in out["ic_by_size"].values())
    typer.echo(
        f"{out['first']} to {out['last']} ({out['months']} months)\n"
        f"IC {out['ic']:.3f} (t {out['ic_t']:.1f}), right in {out['years_right']} of "
        f"{out['years']} years\n"
        f"slope {out['slope']:.2f} (losers {out['slope_losers']:.2f}, winners "
        f"{out['slope_winners']:.2f}), R² {out['r2']:+.3%}\n"
        f"best-worst tenth {out['tenth_yr']:.1%}/yr, IC small/mid/large {sizes}"
    )


def _name(stem: str, tag: str, suffix: str) -> str:
    """``stem.suffix``, or ``stem-tag.suffix`` for a named variant."""
    return f"{stem}-{tag}.{suffix}" if tag else f"{stem}.{suffix}"
