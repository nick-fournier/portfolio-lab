"""``plab forecast ...``: the next-month forecasting study (``research.forecasting``).

``study`` runs walk-forward models, ``review`` checks calibration and stock size, ``blend``
compares two-model blends, and ``publish`` writes the Forecasts page's summary.
"""

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from portfolio_lab.core.config import get_settings
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.research import forecast_report, forecasting
from portfolio_lab.research.panel import Panel

forecast_app = typer.Typer(help="Can next month's stock returns be forecast? (research study)")


@forecast_app.command("study")
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
        write_parquet_atomic(forecasts, out / f"{name}{tag}.forecasts.parquet")
        variants = [(name + tag, months)]
        if "correction" in forecasts.columns:  # stacked: also grade the half-strength version
            half = forecasts.with_columns(pl.col("forecast") - 0.5 * pl.col("correction"))
            write_parquet_atomic(half, out / f"{name}_half{tag}.forecasts.parquet")
            variants.append((f"{name}_half{tag}", forecasting.grade_months(half)))
        if scores:
            table = pl.DataFrame(scores)
            write_parquet_atomic(table, out / f"{name}{tag}.importance.parquet")
            with pl.Config(tbl_rows=20, float_precision=4):
                typer.echo(f"{name}: IC drop when each group is shuffled (mean over years)")
                typer.echo(table.group_by("group").agg(pl.col("drop").mean(),
                           (pl.col("drop") > 0).mean().alias("years_helped"))
                           .sort("drop", descending=True))  # fmt: skip
        for label, graded in variants:
            summary = forecasting.summarize(graded)
            for part in ("all", "first_half", "second_half"):
                rows.append({"model": label, "period": part, **summary[part]})
            rows.append({"model": label, "period": "years IC > 0",
                         "ic": summary["years_ic_positive"]})  # fmt: skip
    with pl.Config(tbl_rows=60, tbl_cols=20, float_precision=3, tbl_width_chars=200):
        typer.echo(pl.DataFrame(rows))


@forecast_app.command("review")
def forecast_review_cmd(
    model: Annotated[str, typer.Argument(help="Saved study run, e.g. linear_regime-no_size.")],
) -> None:
    """Calibration and size breakdown of a saved forecast study run's forecasts."""
    folder = get_settings().data_dir / "results" / "forecast_study"
    forecasts = pl.read_parquet(folder / f"{model}.forecasts.parquet")
    calibrated = forecasting.calibrate(forecasts)
    raw = forecasting.summarize(forecasting.grade_months(
        forecasts.filter(pl.col("date") >= calibrated["date"].min())))  # fmt: skip
    cal = forecasting.summarize(forecasting.grade_months(calibrated))
    factors = calibrated.group_by(pl.col("date").dt.year().alias("year")).agg(
        pl.col("factor").first()).sort("year")  # fmt: skip
    first, second = cal["first_half"]["slope"], cal["second_half"]["slope"]
    lo, hi = factors["factor"].min(), factors["factor"].max()
    typer.echo(f"Calibration ({calibrated['date'].min()}+): slope raw {raw['all']['slope']:.2f}"
               f" -> calibrated {cal['all']['slope']:.2f} (halves {first:.2f}, {second:.2f});"
               f" factor {lo:.2f} to {hi:.2f}")  # fmt: skip
    rows = [{"stocks": name, **s["all"], "years_right": s["years_ic_positive"]}
            for name, s in forecasting.by_size(forecasts).items()]  # fmt: skip
    with pl.Config(tbl_rows=10, tbl_cols=12, float_precision=3, tbl_width_chars=200):
        typer.echo(pl.DataFrame(rows))


@forecast_app.command("blend")
def forecast_blend_cmd(
    first: Annotated[str, typer.Argument(help="Saved study run (weight goes on this one).")],
    second: Annotated[str, typer.Argument(help="Another saved study run.")],
) -> None:
    """Blend two saved forecast study runs: 50/50 ranks vs a walk-forward-learned weight."""
    folder = get_settings().data_dir / "results" / "forecast_study"
    a, b = (pl.read_parquet(folder / f"{n}.forecasts.parquet") for n in (first, second))
    rows = []
    for label, frame in (
        (first, forecasting.blend(a, b, 1.0)),
        (second, forecasting.blend(a, b, 0.0)),
        ("50/50", forecasting.blend(a, b, 0.5)),
        ("learned weight", forecasting.blend(a, b, None)),
    ):
        months = forecasting.grade_months(frame)
        s = forecasting.summarize(months)
        years = months.group_by(pl.col("date").dt.year().alias("y")).agg(pl.col("ic").mean())
        crisis = {f"ic_{y}": years.filter(pl.col("y") == y)["ic"].item() for y in (2009, 2020)}
        rows.append({"forecast": label, "ic": s["all"]["ic"], "t": s["all"]["ic_t"],
                     "first_half": s["first_half"]["ic"], "second_half": s["second_half"]["ic"],
                     "years_right": s["years_ic_positive"], "spread_yr": s["all"]["spread_yr"],
                     **crisis})  # fmt: skip
        if label == "learned weight":
            w = frame.group_by(pl.col("date").dt.year().alias("y")).agg(pl.col("weight").first())
            typer.echo("learned weight on first, by year: "
                       + ", ".join(f"{y}: {v:.1f}" for y, v in w.sort("y").rows()))  # fmt: skip
    with pl.Config(tbl_rows=10, tbl_cols=12, float_precision=3, tbl_width_chars=200):
        typer.echo(pl.DataFrame(rows))


@forecast_app.command("publish")
def forecast_publish_cmd(
    dest: Annotated[Path, typer.Option(help="Main data directory to publish the summary to.")],
) -> None:
    """Summarize the forecasting study for the Forecasts page (research.forecast_report)."""
    settings = get_settings()
    paths = DataPaths(settings.data_dir)
    env = pl.read_parquet(paths.environment) if paths.environment.exists() else None
    source = settings.data_dir / "results" / "forecast_study"
    target = DataPaths(dest).root / "results" / "forecast_study"
    volatility = pl.read_parquet(paths.features, columns=["date", "symbol", "volatility"])
    typer.echo(forecast_report.publish(source, target, env, volatility))
