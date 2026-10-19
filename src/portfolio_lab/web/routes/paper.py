"""Paper trading page: the production strategy on Alpaca's paper account (dummy money).

The account is compared from its first rebalance with the model it holds now (each rebalance
saves its model, ``jobs.models.model_record``), against that model's backtest, so switching
production starts a fresh comparison instead of mixing two models.
"""

import json
from datetime import date
from pathlib import Path

import polars as pl
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.backtest.engine import label as strategy_label
from portfolio_lab.backtest.results import list_runs, load_run
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data import ids, reader
from portfolio_lab.jobs.models import CLASS_1, PAPER_STRATEGY, run_label
from portfolio_lab.strategies.base import create
from portfolio_lab.web.charts import growth_figure

router = APIRouter()

#: The model of rebalances saved before each one recorded its model: the account's first two
#: (2026-09-30 and 2026-10-01) were class 1's.
UNRECORDED = CLASS_1


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def _name(model: tuple[str, dict]) -> str:
    return strategy_label(create(model[0], **model[1]))


def _held(rebalances: pl.DataFrame) -> tuple[str | None, date | None]:
    """The model the account holds (its latest rebalance's, by its current name) and since when.

    "Since" is the first rebalance of the latest unbroken run of that model.
    """
    if not rebalances.height:
        return None, None
    rows = rebalances.sort("date", descending=True)
    records = rows["model"].to_list() if "model" in rows.columns else [None] * rows.height
    models = [run_label(json.loads(r)) if r else _name(UNRECORDED) for r in records]
    since = rows["date"][0]
    for day, model in zip(rows["date"], models, strict=True):
        if model != models[0]:
            break
        since = day
    return models[0], since


def _backtest(data_dir: Path, model: str) -> pl.DataFrame:
    """Daily returns of ``model``'s latest backtest (empty if none)."""
    for run in list_runs(data_dir, latest_only=True):  # newest first
        if run_label(run["meta"]) == model:
            return load_run(data_dir, run["meta"]["run_id"]).daily.select("date", "ret")
    return pl.DataFrame()


def _spy(paths: DataPaths, start: date) -> pl.DataFrame:
    """SPY's daily returns after ``start``."""
    if not paths.ids.exists():
        return pl.DataFrame()
    found = ids.lookup(ids.Ids.load(paths.ids), pl.DataFrame({"ticker": ["SPY"], "date": [start]}))
    if found["sid"][0] is None:
        return pl.DataFrame()
    rows = reader.read(paths.root, "prices", ["ret_cc"], start, sids=[found["sid"][0]])
    return rows.filter(pl.col("date") > start).select("date", pl.col("ret_cc").alias("ret"))


def _growth(snapshots: pl.DataFrame, others: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Growth of $1 since the first snapshot: paper equity, and each return series."""
    start = snapshots["date"].min()
    equity = pl.col("equity")
    frames = [snapshots.sort("date").select(
        "date", pl.lit("paper").alias("key"), (equity / equity.first()).alias("growth")
    )]  # fmt: skip
    for key, daily in others.items():
        if daily.height:
            rows = (
                daily.filter(pl.col("date") > start)
                .sort("date")
                .select(
                    "date", pl.lit(key).alias("key"), (1 + pl.col("ret")).cum_prod().alias("growth")
                )
            )
            first = pl.DataFrame({"date": [start], "key": [key], "growth": [1.0]})
            frames.append(pl.concat([first, rows]))
    return pl.concat(frames)


def _latest(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("date") == frame["date"].max()) if frame.height else frame


def _holdings(folder: Path, equity: float) -> list[dict]:
    """Latest positions' weights beside the latest target weights."""
    held, target = (
        _latest(_read(folder / "positions.parquet")),
        _latest(_read(folder / "targets.parquet")),
    )
    now = (
        held.select("symbol", (pl.col("market_value") / equity).alias("now"))
        if held.height
        else pl.DataFrame(schema={"symbol": pl.String, "now": pl.Float64})
    )
    goal = (
        target.select("symbol", pl.col("weight").alias("target"))
        if target.height
        else pl.DataFrame(schema={"symbol": pl.String, "target": pl.Float64})
    )
    table = now.join(goal, on="symbol", how="full", coalesce=True)
    return table.sort("target", descending=True, nulls_last=True).to_dicts()


def _last_orders(folder: Path) -> tuple[date | None, list[dict]]:
    """The most recent rebalance's date and its orders (status and fills)."""
    rebalances, orders = _read(folder / "rebalances.parquet"), _read(folder / "orders.parquet")
    if not (rebalances.height and orders.height):
        return None, []
    last = rebalances["date"].max()
    mine = orders.filter(pl.col("client_order_id").str.starts_with(f"pl-{last:%Y%m%d}-"))
    return last, mine.sort("side", "symbol").to_dicts()


def page_context(data_dir: Path, trading_dir: Path | None = None) -> dict:
    """The page's model, summary, chart, holdings and last rebalance (see module docs)."""
    paths = DataPaths(data_dir, trading_dir)
    snapshots = _read(paths.paper / "snapshots.parquet")
    snapshots = snapshots.sort("date") if snapshots.height else snapshots
    production = _name(PAPER_STRATEGY)
    held, since = _held(_read(paths.paper / "rebalances.parquet"))
    held = held or production
    context: dict = {"held": held, "production": production}
    if snapshots.height:
        own = snapshots.filter(pl.col("date") >= since) if since else snapshots
        own = own if own.height else snapshots
        first, latest = own.row(0, named=True), own.row(-1, named=True)
        opened = snapshots.row(0, named=True)
        context["summary"] = {
            "start": first["date"], "date": latest["date"], "equity": latest["equity"],
            "cash": latest["cash"], "ret": latest["equity"] / first["equity"] - 1,
            "opened": opened["date"], "total": latest["equity"] / opened["equity"] - 1,
        }  # fmt: skip
        backtest = _backtest(data_dir, held)
        growth = _growth(own, {"SPY": _spy(paths, first["date"]), "backtest": backtest})
        through = f" (through {backtest['date'].max()})" if backtest.height else ""
        names = {"paper": "Paper account", "SPY": "SPY",
                 "backtest": f"Backtest, {held}{through}"}  # fmt: skip
        if growth["date"].n_unique() > 1:
            context["chart"] = growth_figure(growth, names, set(names), skip_weekends=True)
        context["holdings"] = _holdings(paths.paper, latest["equity"])
        context["last_rebalance"], context["orders"] = _last_orders(paths.paper)
    return context


@router.get("/paper", response_class=HTMLResponse)
def paper_page(request: Request) -> HTMLResponse:
    """Render paper equity against SPY and the backtest, holdings and the last orders."""
    context = page_context(request.app.state.data_dir, request.app.state.trading_dir)
    return request.app.state.templates.TemplateResponse(request, "paper.html", context)
