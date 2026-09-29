"""Backtest run pages: the run list (with an HTMX strategy filter) and run detail."""

import re

import polars as pl
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.backtest.results import list_runs, load_run
from portfolio_lab.web.charts import drawdown_figure, equity_figure, weights_figure

router = APIRouter()

#: Run ids are ``<timestamp>-<strategy>-<hash>``; anything else is rejected before any I/O.
_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
HOLDINGS_SHOWN = 25


@router.get("/runs", response_class=HTMLResponse)
def runs_list(request: Request, strategy: str | None = None, history: bool = False) -> HTMLResponse:
    """List runs, newest first: the latest per configuration, or every run with ``history``.

    HTMX requests (the filter controls) get only the table.
    """
    runs = list_runs(request.app.state.data_dir, latest_only=not history)
    strategies = sorted({r["meta"]["strategy"] for r in runs})
    shown = [r for r in runs if not strategy or r["meta"]["strategy"] == strategy]
    template = "_runs_table.html" if request.headers.get("HX-Request") else "runs.html"
    return request.app.state.templates.TemplateResponse(
        request,
        template,
        {"runs": shown, "strategies": strategies, "selected": strategy, "history": history},
    )


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str) -> HTMLResponse:
    """Charts, metrics, caveats and holdings for one run."""
    if not _RUN_ID.match(run_id) or run_id in {".", ".."}:
        raise HTTPException(status_code=404)
    try:
        result = load_run(request.app.state.data_dir, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc

    holdings = []
    if not result.weights.is_empty():
        latest = result.weights["date"].max()
        holdings = (
            result.weights.filter(pl.col("date") == latest)
            .sort("weight", descending=True)
            .head(HOLDINGS_SHOWN)
            .rows(named=True)
        )
    meta = result.meta
    return request.app.state.templates.TemplateResponse(
        request,
        "run.html",
        {
            "meta": meta,
            "metrics": result.metrics,
            "holdings": holdings,
            "n_positions": result.weights.filter(
                pl.col("date") == result.weights["date"].max()
            ).height
            if not result.weights.is_empty()
            else 0,
            "equity": equity_figure(result.daily, meta["strategy"], meta.get("benchmark", "")),
            "drawdown": drawdown_figure(result.daily),
            "weights_chart": weights_figure(result.weights),
        },
    )
