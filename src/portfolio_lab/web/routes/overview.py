"""Landing page (strategy overview with a comparison chart) and the About page."""

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.backtest.results import list_runs, load_run
from portfolio_lab.web.charts import comparison_figure
from portfolio_lab.web.routes.status import data_freshness

router = APIRouter()

#: Most strategies drawn on the overview chart, to keep it readable.
MAX_COMPARED = 8


def run_label(meta: dict) -> str:
    """Short label for a run: strategy name plus any non-default parameters."""
    params = [f"{k}={v}" for k, v in sorted(meta.get("params", {}).items()) if v is not None]
    return meta["strategy"] + (f" ({', '.join(params)})" if params else "")


@router.get("/", response_class=HTMLResponse)
def overview(request: Request) -> HTMLResponse:
    """Latest run of each strategy configuration, compared on one chart."""
    data_dir = request.app.state.data_dir
    latest = sorted(list_runs(data_dir, latest_only=True), key=lambda r: run_label(r["meta"]))
    series = [
        (run_label(r["meta"]), load_run(data_dir, r["meta"]["run_id"]).daily)
        for r in latest[:MAX_COMPARED]
    ]
    benchmark = latest[0]["meta"].get("benchmark", "SPY") if latest else "SPY"
    prices_path = data_dir / "_status" / "prices.json"
    prices = json.loads(prices_path.read_text()) if prices_path.exists() else None
    return request.app.state.templates.TemplateResponse(
        request,
        "overview.html",
        {
            "runs": latest,
            "labels": {r["meta"]["run_id"]: run_label(r["meta"]) for r in latest},
            "chart": comparison_figure(series, benchmark),
            "data_through": prices.get("max_date") if prices else None,
            "freshness": data_freshness(prices),
        },
    )


@router.get("/about", response_class=HTMLResponse)
def about(request: Request) -> HTMLResponse:
    """How the lab works: data, backtests, strategies, caveats and roadmap."""
    return request.app.state.templates.TemplateResponse(request, "about.html", {})
