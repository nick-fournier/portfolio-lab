"""Landing page (strategy overview with a comparison chart) and the About page."""

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.backtest.results import list_runs, load_run
from portfolio_lab.web.charts import comparison_figure
from portfolio_lab.web.routes.status import data_freshness

router = APIRouter()

#: Strategies drawn when the overview opens (besides buy-and-hold); the rest start hidden
#: and can be shown from the legend, so the chart stays readable.
SHOWN = 4


def run_label(meta: dict) -> str:
    """Short label for a run: the engine's label, else the name plus all set parameters.

    Buy-and-hold is labeled with what it holds, e.g. ``buy_hold (SPY)``.
    """
    params = meta.get("params", {})
    if meta["strategy"] == "buy_hold":
        return f"buy_hold ({params.get('symbol') or 'SPY'})"
    if meta.get("label"):
        return meta["label"]
    params = [f"{k}={v}" for k, v in sorted(params.items()) if v is not None]
    return meta["strategy"] + (f" ({', '.join(params)})" if params else "")


@router.get("/", response_class=HTMLResponse)
def overview(request: Request) -> HTMLResponse:
    """Latest run of each strategy configuration, compared on one chart."""
    data_dir = request.app.state.data_dir
    latest = sorted(list_runs(data_dir, latest_only=True), key=lambda r: run_label(r["meta"]))
    benchmark = latest[0]["meta"].get("benchmark", "SPY") if latest else "SPY"
    holds_benchmark = {
        r["meta"]["run_id"]
        for r in latest
        if r["meta"]["strategy"] == "buy_hold"
        and r["meta"].get("params", {}).get("symbol", benchmark) == benchmark
    }
    others = [r for r in latest if r["meta"]["run_id"] not in holds_benchmark]
    best = sorted(others, key=lambda r: -(r["metrics"].get("sharpe") or float("-inf")))
    shown = holds_benchmark | {r["meta"]["run_id"] for r in best[:SHOWN]}
    series = [
        (
            run_label(r["meta"]),
            load_run(data_dir, r["meta"]["run_id"]).daily,
            r["meta"]["run_id"] in shown,
            r["meta"]["run_id"] in holds_benchmark,
        )
        for r in latest
    ]
    prices_path = data_dir / "_status" / "prices.json"
    prices = json.loads(prices_path.read_text()) if prices_path.exists() else None
    return request.app.state.templates.TemplateResponse(
        request,
        "overview.html",
        {
            "runs": latest,
            "labels": {r["meta"]["run_id"]: run_label(r["meta"]) for r in latest},
            "chart": comparison_figure(series, None if holds_benchmark else benchmark),
            "data_through": prices.get("max_date") if prices else None,
            "freshness": data_freshness(prices),
        },
    )


@router.get("/about", response_class=HTMLResponse)
def about(request: Request) -> HTMLResponse:
    """How the lab works: data, backtests, strategies, caveats and roadmap."""
    return request.app.state.templates.TemplateResponse(request, "about.html", {})
