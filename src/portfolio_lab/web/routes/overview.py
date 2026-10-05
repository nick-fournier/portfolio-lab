"""Landing page (strategy overview with a comparison chart) and the About page."""

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.web import series
from portfolio_lab.web.charts import series_figure
from portfolio_lab.web.routes.status import data_freshness

router = APIRouter()

#: Market references on the overview, drawn with the production models; every other
#: strategy is listed but starts hidden on the chart (shown from the legend).
MARKET = ("SPY", "QQQ")


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


def page_context(data_dir: Path) -> dict:
    """Chart and table for the overview, rebuilt when the data changes."""

    def build() -> dict:
        everything = series.load(data_dir)
        shown = [s for s in everything.values() if s.category == "ours"]
        shown += [everything[k] for k in MARKET if k in everything]
        rows = series.table(shown)
        visible = {*MARKET, *series.production_keys()}
        return {"rows": rows, "chart": series_figure(shown, visible),
                "since": series.first_date(shown) if shown else None}  # fmt: skip

    return series.cached(f"overview:{data_dir}", series.sources(data_dir), build)


@router.get("/", response_class=HTMLResponse)
def overview(request: Request) -> HTMLResponse:
    """Our strategies and the market on one chart and one table, longest history first."""
    data_dir = request.app.state.data_dir
    prices_path = data_dir / "_status" / "prices.json"
    prices = json.loads(prices_path.read_text()) if prices_path.exists() else None
    return request.app.state.templates.TemplateResponse(
        request,
        "overview.html",
        {
            **page_context(data_dir),
            "data_through": prices.get("max_date") if prices else None,
            "freshness": data_freshness(prices),
        },
    )


@router.get("/about", response_class=HTMLResponse)
def about(request: Request) -> HTMLResponse:
    """How the lab works: data, backtests, strategies, caveats and roadmap."""
    return request.app.state.templates.TemplateResponse(request, "about.html", {})
