"""Landing page (strategy overview with a comparison chart) and the About page."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from portfolio_lab.web import series
from portfolio_lab.web.charts import series_figure
from portfolio_lab.web.routes.status import data_freshness, read_job

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
    prices = read_job(data_dir, "prices")
    return request.app.state.templates.TemplateResponse(
        request,
        "overview.html",
        {
            **page_context(data_dir),
            "data_through": prices.get("max_date") if prices else None,
            "freshness": data_freshness(prices),
        },
    )


@router.get("/how-it-works", response_class=HTMLResponse)
def how_it_works(request: Request) -> HTMLResponse:
    """How the lab works: data, universe, backtests, the two strategies, caveats."""
    return request.app.state.templates.TemplateResponse(request, "how_it_works.html", {})


@router.get("/about")
def about() -> RedirectResponse:
    """The page's old address."""
    return RedirectResponse("/how-it-works", status_code=301)
