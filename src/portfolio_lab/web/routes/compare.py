"""Make versus buy: our strategies against funds anyone can buy."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.research.funds import CATEGORIES
from portfolio_lab.web import series
from portfolio_lab.web.charts import series_figure

router = APIRouter()

#: Funds always drawn on the chart (besides the production models and the best other fund).
ALWAYS_SHOWN = ("SPY", "QQQ")


def _visible(rows: list[dict]) -> set[str]:
    """SPY, QQQ, the production models and the best other fund by 10-year return."""
    funds = [r for r in rows if r["category"] != "ours" and r["key"] not in ALWAYS_SHOWN]
    best = sorted(funds, key=lambda r: -(r["10Y"] if r["10Y"] is not None else -9))[:1]
    return {*ALWAYS_SHOWN, *series.production_keys(), *(r["key"] for r in best)}


def page_context(data_dir: Path) -> dict:
    """Chart and table for the Compare page, rebuilt when the data changes."""

    def build() -> dict:
        if not any(p.exists() for p in series.sources(data_dir)[1:]):  # no fund comparison yet
            return {}
        everything = list(series.load(data_dir).values())
        rows = series.table(everything)
        order = {c: k for k, c in enumerate(CATEGORIES)}
        by_type = sorted(rows, key=lambda r: (order.get(r["category"], 99), r["name"]))
        groups = {r["key"]: CATEGORIES.get(r["category"], "Other") for r in by_type}
        return {"rows": rows, "since": series.first_date(everything),
                "chart": series_figure(everything, _visible(rows), groups)}  # fmt: skip

    return series.cached(f"compare:{data_dir}", series.sources(data_dir), build)


@router.get("/compare", response_class=HTMLResponse)
def compare(request: Request) -> HTMLResponse:
    """One chart and one table: our strategies and every fund, over 1, 5, 10, 20 years and max."""
    # a copy: rendering adds the request to the context, which must not reach the cache
    context = {**page_context(request.app.state.data_dir)}
    return request.app.state.templates.TemplateResponse(request, "compare.html", context)
