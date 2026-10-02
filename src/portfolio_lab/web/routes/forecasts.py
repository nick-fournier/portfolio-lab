"""Forecasts page: can next month's stock returns be predicted? (``research.forecast_report``).

Reads the study summary published by ``plab forecast publish``. Formerly "Models"; ``/models``
redirects here.
"""

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from portfolio_lab.research.forecast_report import SUMMARY
from portfolio_lab.web import series
from portfolio_lab.web.charts import forecast_fit_figure, forecast_years_figure

router = APIRouter()
ALPHA = "\N{GREEK SMALL LETTER ALPHA}"


def page_context(data_dir: Path) -> dict:
    """The summary and its chart (rebuilt when the summary changes)."""
    path = data_dir / "results" / "forecast_study" / SUMMARY

    def build() -> dict:
        if not path.exists():
            return {}
        summary = json.loads(path.read_text())
        top = max((g["drop"] for g in summary["groups"]), default=1.0) or 1.0
        for g in summary["groups"]:
            g["width"] = max(0.0, g["drop"]) / top * 100
        fit = forecast_fit_figure(summary["fit"]) if summary.get("fit") else None
        grinold = None
        if summary.get("grinold"):
            grinold = forecast_fit_figure(
                summary["grinold"], f"Grinold expected return {ALPHA}, % next month"
            )
        return {"s": summary, "chart": forecast_years_figure(summary["yearly"]), "fit": fit,
                "grinold": grinold}  # fmt: skip

    return series.cached(f"forecasts:{data_dir}", [path], build)


@router.get("/models", include_in_schema=False)
def models_redirect() -> RedirectResponse:
    """The page's old address."""
    return RedirectResponse("/forecasts", status_code=301)


@router.get("/forecasts", response_class=HTMLResponse)
def forecasts(request: Request) -> HTMLResponse:
    """Headline, ranking by year, what it uses, where it works, and the models tried."""
    context = page_context(request.app.state.data_dir)
    return request.app.state.templates.TemplateResponse(request, "forecasts.html", context)
