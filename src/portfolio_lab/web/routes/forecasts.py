"""Forecasts page: next month's return for every stock (``research.forecaster``).

Reads the summary the nightly derive step writes (``research.forecaster.report``).
"""

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.web import series
from portfolio_lab.web.charts import (
    forecast_fit_figure,
    forecast_order_figure,
    forecast_tenths_figure,
    forecast_trailing_figure,
    forecast_years_figure,
)

router = APIRouter()
#: Written by ``research.forecaster.report`` (not imported: the site loads no model code).
SUMMARY = "summary.json"
#: Keys the page needs; an older summary lacks some until the nightly rebuild.
REQUIRED = {"pieces", "yearly", "bins", "grinold_bins", "trailing", "tenths"}


def page_context(data_dir: Path) -> dict:
    """The summary and its charts (rebuilt when the summary changes)."""
    path = DataPaths(data_dir).forecaster / SUMMARY

    def build() -> dict:
        if not path.exists():
            return {}
        s = json.loads(path.read_text())
        if not s.keys() >= REQUIRED:  # written by older code: the nightly rebuild replaces it
            return {"outdated": True}
        return {"s": s, "fit": forecast_fit_figure(s["bins"]),
                "grinold": forecast_order_figure(s["grinold_bins"]),
                "years": forecast_years_figure(s["yearly"]),
                "trailing": forecast_trailing_figure(s["trailing"]),
                "tenths": forecast_tenths_figure(s["tenths"])}  # fmt: skip

    return series.cached(f"forecasts:{data_dir}", [path], build)


@router.get("/forecasts", response_class=HTMLResponse)
def forecasts(request: Request) -> HTMLResponse:
    """How the forecaster works and how it has done."""
    # a copy: rendering adds the request to the context, which must not reach the cache
    context = {**page_context(request.app.state.data_dir)}
    return request.app.state.templates.TemplateResponse(request, "forecasts.html", context)
