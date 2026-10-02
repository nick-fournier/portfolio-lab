"""Forecasts page: how reliably the return-forecasting models called direction, out of sample.

Formerly "Models", renamed so it isn't confused with the strategy (``/models`` redirects).
"""

from pathlib import Path

import polars as pl
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.web.charts import calibration_figure, importance_figure

router = APIRouter()

HORIZONS = {21: "Next month"}  # production rebalances monthly
POOLS = {"all": "All eligible stocks", "top500": "500 most liquid stocks"}
#: Models shown in the calibration chart (raw and calibrated tree model, and the baseline).
CALIBRATION_MODELS = ("gbm", "gbm+cal", "cfo_to_assets")
CALIBRATED = pl.col("model").str.ends_with("+cal")


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def _calibrated_ece(summary: pl.DataFrame) -> dict[str, float]:
    """Calibration error after calibration, by base model name (all stocks)."""
    rows = summary.filter(CALIBRATED & (pl.col("pool") == "all"))
    return {m.removesuffix("+cal"): e for m, e in rows.select("model", "ece").iter_rows()}


def _pools(h: pl.DataFrame) -> list[dict]:
    """Per pool, the uncalibrated models' rows, best ranking accuracy first."""
    return [
        {
            "title": title,
            "rows": h.filter((pl.col("pool") == pool) & ~CALIBRATED)
            .sort("auc", descending=True)
            .to_dicts(),
        }
        for pool, title in POOLS.items()
    ]


def _section(
    horizon: int,
    title: str,
    summary: pl.DataFrame,
    calibration: pl.DataFrame,
    importance: pl.DataFrame,
) -> dict | None:
    """Tables and charts for one horizon (``None`` without results)."""
    h = summary.filter(pl.col("horizon") == horizon)
    if h.is_empty():
        return None
    bins = calibration.filter(
        (pl.col("horizon") == horizon) & pl.col("model").is_in(list(CALIBRATION_MODELS))
    )
    gbm = (
        importance.filter((pl.col("horizon") == horizon) & (pl.col("model") == "gbm"))
        if importance.height
        else importance
    )
    return {
        "key": horizon,
        "title": title,
        "pools": _pools(h),
        "ece_cal": _calibrated_ece(h),
        "calibration": calibration_figure(bins) if bins.height else None,
        "importance": importance_figure(gbm) if gbm.height else None,
    }


@router.get("/models", include_in_schema=False)
def models_redirect() -> RedirectResponse:
    """The page's old address."""
    return RedirectResponse("/forecasts", status_code=301)


@router.get("/forecasts", response_class=HTMLResponse)
def forecasts(request: Request) -> HTMLResponse:
    """Render reliability tables, calibration charts and input importance per horizon."""
    folder = DataPaths(request.app.state.data_dir).models
    summary = _read(folder / "summary.parquet")
    sections = []
    if summary.height:
        calibration = _read(folder / "calibration.parquet")
        importance = _read(folder / "importance.parquet")
        for horizon, title in HORIZONS.items():
            section = _section(horizon, title, summary, calibration, importance)
            if section:
                sections.append(section)
    return request.app.state.templates.TemplateResponse(
        request, "forecasts.html", {"sections": sections}
    )
