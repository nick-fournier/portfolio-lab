"""Data and job status page, plus a health check for the container."""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from portfolio_lab.core.calendar import last_complete_session

router = APIRouter()

#: Status files shown as data jobs, in display order, with the fields worth showing.
DATA_JOBS = {
    "universe": ("asof", "listed", "included"),
    "prices": ("max_date", "rows_written", "symbols_with_data"),
    "benchmarks": ("max_date", "rows_written"),
    "rates": ("max_date", "latest_rate"),
    "verify_prices": ("checked", "repaired"),
}


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def data_freshness(prices: dict[str, Any] | None, now: datetime | None = None) -> str:
    """Return ``"fresh"`` if prices cover the last complete session, else ``"stale"``."""
    if not prices or not prices.get("max_date"):
        return "missing"
    expected = last_complete_session(now or datetime.now(UTC))
    return "fresh" if date.fromisoformat(prices["max_date"]) >= expected else "stale"


@router.get("/status", response_class=HTMLResponse)
def status_page(request: Request) -> HTMLResponse:
    """Show each data job's latest summary and the scheduler's state."""
    status_dir = request.app.state.data_dir / "_status"
    jobs = []
    for name, fields in DATA_JOBS.items():
        info = _read(status_dir / f"{name}.json")
        if info is not None:
            details = {f: info.get(f) for f in fields}
            jobs.append(
                {
                    "name": name,
                    "finished_at": info.get("finished_at"),
                    "details": details,
                    "issues": info.get("issues") or [],
                }
            )
    scheduler = _read(status_dir / "scheduler.json") or {}
    scheduler_jobs = {k: v for k, v in scheduler.items() if isinstance(v, dict)}
    return request.app.state.templates.TemplateResponse(
        request,
        "status.html",
        {
            "jobs": jobs,
            "scheduler": scheduler_jobs,
            "freshness": data_freshness(_read(status_dir / "prices.json")),
        },
    )


@router.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    """Liveness check for the container."""
    return "ok"
