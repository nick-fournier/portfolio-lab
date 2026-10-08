"""Data and job status page, plus a health check for the container."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from portfolio_lab.core.calendar import NEW_YORK, PRICE_UPDATE_DELAY, next_session, session_close

router = APIRouter()

#: Status files shown as data jobs, in display order, with the fields worth showing.
DATA_JOBS = {
    "universe": ("asof", "listed", "included"),
    "prices": ("max_date", "rows_written", "symbols_with_data"),
    "benchmarks": ("max_date", "rows_written"),
    "rates": ("max_date", "latest_rate"),
    "verify_prices": ("checked", "repaired"),
    "fundamentals": ("companies_with_facts", "facts", "latest_filing"),
    "conform": ("alpaca", "nasdaq", "edgar"),
    "quality": ("stage", "hard", "flags", "metrics"),
}


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


#: Prices count as late only once the next scheduled update is this overdue.
GRACE = timedelta(hours=2)


def next_update(through: date) -> datetime:
    """When prices through ``through`` next update: the next close plus the late-print wait."""
    return session_close(next_session(through)) + PRICE_UPDATE_DELAY


def data_freshness(prices: dict[str, Any] | None, now: datetime | None = None) -> dict:
    """Prices' state: ``current`` until the next scheduled update is overdue, then ``late``.

    Returns:
        state (``current``, ``late`` or ``missing``), through (last date with prices) and
        next (when the next update is due, New York time).
    """
    if not prices or not prices.get("max_date"):
        return {"state": "missing"}
    through = date.fromisoformat(prices["max_date"])
    due = next_update(through)
    late = (now or datetime.now(UTC)) > due + GRACE
    return {"state": "late" if late else "current", "through": through,
            "next": due.astimezone(NEW_YORK)}  # fmt: skip


def read_job(data_dir: Path, name: str) -> dict[str, Any] | None:
    """A job's last status: the hive's own jobs, else the fetchers' (in the ingest store)."""
    return _read(data_dir / "_status" / f"{name}.json") or _read(
        data_dir / "ingest" / "_status" / f"{name}.json"
    )


@router.get("/status", response_class=HTMLResponse)
def status_page(request: Request) -> HTMLResponse:
    """Show each data job's latest summary and the scheduler's state."""
    status_dir = request.app.state.data_dir / "_status"
    jobs = []
    for name, fields in DATA_JOBS.items():
        info = read_job(request.app.state.data_dir, name)
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
            "freshness": data_freshness(read_job(request.app.state.data_dir, "prices")),
        },
    )


@router.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    """Liveness check for the container."""
    return "ok"
