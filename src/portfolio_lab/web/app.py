"""FastAPI application factory for the read-only dashboard.

The dashboard only reads the data directory: backtest runs (via ``backtest.results``) and
job status files. It never imports strategy code, so any strategy's runs render the same
way. plotly.js is served from the installed ``plotly`` package; no chart CDN is needed.
"""

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import plotly
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from portfolio_lab.web.routes import (
    compare,
    context,
    forecasts,
    overview,
    paper,
    runs,
    signals,
    status,
    taxes,
)

HERE = Path(__file__).parent
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
#: Shown in place of a missing metric.
MISSING = "\N{EN DASH}"


def _pct(value: float | None, digits: int = 1) -> str:
    """Format a fraction as a percentage, e.g. 0.1234 -> '12.3%'."""
    return MISSING if value is None else f"{value * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 2) -> str:
    """Format a number with thousands separators."""
    return MISSING if value is None else f"{value:,.{digits}f}"


def create_app(data_dir: Path, trading_dir: Path | None = None) -> FastAPI:
    """Build the dashboard app reading from ``data_dir``.

    Args:
        data_dir: The data directory (mounted read-only in production).
        trading_dir: Where trading records live (default ``data_dir/trading``).
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Build the heavy pages in the background, so the first visit is fast.
        def prewarm() -> None:
            overview.page_context(Path(data_dir))
            compare.page_context(Path(data_dir))
            signals.page_context(Path(data_dir))
            taxes.page_data(Path(data_dir))

        threading.Thread(target=prewarm, daemon=True).start()
        yield

    app = FastAPI(
        title="Portfolio lab", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["pct"] = _pct
    templates.env.filters["num"] = _num
    templates.env.globals["plotly_version"] = plotly.__version__
    app.state.data_dir = Path(data_dir)
    app.state.trading_dir = Path(trading_dir) if trading_dir else None
    app.state.templates = templates

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.get("/vendor/plotly.min.js", include_in_schema=False)
    def plotly_js() -> FileResponse:
        """Serve the plotly.js bundle that ships with the Python package."""
        return FileResponse(
            PLOTLY_JS,
            media_type="text/javascript",
            # Pages request it as ?v=<version>, so it can be cached for good.
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    app.include_router(overview.router)
    app.include_router(runs.router)
    app.include_router(signals.router)
    app.include_router(context.router)
    app.include_router(forecasts.router)
    app.include_router(compare.router)
    app.include_router(taxes.router)
    app.include_router(paper.router)
    app.include_router(status.router)
    return app
