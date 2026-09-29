"""Signal scoreboard page: how well each signal's rankings predicted the next month."""

import polars as pl
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.research.scoreboard import HORIZON, summarize
from portfolio_lab.web.charts import cumulative_ic_figure

router = APIRouter()

#: Pools in display order, with their headings.
POOL_TITLES = {"top100": "100 most liquid stocks", "all": "All eligible stocks"}


@router.get("/signals", response_class=HTMLResponse)
def signals(request: Request) -> HTMLResponse:
    """Summary table and cumulative-IC chart per candidate pool."""
    path = DataPaths(request.app.state.data_dir).scoreboard
    scores = pl.read_parquet(path) if path.exists() else None
    pools = []
    if scores is not None and not scores.is_empty():
        table = summarize(scores)
        for pool, title in POOL_TITLES.items():
            rows = table.filter(pl.col("pool") == pool)
            if rows.height:
                chart = cumulative_ic_figure(scores.filter(pl.col("pool") == pool))
                pools.append({"key": pool, "title": title, "rows": rows.to_dicts(), "chart": chart})
    period = (scores["date"].min(), scores["date"].max()) if pools else None
    return request.app.state.templates.TemplateResponse(
        request, "signals.html", {"pools": pools, "period": period, "horizon": HORIZON}
    )
