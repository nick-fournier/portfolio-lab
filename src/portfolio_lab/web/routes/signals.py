"""Signal scoreboard page: how well each signal's rankings predicted the next month."""

import re

import polars as pl
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.research.glossary import EXPECT_TEXT, describe
from portfolio_lab.research.scoreboard import summarize
from portfolio_lab.web.charts import cumulative_ic_figure

router = APIRouter()

#: Pools in display order, with their headings.
POOL_TITLES = {"top100": "100 most liquid stocks", "all": "All eligible stocks"}
#: Horizons in display order, with their headings.
HORIZON_TITLES = {21: "Next month", 63: "Next quarter", 5: "Next week", 1: "Next day"}


@router.get("/signals", response_class=HTMLResponse)
def signals(request: Request) -> HTMLResponse:
    """Summary table and cumulative-IC chart per horizon and candidate pool."""
    path = DataPaths(request.app.state.data_dir).scoreboard
    scores = pl.read_parquet(path) if path.exists() else pl.DataFrame()
    sections = []
    if not scores.is_empty():
        table = summarize(scores)
        for horizon, heading in HORIZON_TITLES.items():
            pools = []
            for pool, title in POOL_TITLES.items():
                rows = table.filter((pl.col("horizon") == horizon) & (pl.col("pool") == pool))
                if rows.height:
                    subset = scores.filter(
                        (pl.col("horizon") == horizon) & (pl.col("pool") == pool)
                    )
                    pools.append(
                        {
                            "key": f"{pool}-{horizon}",
                            "title": title,
                            "rows": rows.to_dicts(),
                            "chart": cumulative_ic_figure(subset),
                        }
                    )
            if pools:
                sections.append({"title": heading, "horizon": horizon, "pools": pools})
    period = (scores["date"].min(), scores["date"].max()) if sections else None
    labels = sorted(scores["signal"].unique()) if not scores.is_empty() else []
    return request.app.state.templates.TemplateResponse(
        request,
        "signals.html",
        {"sections": sections, "period": period, "glossary": _glossary(labels),
         "anchor": anchor},
    )  # fmt: skip


def anchor(label: str) -> str:
    """HTML id for a signal's glossary entry."""
    return "g-" + re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def _glossary(labels: list[str]) -> list[dict]:
    """Glossary entries for the signals on the page, in name order."""
    out = []
    for label in labels:
        entry = describe(label)
        if entry is not None:
            out.append({"label": label, "anchor": anchor(label), "what": entry.what,
                        "why": entry.why, "expect": EXPECT_TEXT[entry.expect]})  # fmt: skip
    return out
