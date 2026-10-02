"""Signals page: how well each signal's ranking of production's candidates predicted returns."""

from pathlib import Path

import polars as pl
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.research.glossary import describe
from portfolio_lab.research.scoreboard import summarize
from portfolio_lab.web.charts import cumulative_ic_figure
from portfolio_lab.web.series import cached

router = APIRouter()

#: The pool shown: production's candidates.
POOL = "top500"
#: Horizons in display order, with their toggle labels.
HORIZON_TITLES = {21: "Next month"}  # production rebalances monthly
#: Signals drawn on the chart at first: the most reliable each way (the rest are in the legend).
CHART_SHOWN = 6


def page_context(data_dir: Path) -> tuple:
    """One table and chart per horizon, and the period covered (rebuilt when data changes)."""
    path = DataPaths(data_dir).scoreboard

    def build() -> tuple:
        scores = pl.read_parquet(path) if path.exists() else pl.DataFrame()
        if not scores.is_empty():
            scores = scores.filter(pl.col("pool") == POOL)
        sections = []
        if not scores.is_empty():
            table = summarize(scores)
            for horizon, title in HORIZON_TITLES.items():
                rows = table.filter(pl.col("horizon") == horizon).sort("ic_t", descending=True)
                if rows.height:
                    subset = scores.filter(pl.col("horizon") == horizon)
                    ranked = rows.sort(pl.col("ic_t").abs(), descending=True)["signal"]
                    shown = set(ranked.head(CHART_SHOWN))
                    sections.append({"key": str(horizon), "title": title,
                                     "rows": [_describe(r) for r in rows.to_dicts()],
                                     "chart": cumulative_ic_figure(subset, shown)})  # fmt: skip
        period = (scores["date"].min(), scores["date"].max()) if sections else None
        return sections, period

    return cached(f"signals:{path}", [path], build)


def _describe(row: dict) -> dict:
    """A table row with the signal's plain-language description and good direction."""
    entry = describe(row["signal"])
    expect = entry.expect if entry else "unclear"
    # A "lower is better" signal is right when the ranking runs backwards (negative IC).
    right = 1 - row["hit"] if expect == "lower" else row["hit"]
    return {**row, "what": entry.what if entry else "", "expect": expect, "right": right}


@router.get("/signals", response_class=HTMLResponse)
def signals(request: Request) -> HTMLResponse:
    """One table per horizon (toggled), with a cumulative-IC chart."""
    sections, period = page_context(request.app.state.data_dir)
    return request.app.state.templates.TemplateResponse(
        request, "signals.html", {"sections": sections, "period": period}
    )
