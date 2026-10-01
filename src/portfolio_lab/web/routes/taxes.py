"""Taxes: is the strategy worth sheltering? Roth vs taxable, at rates the visitor picks.

Reads what ``plab tax-runs --publish`` writes (``results/taxes``) for production, production
holding gains until a year old, and SPY: weekly pre-tax growth and the gains and income each
month realized with no tax paid. The page's JavaScript applies the visitor's two rates with
the formula of ``backtest.tax.approx_after_tax`` (within 0.05 points a year of the exact
lot-by-lot replay), so moving a slider redraws instantly with no server call.
"""

import json
from pathlib import Path

import polars as pl
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.web import series
from portfolio_lab.web.charts import tax_layout

router = APIRouter()

#: Published versions shown, by key: production, production holding gains, SPY held.
VERSIONS = {
    "band0": "Taxable",
    "band0-defer": "Taxable, holding gains a year",
    "spy": "SPY, taxable",
}
START = 100_000.0
#: Rough combined rates (short-term, long-term) by joint taxable income, 2026:
#: federal + 3.8% investment surtax (above $250k) + state.
RATE_GUIDE = (
    ("~$80k", (12, 0), (18, 6)),
    ("~$150k", (22, 15), (31, 24)),
    ("~$300k", (28, 19), (37, 28)),
    ("~$450k", (36, 19), (45, 28)),
    ("~$800k", (41, 24), (51, 34)),
)


def page_data(data_dir: Path) -> dict | None:
    """Everything the page's script needs, as JSON-ready lists (cached until republished)."""
    folder = DataPaths(data_dir).taxes
    index = folder / "index.json"

    def build() -> dict | None:
        if not index.exists():
            return None
        entries = {e["key"]: e for e in json.loads(index.read_text())}
        if "band0" not in entries:
            return None
        out = {}
        for key, name in VERSIONS.items():
            if key not in entries or not (folder / f"{key}.monthly.parquet").exists():
                continue
            nav = series.weekly(
                pl.read_parquet(folder / f"{key}.nav.parquet").rename({"nav": "growth"})
            )
            months = pl.read_parquet(folder / f"{key}.monthly.parquet")
            out[key] = {
                "name": name,
                "x": [d.isoformat() for d in nav["date"].to_list()],
                "nav": nav["growth"].round(6).to_list(),
                "months": {
                    "date": [d.isoformat() for d in months["date"].to_list()],
                    **{c: months[c].round(8).to_list() for c in months.columns if c != "date"},
                },
                "unrealized": entries[key]["unrealized"],
            }
        return {"start": START, "versions": out, "since": out["band0"]["x"][0][:4]}

    return series.cached(f"taxes:{data_dir}", [index], build)


@router.get("/taxes", response_class=HTMLResponse)
def taxes_page(request: Request) -> HTMLResponse:
    """Two rates in, Roth vs taxable vs SPY out."""
    data = page_data(request.app.state.data_dir)
    context = {"data": json.dumps(data) if data else None, "since": data and data["since"],
               "layout": tax_layout(), "guide": RATE_GUIDE}  # fmt: skip
    return request.app.state.templates.TemplateResponse(request, "taxes.html", context)
