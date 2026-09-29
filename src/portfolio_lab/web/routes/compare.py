"""Make versus buy: our strategies against funds anyone can buy."""

import polars as pl
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.research.funds import CATEGORIES, PAIRS
from portfolio_lab.web.charts import growth_figure

router = APIRouter()

#: Funds always drawn on the chart (besides our best strategies and the best fund).
ALWAYS_SHOWN = ("SPY", "QQQ")
OURS_SHOWN = 3


def _pairs(common: pl.DataFrame) -> list[dict]:
    """Each of our strategies beside the fund it competes with most directly."""
    by_key = {r["key"]: r for r in common.to_dicts()}
    out = []
    for label, fund in PAIRS.items():
        ours, theirs = by_key.get(f"ours: {label}"), by_key.get(fund)
        if ours and theirs:
            out.append({"ours": ours, "fund": theirs})
    return out


def _visible(common: pl.DataFrame) -> set[str]:
    """Series shown when the chart opens: SPY, QQQ, the best fund, our best strategies."""
    ranked = common.sort("sharpe", descending=True)
    ours = ranked.filter(pl.col("category") == "ours")["key"].head(OURS_SHOWN).to_list()
    best_fund = ranked.filter(pl.col("category") != "ours")["key"].head(1).to_list()
    return {*ALWAYS_SHOWN, *ours, *best_fund}


@router.get("/compare", response_class=HTMLResponse)
def compare(request: Request) -> HTMLResponse:
    """Render the make-vs-buy tables, head-to-heads and growth chart."""
    folder = DataPaths(request.app.state.data_dir).make_vs_buy
    path = folder / "summary.parquet"
    context: dict = {"categories": CATEGORIES, "periods": []}
    if path.exists():
        summary = pl.read_parquet(path)
        common = summary.filter(pl.col("period") == "common")
        context["pairs"] = _pairs(common)
        for period, title in (("common", "Same period for everyone"),
                              ("full", "Each one's full history since 2017")):  # fmt: skip
            rows = summary.filter(pl.col("period") == period).sort("sharpe", descending=True)
            context["periods"].append({"title": title, "rows": rows.to_dicts()})
        context["common_start"] = common["start"].min()
        growth = pl.read_parquet(folder / "growth.parquet")
        names = dict(summary.select("key", "name").unique("key").iter_rows())
        order = {c: k for k, c in enumerate(CATEGORIES)}
        keyed = common.sort(pl.col("category").replace_strict(order), "name")
        groups = {k: CATEGORIES[c] for k, c in keyed.select("key", "category").iter_rows()}
        context["chart"] = growth_figure(growth, names, _visible(common), groups)
    return request.app.state.templates.TemplateResponse(request, "compare.html", context)
