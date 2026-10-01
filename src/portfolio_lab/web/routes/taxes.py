"""Tax calculator: production after tax, at several levels of stickiness, against SPY.

Reads what ``plab tax-runs --publish`` writes (``results/taxes``): each version's trade log
and pre-tax growth. The backtests are fixed; the taxes are replayed per request with the
visitor's rates (``backtest.tax``), which takes a fraction of a second.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import polars as pl
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from portfolio_lab.backtest.tax import TaxRates, after_tax, events
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.web import series
from portfolio_lab.web.charts import tax_figure

router = APIRouter()

#: Rate presets (short-term, long-term, dividends), combined federal + state + 3.8% NIIT.
PRESETS = {
    "California, ~$370k joint": (37.1, 28.1, 28.1),
    "No state tax, ~$370k joint": (27.8, 18.8, 18.8),
    "California, ~$150k joint": (31.3, 24.3, 24.3),
}
BENCHMARK = "spy"
#: Skip thresholds on offer, in percentage points (as published by ``jobs.taxes``).
BANDS = (0, 1, 2, 5)


class Inputs(BaseModel):
    """The calculator's form (query parameters)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    account: Literal["taxable", "roth"] = "taxable"
    short: float = Field(37.1, ge=0, le=70)
    long: float = Field(28.1, ge=0, le=70)
    dividends: float = Field(28.1, ge=0, le=70)
    start: float = Field(100_000.0, ge=1_000, le=1e9)
    band: int = Field(0, ge=0, le=5)  # a published skip threshold (see BANDS)
    defer: bool = False
    order: Literal["tax", "fifo"] = "tax"
    sell: bool = False

    @property
    def key(self) -> str:
        """The published version these settings select, e.g. ``band2-defer``."""
        return f"band{self.band}" + ("-defer" if self.defer else "")


def _load(data_dir: Path) -> dict:
    """Published versions: index entries with their events and pre-tax growth."""
    folder = DataPaths(data_dir).taxes
    index = folder / "index.json"

    def build() -> dict:
        if not index.exists():
            return {}
        out = {}
        for entry in json.loads(index.read_text()):
            nav = pl.read_parquet(folder / f"{entry['key']}.nav.parquet")
            trades = pl.read_parquet(folder / f"{entry['key']}.trades.parquet")
            out[entry["key"]] = {**entry, "nav": nav, "events": events(trades)}
        return out

    return series.cached(f"taxes:{data_dir}", [index], build)


def version_name(band: int, defer: bool) -> str:
    """E.g. "Skip < 2 pts, hold gains"."""
    name = "Trade every change" if band == 0 else f"Skip < {band} pt" + ("s" if band > 1 else "")
    return name + (", hold gains" if defer else "")


@lru_cache(maxsize=64)
def _replay(data_dir: Path, key: str, inputs: Inputs, stamp: tuple) -> dict:
    """One version's after-tax result (``stamp`` invalidates the cache when files change)."""
    version = _load(data_dir)[key]
    nav = version["nav"]
    rates = TaxRates(inputs.short / 100, inputs.long / 100, inputs.dividends / 100)
    if inputs.account == "roth":
        rates = TaxRates(0.0, 0.0, 0.0, loss_offset=0.0)
    result = after_tax(version["events"], nav, rates, inputs.start, inputs.order, inputs.sell)
    years = (nav["date"][-1] - nav["date"][0]).days / 365.25
    final = result.growth.tail(1)
    gains = max(result.short_gains, 0.0) + max(result.long_gains, 0.0)
    return {
        "key": key,
        "growth": result.growth,
        "before": (final["before"][0] / inputs.start) ** (1 / years) - 1,
        "after": (final["after"][0] / inputs.start) ** (1 / years) - 1,
        "final": final["after"][0],
        "taxes": result.taxes,
        "short_share": max(result.short_gains, 0.0) / gains if gains > 0 else None,
        "unrealized": result.unrealized,
        "traded": (version.get("turnover") or 0.0) / 2,  # one-way: share of the portfolio sold
    }


def calculate(data_dir: Path, inputs: Inputs) -> dict:
    """Every version and SPY under ``inputs``, plus the chart for the chosen version."""
    versions = _load(data_dir)
    if inputs.key not in versions:
        return {}
    stamp = tuple(p.stat().st_mtime_ns for p in [DataPaths(data_dir).taxes / "index.json"])
    rows = []
    for key, version in versions.items():
        row = _replay(data_dir, key, inputs, stamp)
        band = round(version["band"] * 100)
        name = "SPY, held" if key == BENCHMARK else version_name(band, version["defer_short_gains"])
        rows.append({**row, "name": name, "chosen": key == inputs.key})
    chosen = next(r for r in rows if r["chosen"])
    spy = next((r for r in rows if r["key"] == BENCHMARK), None)
    nav = versions[inputs.key]["nav"]
    return {
        "rows": rows,
        "chosen": chosen,
        "since": nav["date"][0],
        "chart": tax_figure(chosen["growth"], spy["growth"] if spy else None),
    }


Form = Annotated[Inputs, Query()]


@router.get("/taxes", response_class=HTMLResponse)
def taxes_page(request: Request, inputs: Form) -> HTMLResponse:
    """The calculator: form plus results (the form refreshes ``/taxes/result``)."""
    context = {"inputs": inputs, "presets": PRESETS, "bands": BANDS,
               **calculate(request.app.state.data_dir, inputs)}  # fmt: skip
    return request.app.state.templates.TemplateResponse(request, "taxes.html", context)


@router.get("/taxes/result", response_class=HTMLResponse)
def taxes_result(request: Request, inputs: Form) -> HTMLResponse:
    """The results block for the form's current values."""
    context = {"inputs": inputs, **calculate(request.app.state.data_dir, inputs)}
    return request.app.state.templates.TemplateResponse(request, "_tax_result.html", context)
