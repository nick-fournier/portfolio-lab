"""Market context page: production's market read and today's conditions.

Also how stock traits and the market behaved in similar conditions before.
"""

import math
from pathlib import Path

import polars as pl
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from portfolio_lab.backtest.results import list_runs
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.research.conditions import CONDITIONS, current_conditions

router = APIRouter()

#: Environment snapshot rows: (column, label, unit). Units: ``points`` (already in %),
#: ``fraction`` (shown as %), ``usd``, ``level``.
SNAPSHOT = (
    ("oil", "Oil (WTI, $/barrel)", "usd"),
    ("natural_gas", "Natural gas ($/MMBtu)", "usd"),
    ("copper", "Copper ($/tonne)", "usd"),
    ("fed_funds", "Fed funds rate", "points"),
    ("yield_2y", "2-year Treasury yield", "points"),
    ("yield_10y", "10-year Treasury yield", "points"),
    ("curve_10y_2y", "Yield curve (10y minus 2y)", "points"),
    ("real_yield_10y", "10-year real yield", "points"),
    ("breakeven_10y", "10-year inflation expectation", "points"),
    ("baa_spread", "Credit spread (Baa minus 10y)", "points"),
    ("vix", "VIX (expected volatility)", "level"),
    ("dollar", "Dollar index (trade-weighted)", "level"),
    ("financial_conditions", "Financial conditions (Chicago Fed, >0 = tight)", "level"),
    ("cpi", "Inflation (CPI, year on year)", "fraction"),
    ("core_cpi", "Core inflation (year on year)", "fraction"),
    ("unemployment", "Unemployment rate", "points"),
    ("payrolls", "Payroll growth (year on year)", "fraction"),
    ("industrial_production", "Industrial production (year on year)", "fraction"),
    ("retail_sales", "Retail sales (year on year)", "fraction"),
    ("consumer_sentiment", "Consumer sentiment (Michigan)", "level"),
    ("market_earnings_yield", "Market earnings yield (our universe)", "fraction"),
    ("equity_risk_premium", "Earnings yield minus 10-year yield", "fraction"),
)
#: Shown in place of a missing value.
MISSING = "\N{EN DASH}"
LOG_SERIES = {"oil", "natural_gas", "copper", "dollar", "jobless_claims", "housing_starts"}


def _fmt(value: float | None, unit: str) -> str:
    """Display a level in its unit."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MISSING
    if unit == "usd":
        return f"${value:,.2f}"
    if unit == "points":
        return f"{value:.2f}%"
    if unit == "fraction":
        return f"{value * 100:.1f}%"
    return f"{value:,.2f}"


def _change(column: str, value: float | None, unit: str) -> str:
    """Display a 3-month change: % for prices, percentage points for rates."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MISSING
    if column in LOG_SERIES:
        return f"{math.exp(value) - 1:+.0%}"
    if unit == "fraction":
        return f"{value * 100:+.1f} pts"
    if unit == "points":
        return f"{value:+.2f} pts"
    return f"{value:+.2f}"


def _snapshot(row: dict) -> list[dict]:
    """The environment table: each series' latest value, 3-month change and percentile."""
    return [
        {"label": label, "value": _fmt(row[column], unit),
         "change": _change(column, row.get(f"{column}_chg3m"), unit),
         "pct": row.get(f"{column}_pct")}
        for column, label, unit in SNAPSHOT
        if column in row
    ]  # fmt: skip


def _now(env: pl.DataFrame) -> list[dict]:
    """Today's bucket for each condition, with its reading in words."""
    out = []
    for c in current_conditions(env):
        column, v = CONDITIONS[c["condition"]][0], c["value"]
        if v is None:
            shown = MISSING
        elif column.endswith("_pct"):
            shown = f"higher than {v:.0%} of its history"
        elif column == "equity_risk_premium":
            shown = f"{v * 100:+.1f}%"
        else:
            shown = f"{v:+.2f} pts"
        out.append({**c, "shown": shown})
    return out


def _conditions(table: pl.DataFrame) -> list[dict]:
    """The traits-by-condition tables, cells in bucket order."""
    out = []
    for name, (_, _, labels) in CONDITIONS.items():
        sub = table.filter(pl.col("condition") == name)
        rows = []
        for trait in sub["trait"].unique(maintain_order=True):
            cells = {r["bucket"]: r for r in sub.filter(pl.col("trait") == trait).to_dicts()}
            rows.append({"trait": trait, "cells": [cells.get(b) for b in labels]})
        if rows:
            out.append({"name": name, "labels": labels, "rows": rows})
    return out


#: What production does in each market state (``research.regimes``).
STATE_TEXT = {
    "normal": "Normal market: production holds its usual best-return-per-risk mix.",
    "bear": "Bear market (S&P 500 below its 200-day average for three month ends and 15%+ off "
    "its high): production holds its lowest-risk mix.",
    "rebound": "Rebound (20%+ off the high with fear easing): production holds its candidates "
    "in equal weights, since beaten-down stocks tend to lead the bounce.",
}


def _market_state(data_dir: Path) -> dict | None:
    """Production's market read at its latest rebalance (from the run's diagnostics)."""
    for run in list_runs(data_dir, latest_only=True):
        meta = run["meta"]
        if meta["strategy"] != "meanvar" or not meta.get("params", {}).get("bear_defense"):
            continue
        states = [g for g in meta.get("diagnostics", {}).get("gauge", []) if "state" in g]
        if states:
            last = states[-1]
            return {"date": last["date"], "state": last["state"],
                    "text": STATE_TEXT.get(last["state"], "")}  # fmt: skip
    return None


@router.get("/context", response_class=HTMLResponse)
def context(request: Request) -> HTMLResponse:
    """Render the market context page from the stored environment and measurements."""
    data_dir = request.app.state.data_dir
    paths = DataPaths(data_dir)
    env = pl.read_parquet(paths.environment) if paths.environment.exists() else None
    has_env = env is not None and env.height > 0
    dial = pl.read_parquet(paths.context_dial) if paths.context_dial.exists() else None
    now = _now(env) if has_env else []
    current = {c["condition"]: c["bucket"] for c in now}
    conditions = (
        _conditions(pl.read_parquet(paths.context_conditions))
        if paths.context_conditions.exists() else []
    )  # fmt: skip
    for c in conditions:
        c["current"] = current.get(c["name"])
        c["dial"] = (
            dial.filter(pl.col("condition") == c["name"]).to_dicts() if dial is not None else []
        )
    return request.app.state.templates.TemplateResponse(
        request,
        "context.html",
        {
            "latest": env["date"].max() if has_env else None,
            "state": _market_state(data_dir),
            "snapshot": _snapshot(env.row(-1, named=True)) if has_env else [],
            "now": now,
            "conditions": conditions,
        },
    )
