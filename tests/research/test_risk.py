from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.core.calendar import rebalance_dates, sessions
from portfolio_lab.research import risk
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import Panel

N_DAYS, PER_SECTOR = 700, 20


def sector_panel(seed=0):
    """40 stocks in two sectors (SIC 28 and 73): market + sector factor + own noise."""
    rng = np.random.default_rng(seed)
    days = sessions(date(2020, 1, 2), date(2023, 12, 29))[:N_DAYS]
    market = rng.normal(0, 0.01, N_DAYS)
    sector_moves = rng.normal(0, 0.015, (N_DAYS, 2))
    symbols = [f"S{i:02d}" for i in range(2 * PER_SECTOR)]
    sec = np.repeat([0, 1], PER_SECTOR)
    ret = market[:, None] + sector_moves[:, sec] + rng.normal(0, 0.01, (N_DAYS, len(symbols)))
    ret = np.column_stack([ret, market])  # SPY last
    adv = np.tile(np.arange(ret.shape[1], 0, -1) * 1e7, (N_DAYS, 1)).astype(float)
    fields = {"close": np.full(ret.shape, 20.0), "ret_cc": ret, "ret_co": np.zeros_like(ret),
              "adv": adv}  # fmt: skip
    eligible = np.ones(ret.shape, dtype=bool)
    eligible[:, -1] = False
    panel = Panel(days, [*symbols, "SPY"], fields, eligible, np.zeros(N_DAYS), symbols)
    month_ends = rebalance_dates(days, "M")
    panel.features = pl.DataFrame(
        [(d, s, 28 if k < PER_SECTOR else 73, float(k), 0.05, 0.5, 0.1, 0.02, 0.05, 1.0)
         for d in month_ends for k, s in enumerate(symbols)],
        schema=["date", "symbol", "sic2", "log_size", "earnings_yield", "book_to_market",
                "mom_12_1", "volatility", "roa", "beta"],
        orient="row",
    )  # fmt: skip
    return panel


def test_risk_model_recovers_sector_structure():
    panel = sector_panel()
    view = DataView(panel, N_DAYS - 1)
    model = risk.fit(view, view.eligible())
    assert set(model.sectors) == {"chemicals_pharma", "business_services"}
    cov = model.cov(model.symbols)
    vol = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vol, vol)
    same = corr[0, 1:PER_SECTOR].mean()
    across = corr[0, PER_SECTOR:].mean()
    assert same > across + 0.2
    # Predicted volatility of an equal-weight portfolio matches its realized volatility.
    w = np.full(len(model.symbols), 1 / len(model.symbols))
    realized = panel.field("ret_cc")[-risk.WINDOW :, :-1] @ w
    assert np.sqrt(w @ cov @ w) == pytest.approx(realized.std() * np.sqrt(252), rel=0.25)


def test_sectors_from_sic_major_groups():
    assert risk.sector(28) == "chemicals_pharma"
    assert risk.sector(36) == "technology"
    assert risk.sector(34) == "manufacturing"
    assert risk.sector(None) == "unclassified" and risk.sector(99) == "unclassified"
