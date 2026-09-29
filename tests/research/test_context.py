from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from portfolio_lab.core.calendar import sessions
from portfolio_lab.data.sources.fred import OBSERVATION_SCHEMA
from portfolio_lab.research.context import environment, market_valuation, sensitivities, tailwinds
from portfolio_lab.research.panel import Panel

D = date


def _obs(rows):
    return pl.DataFrame(rows, schema=OBSERVATION_SCHEMA, orient="row")


def test_environment_is_point_in_time_with_changes_and_yoy():
    rows = [("DCOILWTICO", D(2024, 1, 2), 80.0, D(2024, 1, 3)),
            ("DCOILWTICO", D(2024, 4, 1), 100.0, D(2024, 4, 2)),
            ("UNRATE", D(2024, 3, 1), 4.0, D(2024, 4, 5))]  # fmt: skip
    # Monthly CPI from 100 (Jan 2023) rising 1 point a month, each published mid-next-month.
    for k in range(15):
        month = D(2023 + (k // 12), k % 12 + 1, 1)
        rows.append(("CPIAUCSL", month, 100.0 + k, month + timedelta(days=45)))
    env = environment(_obs(rows), [D(2024, 4, 3), D(2024, 4, 4)])
    first, second = env.row(0, named=True), env.row(1, named=True)
    assert first["oil"] == 100.0
    assert first["oil_chg3m"] == pytest.approx(np.log(100 / 80))
    assert first["unemployment"] is None  # published on 2024-04-05, after both dates
    # Latest CPI known on 2024-04-03: February 2024 (published mid-March) vs February 2023.
    assert first["cpi"] == pytest.approx(113 / 101 - 1)
    assert second["oil"] == 100.0


def _panel(n_days, oil_beta_stock=2.0, seed=0):
    rng = np.random.default_rng(seed)
    days = sessions(D(2020, 1, 2), D(2024, 12, 31))[:n_days]
    oil_log = np.cumsum(rng.normal(0, 0.02, n_days))
    oil_move = np.r_[0.0, np.diff(oil_log)]
    spy = rng.normal(0, 0.01, n_days)
    ret = np.column_stack([
        oil_beta_stock * oil_move + spy + rng.normal(0, 0.001, n_days),  # A: oil-driven
        spy + rng.normal(0, 0.01, n_days),  # B: market only
        spy,  # SPY
    ])  # fmt: skip
    fields = {"close": np.full(ret.shape, 10.0), "ret_cc": ret, "ret_co": np.zeros_like(ret),
              "adv": np.full(ret.shape, 1e9)}  # fmt: skip
    eligible = np.ones(ret.shape, dtype=bool)
    panel = Panel(days, ["A", "B", "SPY"], fields, eligible, np.zeros(n_days), ["A", "B"])
    # Oil level known the same day (available = date) so weekly moves line up with returns.
    obs = _obs([("DCOILWTICO", d, float(np.exp(v)), d) for d, v in zip(days, oil_log, strict=True)])
    return panel, obs


def test_sensitivities_recover_an_oil_driven_stock():
    panel, obs = _panel(600)
    sens = sensitivities(panel, obs, [panel.dates[-1]])
    betas = dict(zip(sens["symbol"], sens["oil_beta"], strict=True))
    assert betas["A"] == pytest.approx(2.0, abs=0.05)
    assert abs(betas["B"]) < 0.3


def test_tailwinds_multiply_beta_by_recent_move():
    day = D(2024, 1, 31)
    sens = pl.DataFrame(
        {"date": [day], "symbol": ["A"], "oil_beta": [2.0], "yield_10y_beta": [0.0],
         "dollar_beta": [0.0], "baa_spread_beta": [0.0]}
    )  # fmt: skip
    env = pl.DataFrame(
        {"date": [day], "oil_chg3m": [0.1], "yield_10y_chg3m": [0.5], "dollar_chg3m": [0.0],
         "baa_spread_chg3m": [0.0]}
    )  # fmt: skip
    row = tailwinds(sens, env).row(0, named=True)
    assert row["oil_tailwind"] == pytest.approx(0.2)
    assert row["macro_tailwind"] == pytest.approx(0.2)


def test_market_valuation_is_an_aggregate():
    f = pl.DataFrame({"date": [D(2024, 1, 31)] * 2, "market_value": [900.0, 100.0],
                      "earnings_yield": [0.02, 0.20], "book_to_market": [0.1, None]})  # fmt: skip
    row = market_valuation(f).row(0, named=True)
    assert row["market_earnings_yield"] == pytest.approx((18 + 20) / 1000)
    assert row["market_book_to_market"] == pytest.approx(0.1)
