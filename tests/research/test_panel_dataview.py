from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.core.calendar import sessions
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import EligibilityRules, Panel

DAYS = sessions(date(2024, 1, 2), date(2024, 1, 31))


def _rows(symbol, closes, volume=1e6):
    return [
        (symbol, d, c, volume, None if i == 0 else c / closes[i - 1] - 1, 0.0)
        for i, (d, c) in enumerate(zip(DAYS, closes, strict=False))
    ]


def _prices(*groups):
    rows = [r for g in groups for r in g]
    return pl.DataFrame(
        rows, schema=["symbol", "date", "close", "volume", "ret_cc", "ret_co"], orient="row"
    )


def test_eligibility_uses_only_trailing_data():
    # PENNY crosses the $5 floor on day 6; THIN never trades enough; SPY is price-only.
    penny = [4.0] * 6 + [6.0] * (len(DAYS) - 6)
    prices = _prices(
        _rows("PENNY", penny),
        _rows("THIN", [20.0] * len(DAYS), volume=10),
        _rows("SPY", [400.0] * len(DAYS)),
    )
    rules = EligibilityRules(min_price=5, min_dollar_volume=1e5, adv_window=4, min_history=3)
    panel = Panel.from_long(prices, ["PENNY", "THIN"], rules=rules)
    col = panel.symbol_index["PENNY"]
    assert not panel.eligible[5, col] and panel.eligible[6, col]
    assert not panel.eligible[:, panel.symbol_index["THIN"]].any()
    assert not panel.eligible[:, panel.symbol_index["SPY"]].any()


def test_min_history_and_rates_alignment():
    prices = _prices(_rows("AAA", [10.0] * len(DAYS)))
    rates = pl.DataFrame({"date": [DAYS[0], DAYS[10]], "rate": [0.0504, 0.0252]})
    rules = EligibilityRules(min_price=1, min_dollar_volume=0, adv_window=2, min_history=5)
    panel = Panel.from_long(prices, ["AAA"], rates, rules)
    assert panel.eligible[:, 0].tolist() == [False] * 4 + [True] * (len(DAYS) - 4)
    assert panel.rf_daily[9] == pytest.approx(0.0504 / 252)
    assert panel.rf_daily[-1] == pytest.approx(0.0252 / 252)


def test_panel_arrays_are_read_only(make_panel):
    panel = make_panel()
    with pytest.raises(ValueError):
        panel.field("ret_cc")[0, 0] = 1.0
    with pytest.raises(ValueError):
        panel.eligible[0, 0] = True


def test_view_windows_end_at_asof(make_panel):
    panel = make_panel()
    view = DataView(panel, 100)
    rets = view.returns(20, ["AAA", "BBB"])
    assert rets.shape == (20, 2)
    assert rets.index[-1].date() == view.asof == panel.dates[100]
    assert DataView(panel, 3).returns(20).shape[0] == 4  # clipped at the start
    assert list(view.returns(5, ["AAA", "NOPE"]).columns) == ["AAA"]


def test_view_prices_rebased_and_close(make_panel):
    panel = make_panel()
    view = DataView(panel, 50)
    prices = view.prices(10, ["AAA"])
    rets = view.returns(10, ["AAA"])["AAA"].to_numpy()
    assert prices["AAA"].iloc[-1] == pytest.approx(np.prod(1 + rets))
    assert view.close(["AAA"])["AAA"] == panel.field("close")[50, panel.symbol_index["AAA"]]
    assert view.risk_free() == pytest.approx(0.05)


def test_view_eligible_and_top_liquid(make_panel):
    panel = make_panel()
    view = DataView(panel, 100)
    assert set(view.eligible()) == {"AAA", "BBB", "CCC"}  # SPY is a benchmark, never eligible
    adv = {s: panel.field("adv")[100, panel.symbol_index[s]] for s in view.eligible()}
    assert view.top_liquid(2) == sorted(adv, key=adv.get, reverse=True)[:2]


def test_view_rejects_out_of_range(make_panel):
    panel = make_panel()
    with pytest.raises(IndexError):
        DataView(panel, len(panel.dates))


def test_fscores_are_point_in_time_and_expire(make_panel):
    panel = make_panel(symbols=("AAA", "BBB"), days=400)
    table = panel.fundamentals
    first = table["filed"].min()
    day = panel.date_index[first]
    # A filing dated asof is not visible yet; the next session it is.
    assert DataView(panel, day).fscores() == {}
    assert DataView(panel, day + 1).fscores() == {"AAA": 9, "BBB": 5}
    # Scores older than max_age_days are dropped; too few signals are dropped.
    assert DataView(panel, day + 1).fscores(max_age_days=0) == {}
    assert DataView(panel, day + 1).fscores(min_signals=10) == {}
    assert DataView(panel, day + 1).fscores(["BBB", "ZZZ"]) == {"BBB": 5}
