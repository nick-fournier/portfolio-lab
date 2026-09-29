from datetime import date

import polars as pl
import pytest

from portfolio_lab.core.calendar import sessions
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import EligibilityRules, Panel
from portfolio_lab.strategies.base import create

DAYS = 300
ASOF = DAYS - 1


def _panel():
    """Five stocks with known paths over the last year (daily returns per segment)."""
    dates = sessions(date(2023, 1, 3), date(2024, 12, 31))[:DAYS]
    paths = {
        "WIN": [0.002] * DAYS,  # steady winner
        "LATE": [0.0] * (DAYS - 21) + [0.03] * 21,  # +86%, all inside the skipped last month
        "FLAT": [0.0] * DAYS,
        "LOSE": [-0.001] * DAYS,
        "OK": [0.001] * DAYS,  # moderate winner
    }
    rows = []
    for symbol, rets in paths.items():
        price = 20.0
        for i, (d, r) in enumerate(zip(dates, rets, strict=True)):
            price *= 1 + r
            rows.append((symbol, d, price, 1e6, None if i == 0 else r, 0.0))
    prices = pl.DataFrame(
        rows, schema=["symbol", "date", "close", "volume", "ret_cc", "ret_co"], orient="row"
    )
    rules = EligibilityRules(min_price=1, min_dollar_volume=0, adv_window=5, min_history=5)
    return Panel.from_long(prices, list(paths), rules=rules)


def test_holds_top_winners_and_skips_the_last_month():
    weights = create("momentum", pool=None, hold=2).target_weights(DataView(_panel(), ASOF))
    assert weights == {"WIN": 0.5, "OK": 0.5}  # LATE's jump is inside the skipped month


def test_without_skip_the_late_jump_counts():
    weights = create("momentum", pool=None, hold=1, skip=0).target_weights(DataView(_panel(), ASOF))
    assert weights == {"LATE": 1.0}  # +86% in the last month beats WIN's +65% year


def test_needs_a_full_lookback():
    assert create("momentum").target_weights(DataView(_panel(), 100)) == {}


def test_rejects_bad_skip():
    with pytest.raises(ValueError, match="skip"):
        create("momentum", lookback=21, skip=21)
