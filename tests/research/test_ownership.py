from datetime import date

import polars as pl
import pytest

from portfolio_lab.research.ownership import ownership_features


def test_ownership_features_are_point_in_time():
    features = pl.DataFrame(
        {"date": [date(2020, 6, 30), date(2020, 12, 31)], "symbol": ["A", "A"],
         "market_value": [1000.0, 1000.0]}
    )  # fmt: skip
    trades = pl.DataFrame(
        {"symbol": ["B", "A", "A", "A"],  # B's filing starts the data's coverage
         "filed": [date(2019, 1, 2), date(2020, 3, 1), date(2020, 6, 30), date(2020, 9, 1)],
         "buy_value": [1.0, 10.0, 50.0, 20.0], "sell_value": [0.0, 0.0, 0.0, 5.0],
         "buys": [1.0, 1.0, 1.0, 1.0]}
    )  # fmt: skip
    holders = pl.DataFrame(
        {"symbol": ["A"] * 3,
         "available": [date(2019, 11, 14), date(2020, 5, 15), date(2020, 11, 14)],
         "holders": [100.0, 110.0, 150.0]}
    )  # fmt: skip
    out = ownership_features(features, trades, holders).sort("date")
    june, dec = out.row(0, named=True), out.row(1, named=True)
    # The filing on June 30 itself is not yet known at June 30's close.
    assert june["insider_buy"] == pytest.approx(0.01)
    # By December only the September purchase is inside the six-month window.
    assert dec["insider_buy"] == pytest.approx(0.02)
    assert dec["insider_net"] == pytest.approx(0.015)
    assert dec["inst_breadth_1y"] == pytest.approx(0.5)
    assert june["inst_breadth_1q"] == pytest.approx(0.1)  # May count vs the November one


def test_insider_features_unknown_before_the_data_starts():
    features = pl.DataFrame(
        {"date": [date(2019, 1, 31), date(2021, 1, 29)], "symbol": ["A", "A"],
         "market_value": [1000.0, 1000.0]}
    )  # fmt: skip
    trades = pl.DataFrame(
        {"symbol": ["B"], "filed": [date(2020, 1, 2)], "buy_value": [1.0], "sell_value": [0.0],
         "buys": [1.0]}
    )  # fmt: skip
    out = ownership_features(features, trades, None).sort("date")
    assert out["insider_buy"].to_list() == [None, 0.0]
