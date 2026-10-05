from datetime import date

import polars as pl
import pytest

from portfolio_lab.research.characteristics.build import COLUMNS, finish

SOURCES = ("market_value", "debtnc", "assets", "cashneq", "rnd", "d_revenue_q",
           "mom_12_1", "revenue", "book_to_market", "cf_yield", "pchcapx", "chato", "chpm",
           "tb_raw")  # fmt: skip


def _grid() -> pl.DataFrame:
    rows = {"date": [date(2020, 1, 31)] * 3, "symbol": ["A", "B", "C"], "sic2": [21, 21, None],
            "sic": [2111, 2150, None]}  # fmt: skip
    rows |= {c: [1.0, 3.0, 5.0] for c in SOURCES}
    rows["cashneq"] = [0.0, 2.0, 2.0]  # A has no cash: its cash ratio is missing, not infinite
    frame = pl.DataFrame(rows)
    return frame.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(c) for c in COLUMNS if c not in frame.columns
    )


def test_finish_adjusts_by_industry_and_flags_sin_stocks():
    out = finish(_grid()).sort("symbol")
    assert out.columns == ["date", "symbol", *COLUMNS]
    assert out["bm_ia"].to_list()[:2] == pytest.approx([-1.0, 1.0])  # minus the industry mean
    assert out["bm_ia"][2] is None  # no industry
    assert out["sin"].to_list() == [1.0, 1.0, None]  # tobacco
    assert out["cashpr"][0] is None
    assert out["herf"][0] == pytest.approx(0.25**2 + 0.75**2)  # revenue shares 1/4 and 3/4
