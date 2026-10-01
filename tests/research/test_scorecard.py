from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.scorecard import scorecard


def test_scorecard_compares_runs_with_the_reference():
    days = [date(2001, 1, 1) + timedelta(days=k) for k in range(1500)]
    rng = np.random.default_rng(0)
    base = rng.normal(0.0004, 0.01, len(days))
    daily = {
        "ref": pl.DataFrame({"date": days, "ret": base}),
        "better": pl.DataFrame({"date": days, "ret": base + 0.0002}),
    }
    table = scorecard(daily, {"ref": 1.0, "better": 2.0}, "ref")
    rows = {r["run"]: r for r in table.to_dicts()}
    assert rows["better"]["cagr"] > rows["ref"]["cagr"]
    years = rows["better"]["years_won_vs_ref"].split("/")
    assert years[0] == years[1]  # wins every calendar year
    assert rows["better"]["worst_3y_vs_ref"] > 0
    assert rows["ref"]["worst_3y_vs_ref"] == pytest.approx(0.0)
    assert rows["better"]["turnover"] == 2.0
