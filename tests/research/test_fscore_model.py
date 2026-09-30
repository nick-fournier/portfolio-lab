from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.fscore_model import growth_labels, pca
from portfolio_lab.research.piotroski import PIOTROSKI, continuous


def test_continuous_score_respects_directions():
    good = {c: (0.9 if d > 0 else 0.1) for c, d in PIOTROSKI.items()}
    bad = {c: 1 - v for c, v in good.items()}
    data = pl.DataFrame([{"date": date(2020, 1, 31), "symbol": s, **v}
                         for s, v in (("GOOD", good), ("BAD", bad))])  # fmt: skip
    scores = dict(continuous(data).select("symbol", "score").iter_rows())
    assert scores["GOOD"] == pytest.approx(0.9) and scores["BAD"] == pytest.approx(0.1)


def test_growth_label_compares_the_same_period_a_year_later():
    states = pl.DataFrame(
        {"cik": [1, 1, 1],
         "filed": [date(2020, 2, 1), date(2020, 5, 1), date(2021, 2, 1)],
         "period_end": [date(2019, 12, 31), date(2020, 3, 31), date(2020, 12, 31)],
         "revenue": [100.0, 105.0, 120.0], "cfo": [10.0, 11.0, 12.0]}
    )  # fmt: skip
    labels = growth_labels(states, pl.DataFrame({"cik": [1], "symbol": ["A"]}))
    assert labels.height == 1  # only the 2019 filing has its year-later match
    row = labels.row(0, named=True)
    assert row["grows"] == 1 and row["known"] == date(2021, 2, 1)


def test_pca_weights_follow_shared_signal_and_skip_empty_metrics():
    rng = np.random.default_rng(0)
    rows = []
    for year in range(2010, 2017):
        for k in range(200):
            health = rng.normal()
            row = {c: float(np.clip(0.5 + 0.2 * health * d + 0.05 * rng.normal(), 0, 1))
                   for c, d in PIOTROSKI.items()}  # fmt: skip
            row["share_issuance"] = float(rng.uniform())  # unrelated noise
            rows.append({"date": date(year, 6, 30), "symbol": f"S{k}", **row,
                         "gross_profitability": None})  # fmt: skip
    data = pl.DataFrame(rows, schema_overrides={"gross_profitability": pl.Float64})
    scores, weights = pca(data, PIOTROSKI | {"gross_profitability": 1})
    first = weights.filter((pl.col("component") == 1) & (pl.col("year") == 2016))
    w = dict(first.select("metric", "weight").iter_rows())
    assert w["roa"] > 0.3 and abs(w["share_issuance"]) < 0.1
    assert w["gross_profitability"] == 0.0  # no data: left out
    assert scores["date"].min().year == 2015
