from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.evaluation import calibrate, scoreboard_rows, summary
from portfolio_lab.research.models import MODELS, training_rows, walk_forward

MONTHS = [date(2017 + k // 12, k % 12 + 1, 28) for k in range(96)]


def _data(n=200, seed=0):
    """Monthly rows where cfo_to_assets (a percentile) drives the chance of beating the
    median; fscore is noise. Sessions are ~21 apart; labels reach 21 sessions ahead."""
    rng = np.random.default_rng(seed)
    frames = []
    for k, day in enumerate(MONTHS):
        signal = rng.random(n)
        fwd = 0.05 * (signal - 0.5) + rng.normal(0, 0.02, n)
        frames.append(pl.DataFrame({
            "date": [day] * n, "symbol": [f"S{j}" for j in range(n)],
            "cfo_to_assets": signal, "fscore": rng.random(n), "top500": [True] * n,
            "session": [21 * k] * n, "fwd_21": fwd, "label_end_21": [21 * k + 21] * n,
        }))  # fmt: skip
    data = pl.concat(frames)
    return data.with_columns(
        (pl.col("fwd_21") > pl.col("fwd_21").median().over("date")).cast(pl.Int8).alias("y_21")
    )


def test_training_rows_never_overlap_the_test_year():
    data = _data()
    start = data.filter(pl.col("date") == date(2021, 1, 28))["session"][0]
    train = training_rows(data, 2021, 21, start)
    assert train["date"].max() < date(2021, 1, 1)
    assert train["label_end_21"].max() < start  # December 2020's label reaches into 2021
    assert date(2020, 12, 28) not in train["date"].to_list()


def test_walk_forward_learns_the_informative_trait():
    data = _data()
    pred, fitted = walk_forward(data, MODELS["cfo_to_assets"], 21)
    assert min(fitted) == 2020 and pred["date"].min() >= date(2020, 1, 1)
    table = summary(pred).row(0, named=True)
    assert table["auc"] > 0.65 and table["top10_hit"] > 0.7 and table["bottom10_hit"] > 0.7
    noise, _ = walk_forward(data, MODELS["fscore"], 21)
    assert summary(noise).row(0, named=True)["auc"] < 0.55


def test_calibration_fixes_overconfidence():
    data = _data()
    pred, _ = walk_forward(data, MODELS["cfo_to_assets"], 21)
    loud = pred.with_columns((0.5 + 3 * (pl.col("p") - 0.5)).clip(0.01, 0.99).alias("p"))
    both = pl.concat([loud, calibrate(loud)], how="diagonal_relaxed")
    ece = dict(summary(both).filter(pl.col("pool") == "all").select("model", "ece").iter_rows())
    assert ece["cfo_to_assets+cal"] < ece["cfo_to_assets"]


def test_scoreboard_rows_have_the_scoreboard_schema():
    pred, _ = walk_forward(_data(), MODELS["cfo_to_assets"], 21)
    rows = scoreboard_rows(pred)
    assert rows.columns == ["signal", "pool", "horizon", "date", "n", "ic", "top", "bottom"]
    assert set(rows["signal"]) == {"model: cfo_to_assets"}
    assert rows["ic"].mean() == pytest.approx(0.5, abs=0.3)
    assert (rows["top"] > rows["bottom"]).mean() > 0.9
