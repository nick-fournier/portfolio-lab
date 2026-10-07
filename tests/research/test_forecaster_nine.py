from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.forecaster import nine, report


def _month_end(k: int) -> date:
    return date(2000 + k // 12, k % 12 + 1, 28)


def _table(months: int = 40, n: int = 200, noise: float = 0.2, seed: int = 0) -> pl.DataFrame:
    """Stock-months whose target is a fixed mix of three of the terms plus noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for m in range(months):
        x = rng.normal(size=(n, len(nine.TERMS)))
        actual = 0.01 + 0.02 * x[:, 0] - 0.01 * x[:, 2] + 0.005 * x[:, 7] + rng.normal(0, noise, n)
        frames.append(pl.DataFrame({
            "date": [_month_end(m)] * n, "symbol": [f"S{k:03d}" for k in range(n)],
            "actual": actual, "has_momentum": [True] * n, "spy12": [0.1] * n,
            **{t: x[:, j] for j, t in enumerate(nine.TERMS)},
        }))  # fmt: skip
    return pl.concat(frames)


def test_walk_starts_after_the_minimum_history_and_recovers_the_signal():
    data = _table()
    out = nine.walk(data)
    months = sorted(set(data["date"].to_list()))
    assert out["date"].min() == months[nine.MIN_MONTHS]
    last = out.filter(pl.col("date") == months[-1])
    truth = data.filter(pl.col("date") == months[-1])
    signal = 0.02 * truth["mom12"] - 0.01 * truth["cfo_assets"] + 0.005 * truth["mom12_x_spy"]
    assert np.corrcoef(last["forecast"].to_numpy(), signal.to_numpy())[0, 1] > 0.95


def test_walk_never_uses_the_month_it_forecasts_or_later():
    data = _table()
    months = sorted(set(data["date"].to_list()))
    cut = months[30]
    changed = data.with_columns(
        pl.when(pl.col("date") >= cut).then(pl.col("actual") * -5).otherwise(pl.col("actual"))
    )
    a = nine.walk(data).filter(pl.col("date") <= cut)
    b = nine.walk(changed).filter(pl.col("date") <= cut)
    assert a["forecast"].to_numpy() == pytest.approx(b["forecast"].to_numpy())


def test_walk_fits_only_stocks_with_a_one_year_momentum():
    data = _table()
    bad = data.with_columns(
        pl.when(pl.col("symbol") == "S000").then(1e6).otherwise(pl.col("actual")).alias("actual"),
        (pl.col("symbol") != "S000").alias("has_momentum"),
    )
    a, b = nine.walk(data.filter(pl.col("symbol") != "S000")), nine.walk(bad)
    b = b.filter(pl.col("symbol") != "S000")
    assert a["forecast"].to_numpy() == pytest.approx(b["forecast"].to_numpy())


def test_inputs_are_standardized_within_each_month_and_keep_their_sign():
    rng = np.random.default_rng(3)
    frame = pl.DataFrame({
        "date": [_month_end(k // 500) for k in range(1000)],
        "x": np.r_[rng.standard_t(2, 500) * 50, rng.standard_t(2, 500)],
    })  # fmt: skip
    out = frame.with_columns(nine._scaled("x", clip=False).alias("z"))
    g = out.group_by("date").agg(pl.col("z").mean().alias("m"), pl.col("z").std().alias("s"))
    assert g["m"].to_numpy() == pytest.approx(0, abs=1e-9)
    assert g["s"].to_numpy() == pytest.approx(1)
    same = out.group_by("date").agg(pl.corr(pl.col("x").rank(), pl.col("z").rank()).alias("r"))
    assert same["r"].to_numpy() == pytest.approx(1)
    clipped = frame.with_columns(nine._scaled("x", clip=True).alias("z"))
    assert clipped["z"].abs().max() < out["z"].abs().max()


def test_summary_grades_three_forecasts_on_shared_stock_months():
    rng = np.random.default_rng(5)
    rows = []
    for m in range(120):
        d = date(2009 + m // 12, m % 12 + 1, 28)
        x = rng.normal(size=500)
        r = 0.02 * x + rng.normal(0, 0.1, 500)
        for k in range(500):
            rows.append((d, f"S{k:03d}", float(r[k] - r.mean()), float(r[k]), float(x[k]),
                         float(rng.normal()), float(k), (k - 250) / 500))  # fmt: skip
    f = pl.DataFrame(rows, schema=["date", "symbol", "actual", "r", "x", "noise", "liq", "size"],
                     orient="row")  # fmt: skip
    forecasts = f.select("date", "symbol", "actual", pl.col("x").alias("forecast"))
    previous = f.select("date", "symbol", "actual", "size", pl.col("noise").alias("linear"),
                        pl.lit(0.0).alias("correction"),
                        pl.lit(None, dtype=pl.Float64).alias("nets"),
                        pl.lit(1).alias("components"))  # fmt: skip
    production = f.select("date", "symbol", (-pl.col("x")).alias("production"))
    liquid = f.select("date", "symbol", pl.col("liq").alias("liquidity"), "r")
    s = report.build(forecasts, previous, production, liquid)
    p = s["pieces"]
    assert set(p) == {"forecast", "previous", "production"}
    assert p["forecast"]["ic"] > 0.1 > abs(p["previous"]["ic"])
    assert p["production"]["ic"] < -0.1
    assert p["forecast"]["liquid_ic"] > 0.1 and p["forecast"]["top_yr"] > p["production"]["top_yr"]
    assert {"ic", "old", "prev"} <= set(s["trailing"][0])
    assert {"top", "p_top", "q_top"} <= set(s["tenths"][0])
    assert s["yearly"][0]["prev"] is not None
