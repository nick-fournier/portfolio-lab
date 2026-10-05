from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.forecaster import grade, linear, walk
from portfolio_lab.research.forecaster.dataset import INPUTS, _expanding_pct


def _month_end(k: int) -> date:
    return date(2000 + k // 12, k % 12 + 1, 28)


def _stocks(months: int = 36, n: int = 300, noise: float = 1.0, seed: int = 0) -> pl.DataFrame:
    """Stock-months whose target is a fixed mix of the first three inputs plus noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for m in range(months):
        x = rng.uniform(-0.5, 0.5, (n, len(INPUTS)))
        signal = x[:, 0] - 0.5 * x[:, 1] + 0.25 * x[:, 2]
        actual = signal + rng.normal(0, noise, n)
        actual -= actual.mean()
        frames.append(pl.DataFrame({
            "date": [_month_end(m)] * n, "symbol": [f"S{k:03d}" for k in range(n)],
            "actual": actual, "y": actual, "size": x[:, 3],
            **{c: x[:, j] for j, c in enumerate(INPUTS)},
        }))  # fmt: skip
    return pl.concat(frames)


def _market(months: int = 36) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    return pl.DataFrame({
        "date": [_month_end(m) for m in range(months)], "vix": rng.uniform(-0.5, 0.5, months),
        "dispersion": rng.uniform(-0.5, 0.5, months), "env_rate": rng.normal(size=months),
        "mkt_bear": rng.integers(0, 2, months).astype(float),
    })  # fmt: skip


def test_month_sums_add_up_to_the_squared_error():
    rng = np.random.default_rng(0)
    x = np.hstack([np.ones((50, 1)), rng.normal(size=(50, 12))])
    y = rng.normal(size=50)
    sums = linear.total([linear.month_sums(x[:25], y[:25]), linear.month_sums(x[25:], y[25:])])
    coef = rng.normal(size=(13, 2))
    direct = ((y[:, None] - x @ coef) ** 2).sum(axis=0)
    assert linear.errors(coef, sums) == pytest.approx(direct)


def test_linear_part_recovers_a_planted_signal():
    stocks = _stocks(noise=0.2)
    x = linear.design(stocks.select(INPUTS).to_numpy(), np.zeros(stocks.height),
                      np.zeros(stocks.height))  # fmt: skip
    months = sorted(set(stocks["date"].to_list()))
    sums = {}
    for d in months:
        rows = (stocks["date"] == d).to_numpy()
        sums[d] = linear.month_sums(x[rows], stocks["y"].to_numpy()[rows])
    coef, k = linear.fit(months, sums)
    assert k in linear.COMPONENTS
    assert coef[1] == pytest.approx(1.0, abs=0.1)
    assert coef[2] == pytest.approx(-0.5, abs=0.1)


def test_walk_forward_uses_only_earlier_months():
    stocks, market = _stocks(), _market()
    saved = []
    start = _month_end(30)
    out = walk.run(stocks, market, start=start, threads=1, skip={_month_end(31)},
                   save=saved.append)  # fmt: skip
    assert sorted(out["date"].unique().to_list()) == [_month_end(m) for m in (30, *range(32, 36))]
    assert len(saved) == 5
    # changing a later month's targets leaves an earlier month's forecast unchanged
    future = stocks.with_columns(
        pl.when(pl.col("date") > start).then(pl.col("y") * -3).otherwise(pl.col("y")).alias("y")
    )
    later = {_month_end(m) for m in range(31, 36)}
    again = walk.run(future, market, start=start, threads=1, skip=later)
    first = out.filter(pl.col("date") == start)
    assert np.allclose(first["linear"].to_numpy(), again["linear"].to_numpy())
    assert np.allclose(first["correction"].to_numpy(), again["correction"].to_numpy())
    graded = grade.grade_months(out.with_columns(pl.col("linear").alias("forecast")))
    assert graded["ic"].mean() > 0.2


def test_latest_month_without_a_target_is_forecast():
    stocks = _stocks().with_columns(
        pl.when(pl.col("date") == _month_end(35)).then(None).otherwise(pl.col(c)).alias(c)
        for c in ("actual", "y")
    )
    out = walk.run(stocks, _market(), start=_month_end(35), threads=1)
    assert out.height == 300 and out["linear"].is_not_null().all()


def _forecasts(correction_noise: float) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for m in range(24):
        signal = rng.normal(0, 1, 200)
        part1 = signal * 0.5
        correction = signal * 0.5 if correction_noise == 0 else rng.normal(0, correction_noise, 200)
        actual = signal + rng.normal(0, 1, 200)
        rows.append(pl.DataFrame({"date": [_month_end(m)] * 200, "linear": part1,
                                  "correction": correction, "actual": actual}))  # fmt: skip
    return pl.concat(rows)


def test_strength_is_chosen_from_earlier_months_only():
    useful = walk.combine(_forecasts(0.0))
    strengths = useful.group_by("date").agg(pl.col("strength").first()).sort("date")
    assert strengths["strength"][0] == 0.0  # nothing earlier to choose from
    assert strengths["strength"][-1] == 1.0
    noise = walk.combine(_forecasts(1.0))
    assert noise.filter(pl.col("date") > _month_end(3))["strength"].max() == 0.0


def test_nearly_perfect_forecasts_grade_nearly_perfectly():
    rng = np.random.default_rng(0)
    frames = []
    for m in range(24):
        f = rng.normal(0, 0.02, 200)
        frames.append(pl.DataFrame({"date": [_month_end(m)] * 200, "forecast": f,
                                    "actual": f + rng.normal(0, 0.001, 200),
                                    "size": rng.uniform(-0.5, 0.5, 200)}))  # fmt: skip
    out = grade.report(pl.concat(frames))
    assert out["ic"] == pytest.approx(1.0, abs=0.01)
    assert out["slope"] == pytest.approx(1.0, abs=0.02)
    assert out["r2"] == pytest.approx(1.0, abs=0.01)
    assert out["years_right"] == out["years"] == 2
    assert all(v == pytest.approx(1.0, abs=0.02) for v in out["ic_by_size"].values())


def test_expanding_percentile_sees_only_the_past():
    assert _expanding_pct([3.0, 1.0, 2.0, 5.0]) == [1.0, 0.5, 2 / 3, 1.0]


def test_nets_join_the_forecast_and_carry_over_between_runs(tmp_path):
    pytest.importorskip("torch")
    from portfolio_lab.research.forecaster.nets import NETS, Nets  # noqa: PLC0415 - optional

    stocks, market = _stocks(noise=0.2), _market()
    state = tmp_path / "nets.pt"
    first = walk.run(stocks, market, start=_month_end(33), threads=1, nets=Nets(state=state),
                     skip={_month_end(m) for m in (34, 35)})  # fmt: skip
    assert state.exists() and first["nets"].is_not_null().all()
    resumed = Nets(state=state)
    assert len(resumed.nets) == NETS  # picks up where the first run stopped
    later = walk.run(stocks, market, start=_month_end(34), threads=1, nets=resumed)
    combined = walk.combine(pl.concat([first, later]))
    assert {"linear_trees", "nets", "forecast"} <= set(combined.columns)
    assert grade.grade_months(combined)["ic"].mean() > 0.3


def test_grinold_rescales_by_earlier_months_only():
    rng = np.random.default_rng(0)
    frames = []
    for m in range(36):
        f = rng.normal(0, 0.02, 300)
        actual = 0.5 * f + rng.normal(0, 0.01, 300)  # forecasts twice too big
        frames.append(
            pl.DataFrame({"date": [_month_end(m)] * 300, "forecast": f, "actual": actual})
        )
    out = walk.calibrate(pl.concat(frames))
    scales = out.group_by("date").agg(pl.col("scale").first()).sort("date")["scale"]
    assert scales[0] == 1.0  # nothing earlier
    assert scales[-1] == pytest.approx(0.5, abs=0.03)
    ranks = out.group_by("date").agg(pl.corr("forecast", "raw_forecast", method="spearman"))
    assert ranks["forecast"].min() == pytest.approx(1.0)
