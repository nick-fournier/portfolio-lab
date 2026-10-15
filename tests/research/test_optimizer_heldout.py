"""Class 3 optimizer: which training months are held out, and cold sampled months."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.optimizer import data


def test_held_block_is_consecutive_and_left_out_of_the_fit():
    from portfolio_lab.research.optimizer import stopping  # noqa: PLC0415

    train = list(range(10, 70))
    fit, block = stopping.held_block(train, 12, np.random.default_rng(0))
    assert len(block) == 12 and block == list(range(block[0], block[0] + 12))
    assert sorted(fit + block) == train and not set(fit) & set(block)
    short = list(range(14))
    assert stopping.held_block(short, 12, np.random.default_rng(0)) == (short, [])


def test_scattered_held_months_are_spread_out_and_left_out_of_the_fit():
    from portfolio_lab.research.optimizer import stopping  # noqa: PLC0415

    train = list(range(10, 130))
    fit, held = stopping.held_months(train, 12, np.random.default_rng(3), scatter=True)
    assert len(held) == 12 and held == sorted(held)
    assert sorted(fit + held) == train and not set(fit) & set(held)
    assert held != list(range(held[0], held[0] + 12))  # not one block
    assert stopping.held_months(train, 0, np.random.default_rng(3), scatter=True) == (train, [])
    assert stopping.held_months(train, 0, np.random.default_rng(3)) == (train, [])


def test_cold_listed_months_stand_alone_and_holdout_seed_moves_the_held_months():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    rng = np.random.default_rng(23)
    months = [
        data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCDEFGH"),
                   rng.uniform(-0.5, 0.5, (8, 2)).astype(np.float32),
                   rng.normal(0.01, 0.05, 8).astype(np.float32))
        for k in range(30)
    ]  # fmt: skip
    stock = [(np.full(8, 0.005, np.float32), np.full(8, 0.05, np.float32))] * 30
    base = dict(
        window=20,
        first_epochs=5,
        start=date(2001, 9, 1),
        costs="table.parquet",
        holdings_input=True,
        holdout=6,
        cold=True,
        watch=True,
        scatter=True,
    )

    def run(**kw):
        return walk(months, Settings("linear", **base, **kw), stock_costs=stock)

    _, log_all, _ = run()
    picked = (date(2002, 1, 1), date(2002, 4, 1))
    w_two, log_two, _ = run(only=picked)
    assert log_two["date"].to_list() == list(picked)
    w_one, _, _ = run(only=picked[1:])
    for d in picked:  # a cold month does not depend on the months that ran before it
        assert log_two.filter(pl.col("date") == d)["test_curve"][0].to_list() == pytest.approx(
            log_all.filter(pl.col("date") == d)["test_curve"][0].to_list(), abs=1e-6
        )
    late = pl.col("date") == picked[1]  # traded holdings differ from a full walk's by design
    assert np.allclose(w_two.filter(late)["weight"].to_numpy(),
                       w_one.filter(late)["weight"].to_numpy(), atol=1e-6)  # fmt: skip
    _, other, _ = run(only=picked, holdout_seed=7)
    assert other["held_curve"].to_list() != log_two["held_curve"].to_list()
    _, none, _ = run(only=picked, holdout=0)  # nothing held out: the test month still logged
    assert all(len(c) == 0 for c in none["held_curve"])
    assert all(len(c) == 5 for c in none["test_curve"])
