"""Class 3 optimizer: costs in the loss, init, grid, schedules, early stopping, seed averaging."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.optimizer import data


def test_net_return_charges_the_spread_on_turnover_from_last_month_and_a_fee_on_shorts():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import costs  # noqa: PLC0415

    before = data.Month(date(2000, 1, 31), ["A", "B", "C"], np.zeros((3, 1)), np.zeros(3))
    now = data.Month(date(2000, 2, 29), ["B", "A", "D"], np.zeros((3, 1)), np.zeros(3))
    link = costs.links([before, now])[1]
    w_before = torch.tensor([0.5, -0.2, 0.3])  # A, B, C
    w = torch.tensor([-0.2, 0.6, 0.1])  # B unchanged, A +0.1, D bought; C (0.3) sold
    r = torch.tensor([0.0, 0.1, 0.0])
    net = costs.net_return(w, r, w_before, link, spread=0.01, short_fee=0.12)
    turnover = 0.0 + 0.1 + 0.1 + 0.3
    assert float(net) == pytest.approx(0.06 - 0.01 * turnover - 0.01 * 0.2)
    assert float(costs.net_return(w, r, None, None, 0.01, 0.0)) == pytest.approx(0.06 - 0.009)


def test_walk_with_trading_costs_trades_less_than_without():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    rng = np.random.default_rng(5)
    months = []
    for k in range(30):
        x = rng.uniform(-0.5, 0.5, (8, 2)).astype(np.float32)
        r = (0.1 * x[:, 0] + rng.normal(0, 0.05, 8)).astype(np.float32)
        months.append(data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCDEFGH"), x, r))
    base = dict(window=12, first_epochs=100, warm_epochs=5, lr=0.02, start=date(2000, 1, 1))

    stock = [(np.full(8, 0.05, np.float32), np.full(8, 0.5, np.float32))] * 30

    def traded(**kw):
        w = walk(months, Settings("linear", **base, **kw), stock_costs=stock)[0]
        w = w.pivot("symbol", index="date").drop("date").to_numpy()
        return float(np.abs(np.diff(w, axis=0)).sum())

    assert traded(costs="table.parquet") < traded()


def test_per_stock_costs_charge_each_stock_its_own_rate_and_sold_stocks_last_months():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import costs  # noqa: PLC0415

    before = data.Month(date(2000, 1, 31), ["A", "B", "C"], np.zeros((3, 1)), np.zeros(3))
    now = data.Month(date(2000, 2, 29), ["B", "A", "D"], np.zeros((3, 1)), np.zeros(3))
    link = costs.links([before, now])[1]
    w_before = torch.tensor([0.5, -0.2, 0.3])  # A, B, C
    w = torch.tensor([-0.4, 0.6, 0.1])  # B -0.2 more short, A +0.1, D bought; C sold
    r = torch.zeros(3)
    spread = torch.tensor([0.02, 0.01, 0.05])  # this month: B, A, D
    spread_before = torch.tensor([0.0, 0.0, 0.03])  # last month: A, B, C
    fee = torch.tensor([1.2, 0.0, 0.0])  # only B costs to borrow
    net = costs.net_return(w, r, w_before, link, spread, fee, spread_before)
    traded = 0.02 * 0.2 + 0.01 * 0.1 + 0.05 * 0.1 + 0.03 * 0.3
    assert float(net) == pytest.approx(-traded - 1.2 / 12 * 0.4)


def test_align_puts_costs_in_row_order_and_fills_missing_stocks():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import costs  # noqa: PLC0415

    month = data.Month(date(2000, 1, 31), ["C", "A", "B"], np.zeros((3, 1)), np.zeros(3))
    table = pl.DataFrame({"date": [date(2000, 1, 31)] * 2, "symbol": ["A", "B"],
                          "trade_cost": [0.01, 0.03], "borrow_fee": [0.003, 0.5]})  # fmt: skip
    ((trade, fee, easy),) = costs.align([month], table)
    assert trade.tolist() == pytest.approx([0.02, 0.01, 0.03])  # C: the month's median
    assert fee.tolist() == pytest.approx([1.0, 0.003, 0.5])  # C: not lendable
    assert easy.tolist() == [1.0, 1.0, 1.0]  # no easy_chance column: all
    table = table.with_columns(pl.Series("easy_chance", [0.95, 0.4]))
    ((_, _, easy),) = costs.align([month], table)
    assert easy.tolist() == pytest.approx([0.0, 0.95, 0.4])  # C missing: 0


def test_walk_with_per_stock_costs_needs_them_and_runs_with_them():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    rng = np.random.default_rng(6)
    months = [
        data.Month(
            date(2000 + k // 12, k % 12 + 1, 1),
            list("ABCD"),
            rng.uniform(-0.5, 0.5, (4, 2)).astype(np.float32),
            rng.normal(0.01, 0.05, 4).astype(np.float32),
        )
        for k in range(16)
    ]
    settings = Settings("linear", window=12, first_epochs=5, warm_epochs=1,
                        start=date(2000, 1, 1), costs="table.parquet")  # fmt: skip
    with pytest.raises(ValueError, match="stock_costs"):
        walk(months, settings)
    stock = [(np.full(4, 0.01, np.float32), np.array([0.003, 1, 1, 1], np.float32))] * 16
    weights, log, _ = walk(months, settings, stock_costs=stock)
    assert log.height == 4 and weights["weight"].is_finite().all()


def test_init_keeps_rows_their_size_through_ten_blocks():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415

    x = torch.rand(500, 100) - 0.5
    size = float(x.norm(dim=1).mean())

    torch.manual_seed(0)
    y = x
    with torch.no_grad():
        for block in Optimizer("transformer", 100, 10, 256).blocks:
            y = block(y)
    assert float(y.norm(dim=1).mean()) / size == pytest.approx(1.0, abs=0.05)


def test_grid_records_every_epoch_and_cosine_decays_to_the_floor():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import grid  # noqa: PLC0415

    config = grid.Config("c", blocks=2, lr=1e-2, schedule="cosine", lr_min=1e-4, epochs=5)
    assert grid.lr_at(config, 0) == pytest.approx(1e-2)
    assert grid.lr_at(config, 4) == pytest.approx(1e-4)
    with pytest.raises(ValueError, match="unknown"):
        grid.configs_from([{"name": "x", "learning_rate": 1}])
    rng = np.random.default_rng(10)
    months = []
    for k in range(20):
        x = rng.uniform(-0.5, 0.5, (6, 3)).astype(np.float32)
        r = (0.1 * x[:, 0] + rng.normal(0, 0.03, 6)).astype(np.float32)
        months.append(data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCDEF"), x, r))
    setups = [config, grid.Config("small", blocks=1, epochs=3, stocks=4)]
    curves = grid.grid(months, [16, 18], setups, window=12, validation=3, device="cpu")
    assert curves.height == 2 * 5 + 2 * 3
    assert {"train_loss", "val_loss", "test_ret", "row_max", "gross", "seconds"} <= set(
        curves.columns
    )
    summary = grid.summary(curves)
    assert summary["config"].to_list() == ["c", "small"]
    assert (summary["best_epoch"] >= 1).all()


def test_walk_cosine_schedule_and_warm_rate():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import walk as wk  # noqa: PLC0415

    assert wk.rate("cosine", 1e-2, 1e-4, 0, 5) == pytest.approx(1e-2)
    assert wk.rate("cosine", 1e-2, 1e-4, 4, 5) == pytest.approx(1e-4)
    assert wk.rate("flat", 1e-2, 1e-4, 4, 5) == pytest.approx(1e-2)
    with pytest.raises(ValueError, match="schedule"):
        wk.rate("step", 1e-2, 1e-4, 0, 5)
    rng = np.random.default_rng(11)
    months = [
        data.Month(
            date(2000 + k // 12, k % 12 + 1, 1),
            list("ABCD"),
            rng.uniform(-0.5, 0.5, (4, 2)).astype(np.float32),
            rng.normal(0.01, 0.05, 4).astype(np.float32),
        )
        for k in range(15)
    ]
    weights, log, _ = wk.walk(months, wk.Settings("mlp", window=12, first_epochs=6,
                              warm_epochs=2, start=date(2000, 1, 1), schedule="cosine",
                              warm_lr=1e-3))  # fmt: skip
    assert log.height == 3 and weights["weight"].is_finite().all()


def test_seed_average_leaves_out_a_dud_seed_month_and_saves_the_average(tmp_path):
    from portfolio_lab.research.optimizer.cli import _average  # noqa: PLC0415

    d1, d2 = date(2020, 1, 31), date(2020, 2, 29)
    for name, loss, w in (
        ("seed-0", [0.5, 0.5], [2.0, -2.0]),
        ("seed-1", [0.4, float("nan")], [1.0, 3.0]),
    ):
        run = tmp_path / name
        run.mkdir()
        pl.DataFrame({"date": [d1, d2], "ret": [0.1, 0.2], "long": [1.0, 1.0], "short": [-1.0, 0.0],
                      "loss": loss}).write_parquet(run / "log.parquet")  # fmt: skip
        weights = pl.DataFrame({"date": [d1, d1, d2, d2], "symbol": ["A", "B", "A", "B"],
                                "weight": [w[0], w[1], w[0], w[1]]})  # fmt: skip
        weights.write_parquet(run / "weights.parquet")
    line = _average([tmp_path / "seed-0", tmp_path / "seed-1"])
    assert "1 dud seed-months" in line
    avg = pl.read_parquet(tmp_path / "weights.parquet").sort("date", "symbol")
    jan = avg.filter(pl.col("date") == d1)["weight"].to_list()
    feb = avg.filter(pl.col("date") == d2)["weight"].to_list()
    assert jan == pytest.approx([(0.5 + 0.25) / 2, (-0.5 + 0.75) / 2])  # both seeds at $1 gross
    assert feb == pytest.approx([0.5, -0.5])  # seed-1's February is a dud: seed-0 alone


def test_grid_with_costs_and_a_short_hurdle_shorts_less_and_reports_the_long_side():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import grid  # noqa: PLC0415

    rng = np.random.default_rng(13)
    months = []
    for k in range(20):
        x = rng.uniform(-0.5, 0.5, (8, 3)).astype(np.float32)
        r = (0.1 * x[:, 0] + rng.normal(0, 0.03, 8)).astype(np.float32)
        months.append(data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCDEFGH"), x, r))
    stock = [
        (np.full(8, 0.001, np.float32), np.full(8, 0.003, np.float32), np.ones(8, np.float32))
    ] * 20
    base = dict(blocks=1, lr=1e-2, epochs=15, costs=True)
    setups = [grid.Config("free", **base), grid.Config("long", short_hurdle=50.0, **base)]
    curves = grid.grid(months, [16, 18], setups, window=12, validation=3, device="cpu",
                       stock_costs=stock)  # fmt: skip
    last = (
        curves.filter(pl.col("epoch") == 15)
        .group_by("config")
        .agg(pl.col("val_short_share").mean())
    )
    share = dict(last.rows())
    assert share["long"] < share["free"]
    assert {"val_long_sharpe", "test_long_ret"} <= set(curves.columns)
    with pytest.raises(ValueError, match="stock_costs"):
        grid.fit(months, 16, setups[0], window=12, validation=3, device="cpu")


def test_max_return_zeroes_garbage_training_returns_only():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, _data  # noqa: PLC0415

    m = data.Month(date(2000, 1, 31), ["A", "B"], np.zeros((2, 1), np.float32),
                   np.array([948.0, 0.05], np.float32))  # fmt: skip
    ((_, r),) = _data([m], Settings("linear", max_return=3.0), "cpu")
    assert r.tolist() == pytest.approx([0.0, 0.05])
    assert m.r[0] == 948.0  # the month itself (the test month's return) is untouched
    ((_, r),) = _data([m], Settings("linear"), "cpu")
    assert float(r[0]) == 948.0
    assert torch.is_tensor(r)


def test_held_block_is_consecutive_and_left_out_of_the_fit():
    from portfolio_lab.research.optimizer import stopping  # noqa: PLC0415

    train = list(range(10, 70))
    fit, block = stopping.held_block(train, 12, np.random.default_rng(0))
    assert len(block) == 12 and block == list(range(block[0], block[0] + 12))
    assert sorted(fit + block) == train and not set(fit) & set(block)
    short = list(range(14))
    assert stopping.held_block(short, 12, np.random.default_rng(0)) == (short, [])


def test_fit_stopped_reads_the_smoothed_score_and_rolls_back_to_the_best_window():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import stopping  # noqa: PLC0415

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    adam = torch.optim.Adam(model.parameters())
    # raw scores: the spike at pass 2 must not win; means of 3 bottom at pass 4 (2.0, tied at 5)
    scores = iter([9.0, 0.5, 3.0, 2.5, 0.5, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0])

    def one_pass(p):
        with torch.no_grad():
            model.weight += 1  # the weight counts the passes
        return [float(model.weight)]

    losses, passes, best, curve = stopping.fit_stopped(
        model, adam, one_pass, lambda: next(scores), True, most=40, patience=5, smooth=3
    )
    assert (passes, best, losses) == (4, 2.0, [4.0])
    assert float(model.weight) == 4.0  # rolled back from pass 9
    assert curve == [9.0, 0.5, 3.0, 2.5, 0.5, 4.0, 3.0, 3.0, 3.0]
    calls = []
    losses, passes, best, curve = stopping.fit_stopped(
        model, adam, lambda p: calls.append(p) or [0.0], lambda: 0.0, False, most=4, patience=3
    )
    assert calls == [0, 1, 2, 3] and passes == 4 and np.isnan(best) and curve == []


def test_walk_with_early_stopping_logs_passes_and_leaves_the_book_to_training():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    rng = np.random.default_rng(19)
    months = [
        data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCDEFGH"),
                   rng.uniform(-0.5, 0.5, (8, 2)).astype(np.float32),
                   rng.normal(0.01, 0.05, 8).astype(np.float32))
        for k in range(30)
    ]  # fmt: skip
    stock = [(np.full(8, 0.005, np.float32), np.full(8, 0.05, np.float32))] * 30
    settings = Settings("linear", window=20, first_epochs=8, warm_epochs=8,
                        start=date(2001, 9, 1), costs="table.parquet", holdings_input=True,
                        stop=True, patience=2, holdout=6, smooth=2)  # fmt: skip
    _, log, _ = walk(months, settings, stock_costs=stock)
    assert log.height == 10
    assert all(2 <= p <= 8 for p in log["passes"]) and log["held_loss"].is_finite().all()
    assert all(len(c) >= p for c, p in zip(log["held_curve"], log["passes"], strict=True))
    _, plain, _ = walk(months, Settings("linear", window=20, first_epochs=8, warm_epochs=8,
                                        start=date(2001, 9, 1)))  # fmt: skip
    assert (plain["passes"] == 8).all() and plain["held_loss"].is_nan().all()
