"""Class 3 optimizer: costs, short controls, partial trading, retraining, init, grid, schedules."""

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

    def traded(**kw):
        w = walk(months, Settings("linear", **base, **kw))[0].pivot("symbol", index="date")
        return float(np.abs(np.diff(w.drop("date").to_numpy(), axis=0)).sum())

    assert traded(spread=0.05, short_fee=0.5) < traded()


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


def test_partial_trading_moves_holdings_part_way_from_last_month_at_the_learned_pace():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    assert float(Optimizer("linear", 2, partial=True).pace()) == pytest.approx(0.881, abs=1e-3)
    with pytest.raises(ValueError, match="partial"):
        Optimizer("linear", 2).pace()
    rng = np.random.default_rng(7)
    months = []
    for k in range(18):
        names = ["A", "B", "C", "D"] if k % 2 else ["A", "B", "C", "E"]  # D and E come and go
        x = rng.uniform(-0.5, 0.5, (4, 2)).astype(np.float32)
        r = (0.1 * x[:, 0] + rng.normal(0, 0.03, 4)).astype(np.float32)
        months.append(data.Month(date(2000 + k // 12, k % 12 + 1, 1), names, x, r))
    settings = Settings("linear", window=12, first_epochs=20, warm_epochs=2, lr=0.05,
                        start=date(2000, 1, 1), spread=0.01, partial=True)  # fmt: skip
    weights, log, model = walk(months, settings)
    a = float(model.pace())
    assert log["pace"].to_list()[-1] == pytest.approx(a)
    last, prev = months[-1], months[-2]
    held = dict(weights.filter(pl.col("date") == prev.date).select("symbol", "weight").rows())
    with torch.no_grad():
        target = model.weights(torch.from_numpy(last.x)).numpy()
    pairs = zip(last.symbols, target, strict=True)
    expected = [(1 - a) * held.get(s, 0.0) + a * t for s, t in pairs]
    got = weights.filter(pl.col("date") == last.date)["weight"].to_list()
    assert got == pytest.approx(expected, rel=1e-4, abs=1e-6)


def test_short_controls_keep_unshortable_stocks_and_no_shorts_at_or_above_zero():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    rng = np.random.default_rng(8)
    months = []
    for k in range(16):
        x = rng.uniform(-0.5, 0.5, (4, 2)).astype(np.float32)
        r = (0.1 * x[:, 0] + rng.normal(0, 0.03, 4)).astype(np.float32)  # input 0: long or short
        months.append(data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCD"), x, r))
    easy = np.array([0.99, 0.95, 0.5, 0.1], np.float32)
    stock = [(np.full(4, 0.001, np.float32), np.full(4, 0.003, np.float32), easy)] * 16
    base = dict(window=12, first_epochs=50, warm_epochs=3, lr=0.05, start=date(2000, 1, 1),
                costs="table.parquet", partial=True)  # fmt: skip
    with pytest.raises(ValueError, match="stock_costs"):
        walk(months, Settings("linear", short_min_easy=0.9, **base))
    w = walk(months, Settings("linear", short_min_easy=0.9, **base), stock_costs=stock)[0]
    assert (w.filter(pl.col("symbol").is_in(["C", "D"]))["weight"] >= 0).all()
    assert (w.filter(pl.col("symbol").is_in(["A", "B"]))["weight"] < 0).any()  # still shorts
    settings = Settings("linear", no_shorts=True, short_hurdle=1.0, **base)
    assert (walk(months, settings, stock_costs=stock)[0]["weight"] >= 0).all()


def test_fresh_retraining_makes_each_month_independent_of_the_others():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    rng = np.random.default_rng(9)
    months = []
    for k in range(18):
        x = rng.uniform(-0.5, 0.5, (5, 2)).astype(np.float32)
        r = (0.1 * x[:, 0] + rng.normal(0, 0.03, 5)).astype(np.float32)
        months.append(data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCDE"), x, r))
    base = dict(window=12, first_epochs=10, lr=0.01, start=date(2000, 1, 1), fresh=True)
    with pytest.raises(ValueError, match="fresh"):
        walk(months, Settings("mlp", every=2, **{**base, "fresh": False}))
    every = walk(months, Settings("mlp", every=2, **base))[0]
    full = walk(months, Settings("mlp", **base))[0]
    assert every["date"].unique().sort().to_list() == [m.date for m in months[12::2]]
    shared = full.filter(pl.col("date").is_in(every["date"].unique().implode()))
    assert shared["weight"].to_list() == pytest.approx(every["weight"].to_list(), rel=1e-5)


def test_std_init_keeps_rows_their_size_through_ten_blocks_and_variance_init_grows_them():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415

    x = torch.rand(500, 100) - 0.5
    size = float(x.norm(dim=1).mean())

    def grown(init):
        torch.manual_seed(0)
        y = x
        with torch.no_grad():
            for block in Optimizer("transformer", 100, 10, 256, init=init).blocks:
                y = block(y)
        return float(y.norm(dim=1).mean()) / size

    assert grown("std") == pytest.approx(1.0, abs=0.05)
    assert grown("variance") > 20
    with pytest.raises(ValueError, match="init"):
        Optimizer("linear", 3, init="var")


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

    s = wk.Settings("linear", lr=1e-2, schedule="cosine", lr_min=1e-4, warm_lr=1e-3)
    assert wk._lr(s, 1e-2, 0, 5) == pytest.approx(1e-2)
    assert wk._lr(s, 1e-2, 4, 5) == pytest.approx(1e-4)
    assert wk._start_lr(s, trained=False) == pytest.approx(1e-2)
    assert wk._start_lr(s, trained=True) == pytest.approx(1e-3)  # continuing last month's model
    assert wk._start_lr(wk.Settings("linear", lr=1e-2, warm_lr=1e-3, fresh=True), True) == 1e-2
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
