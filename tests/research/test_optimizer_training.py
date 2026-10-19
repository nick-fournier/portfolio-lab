"""Class 3 optimizer: costs in the loss, init, schedules, held-out scoring, seed averaging."""

import math
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

    month = data.Month(date(2000, 1, 31), ["C", "A", "B", "D"], np.zeros((4, 1)), np.zeros(4))
    table = pl.DataFrame({"date": [date(2000, 1, 31)] * 3, "symbol": ["A", "B", "D"],
                          "trade_cost": [0.01, 0.03, None],
                          "borrow_fee": [0.003, 0.5, float("nan")]})  # fmt: skip
    ((trade, fee, easy),) = costs.align([month], table)
    assert trade.tolist() == pytest.approx([0.02, 0.01, 0.03, 0.02])  # C, D: the month's median
    assert costs.HARD_TO_BORROW == 12.0  # the cost table's hard-to-borrow fee, 1,200% a year
    assert fee.tolist() == pytest.approx([12.0, 0.003, 0.5, 12.0])  # C, D: hard to borrow
    ((_, no_fees, _),) = costs.align([month], table.drop("borrow_fee"))
    assert no_fees.tolist() == [12.0] * 4  # a table without fees: every stock
    later = data.Month(date(2000, 2, 29), ["A"], np.zeros((1, 1)), np.zeros(1))
    ((trade_later, fee_later, easy_later),) = costs.align([later], table)  # a month it lacks
    assert trade_later.tolist() == pytest.approx([0.005])
    assert fee_later.tolist() == [12.0] and easy_later.tolist() == [0.0]
    assert easy.tolist() == [1.0] * 4  # no easy_chance column: every stock of the month
    table = table.with_columns(pl.Series("easy_chance", [0.95, 0.4, None]))
    ((_, _, easy),) = costs.align([month], table)
    assert easy.tolist() == pytest.approx([0.0, 0.95, 0.4, 0.0])  # C missing, D without: 0
    ((_, _, easy_later),) = costs.align([later], table)
    assert easy_later.tolist() == [0.0]


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
    from portfolio_lab.research.optimizer.seedavg import average as _average  # noqa: PLC0415

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


def test_fit_watched_runs_every_pass_and_logs_the_best_smoothed_score():
    from portfolio_lab.research.optimizer import stopping  # noqa: PLC0415

    # raw scores: the dip at pass 2 must not win; means of 3 bottom at pass 4 (2.0, tied at 5)
    raw = [9.0, 0.5, 3.0, 2.5, 0.5, 4.0, 3.0, 3.0, 3.0]
    scores = iter(raw)
    calls = []
    losses, passes, best, curve = stopping.fit_watched(
        lambda p: calls.append(p) or [float(p)], lambda: next(scores), True, len(raw)
    )
    assert calls == list(range(9)) and passes == 9 and losses == [8.0]  # the last pass is kept
    assert best == pytest.approx(2.0) and curve == raw
    assert stopping.best_smoothed([1.0, math.nan, 1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert stopping.best_smoothed([1.0, 1.0]) == math.inf  # no full window
    calls = []
    losses, passes, best, curve = stopping.fit_watched(
        lambda p: calls.append(p) or [0.0], lambda: 0.0, False, 4
    )
    assert calls == [0, 1, 2, 3] and passes == 4 and np.isnan(best) and curve == []


def test_keeper_keeps_the_best_smoothed_pass_and_stops_on_patience():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import stopping  # noqa: PLC0415
    from portfolio_lab.research.optimizer.averaging import Average  # noqa: PLC0415

    model = torch.nn.Linear(2, 1)
    adam = torch.optim.Adam(model.parameters())
    average = Average(model, 0.5)

    def one_pass(p):  # pass p leaves the weights at p and folds them into the average
        with torch.no_grad():
            model.weight.fill_(float(p))
        average.update()
        return [float(p)]

    # smoothed over 3: 4.0 at pass 2, 3.0 at 3, 2.67 at 4 (best), 3.0 at 5, 4.0 at 6, 5.0 at 7
    raw = iter([5.0, 4.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    keeper = stopping.Keeper(model, adam, average, patience=3)
    losses, passes, best, curve = stopping.fit_watched(
        one_pass, lambda: next(raw), True, 10, keeper
    )
    assert passes == 8 and keeper.best_pass == 4 and best == pytest.approx(8 / 3)  # 4 + 3
    assert losses == [7.0] and len(curve) == 8  # the last pass run was 7; its loss is logged
    assert float(model.weight.detach()[0, 0]) == 4.0 and average.steps == 5  # pass 4's state
    keeper = stopping.Keeper(model, adam, None, patience=2)  # no finite score: nothing kept
    _, passes, _, _ = stopping.fit_watched(one_pass, lambda: math.nan, True, 3, keeper)
    assert passes == 3 and keeper.best_pass == -1 and float(model.weight.detach()[0, 0]) == 2.0
    a, b = torch.tensor([0.1, 0.5, 0.3, 0.9]), torch.tensor([1.0, 3.0, 2.0, 4.0])
    assert float(stopping.rank_corr(a, b)) == pytest.approx(1.0)
    assert float(stopping.rank_corr(a, -b)) == pytest.approx(-1.0)
    assert abs(float(stopping.rank_corr(a, torch.tensor([1.0, 4.0, 3.0, 2.0])))) < 1.0
    # every 3rd month end held out for good: fixed by position, the random stream untouched
    rng = np.random.default_rng(5)
    fit, block = stopping.held_months(list(range(7, 20)), 12, rng, scatter=True, every=3)
    assert block == [9, 12, 15, 18] and fit == [k for k in range(7, 20) if k % 3]
    assert rng.integers(1000) == np.random.default_rng(5).integers(1000)


def test_walk_logs_passes_gradients_and_with_watch_the_held_out_curve():
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
    _, plain, _ = walk(months, Settings("linear", window=20, first_epochs=8, warm_epochs=8,
                                        start=date(2001, 9, 1)))  # fmt: skip
    assert (plain["passes"] == 8).all() and plain["held_loss"].is_nan().all()
    assert (plain["grad_max"] >= plain["grad_median"]).all() and (plain["clipped"] <= 1).all()
    _, big, _ = walk(months, Settings("linear", window=20, first_epochs=8, warm_epochs=8,
                                      start=date(2001, 9, 1), clip=1e-9))  # fmt: skip
    assert (big["clipped"] == 1.0).all()  # every step's gradient is over a tiny limit
    watched = Settings("linear", window=20, first_epochs=6, warm_epochs=2,
                       start=date(2001, 9, 1), costs="table.parquet", holdings_input=True,
                       holdout=6, cold=True, watch=True)  # fmt: skip
    _, cold, _ = walk(months, watched, stock_costs=stock)
    assert (cold["passes"] == 6).all()  # cold: every month gets first_epochs, all run
    assert all(len(c) == 6 for c in cold["held_curve"])
    assert all(len(c) == 6 for c in cold["test_curve"].drop_nulls())
    assert cold["held_loss"].is_finite().all()
    kept = Settings("linear", window=20, first_epochs=6, warm_epochs=2, start=date(2001, 9, 1),
                    costs="table.parquet", holdings_input=True, holdout=6, cold=True,
                    watch=True, keep_best=True, keep_patience=2, average=0.9)  # fmt: skip
    _, best, _ = walk(months, kept, stock_costs=stock)
    assert (best["passes"] <= 6).all() and (best["best_pass"] < best["passes"]).all()
    assert (best["best_pass"] >= 0).all() and (best["passes"] < 6).any()  # stopped early somewhere
    assert all(-1 <= v <= 1 for c in best["held_curve"] for v in c)  # minus the held-out ranking
    with pytest.raises(ValueError, match="keep_best needs watch"):
        walk(months, Settings("linear", window=20, first_epochs=2, warm_epochs=2,
                              start=date(2001, 9, 1), keep_best=True))  # fmt: skip


def test_auto_clipping_sets_the_bar_once_from_the_first_steps_and_holds_it():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.clipping import Clipper  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    w = torch.nn.Parameter(torch.zeros(4))
    clipper = Clipper("auto", percentile=90, calibrate=10)
    w.grad = torch.full((4,), 500.0)  # a start-up transient before arming: not recorded
    clipper([w])
    clipper.arm()
    sizes = [1.0] * 9 + [2.0] + [1000.0, 1.0, 300.0]
    out = []
    for size in sizes:
        w.grad = torch.full((4,), size / 2)  # gradient norm = size
        out.append(clipper([w]))
        out[-1] = (float(out[-1][0]), float(out[-1][1]), float(w.grad.norm()))
    assert out[5] == (1.0, float("inf"), 1.0)  # calibrating: no limit
    assert clipper.fixed == 1.0  # the 90th percentile of the 10 armed sizes, then frozen
    assert out[10][0] == 1000.0 and out[10][1] == 1.0 and out[10][2] == pytest.approx(1.0)
    assert out[12][0] == 300.0 and out[12][1] == 1.0  # later sizes never move the bar
    with pytest.raises(ValueError, match="clip"):
        Clipper(-1)
    with pytest.raises(ValueError, match="clip"):
        Clipper("max")
    rng = np.random.default_rng(20)
    months = [
        data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCD"),
                   rng.uniform(-0.5, 0.5, (4, 2)).astype(np.float32),
                   rng.normal(0.01, 0.05, 4).astype(np.float32))
        for k in range(16)
    ]  # fmt: skip
    _, log, _ = walk(months, Settings("linear", window=12, first_epochs=5, warm_epochs=50,
                                      start=date(2000, 1, 1), clip="auto"))  # fmt: skip
    assert log["limit"][0] == float("inf")  # the cold start is not clipped
    assert log["limit"][1:].is_finite().all() and log["limit"][1:].n_unique() == 1
    assert (log["clipped"] <= 0.5).all()


def test_parameter_average_is_bias_corrected_swaps_in_and_rolls_back_with_the_guard():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import stopping  # noqa: PLC0415
    from portfolio_lab.research.optimizer.averaging import Average  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(5.0)
    avg = Average(model, 0.5)
    avg.update()
    assert avg.values()[0].item() == pytest.approx(5.0)  # one step: the value itself
    with torch.no_grad():
        model.weight.fill_(1.0)
    avg.update()  # (0.5 x 5 + 0.5 x 1) / (1 - 0.25) scaled from the zero start
    assert avg.values()[0].item() == pytest.approx((0.25 * 5 + 0.5 * 1) / 0.75)
    with avg.swapped():
        assert model.weight.item() == pytest.approx((0.25 * 5 + 0.5 * 1) / 0.75)
    assert model.weight.item() == 1.0
    with pytest.raises(ValueError, match="decay"):
        Average(model, 1.0)
    adam = torch.optim.Adam(model.parameters())
    guard = stopping.Guard(model, adam, avg)
    kept = avg.values()[0].item()

    def blow_up(p):
        with torch.no_grad():
            model.weight.fill_(math.nan)
        avg.update()
        return [0.0]

    assert guard(blow_up, 0) == [] and guard.rollbacks == 1
    assert model.weight.item() == 1.0 and avg.steps == 2  # both rolled back
    assert avg.values()[0].item() == pytest.approx(kept)
    rng = np.random.default_rng(21)
    months = [
        data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCD"),
                   rng.uniform(-0.5, 0.5, (4, 2)).astype(np.float32),
                   rng.normal(0.01, 0.05, 4).astype(np.float32))
        for k in range(16)
    ]  # fmt: skip
    base = dict(window=12, first_epochs=10, warm_epochs=2, start=date(2000, 1, 1), seed=3)
    held, _, _ = walk(months, Settings("linear", **base, average=0.9))
    latest, _, _ = walk(months, Settings("linear", **base))
    assert held["weight"].is_finite().all()
    assert held["weight"].to_list() != latest["weight"].to_list()


def test_unit_start_scales_a_fresh_model_so_its_first_book_has_gross_one():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import walkbits  # noqa: PLC0415
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    rng = np.random.default_rng(3)
    months = [data.Month(date(2000, k + 1, 1), list("ABCDEF"),
                         rng.uniform(-0.5, 0.5, (6, 4)).astype(np.float32),
                         rng.normal(0, 0.05, 6).astype(np.float32)) for k in range(6)]  # fmt: skip
    pairs = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    torch.manual_seed(0)
    model = Optimizer("mlp", 4, blocks=2, hidden=8)
    before = float(model(pairs[3][0]).abs().sum())
    gross = walkbits.unit_start(model, pairs, None, months, Settings("mlp", start=date(2000, 5, 1)))
    assert gross == pytest.approx(before)  # the month before the first traded month (May): April
    assert float(model(pairs[3][0]).abs().sum()) == pytest.approx(1.0, rel=1e-5)


def test_fixed_readout_holds_the_readouts_length_through_training():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    rng = np.random.default_rng(11)
    months = [data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCDEFGH"),
                         rng.uniform(-0.5, 0.5, (8, 3)).astype(np.float32),
                         rng.normal(0.01, 0.05, 8).astype(np.float32))
              for k in range(16)]  # fmt: skip
    base = dict(window=12, first_epochs=4, warm_epochs=2, lr=0.05, start=date(2001, 1, 1))
    _, _, held = walk(months, Settings("mlp", blocks=1, hidden=8, fixed_readout=True, **base))
    assert float(held.lam.detach().norm()) == pytest.approx(held.readout_length, rel=1e-5)
    _, _, free = walk(months, Settings("mlp", blocks=1, hidden=8, **base))
    assert abs(float(free.lam.detach().norm()) - free.readout_length) > 1e-3  # it moves without
    assert torch.isfinite(held.lam).all()


def test_normalize_bounds_the_raw_weights_however_large_the_block_weights_grow():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415

    torch.manual_seed(2)
    x = torch.rand(40, 6) - 0.5
    plain, normed = Optimizer("transformer", 6, 3, 16), Optimizer("transformer", 6, 3, 16,
                                                                  normalize=True)  # fmt: skip
    normed.load_state_dict(plain.state_dict())
    assert not torch.equal(plain(x), normed(x))
    with torch.no_grad():
        for m in (plain, normed):
            for name, q in m.named_parameters():
                if name != "lam":
                    q.mul_(30.0)  # block weights grown 30x
    # |norm(y) . lam| <= |norm(y)| |lam|, and |norm(y)| = sqrt(D)
    bound = float(normed.lam.norm()) * 6**0.5
    assert float(normed(x).abs().max()) <= bound * (1 + 1e-4)
    assert float(plain(x).abs().max()) > 100 * bound  # without it the weights grow with the blocks
