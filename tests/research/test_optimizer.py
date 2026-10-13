from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.research.optimizer import data


def test_ranks_fill_minus_half_to_half_with_ties_averaged_and_missing_in_the_middle():
    x = np.array([[1.0], [2.0], [2.0], [np.nan], [5.0]])
    ranked = data._ranked(x)[:, 0]
    # four values: ranks 1, 2.5, 2.5, 4 -> (rank - 0.5) / 4 - 0.5
    assert ranked.tolist() == pytest.approx([-0.375, 0.0, 0.0, 0.0, 0.375])


def _table() -> pl.DataFrame:
    rows = []
    for day in (date(2020, 1, 31), date(2020, 2, 28)):
        for k in range(6):
            rows.append({"date": day, "symbol": f"S{k}", "sic2": 10, "market_value": 10.0 + k,
                         "nyse": True, "eligible": k != 5, "adv": float(k),
                         "actual": 0.01 * k})  # fmt: skip
    return pl.DataFrame(rows)


def test_months_take_the_most_liquid_eligible_stocks_and_rank_among_them():
    months = data.months(_table(), pool=3)
    assert [m.date for m in months] == [date(2020, 1, 31), date(2020, 2, 28)]
    first = months[0]
    assert first.symbols == ["S4", "S3", "S2"]  # S5 is ineligible
    assert first.x[:, 0].tolist() == pytest.approx([1 / 3, 0.0, -1 / 3])  # market_value
    assert first.r.tolist() == pytest.approx([0.04, 0.03, 0.02])


def test_the_paper_universe_drops_nano_caps_below_the_1st_percentile_of_nyse_values():
    months = data.months(_table())
    # NYSE values 10..15: the 1st percentile is 10.05, so S0 (10) is a nano cap
    assert months[0].symbols == ["S1", "S2", "S3", "S4", "S5"]


def test_the_three_models_give_weights_that_are_positive_and_sum_to_one():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.model import MODELS, Optimizer, loss  # noqa: PLC0415

    x = torch.rand(7, 4) - 0.5
    r = torch.rand(7) * 0.1
    for kind in MODELS:
        model = Optimizer(kind, width=4, blocks=2, hidden=8, long_only=True)
        w = model.weights(x)
        assert w.shape == (7,)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
        assert bool((w > 0).all())
        assert float(loss(model, x, r)) == pytest.approx(float((1 - w @ r) ** 2), rel=1e-6)
    assert not any(b.attention for b in Optimizer("mlp", 4, 2, 8).blocks)
    assert len(Optimizer("linear", 4, 2, 8).blocks) == 0


def test_capped_softmax_holds_every_weight_at_or_under_the_cap_and_keeps_proportions():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.model import Optimizer, capped_softmax  # noqa: PLC0415

    scores = torch.tensor([5.0, 4.0, 1.0, 0.5, 0.0, 0.0, -1.0, -2.0, -3.0, -3.0, -4.0, -5.0])
    w = capped_softmax(scores, 0.2)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(w.max()) == pytest.approx(0.2, abs=1e-6)
    free = torch.softmax(scores[w < 0.2 - 1e-6], dim=0)  # the uncapped keep their proportions
    assert torch.allclose(w[w < 0.2 - 1e-6] / w[w < 0.2 - 1e-6].sum(), free, atol=1e-6)
    assert torch.equal(capped_softmax(torch.zeros(10), 0.2), torch.full((10,), 0.1))
    model = Optimizer("mlp", 3, 1, 8, long_only=True, cap=0.1)
    x = torch.rand(40, 3) - 0.5
    model.lam.data *= 100  # extreme scores: an uncapped softmax would hold one stock
    w = model.weights(x)
    assert float(w.max()) <= 0.1 + 1e-6 and float(w.sum()) == pytest.approx(1.0, abs=1e-5)
    (w @ torch.rand(40)).backward()  # trains through the cap
    assert model.lam.grad is not None and bool(model.lam.grad.abs().sum() > 0)


def test_names_fix_the_concentration_whatever_the_scores_size():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415

    torch.manual_seed(0)
    model = Optimizer("linear", 3, long_only=True, cap=0.1, names=25)
    x = torch.rand(3000, 3) - 0.5
    w = model.weights(x)
    model.lam.data *= 1e6  # runaway scores: same portfolio
    assert torch.allclose(model.weights(x), w, atol=1e-6)
    assert 20 < float(1 / (w**2).sum()) < 35  # near 25; the cap shifts it a little
    w.sum().backward()


def test_objectives_prefer_the_steadier_of_two_equal_return_streams():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import objectives  # noqa: PLC0415

    steady = torch.tensor([0.01, 0.012, 0.008, 0.011, 0.009, 0.01])
    wild = torch.tensor([0.10, -0.08, 0.09, -0.07, 0.08, -0.06])  # same mean, deep losses
    for name in objectives.OBJECTIVES:
        assert objectives.grade(name, steady) < objectives.grade(name, wild), name
    with pytest.raises(ValueError, match="unknown objective"):
        objectives.grade("nope", steady)


def test_batch_steps_cover_every_training_month_once_in_consecutive_blocks():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, _steps  # noqa: PLC0415

    train = list(range(10, 70))
    steps = _steps(train, Settings("linear", objective="sharpe", batch=12),
                   np.random.default_rng(3))  # fmt: skip
    covered = sorted(k for s in steps for k in s)
    assert set(covered) <= set(train) and len(covered) == len(set(covered)) >= 55
    assert all(s == list(range(s[0], s[0] + len(s))) and 3 <= len(s) <= 12 for s in steps)
    paper = _steps(train, Settings("linear"), np.random.default_rng(3))
    assert sorted(k for s in paper for k in s) == train


def test_fast_attention_matches_the_written_out_formula_to_bfloat16_precision():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.model import Block  # noqa: PLC0415

    block = Block(width=4, hidden=8, attention=True)
    y = torch.rand(9, 4) - 0.5
    written = y + block.peers(y) @ (y @ block.v)
    written = written + torch.relu(written @ block.w1 + block.b1) @ block.w2 + block.b2
    assert torch.allclose(block(y), written, atol=1e-6)
    block.fast = True
    assert torch.allclose(block(y), written, atol=2e-2)
    scaled = Block(width=4, hidden=8, attention=True, scaled=True)
    logits = (y @ scaled.q) @ (y @ scaled.k).T / 2  # sqrt(4)
    assert torch.allclose(scaled.peers(y), torch.softmax(logits, dim=-1), atol=1e-6)
    written = y + scaled.peers(y) @ (y @ scaled.v)
    written = written + torch.relu(written @ scaled.w1 + scaled.b1) @ scaled.w2 + scaled.b2
    scaled.fast = True
    assert torch.allclose(scaled(y), written, atol=2e-2)


def test_walk_trains_on_earlier_months_and_learns_to_favor_the_stock_that_keeps_winning():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    rng = np.random.default_rng(0)
    months = []
    for k in range(30):
        x = rng.uniform(-0.5, 0.5, (5, 2)).astype(np.float32)
        r = np.where(x[:, 0] > 0.2, 0.05, -0.01).astype(np.float32)  # input 0 predicts return
        months.append(data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCDE"), x, r))
    settings = Settings("linear", window=12, first_epochs=200, warm_epochs=5, lr=0.05,
                        start=date(2000, 1, 1), long_only=True)  # fmt: skip
    weights, log, _ = walk(months, settings)
    assert log["date"].to_list() == [m.date for m in months[12:]]  # needs 12 earlier months
    assert weights.group_by("date").agg(pl.col("weight").sum())["weight"].to_list() == (
        pytest.approx([1.0] * 18)
    )
    assert log["ret"].mean() > 0.02  # it learned to hold the stocks with input 0 high


def test_scaled_long_only_weights_are_m_times_a_mix_and_the_walk_saves_the_mix():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    model = Optimizer("mlp", 3, 1, 8, long_only=True, scale=True)
    with torch.no_grad():
        model.log_m.fill_(np.log(0.5))
    w = model.weights(torch.rand(6, 3) - 0.5)
    assert float(w.sum()) == pytest.approx(0.5, abs=1e-6)
    assert bool((w > 0).all())
    rng = np.random.default_rng(1)
    months = [
        data.Month(
            date(2000 + k // 12, k % 12 + 1, 1),
            list("ABCDE"),
            rng.uniform(-0.5, 0.5, (5, 2)).astype(np.float32),
            rng.normal(0.01, 0.05, 5).astype(np.float32),
        )
        for k in range(16)
    ]
    settings = Settings(
        "linear",
        window=12,
        first_epochs=20,
        warm_epochs=2,
        lr=0.05,
        start=date(2000, 1, 1),
        long_only=True,
        scale=True,
    )
    weights, log, _ = walk(months, settings)
    sums = weights.group_by("date").agg(pl.col("weight").sum())["weight"].to_list()
    assert sums == pytest.approx([1.0] * 4)
    assert (log["m"] > 0).all()


def test_softplus_long_only_weights_are_positive_scores_over_n_and_the_walk_saves_the_mix():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    model = Optimizer("linear", 2, long_only=True, softplus=True)
    with torch.no_grad():
        model.lam.copy_(torch.tensor([1.0, 0.0]))
    w = model.weights(torch.tensor([[5.0, 0.0], [-5.0, 0.0], [0.0, 0.0]]))
    expected = [np.log1p(np.exp(5.0)) / 3, np.log1p(np.exp(-5.0)) / 3, np.log(2.0) / 3]
    assert w.tolist() == pytest.approx(expected, rel=1e-5)
    rng = np.random.default_rng(2)
    months = []
    for k in range(30):
        x = rng.uniform(-0.5, 0.5, (5, 2)).astype(np.float32)
        r = np.where(x[:, 0] > 0.2, 0.05, -0.01).astype(np.float32)  # input 0 predicts return
        months.append(data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCDE"), x, r))
    settings = Settings("linear", window=12, first_epochs=200, warm_epochs=5, lr=0.05,
                        start=date(2000, 1, 1), long_only=True, softplus=True)  # fmt: skip
    weights, log, _ = walk(months, settings)
    assert (weights["weight"] > 0).all()
    sums = weights.group_by("date").agg(pl.col("weight").sum())["weight"].to_list()
    assert sums == pytest.approx([1.0] * 18)  # the mix; the size is logged as m
    assert log["ret"].mean() > 0.02  # it learned to hold the stocks with input 0 high


def test_subset_inference_runs_the_trained_model_on_the_subset_rows_alone():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    rng = np.random.default_rng(3)
    months = []
    for k in range(16):
        x = rng.uniform(-0.5, 0.5, (5, 2)).astype(np.float32)
        r = rng.normal(0.01, 0.05, 5).astype(np.float32)
        months.append(data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCDE"), x, r))
    settings = Settings("transformer", window=12, first_epochs=5, warm_epochs=1,
                        start=date(2000, 1, 1))  # fmt: skip
    subsets = {m.date: {"A", "C"} for m in months}
    weights, log, model = walk(months, settings, subsets=subsets)
    last = weights.filter(pl.col("date") == months[-1].date)
    assert last.filter(pl.col("symbol").is_in(["B", "D", "E"]))["subset_weight"].is_null().all()
    with torch.no_grad():  # the final month: the model on rows A and C only
        expected = model.weights(torch.from_numpy(months[-1].x[[0, 2]])).numpy()
    got = last.filter(pl.col("symbol").is_in(["A", "C"]))["subset_weight"].to_numpy()
    assert got == pytest.approx(expected, rel=1e-5)
    assert (log["stocks_subset"] == 2).all()


def test_subset_training_only_ignores_other_rows_and_finetune_leaves_the_main_model_alone():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    rng = np.random.default_rng(4)
    months = []
    for k in range(16):
        x = rng.uniform(-0.5, 0.5, (5, 2)).astype(np.float32)
        r = rng.normal(0.01, 0.05, 5).astype(np.float32)
        months.append(data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCDE"), x, r))
    subsets = {m.date: {"A", "C", "D"} for m in months}
    base = dict(window=12, first_epochs=5, warm_epochs=2, start=date(2000, 1, 1))

    def run(months, **kw):
        return walk(months, Settings("mlp", **base, **kw), subsets=subsets)[0]

    plain, tuned = run(months), run(months, subset_train="finetune")
    assert plain["weight"].to_list() == tuned["weight"].to_list()  # main model unchanged
    assert plain["subset_weight"].to_list() != tuned["subset_weight"].to_list()
    other = [  # returns of stocks outside the subset changed: "only" never sees them
        data.Month(m.date, m.symbols, m.x, np.where(np.isin(m.symbols, ["B", "E"]), 0.5, m.r))
        for m in months
    ]
    only = run(months, subset_train="only")["subset_weight"].to_numpy()
    assert run(other, subset_train="only")["subset_weight"].to_numpy() == pytest.approx(
        only, nan_ok=True
    )


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
