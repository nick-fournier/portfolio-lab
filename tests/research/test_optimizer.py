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


def test_temperature_makes_weights_depend_only_on_the_ranking_not_the_scores_size():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415

    model = Optimizer("linear", 3, long_only=True, cap=0.1, temperature=0.45)
    x = torch.rand(3000, 3) - 0.5
    w = model.weights(x)
    model.lam.data *= 1e6  # runaway scores: same portfolio
    assert torch.allclose(model.weights(x), w, atol=1e-6)
    assert 15 < float(1 / (w**2).sum()) < 60  # about 3000 exp(-1 / 0.45^2) = 25 names
    w.sum().backward()


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
