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


def test_the_paper_universe_drops_nano_caps_below_the_1st_percentile_of_nyse_values():
    months = data.months(_table())
    # NYSE values 10..15: the 1st percentile is 10.05, so S0 (10) is a nano cap
    assert months[0].symbols == ["S1", "S2", "S3", "S4", "S5"]


def test_universe_floors_drop_cheap_thin_and_listed_stocks_before_ranking():
    table = _table().with_columns(pl.Series("prc", [0.5, 3.0, 6.0, 9.0, 12.0, 15.0] * 2))
    plain = data.months(table, ["market_value"])[0]
    assert plain.symbols == ["S1", "S2", "S3", "S4", "S5"]
    floors = {"min_price": 5.0, "min_adv": 3.0, "drop": frozenset({"S4"})}
    cut = data.months(table, ["market_value"], **floors)[0]
    assert cut.symbols == ["S3", "S5"]  # S1 under $5, S2 thin, S4 listed out
    assert cut.x[:, 0].tolist() == pytest.approx([-0.25, 0.25])  # ranked among the two left
    assert data.months(table.with_columns(pl.lit(None, pl.Float64).alias("adv")), ["market_value"],
                       min_adv=1.0) == []  # a missing volume fails the floor  # fmt: skip


def test_the_three_models_give_one_weight_per_stock_and_the_paper_loss():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.model import MODELS, Optimizer, loss  # noqa: PLC0415

    x = torch.rand(7, 4) - 0.5
    r = torch.rand(7) * 0.1
    for kind in MODELS:
        model = Optimizer(kind, width=4, blocks=2, hidden=8)
        w = model(x)
        assert w.shape == (7,)
        assert float(loss(model, x, r)) == pytest.approx(float((1 - w @ r) ** 2), rel=1e-6)
    assert not any(b.attention for b in Optimizer("mlp", 4, 2, 8).blocks)
    assert len(Optimizer("linear", 4, 2, 8).blocks) == 0
    with pytest.raises(ValueError, match="unknown model"):
        Optimizer("cnn", 4)


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


def test_walk_trains_on_earlier_months_and_learns_to_favor_the_stocks_that_keep_winning():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    rng = np.random.default_rng(0)
    months = []
    for k in range(30):
        x = rng.uniform(-0.5, 0.5, (5, 2)).astype(np.float32)
        r = np.where(x[:, 0] > 0.2, 0.05, -0.01).astype(np.float32)  # input 0 predicts return
        months.append(data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCDE"), x, r))
    settings = Settings("linear", window=12, first_epochs=200, warm_epochs=5, lr=0.05,
                        start=date(2000, 1, 1))  # fmt: skip
    weights, log, _ = walk(months, settings)
    assert log["date"].to_list() == [m.date for m in months[12:]]  # needs 12 earlier months
    assert weights.height == 18 * 5
    assert log["ret"].mean() > 0.02  # it learned to hold the stocks with input 0 high
