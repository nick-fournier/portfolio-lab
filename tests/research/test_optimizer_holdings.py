"""Class 3 optimizer: carried holdings, holdings and spread inputs."""

from datetime import date

import numpy as np
import pytest

from portfolio_lab.research.optimizer import data


def test_holding_drifts_last_months_book_onto_this_months_stocks():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import costs, holdings  # noqa: PLC0415

    x = np.zeros((3, 1), np.float32)
    before = data.Month(date(2000, 1, 31), ["A", "B", "C"], x, np.zeros(3, np.float32))
    now = data.Month(date(2000, 2, 29), ["C", "D", "A"], x, np.zeros(3, np.float32))
    link = costs.links([before, now])[1]
    held = holdings.holding(torch.tensor([0.2, -0.1, 0.3]), torch.tensor([0.1, 0.0, -0.5]), link, 3)
    # C: 0.3 x 0.5, D: new, A: 0.2 x 1.1; B was sold; scaled to N h / sum |h|
    assert held.tolist() == pytest.approx([0.15 * 3 / 0.37, 0.0, 0.22 * 3 / 0.37])
    assert holdings.holding(None, torch.zeros(3), link, 3).tolist() == [0.0, 0.0, 0.0]


def test_grid_carry_and_holdings_input_score_held_out_months_from_the_carried_book():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import grid  # noqa: PLC0415

    rng = np.random.default_rng(17)
    pool = list("ABCDEFGHIJ")
    months, stock = [], []
    for k in range(20):
        names = [pool[(k + i) % 10] for i in range(8)]  # one stock leaves, one joins each month
        x = rng.uniform(-0.5, 0.5, (8, 3)).astype(np.float32)
        r = (0.1 * x[:, 0] + rng.normal(0, 0.03, 8)).astype(np.float32)
        months.append(data.Month(date(2000 + k // 12, k % 12 + 1, 1), names, x, r))
        stock.append((rng.uniform(0.001, 0.02, 8).astype(np.float32),
                      np.full(8, 0.05, np.float32), np.ones(8, np.float32)))  # fmt: skip
    base = dict(blocks=1, lr=1e-2, epochs=3, costs=True)
    setups = [grid.Config("costs", **base), grid.Config("carry", carry=True, **base),
              grid.Config("inputs", holdings_input=True, **base),
              grid.Config("both", carry=True, holdings_input=True, **base)]  # fmt: skip
    curves = grid.grid(months, [16, 18], setups, window=12, validation=3, device="cpu",
                       stock_costs=stock)  # fmt: skip
    measures = ["val_net_sharpe", "val_carry_sharpe", "val_net_ret", "val_gross_ret",
                "val_spread", "val_borrow", "val_turnover"]  # fmt: skip
    assert set(measures) <= set(curves.columns)
    assert curves["config"].unique().len() == 4
    assert all(np.isfinite(curves[c].to_numpy()).all() for c in measures)
    assert (curves["val_turnover"] > 0).all()
    assert (curves["val_net_ret"] <= curves["val_gross_ret"]).all()
    with pytest.raises(ValueError, match="carry"):
        grid.fit(months, 16, grid.Config("x", carry=True), window=12, validation=3, device="cpu")


def test_walk_holdings_input_feeds_the_real_book_to_the_test_month_and_needs_costs():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace  # noqa: PLC0415

    from portfolio_lab.research.optimizer import costs, holdings  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    rng = np.random.default_rng(18)
    pool = list("ABCDEF")
    months = [
        data.Month(
            date(2000 + k // 12, k % 12 + 1, 1),
            [pool[(k + i) % 6] for i in range(4)],  # one stock leaves, one joins each month
            rng.uniform(-0.5, 0.5, (4, 2)).astype(np.float32),
            rng.normal(0.01, 0.05, 4).astype(np.float32),
        )
        for k in range(16)
    ]
    stock = [(rng.uniform(0.001, 0.02, 4).astype(np.float32), np.full(4, 0.05, np.float32))] * 16
    settings = Settings("transformer", window=12, first_epochs=3, warm_epochs=1,
                        start=date(2000, 1, 1), costs="table.parquet",
                        holdings_input=True)  # fmt: skip
    weights, log, model = walk(months, settings, stock_costs=stock)
    assert log.height == 4 and weights["weight"].is_finite().all()
    assert model(torch.zeros(4, 4)).shape == (4,)  # 2 traits + holding + spread
    with pytest.raises(ValueError, match="holdings_input"):
        walk(months, Settings("linear", window=12, start=date(2000, 1, 1), holdings_input=True))
    with pytest.raises(ValueError, match="borrow_input"):
        walk(months, Settings("linear", window=12, start=date(2000, 1, 1), costs="t.parquet",
                              borrow_input=True), stock_costs=stock)  # fmt: skip
    _, _, fed = walk(months, Settings("transformer", window=12, first_epochs=3, warm_epochs=1,
                                      start=date(2000, 1, 1), costs="table.parquet",
                                      holdings_input=True, borrow_input=True),
                     stock_costs=stock)  # fmt: skip
    assert fed(torch.zeros(4, 5)).shape == (4,)  # + borrow fee
    x = holdings.with_holdings(tensors := [(torch.from_numpy(m.x), torch.from_numpy(m.r))
                                            for m in months],
                               costs.links(months),
                               [(torch.from_numpy(s), torch.from_numpy(f)) for s, f in stock],
                               {}, 3, borrow=True)  # fmt: skip
    assert x.shape == (4, 5) and x[:, 4].tolist() == pytest.approx([np.log(0.05 / 0.03)] * 4)
    # the test month's holding column is last month's real book after its returns
    spreads = [(torch.from_numpy(s), torch.from_numpy(f)) for s, f in stock]
    pricing = SimpleNamespace(links=costs.links(months), book={}, stock=spreads)
    last = torch.tensor([0.3, -0.2, 0.1, 0.4])
    got = holdings.pick(lambda x: x[:, 2], tensors, pricing, last, 13)
    want = holdings.holding(last, tensors[12][1], pricing.links[13], 4)
    assert got.tolist() == pytest.approx(want.tolist())
