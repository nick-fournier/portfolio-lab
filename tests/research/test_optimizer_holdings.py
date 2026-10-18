"""Class 3 optimizer: holdings, spread and borrow fee inputs."""

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
    pricing = SimpleNamespace(links=costs.links(months), book={}, stock=spreads, borrow=False)
    last = torch.tensor([0.3, -0.2, 0.1, 0.4])
    got = holdings.pick(lambda x: x[:, 2], tensors, pricing, last, 13)
    want = holdings.holding(last, tensors[12][1], pricing.links[13], 4)
    assert got.tolist() == pytest.approx(want.tolist())


def test_the_extra_columns_can_drop_the_holding_rank_the_spread_and_flag_hard_borrows():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import costs, holdings  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    x = np.zeros((4, 2), np.float32)
    months = [data.Month(date(2000, k + 1, 1), list("ABCD"), x, np.full(4, 0.01, np.float32))
              for k in range(3)]  # fmt: skip
    tensors = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    spread = torch.tensor([0.004, 0.001, 0.02, 0.008])
    fee = torch.tensor([0.0, 12.0, 0.0, 0.0])  # B is hard to borrow
    stock = [(spread, fee)] * 3
    book = {1: torch.tensor([0.5, -0.5, 0.0, 0.0])}
    args = (tensors, costs.links(months), stock, book, 2, True)
    logs = holdings.with_holdings(*args)
    assert logs.shape == (4, 5)  # 2 traits + holding + log spread + log fee
    assert torch.equal(holdings.with_holdings(*args, holdings.LOGS), logs)  # the default
    same = holdings.with_holdings(*args, holdings.Columns(holding=False))
    assert same.shape == (4, 4) and torch.equal(same, logs[:, [0, 1, 3, 4]])  # holding left out
    ranked = holdings.with_holdings(*args, holdings.Columns(spread_rank=True))
    assert ranked[:, 3].tolist() == pytest.approx(
        [-0.125, -0.375, 0.375, 0.125]
    )  # 2nd, 1st, 4th, 3rd
    held = holdings.with_holdings(*args, holdings.Columns(holding_rank=True))
    # holding after returns: A long, B short, C and D not held (tied in the middle)
    assert held[:, 2].tolist() == pytest.approx([0.375, -0.375, 0.0, 0.0])
    assert torch.equal(held[:, [0, 1, 3, 4]], logs[:, [0, 1, 3, 4]])
    ties = holdings.with_holdings(tensors, costs.links(months),
                                  [(torch.tensor([0.004, 0.001, 0.004, 0.004]), fee)] * 3,
                                  book, 2, True, holdings.Columns(spread_rank=True))  # fmt: skip
    assert ties[:, 3].tolist() == pytest.approx([0.125, -0.375, 0.125, 0.125])  # ties share
    flagged = holdings.with_holdings(*args, holdings.Columns(borrow_flag=True))
    assert flagged[:, 4].tolist() == pytest.approx([-0.25, 0.75, -0.25, -0.25])  # centred flag
    assert torch.equal(flagged[:, :4], logs[:, :4])
    # the walk builds the model one column narrower without the holding column, and checks
    rng = np.random.default_rng(3)
    walk_months = [
        data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCD"),
                   rng.uniform(-0.5, 0.5, (4, 2)).astype(np.float32),
                   rng.normal(0.01, 0.05, 4).astype(np.float32))
        for k in range(14)
    ]  # fmt: skip
    costs_ = [(rng.uniform(0.001, 0.02, 4).astype(np.float32), np.full(4, 0.05, np.float32))] * 14
    base = Settings("linear", window=12, first_epochs=2, start=date(2000, 1, 1),
                    costs="t.parquet", holdings_input=True, holding_column=False)  # fmt: skip
    _, _, model = walk(walk_months, base, stock_costs=costs_)
    assert model(torch.zeros(4, 3)).shape == (4,)  # 2 traits + spread
    with pytest.raises(ValueError, match="borrow_flag"):
        walk(walk_months, Settings("linear", window=12, start=date(2000, 1, 1),
                                   costs="t.parquet", holdings_input=True, borrow_flag=True),
             stock_costs=costs_)  # fmt: skip
    with pytest.raises(ValueError, match="spread_rank"):
        walk(walk_months, Settings("linear", window=12, start=date(2000, 1, 1),
                                   costs="t.parquet", spread_rank=True),
             stock_costs=costs_)  # fmt: skip
