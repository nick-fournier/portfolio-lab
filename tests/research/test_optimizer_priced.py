"""Class 3 optimizer: the book each training month trades from (research.optimizer.priced)."""

from datetime import date

import numpy as np
import pytest

from portfolio_lab.research.optimizer import data


def _months(n=6, seed=4):
    rng = np.random.default_rng(seed)
    return [
        data.Month(date(2000 + k // 12, k % 12 + 1, 1), list("ABCD"),
                   rng.uniform(-0.5, 0.5, (4, 2)).astype(np.float32),
                   rng.normal(0.01, 0.1, 4).astype(np.float32))
        for k in range(n)
    ]  # fmt: skip


def test_a_month_trades_from_the_stored_book_after_its_returns():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import costs, priced  # noqa: PLC0415
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    months = _months()
    stock = [(np.full(4, 0.01, np.float32), np.full(4, 0.05, np.float32))] * len(months)
    s = Settings("linear", costs="t.parquet", holdings_input=True)
    p = priced.pricing(months, s, "cpu", stock)
    pairs = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    model = Optimizer("linear", 2 + 2)
    r1 = pairs[1][1]
    p.book[1] = torch.tensor([1.0, -1.0, 0.5, 0.0])
    assert p.held().get(1) is p.book[1] and p.held().get(3) is None  # a view of the table
    assert not hasattr(p.held(), "__setitem__")  # read-only
    held = priced.held_going_in(model, pairs, p, 2)
    assert torch.allclose(held, p.book[1] * (1 + r1))  # the stored book, grown by its returns
    assert priced.held_going_in(model, pairs, p, 0) is None  # the first month starts in cash
    w = torch.tensor([2.5, 0.0, 0.0, -1.0])
    turnover = (w - p.book[1] * (1 + r1)).abs().sum()  # same stocks both months
    expect = w @ pairs[2][1] - 0.01 * turnover - 0.05 / 12 * 1.0
    assert float(priced.net(model, pairs, p, s, 2, w)) == pytest.approx(float(expect), rel=1e-5)
    other = {1: torch.tensor([2.0, 0.0, 0.0, -2.0])}  # any book with get(month) can stand in
    assert torch.allclose(priced.held_going_in(model, pairs, p, 2, other), other[1] * (1 + r1))
    assert costs.carry(other[1], p.links[2]).tolist() == other[1].tolist()


def test_the_loss_stores_each_months_book_and_trades_from_the_month_before():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import holdings, priced  # noqa: PLC0415
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    months = _months()
    stock = [(np.full(4, 0.01, np.float32), np.full(4, 0.05, np.float32))] * len(months)
    s = Settings("linear", costs="t.parquet", holdings_input=True)
    p = priced.pricing(months, s, "cpu", stock)
    pairs = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    torch.manual_seed(0)
    model = Optimizer("linear", 2 + 2)
    p.book[1] = torch.tensor([1.0, -1.0, 0.5, 0.0])
    got = priced.loss(model, pairs, p, s, 2)
    w = model(holdings.inputs(pairs, p, p.held(), 2))
    assert torch.equal(p.book[2], w.detach())  # the step's book goes in the table
    want = (1 - priced.net(model, pairs, p, s, 2, w)) ** 2
    assert float(got) == pytest.approx(float(want), rel=1e-6)


def test_walk_hands_back_the_averaged_model_that_traded_the_last_month():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    months = _months(20)
    s = Settings("linear", window=12, first_epochs=5, start=date(2001, 6, 1), average=0.9)
    weights, _, model = walk(months, s)
    last = weights.filter(weights["date"] == weights["date"].max())["weight"].to_numpy()
    with torch.no_grad():
        again = model(torch.from_numpy(months[-1].x)).numpy()
    assert np.allclose(again, last, atol=1e-6)


def test_pricing_refuses_a_traded_month_charge_or_start_book_that_cannot_work():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import priced, realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    months = _months()
    stock = [(np.full(4, 0.01, np.float32), np.full(4, 0.05, np.float32))] * len(months)
    market = [realism.Market(torch.ones(4), torch.ones(4), torch.ones(4), 0.0)] * len(months)
    base = {"costs": "t.parquet"}
    for bad, match in (
        ({"trade_margin": True}, "trade_margin needs"),  # no term to charge: it crashed
        ({"trade_margin": True, "rebate": True}, "trade_margin needs"),  # a no-op
        ({"gross_limit": 1.25, "start_book": "cash"}, "start_book needs holdings_input"),
        ({"holdings_input": True, "start_book": "market"}, "market start needs a realism"),
    ):
        with pytest.raises(ValueError, match=match):
            priced.pricing(months, Settings("linear", **base, **bad), "cpu", stock, market)
    good = Settings("linear", **base, holdings_input=True, gross_limit=1.25, trade_margin=True,
                    start_book="market")  # fmt: skip
    assert priced.pricing(months, good, "cpu", stock, market).market is not None
