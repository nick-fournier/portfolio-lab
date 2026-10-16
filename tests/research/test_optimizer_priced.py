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


def test_a_month_trades_from_the_stored_book_after_its_returns_and_real_trades_win():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import costs, priced  # noqa: PLC0415
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    months = _months()
    stock = [(np.full(4, 0.01, np.float32), np.full(4, 0.05, np.float32))] * len(months)
    s = Settings("linear", costs="t.parquet", holdings_input=True, real_book=True)
    p = priced.pricing(months, s, "cpu", stock)
    pairs = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    model = Optimizer("linear", 2 + 2)
    r1 = pairs[1][1]
    p.book[1] = torch.tensor([1.0, -1.0, 0.5, 0.0])
    held = priced.held_going_in(model, pairs, p, 2)
    assert torch.allclose(held, p.book[1] * (1 + r1))  # the stored book, grown by its returns
    p.traded[1] = torch.tensor([2.0, 0.0, 0.0, -2.0])
    assert torch.allclose(priced.held_going_in(model, pairs, p, 2), p.traded[1] * (1 + r1))
    assert priced.held_going_in(model, pairs, p, 0) is None  # the first month starts in cash
    w = torch.tensor([2.5, 0.0, 0.0, -1.0])
    turnover = (w - p.traded[1] * (1 + r1)).abs().sum()  # same stocks both months
    expect = w @ pairs[2][1] - 0.01 * turnover - 0.05 / 12 * 1.0
    assert float(priced.net(model, pairs, p, s, 2, w)) == pytest.approx(float(expect), rel=1e-5)
    assert costs.carry(p.traded[1], p.links[2]).tolist() == p.traded[1].tolist()


def test_real_book_needs_holdings_and_keeps_every_traded_month():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import priced  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    months = _months(30)
    stock = [(np.full(4, 0.01, np.float32), np.full(4, 0.05, np.float32))] * 30
    with pytest.raises(ValueError, match="real_book"):
        priced.pricing(months, Settings("linear", costs="t.parquet", real_book=True), "cpu", stock)
    base = dict(window=12, first_epochs=3, start=date(2001, 6, 1), costs="t.parquet",
                holdings_input=True, cold=True)  # fmt: skip
    w0, _, _ = walk(months, Settings("linear", **base), stock_costs=stock)
    w1, _, _ = walk(months, Settings("linear", **base, real_book=True), stock_costs=stock)
    first = w0["date"].min()  # nothing traded yet: the same; later months train on real books
    assert np.allclose(w0.filter(w0["date"] == first)["weight"].to_numpy(),
                       w1.filter(w1["date"] == first)["weight"].to_numpy(), atol=1e-6)  # fmt: skip
    assert not np.allclose(w0["weight"].to_numpy(), w1["weight"].to_numpy())
