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


def test_a_displaced_book_moves_by_the_asked_share_of_the_book():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import priced  # noqa: PLC0415

    book = torch.randn(20000)
    moved = priced.displaced(book, 0.4, np.random.default_rng(0))
    assert float((moved - book).abs().sum() / book.abs().sum()) == pytest.approx(0.4, abs=0.01)
    assert torch.equal(priced.displaced(book, 0.0, np.random.default_rng(0)), book)


def test_inherit_trades_from_one_displaced_book_for_both_the_input_and_the_cost():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import holdings, priced  # noqa: PLC0415
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    months = _months()
    stock = [(np.full(4, 0.01, np.float32), np.full(4, 0.05, np.float32))] * len(months)
    s = Settings("linear", costs="t.parquet", holdings_input=True, inherit=0.3)
    pairs = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    torch.manual_seed(0)
    model = Optimizer("linear", 2 + 2)
    with pytest.raises(ValueError, match="inherit"):
        priced.pricing(months, Settings("linear", costs="t.parquet", inherit=0.3), "cpu", stock)
    p = priced.pricing(months, s, "cpu", stock)
    p.book[1] = torch.tensor([1.0, -1.0, 0.5, 0.0])
    got = priced.loss(model, pairs, p, s, 2, np.random.default_rng(5))
    rng = np.random.default_rng(5)  # the same draws by hand: the size, then the noise
    moved = {1: priced.displaced(torch.tensor([1.0, -1.0, 0.5, 0.0]), rng.uniform(0, 0.6), rng)}
    w = model(holdings.inputs(pairs, p, moved, 2))
    want = (1 - priced.net(model, pairs, p, s, 2, w, held=moved)) ** 2
    assert float(got.detach()) == pytest.approx(float(want.detach()), rel=1e-6)
    plain = priced.loss(model, pairs, p, s, 2)  # no stream (held-out scoring): the real book
    real = model(holdings.inputs(pairs, p, p.held(), 2))
    assert float(plain) == pytest.approx(float((1 - priced.net(model, pairs, p, s, 2, real)) ** 2))
    assert float(got) != pytest.approx(float(plain))


def test_inherit_has_its_own_random_stream_and_changes_the_training():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    months = _months(30)
    stock = [(np.full(4, 0.01, np.float32), np.full(4, 0.05, np.float32))] * 30
    base = dict(window=12, first_epochs=3, start=date(2001, 6, 1), costs="t.parquet",
                holdings_input=True, cold=True, stock_drop=0.25)  # fmt: skip
    w0, _, _ = walk(months, Settings("linear", **base), stock_costs=stock)
    tiny, _, _ = walk(months, Settings("linear", **base, inherit=1e-9), stock_costs=stock)
    w1, _, _ = walk(months, Settings("linear", **base, inherit=0.5), stock_costs=stock)
    # same month order and stock drops: a vanishing displacement changes nothing
    assert np.allclose(w0["weight"].to_numpy(), tiny["weight"].to_numpy(), atol=1e-6)
    assert not np.allclose(w0["weight"].to_numpy(), w1["weight"].to_numpy(), atol=1e-4)


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
