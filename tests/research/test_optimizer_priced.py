"""Class 3 optimizer: the book each training month trades from (research.optimizer.priced)."""

from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
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


def test_the_windows_first_month_trades_from_cash_as_its_holding_input_says():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import holdings, priced, realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, _held_loss  # noqa: PLC0415

    months = _months(8)
    stock = [(np.full(4, 0.01, np.float32), np.full(4, 0.05, np.float32))] * len(months)
    market = [realism.Market(torch.full((4,), 10.0), torch.full((4,), 0.02),
                             torch.full((4,), 1e6), 0.0)] * len(months)  # fmt: skip
    s = Settings("linear", costs="t.parquet", holdings_input=True, impact=1.0)
    p = priced.pricing(months, s, "cpu", stock, market)
    pairs = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    torch.manual_seed(0)
    model = Optimizer("linear", 2 + 2)
    lo, t = 3, 7  # a cold month: its table is filled from the window's first month on
    holdings.fill(pairs, p, range(lo, t))
    assert lo - 1 not in p.book
    assert holdings.inputs(pairs, p, p.held(), lo)[:, 2].abs().sum() == 0  # a cash holding input
    assert priced.held_going_in(model, pairs, p, lo) is None  # and trades from cash
    later = priced.held_going_in(model, pairs, p, lo + 1)
    assert torch.allclose(later, p.book[lo] * (1 + pairs[lo][1]))  # later months: the table

    def from_cash(w):  # same stocks both months: every dollar of w is traded
        r = pairs[lo][1]
        return (w @ r - 0.01 * w.abs().sum() - 0.05 / 12 * torch.relu(-w).sum()
                - realism.charges(w, torch.zeros(4), p.market[lo], s, s.equity))  # fmt: skip

    w = torch.tensor([2.5, 0.0, 0.3, -1.0])
    zeros = {lo - 1: torch.zeros(4)}  # an all-zero book
    got = float(priced.net(model, pairs, p, s, lo, w))
    assert got == pytest.approx(float(from_cash(w)), rel=1e-5)
    assert got == pytest.approx(float(priced.net(model, pairs, p, s, lo, w, zeros)), rel=1e-5)
    own = model(holdings.inputs(pairs, p, p.held(), lo)).detach()  # the model's book for lo
    want = float((1 - from_cash(own)) ** 2)
    assert _held_loss(model, pairs, p, s, [lo]) == pytest.approx(want, rel=1e-5)  # held out
    assert float(priced.loss(model, pairs, p, s, lo).detach()) == pytest.approx(want, rel=1e-5)


def test_a_cold_months_pass_one_trades_from_cash_until_the_month_before_is_visited(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace  # noqa: PLC0415

    from portfolio_lab.research.optimizer import holdings, priced  # noqa: PLC0415
    from portfolio_lab.research.optimizer import walk as walking  # noqa: PLC0415
    from portfolio_lab.research.optimizer.clipping import Clipper  # noqa: PLC0415
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415

    months = _months(8)
    stock = [(np.full(4, 0.01, np.float32), np.full(4, 0.05, np.float32))] * len(months)
    s = walking.Settings("linear", costs="t.parquet", holdings_input=True)
    p = priced.pricing(months, s, "cpu", stock)  # a cold month: its table starts empty
    pairs = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    torch.manual_seed(0)
    model = Optimizer("linear", 2 + 2)
    lo, t, order = 2, 7, [4, 2, 3, 5, 6]  # 4 before 3: its month before is not yet visited
    steps, real = [], priced.loss

    def watched(net, data, pricing, settings, k):  # each step: what it was shown and priced from
        x = holdings.inputs(data, pricing, pricing.held(), k)
        own = net(x).detach()  # the model's book for k as it stands
        cash = (1 - priced.net(net, data, pricing, settings, k, own, {})) ** 2  # from cash
        table = {j: w.clone() for j, w in pricing.book.items()}
        held = priced.held_going_in(net, data, pricing, k)
        value = real(net, data, pricing, settings, k)
        assert torch.equal(pricing.book[k], own)  # the step writes the model's book over it
        steps.append((k, table, x[:, 2].clone(), held, float(cash), float(value.detach()), own))
        return value

    monkeypatch.setattr(priced, "loss", watched)
    walking._fit(
        model, torch.optim.Adam(model.parameters()), pairs, list(range(lo, t)), 1, s,
        SimpleNamespace(permutation=lambda months: list(order)), p, start_lr=0.01,
        span=range(lo, t), clipper=Clipper(None), average=None, test=-1, hold_rng=None, raw=pairs,
    )  # fmt: skip
    assert [k for k, *_ in steps] == order
    filled = steps[0][1]  # the table as the fill left it, before any step
    assert sorted(filled) == list(range(lo, t))
    assert all(torch.equal(w, torch.zeros(4)) for w in filled.values())  # cash, not the model's
    written, visited = {}, []
    for k, table, holding, held, cash, value, own in steps:
        if k - 1 in written:  # the month before was visited: its book as that step wrote it
            visited.append(k)
            assert torch.equal(table[k - 1], written[k - 1])
            want = holdings.holding(written[k - 1], pairs[k - 1][1], p.links[k], 4)
            assert torch.allclose(holding, want) and float(holding.abs().sum()) > 0
            assert torch.allclose(held, written[k - 1] * (1 + pairs[k - 1][1]))
        else:  # not yet visited (2: before the window): a zero holding, priced from cash
            assert float(holding.abs().sum()) == 0
            assert held is None or float(held.abs().sum()) == 0
            assert value == pytest.approx(cash, rel=1e-6)
        written[k] = own
    assert visited == [3, 5, 6]


def test_fill_gives_cash_only_to_months_not_in_the_table():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import holdings, priced  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    months = _months(8)
    stock = [(np.full(4, 0.01, np.float32), np.full(4, 0.05, np.float32))] * len(months)
    p = priced.pricing(months, Settings("linear", costs="t.parquet", holdings_input=True),
                       "cpu", stock)  # fmt: skip
    pairs = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    kept = torch.tensor([1.0, -1.0, 0.5, 0.0])
    p.book[3] = kept  # a warm walk: a month already visited keeps its book
    holdings.fill(pairs, p, range(2, 7))
    assert p.book[3] is kept and sorted(p.book) == [2, 3, 4, 5, 6]
    assert all(torch.equal(p.book[k], torch.zeros(4)) for k in (2, 4, 5, 6))


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


def test_loss_gross_takes_the_loss_on_the_sized_book_with_a_chosen_risk_aversion():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import priced, realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    rng = np.random.default_rng(5)
    months = [
        data.Month(date(2000, k + 1, 1), list("ABCD"),
                   rng.uniform(-0.5, 0.5, (4, 2)).astype(np.float32),
                   rng.normal(0.01, 0.05, 4).astype(np.float32))
        for k in range(3)
    ]  # fmt: skip
    stock = [(np.full(4, 0.002, np.float32), np.zeros(4, np.float32))] * 3
    prices = (1.0, 4.0, 10.0, 50.0)
    table = pl.DataFrame([
        {"date": m.date, "symbol": s, "prc": p, "rvol_21d": 0.02, "adv": 1e6, "tbill": 0.004}
        for m in months for s, p in zip(m.symbols, prices, strict=True)
    ])  # fmt: skip
    market = realism.align(months, table)
    base = Settings("linear", costs="t.parquet", holdings_input=True, rebate=True, trade_gross=1.0)
    sized = replace(base, loss_gross=1.0, risk_aversion=20.0)
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1, bias=False)  # 2 traits + holding + spread -> a weight

    def net(x):
        return model(x)[:, 0] * 50  # a free-size book, far above 1x

    pricing = priced.pricing(months, sized, "cpu", stock, market)
    tensors = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    pricing.book[0] = torch.zeros(4)
    got = priced.loss(net, tensors, pricing, sized, 1)
    w = pricing.book[1]  # the table holds the SIZED book: 1.0x gross, or less at the 0.5 cap
    gross, need = float(w.abs().sum()), float(realism.requirement(w, market[1].price))
    assert gross == pytest.approx(1.0, rel=1e-5) or (gross < 1.0 and need == pytest.approx(0.5))
    r = priced.net(net, tensors, pricing, sized, 1, w)
    assert float(got) == pytest.approx(float(-r + 10 * r * r))
    plain = priced.pricing(months, base, "cpu", stock, market)
    plain.book[0] = torch.zeros(4)
    priced.loss(net, tensors, plain, base, 1)
    assert float(plain.book[1].abs().sum()) > 10  # loss_gross 0: the model's own size, as before
    for bad in (replace(base, loss_gross=1.0, trade_gross=0.0),
                replace(base, loss_gross=1.0, trade_gross=2.0), replace(sized, price_gross=1.0),
                replace(sized, risk_aversion=0.0)):  # fmt: skip
        with pytest.raises(ValueError, match=r"loss_gross|risk_aversion"):
            priced.pricing(months, bad, "cpu", stock, market)
    assert priced.objective(torch.tensor(0.02), base) == pytest.approx((1 - 0.02) ** 2)
