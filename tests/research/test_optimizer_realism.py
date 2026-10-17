"""Class 3 optimizer: the account's size-dependent costs in the loss (optimizer.realism)."""

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


def _market_table(months, prices=(1.0, 4.0, 10.0, 50.0)):
    return pl.DataFrame([
        {"date": m.date, "symbol": s, "prc": p, "rvol_21d": 0.02, "adv": 1e6, "tbill": 0.004}
        for m in months for s, p in zip(m.symbols, prices, strict=True)
    ])  # fmt: skip


def test_maintenance_requirement_follows_alpacas_price_tables():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import realism  # noqa: PLC0415

    price = torch.tensor([1.0, 4.0, 10.0, 50.0])
    longs = torch.tensor([1.0, 1.0, 1.0, 1.0])
    # longs: $1 -> 100%, $4 -> 50%, $10 and $50 -> 30%
    assert float(realism.requirement(longs, price)) == pytest.approx(1.0 + 0.5 + 0.3 + 0.3)
    # shorts: $1 -> $2.50 a share = 250%; $4 -> 100%; $10 -> $5 a share = 50%; $50 -> 30%
    assert float(realism.requirement(-longs, price)) == pytest.approx(2.5 + 1.0 + 0.5 + 0.3)


def test_each_charge_matches_its_formula_and_is_off_by_default():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    m = realism.Market(torch.tensor([1.0, 4.0, 10.0, 50.0]), torch.full((4,), 0.02),
                       torch.full((4,), 1e6), 0.004)  # fmt: skip
    w = torch.tensor([0.9, 0.6, -0.3, -0.2])  # longs 1.5, shorts 0.5
    held = torch.tensor([0.5, 0.6, -0.3, 0.0])
    base = Settings("linear")
    assert float(realism.charges(w, held, m, base, 1e5)) == 0.0
    assert float(realism.charges(w, held, m, replace(base, rebate=True), 1e5)) == pytest.approx(
        0.004 * 0.5)  # fmt: skip
    assert float(realism.charges(w, held, m, replace(base, margin_spread=0.06), 1e5)) == \
        pytest.approx(0.06 / 12 * 0.5)  # fmt: skip
    req = 0.9 * 1.0 + 0.6 * 0.5 + 0.3 * 0.5 + 0.2 * 0.3
    got = realism.charges(w, held, m, replace(base, margin_cushion=0.6, cushion_rate=0.1), 1e5)
    assert float(got) == pytest.approx(0.1 * (req - 0.6))
    steep = replace(base, margin_cushion=0.6, cushion_rate=0.1, cushion_curve=1.0)
    over = req - 0.6
    assert float(realism.charges(w, held, m, steep, 1e5)) == pytest.approx(0.1 * over + over**2)
    under = replace(steep, margin_cushion=5.0)  # nothing above the cushion: no charge
    assert float(realism.charges(w, held, m, under, 1e5)) == 0.0
    t = np.array([0.4, 0.0, 0.0, 0.2])  # dollars traded per $1 of equity
    impact = (0.02 * np.sqrt(t * 1e5 / 1e6) * t).sum()
    assert float(realism.charges(w, held, m, replace(base, impact=1.0), 1e5)) == pytest.approx(
        impact, rel=1e-4)  # fmt: skip


def test_align_reads_the_market_columns_and_fills_gaps_with_the_month_median():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import realism  # noqa: PLC0415

    months = _months(2)
    table = _market_table(months).with_columns(
        pl.when(pl.col("symbol") == "B").then(None).otherwise(pl.col("prc")).alias("prc")
    )
    (first, _) = realism.align(months, table)
    assert first.price.tolist() == pytest.approx([1.0, 10.0, 10.0, 50.0])  # B: median of 1, 10, 50
    assert first.tbill == pytest.approx(0.004)
    with pytest.raises(ValueError, match="needs"):
        realism.align(months, table.drop("tbill"))


def test_the_loss_subtracts_the_charges_and_needs_costs_and_market_data():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import costs, priced, realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    months = _months()
    stock = [(np.full(4, 0.01, np.float32), np.full(4, 0.05, np.float32))] * len(months)
    market = realism.align(months, _market_table(months))
    s = Settings("linear", costs="t.parquet", holdings_input=True, rebate=True,
                 margin_spread=0.06, margin_cushion=0.6, impact=1.0)  # fmt: skip
    with pytest.raises(ValueError, match="market"):
        priced.pricing(months, s, "cpu", stock)
    with pytest.raises(ValueError, match="need costs"):
        priced.pricing(months, replace(s, costs=None, holdings_input=False), "cpu", None, market)
    p = priced.pricing(months, s, "cpu", stock, market)
    plain = priced.pricing(months, replace(s, rebate=False, margin_spread=0.0, margin_cushion=0.0,
                                           impact=0.0), "cpu", stock)  # fmt: skip
    pairs = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    model = Optimizer("linear", 2 + 2)
    p.book[1] = plain.book[1] = torch.tensor([1.0, -1.0, 0.5, 0.0])
    w = torch.tensor([2.5, 0.0, 0.3, -1.0])
    held = costs.carry(p.book[1] * (1 + pairs[1][1]), p.links[2])
    want = priced.net(model, pairs, plain, s, 2, w) - realism.charges(w, held, p.market[2], s, 1e5)
    assert float(priced.net(model, pairs, p, s, 2, w)) == pytest.approx(float(want), rel=1e-5)


def test_a_walk_with_every_charge_runs_and_dear_margin_shrinks_the_book():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    months = _months(30)
    stock = [(np.full(4, 0.001, np.float32), np.full(4, 0.003, np.float32))] * 30
    market = realism.align(months, _market_table(months))
    base = dict(window=12, first_epochs=20, start=date(2001, 6, 1), costs="t.parquet",
                holdings_input=True, cold=True, lr=0.05)  # fmt: skip
    free, _, _ = walk(months, Settings("linear", **base), stock_costs=stock)
    dear, log, _ = walk(months, Settings("linear", **base, rebate=True, margin_spread=200.0,
                                         margin_cushion=0.3, impact=1.0),
                        stock_costs=stock, market=market)  # fmt: skip
    assert np.isfinite(dear["weight"].to_numpy()).all() and log["rollbacks"].sum() == 0
    size = lambda w: w.group_by("date").agg(pl.col("weight").abs().sum())["weight"].median()  # noqa: E731
    assert size(dear) < size(free)


def test_margin_at_the_brokers_rate_and_idle_cash_at_its_cash_rate():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    m = realism.Market(torch.full((2,), 10.0), torch.full((2,), 0.02), torch.full((2,), 1e6), 0.004)
    base = Settings("linear")
    levered, partial = torch.tensor([1.5, -0.2]), torch.tensor([0.7, -0.2])
    # 6.5% a year = 0.5417% a month, minus the 0.4% T-bill, on the $0.50 borrowed
    got = realism.charges(levered, levered, m, replace(base, margin_rate=0.065), 1e5)
    assert float(got) == pytest.approx((0.065 / 12 - 0.004) * 0.5)
    assert float(realism.charges(partial, partial, m, replace(base, margin_rate=0.065), 1e5)) == 0
    # the broker's rate wins over a fixed spread; a T-bill above the rate charges nothing extra
    both = replace(base, margin_rate=0.065, margin_spread=0.5)
    assert float(realism.charges(levered, levered, m, both, 1e5)) == pytest.approx(float(got))
    high = realism.Market(m.price, m.vol, m.adv, 0.006)
    rate = replace(base, margin_rate=0.065)
    assert float(realism.charges(levered, levered, high, rate, 1e5)) == 0
    # $0.30 of idle cash earns 3.56% a year instead of the 0.4% T-bill
    cash = realism.charges(partial, partial, m, replace(base, cash_rate=0.0356), 1e5)
    assert float(cash) == pytest.approx((0.004 - 0.0356 / 12) * 0.3)
    assert realism.on(replace(base, cash_rate=0.0)) and not realism.on(base)


def test_the_alpaca_profile_sets_every_realism_term():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    s = realism.with_profile(Settings("linear"), "alpaca")
    assert {k: getattr(s, k) for k in realism.ALPACA} == realism.ALPACA
    with pytest.raises(ValueError, match="profile"):
        realism.with_profile(Settings("linear"), "robinhood")


def test_the_size_bound_holds_both_caps_and_cheap_shorts_lower_the_cap():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace  # noqa: PLC0415

    from portfolio_lab.research.optimizer import realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415

    torch.manual_seed(0)
    model = Optimizer("mlp", 3, blocks=1, hidden=8, sized=True)
    for trial in range(20):
        price = torch.rand(50) * 60 + 0.5
        x = torch.randn(50, 3) * (1 + trial)
        m = realism.Market(price, torch.full((50,), 0.02), torch.full((50,), 1e6), 0.004)
        p = SimpleNamespace(bound=(1.25, 0.5), market=[m])
        with torch.no_grad():
            model.size[...] = 8.0  # sigmoid ~1: at the cap
            w = realism.book(model, x, p, 0)
        assert float(w.abs().sum()) <= 1.25 + 1e-5
        assert float(realism.requirement(w, price)) <= 0.5 + 1e-5
        assert float(w.abs().sum()) == pytest.approx(1.25, rel=1e-3) or \
            float(realism.requirement(w, price)) == pytest.approx(0.5, rel=1e-3)  # fmt: skip
    # all short in $1 stocks: $2.50 a share = 250% per dollar -> cap 0.5 / 2.5 = $0.20 of gross
    cheap = SimpleNamespace(bound=(1.25, 0.5), market=[realism.Market(
        torch.ones(4), torch.full((4,), 0.02), torch.full((4,), 1e6), 0.0)])  # fmt: skip
    short = Optimizer("linear", 1, sized=True)
    with torch.no_grad():
        short.lam[...] = -1.0
        short.size[...] = 20.0
        assert float(realism.book(short, torch.ones(4, 1), cheap, 0).abs().sum()) == pytest.approx(
            0.2, rel=1e-4)  # fmt: skip
    plain = SimpleNamespace(bound=None, market=None)  # no bound: the model's output as it is
    assert torch.equal(realism.book(short, torch.ones(4, 1), plain, 0),
                       short(torch.ones(4, 1)))  # fmt: skip


def test_the_size_bound_trains_both_parts_and_holds_on_new_months():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    model = Optimizer("linear", 2, sized=True)
    m = realism.Market(torch.tensor([3.0, 40.0]), torch.full((2,), 0.02), torch.full((2,), 1e6), 0)
    p = type("P", (), {"bound": (1.25, 0.5), "market": [m]})()
    w = realism.book(model, torch.tensor([[1.0, 0.5], [-0.3, 2.0]]), p, 0)
    (w @ torch.tensor([0.1, -0.05])).backward()
    assert model.size.grad is not None and model.lam.grad is not None
    assert float(model.size.grad.abs()) > 0 and float(model.lam.grad.abs().sum()) > 0

    months = _months(30)
    prices = (1.0, 4.0, 10.0, 50.0)
    stock = [(np.full(4, 0.001, np.float32), np.full(4, 0.003, np.float32))] * 30
    market = realism.align(months, _market_table(months, prices))
    s = Settings("linear", window=12, first_epochs=10, start=date(2001, 6, 1), costs="t.parquet",
                 holdings_input=True, cold=True, lr=0.05, size_bound=True)  # fmt: skip
    weights, log, _ = walk(months, s, stock_costs=stock, market=market)
    for (_,), g in weights.group_by("date"):
        w = torch.tensor(g["weight"].to_numpy(), dtype=torch.float32)
        assert float(w.abs().sum()) <= 1.25 + 1e-5
        assert float(realism.requirement(w, torch.tensor(prices))) <= 0.5 + 1e-5
    assert log["rollbacks"].sum() == 0
