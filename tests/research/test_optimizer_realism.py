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
    req = 0.9 * 1.0 + 0.6 * 0.5 + 0.3 * 0.5 + 0.2 * 0.3
    cushion = replace(base, margin_cushion=0.6, cushion_rate=0.1)
    assert float(realism.charges(w, held, m, cushion, 1e5)) == pytest.approx(0.1 * (req - 0.6))
    under = replace(cushion, margin_cushion=5.0)  # nothing above the cushion: no charge
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
                 margin_rate=0.065, margin_cushion=0.6, impact=1.0)  # fmt: skip
    with pytest.raises(ValueError, match="market"):
        priced.pricing(months, s, "cpu", stock)
    with pytest.raises(ValueError, match="need costs"):
        priced.pricing(months, replace(s, costs=None, holdings_input=False), "cpu", None, market)
    p = priced.pricing(months, s, "cpu", stock, market)
    plain = priced.pricing(months, replace(s, rebate=False, margin_rate=0.0, margin_cushion=0.0,
                                           impact=0.0), "cpu", stock)  # fmt: skip
    pairs = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    model = Optimizer("linear", 2 + 2)
    p.book[1] = plain.book[1] = torch.tensor([1.0, -1.0, 0.5, 0.0])
    w = torch.tensor([2.5, 0.0, 0.3, -1.0])  # longs 2.8: margin interest on 1.8
    held = costs.carry(p.book[1] * (1 + pairs[1][1]), p.links[2])
    assert float(realism.charges(w, held, p.market[2], s, 1e5)) > 0
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
    dear, log, _ = walk(months, Settings("linear", **base, rebate=True, margin_rate=200.0,
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
    # a T-bill above the broker's rate charges nothing extra
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
    # the terms given explicitly stay, 0 too; the profile fills the rest; none: no profile
    given = Settings("linear", gross_limit=0.0, margin_rate=0.08)
    kept = realism.with_profile(given, "alpaca", keep={"gross_limit", "margin_rate"})
    assert {k: getattr(kept, k) for k in realism.ALPACA} == realism.ALPACA | {
        "gross_limit": 0.0, "margin_rate": 0.08}  # fmt: skip
    assert realism.with_profile(given, "none") == realism.with_profile(given, "") == given


def test_the_gross_limit_is_charged_like_the_cushion():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    m = realism.Market(torch.full((2,), 50.0), torch.full((2,), 0.02), torch.full((2,), 1e6), 0)
    w = torch.tensor([1.0, -0.5])  # gross 1.5; requirement 0.3 x 1.5 = 0.45
    s = Settings("linear", gross_limit=1.25, cushion_rate=0.1)
    assert float(realism.penalty(w, m, s)) == pytest.approx(0.1 * 0.25)
    both = replace(s, margin_cushion=0.4)  # 0.05 over the cushion too
    want = 0.1 * 0.25 + 0.1 * 0.05
    assert float(realism.penalty(w, m, both)) == pytest.approx(want, rel=1e-5)
    assert float(realism.charges(w, w, m, both, 1e5)) == pytest.approx(want, rel=1e-5)


def test_the_traded_months_penalty_is_its_books_and_trading_on_it_shrinks_the_book():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import holdings, priced, realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.model import Optimizer  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    months = _months(30)
    stock = [(np.full(4, 0.001, np.float32), np.full(4, 0.003, np.float32))] * 30
    market = realism.align(months, _market_table(months))
    s = Settings("linear", costs="t.parquet", holdings_input=True, margin_cushion=0.5,
                 gross_limit=1.25, invest_floor=1.0, cushion_rate=0.1)  # fmt: skip
    p = priced.pricing(months, s, "cpu", stock, market)
    pairs = [(torch.from_numpy(m.x), torch.from_numpy(m.r)) for m in months]
    model = Optimizer("linear", 2 + 2)
    p.book[28] = torch.tensor([2.0, -1.0, 0.5, 0.0])
    w = realism.book(model, holdings.inputs(pairs, p, p.held(), 29), p, 29)
    assert torch.equal(w, model(holdings.inputs(pairs, p, p.held(), 29)))  # the model's output
    assert float(priced.trade_penalty(model, pairs, p, s, 29)) == pytest.approx(
        float(realism.penalty(w, p.market[29], s)))  # fmt: skip
    floor = replace(s, margin_cushion=0.0, gross_limit=0.0)  # the investment floor alone
    short = torch.relu(1.0 - torch.relu(w).sum())  # longs below $1 of equity
    assert float(priced.trade_penalty(model, pairs, p, floor, 29)) == pytest.approx(
        float(0.1 * short))  # fmt: skip
    real = torch.tensor([0.1, 0.0, 0.0, -0.1])  # the real traded book wins over the table's
    w_real = realism.book(model, holdings.inputs(pairs, p, {28: real}, 29), p, 29)
    assert float(priced.trade_penalty(model, pairs, p, s, 29, real)) == pytest.approx(
        float(realism.penalty(w_real, p.market[29], s)))  # fmt: skip

    base = dict(window=12, first_epochs=20, start=date(2001, 6, 1), costs="t.parquet",
                holdings_input=True, cold=True, lr=0.05, margin_cushion=0.5, gross_limit=1.25,
                cushion_rate=5.0)  # fmt: skip
    free, _, _ = walk(months, Settings("linear", **base), stock_costs=stock, market=market)
    kept, log, _ = walk(months, Settings("linear", **base, trade_margin=True), stock_costs=stock,
                        market=market)  # fmt: skip
    size = lambda w: w.group_by("date").agg(pl.col("weight").abs().sum())["weight"]  # noqa: E731
    assert log["rollbacks"].sum() == 0 and np.isfinite(kept["weight"].to_numpy()).all()
    assert float(size(kept).max()) < float(size(free).max())


def test_a_walk_can_start_from_cash_or_the_market_instead_of_its_own_guess():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings, walk  # noqa: PLC0415

    mv = torch.tensor([5.0, 1.0, 3.0, 0.0])
    m = realism.Market(torch.ones(4), torch.ones(4), torch.ones(4), 0.0, mv)
    assert realism.start_book("guess", m, 4) is None
    assert realism.start_book("cash", None, 4).tolist() == [0.0, 0.0, 0.0, 0.0]
    assert realism.start_book("market", m, 4).tolist() == pytest.approx([5 / 9, 1 / 9, 3 / 9, 0.0])
    with pytest.raises(ValueError, match="market_value"):
        realism.start_book("market", realism.Market(m.price, m.vol, m.adv, 0.0), 4)
    with pytest.raises(ValueError, match="start_book"):
        realism.start_book("index", m, 4)

    months = _months(30)
    stock = [(np.full(4, 0.001, np.float32), np.full(4, 0.003, np.float32))] * 30
    table = _market_table(months).with_columns(pl.lit(1e9).alias("market_value"))
    market = realism.align(months, table)
    assert market[0].mv is not None
    base = dict(window=12, first_epochs=3, start=date(2001, 6, 1), costs="t.parquet",
                holdings_input=True, cold=True, impact=1.0)  # fmt: skip
    runs = {k: walk(months, Settings("linear", **base, start_book=k), stock_costs=stock,
                    market=market)[0] for k in ("guess", "cash", "market")}  # fmt: skip
    first = min(runs["guess"]["date"])
    one = {k: w.filter(pl.col("date") == first)["weight"].to_numpy() for k, w in runs.items()}
    assert not np.allclose(one["guess"], one["cash"])
    assert not np.allclose(one["cash"], one["market"])


def test_the_investment_floor_charges_idle_cash_like_the_caps():
    torch = pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import realism  # noqa: PLC0415
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    m = realism.Market(torch.full((3,), 50.0), torch.full((3,), 0.02), torch.full((3,), 1e6), 0)
    s = Settings("linear", invest_floor=1.0, cushion_rate=0.1)
    half = torch.tensor([0.3, 0.2, -0.4])  # longs 0.5: idle cash 0.5
    assert float(realism.penalty(half, m, s)) == pytest.approx(0.1 * 0.5)
    assert float(realism.penalty(torch.tensor([0.7, 0.4, -0.2]), m, s)) == 0.0  # longs 1.1
    assert realism.on(s) and not realism.on(replace(s, invest_floor=0.0))
