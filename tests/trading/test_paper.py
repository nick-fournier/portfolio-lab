from datetime import date

import httpx
import polars as pl
import pytest

from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.trading import paper
from portfolio_lab.trading.broker import Order, PaperBroker
from portfolio_lab.trading.paper import MAX_INVESTED, due, plan_orders


def test_plan_orders_closes_dropped_names_sells_first_and_skips_dust():
    targets = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    held = {"AAA": 60_000.0, "BBB": 29_750.0, "OLD": 5_000.0}
    prices = {"AAA": 100.0, "BBB": 50.0, "CCC": 7.0}
    closes, orders = plan_orders(targets, 100_000.0, held, prices, lambda s: s != "CCC")
    assert closes == ["OLD"]
    assert [(o.symbol, o.side) for o in orders] == [("AAA", "sell"), ("CCC", "buy")]
    assert orders[0].notional == pytest.approx(60_000 - 0.5 * MAX_INVESTED * 100_000)
    assert orders[1].qty == 2828.0  # whole shares: CCC isn't fractionable
    # BBB is within the minimum trade size of its target, so it is left alone.


def test_plan_orders_never_invests_more_than_the_cap():
    _, orders = plan_orders({"AAA": 1.0}, 50_000.0, {}, {"AAA": 10.0}, lambda s: True)
    assert sum(o.notional for o in orders) == pytest.approx(MAX_INVESTED * 50_000)


def test_due_on_month_ends_once_or_immediately_when_empty():
    sessions = [date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 1)]
    assert due(date(2026, 9, 30), sessions, set(), holding=True)
    assert not due(date(2026, 9, 30), sessions, {date(2026, 9, 30)}, holding=True)
    assert not due(date(2026, 9, 29), sessions, {date(2026, 8, 31)}, holding=True)
    assert due(date(2026, 9, 29), sessions, set(), holding=False)  # first run


def test_latest_session_is_not_a_month_end_just_because_data_stops_there():
    # Data through Thu Oct 1: October's later sessions haven't happened yet.
    sessions = [date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 1)]
    assert not due(date(2026, 10, 1), sessions, {date(2026, 9, 30)}, holding=False)
    assert not due(date(2026, 10, 1), sessions, {date(2026, 9, 30)}, holding=True)
    assert due(date(2026, 10, 30), [*sessions, date(2026, 10, 30)], {date(2026, 9, 30)}, True)


def test_broker_refuses_anything_but_paper():
    settings = Settings(alpaca_paper_url="https://api.alpaca.markets",
                        ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY="s")  # fmt: skip
    with pytest.raises(RuntimeError, match="paper"):
        PaperBroker(settings)


def test_resent_order_is_not_placed_twice():
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(200, json={"id": "1"})
        return httpx.Response(422, json={"message": "client_order_id must be unique"})

    client = RateLimitedClient(
        base_url="https://paper-api.alpaca.markets",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    settings = Settings(ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY="s")
    broker = PaperBroker(settings, client=client)
    order = Order("AAA", "buy", notional=100.0)
    assert broker.submit(order, "pl-x") == {"id": "1"}
    assert broker.submit(order, "pl-x") is None


class FakeBroker:
    def __init__(self):
        self.submitted, self.closed, self.cancelled = [], [], 0

    def account(self):
        return {"equity": "100000", "cash": "100000"}

    def positions(self):
        return []

    def orders(self, after):
        return []

    def fractionable(self, symbol):
        return True

    def cancel_open_orders(self):
        self.cancelled += 1

    def close_position(self, symbol):
        self.closed.append(symbol)

    def submit(self, order, client_id):
        self.submitted.append((order, client_id))
        return {"id": client_id}


class FixedStrategy:
    def target_weights(self, view):
        return {"AAA": 0.6, "BBB": 0.4}


def test_run_rebalances_once_and_records(tmp_path, make_panel, monkeypatch):
    panel = make_panel()
    monkeypatch.setattr(paper.Panel, "load", classmethod(lambda cls, d: panel))
    broker = FakeBroker()
    first = paper.run(tmp_path, FixedStrategy(), broker)
    assert first["rebalanced"] and len(broker.submitted) == 2
    again = paper.run(tmp_path, FixedStrategy(), broker)
    assert not again["rebalanced"] and len(broker.submitted) == 2
    log = pl.read_parquet(tmp_path / "trading" / "paper" / "rebalances.parquet")
    assert log.height == 1 and log["date"][0] == panel.dates[-1]


def test_dry_run_places_nothing(tmp_path, make_panel, monkeypatch):
    panel = make_panel()
    monkeypatch.setattr(paper.Panel, "load", classmethod(lambda cls, d: panel))
    broker = FakeBroker()
    plan = paper.run(tmp_path, FixedStrategy(), broker, dry_run=True)
    assert len(plan["orders"]) == 2 and not broker.submitted and broker.cancelled == 0
    assert not (tmp_path / "trading").exists()


def test_paper_records_follow_the_trading_dir(tmp_path):
    assert DataPaths(tmp_path).paper == tmp_path / "trading" / "paper"
    assert DataPaths(tmp_path, tmp_path / "ops").paper == tmp_path / "ops" / "paper"
