"""Paper trading: run the production strategy on Alpaca's paper account (dummy money).

Every night after the daily price ingest, :func:`run`:

1. records the account (equity, cash) and positions, and refreshes the status and fills
   of recent orders;
2. on the last trading day of a month (or on the first run, with nothing held yet),
   computes the strategy's target weights from data through that close and queues market
   orders, which execute at the next open, as the backtest assumes.

Orders are planned by :func:`plan_orders`: positions that left the target are closed,
the rest are sized in dollars (whole shares where fractions aren't allowed), trades below
:data:`MIN_TRADE_DOLLARS` or :data:`MIN_TRADE_SHARE` of equity are skipped, and at most
:data:`MAX_INVESTED` of equity is invested, so the account never borrows. A month end is
rebalanced once, even if the job runs again.
"""

import logging
import math
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from portfolio_lab.core.calendar import next_session
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import upsert_parquet
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.panel import Panel
from portfolio_lab.trading.broker import Order, PaperBroker

log = logging.getLogger(__name__)

MAX_INVESTED = 0.99
MIN_TRADE_DOLLARS = 100.0
MIN_TRADE_SHARE = 0.002
#: Orders submitted within this many days are refreshed each night (status, fills).
ORDER_LOOKBACK_DAYS = 45


def plan_orders(
    targets: dict[str, float],
    equity: float,
    held: dict[str, float],
    prices: dict[str, float],
    fractionable: Callable[[str], bool],
) -> tuple[list[str], list[Order]]:
    """Positions to close and orders to place to move ``held`` to ``targets``.

    Args:
        targets: Symbol -> target weight (summing to at most 1).
        equity: Account equity in dollars.
        held: Symbol -> current market value.
        prices: Symbol -> latest price.
        fractionable: Whether a symbol may be traded in fractional shares.

    Returns:
        (symbols to close entirely, orders), sells before buys.
    """
    closes = sorted(s for s in held if s not in targets)
    floor = max(MIN_TRADE_DOLLARS, MIN_TRADE_SHARE * equity)
    orders = []
    for symbol, weight in sorted(targets.items()):
        diff = weight * MAX_INVESTED * equity - held.get(symbol, 0.0)
        if abs(diff) < floor:
            continue
        side = "buy" if diff > 0 else "sell"
        if fractionable(symbol):
            orders.append(Order(symbol, side, notional=round(abs(diff), 2)))
        elif (shares := math.floor(abs(diff) / prices[symbol])) > 0:
            orders.append(Order(symbol, side, qty=float(shares)))
    return closes, sorted(orders, key=lambda o: o.side != "sell")


def _append(path: Path, frame: pl.DataFrame, keys: tuple[str, ...]) -> None:
    if frame.height:
        upsert_parquet(frame, path, keys)


def _record_account(folder: Path, broker: PaperBroker, session: date) -> dict[str, float]:
    """Save tonight's account snapshot and positions; return symbol -> market value."""
    account = broker.account()
    positions = broker.positions()
    _append(folder / "snapshots.parquet", pl.DataFrame(
        {"date": [session], "equity": [float(account["equity"])],
         "cash": [float(account["cash"])], "positions": [len(positions)]}
    ), ("date",))  # fmt: skip
    _append(folder / "positions.parquet", pl.DataFrame(
        {"date": [session] * len(positions), "symbol": [p["symbol"] for p in positions],
         "qty": [float(p["qty"]) for p in positions],
         "market_value": [float(p["market_value"]) for p in positions]},
        schema={"date": pl.Date, "symbol": pl.String, "qty": pl.Float64,
                "market_value": pl.Float64},
    ), ("date", "symbol"))  # fmt: skip
    return {p["symbol"]: float(p["market_value"]) for p in positions}


def _refresh_orders(folder: Path, broker: PaperBroker, session: date) -> None:
    """Update recent orders' status and fills."""
    after = (session - timedelta(days=ORDER_LOOKBACK_DAYS)).isoformat() + "T00:00:00Z"
    rows = [
        {"id": o["id"], "client_order_id": o["client_order_id"], "symbol": o["symbol"],
         "side": o["side"], "notional": _float(o.get("notional")), "qty": _float(o.get("qty")),
         "status": o["status"], "submitted_at": o.get("submitted_at"),
         "filled_qty": _float(o.get("filled_qty")),
         "filled_avg_price": _float(o.get("filled_avg_price")), "filled_at": o.get("filled_at")}
        for o in broker.orders(after)
    ]  # fmt: skip
    if rows:
        _append(folder / "orders.parquet", pl.DataFrame(rows, infer_schema_length=None), ("id",))


def _float(value: Any) -> float | None:
    return float(value) if value not in (None, "") else None


def due(session: date, sessions: list[date], rebalanced: set[date], holding: bool) -> bool:
    """Rebalance on a month's last session not yet done, or right away when holding nothing.

    A month's last session comes from the exchange calendar, not from ``sessions``: the
    latest session in the data is not a month end just because the month's later sessions
    haven't happened yet.
    """
    if session in rebalanced:
        return False
    month_end = next_session(session).month != session.month
    return month_end or (not holding and not rebalanced)


def run(
    data_dir: Path,
    strategy: Any,
    broker: PaperBroker,
    refresh: Callable[[], Any] | None = None,
    dry_run: bool = False,
    trading_dir: Path | None = None,
) -> dict[str, Any]:
    """Record the account and, when due, rebalance to ``strategy``'s targets (module docs).

    Args:
        data_dir: The data directory (prices, features, and where records are kept).
        strategy: The strategy object (production, ``jobs.tasks.PRODUCTION``).
        broker: The paper account.
        refresh: Called before computing targets on a rebalance day (rebuilds the monthly
            features so the month just closed is included); skipped in a dry run.
        dry_run: Plan the rebalance now and return the orders without placing them or
            writing records.
        trading_dir: Where trading records live (default ``data_dir/trading``).
    """
    folder = DataPaths(data_dir, trading_dir).paper
    panel = Panel.load(data_dir)
    session = panel.dates[-1]
    if dry_run:
        held = {p["symbol"]: float(p["market_value"]) for p in broker.positions()}
    else:
        held = _record_account(folder, broker, session)
        _refresh_orders(folder, broker, session)
    log_path = folder / "rebalances.parquet"
    done = set(pl.read_parquet(log_path)["date"]) if log_path.exists() else set()
    if not dry_run and not due(session, panel.dates, done, bool(held)):
        return {"session": session, "rebalanced": False, "positions": len(held)}
    if refresh is not None and not dry_run:
        refresh()
        panel = Panel.load(data_dir)
    targets = strategy.target_weights(DataView(panel, panel.date_index[session]))
    if not targets:
        raise RuntimeError(f"strategy produced no targets for {session}")
    equity = float(broker.account()["equity"])
    i = panel.date_index[session]
    prices = {panel.names[s]: float(panel.field("close")[i, panel.symbol_index[s]])
              for s in targets}  # fmt: skip
    targets = {panel.names[s]: w for s, w in targets.items()}
    closes, orders = plan_orders(targets, equity, held, prices, broker.fractionable)
    if dry_run:
        return {"session": session, "equity": equity, "targets": targets, "closes": closes,
                "orders": orders}  # fmt: skip
    broker.cancel_open_orders()
    for symbol in closes:
        broker.close_position(symbol)
    for order in orders:
        broker.submit(order, f"pl-{session:%Y%m%d}-{order.symbol}-{order.side}")
    _append(log_path, pl.DataFrame(
        {"date": [session], "targets": [len(targets)], "closes": [len(closes)],
         "orders": [len(orders)], "equity": [equity]}
    ), ("date",))  # fmt: skip
    _append(folder / "targets.parquet", pl.DataFrame(
        {"date": [session] * len(targets), "symbol": list(targets),
         "weight": [float(w) for w in targets.values()]}
    ), ("date", "symbol"))  # fmt: skip
    log.info("paper rebalance %s: %d targets, %d closes, %d orders", session, len(targets),
             len(closes), len(orders))  # fmt: skip
    return {"session": session, "rebalanced": True, "targets": len(targets),
            "closes": len(closes), "orders": len(orders)}  # fmt: skip
