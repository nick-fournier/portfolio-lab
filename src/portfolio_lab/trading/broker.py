"""Alpaca paper-trading account: read the account, cancel, close and place orders.

Only the paper endpoint is accepted (:data:`PAPER_HOST`), so this module cannot touch
real money. Orders carry a deterministic client id, so a resent order is rejected by
Alpaca instead of being placed twice.
"""

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from portfolio_lab.core.config import Settings
from portfolio_lab.core.http import RateLimitedClient

log = logging.getLogger(__name__)

PAPER_HOST = "paper-api.alpaca.markets"


@dataclass(frozen=True)
class Order:
    """An order to place: a dollar amount (``notional``) or a share count (``qty``)."""

    symbol: str
    side: str  # "buy" or "sell"
    notional: float | None = None
    qty: float | None = None


class PaperBroker:
    """Thin client for the Alpaca paper account (see module docs).

    Raises:
        RuntimeError: If the configured trading URL is not Alpaca's paper endpoint.
    """

    def __init__(self, settings: Settings, client: RateLimitedClient | None = None):
        if PAPER_HOST not in settings.alpaca_paper_url:
            raise RuntimeError(f"refusing to trade outside paper ({settings.alpaca_paper_url})")
        self._client = client or RateLimitedClient(
            base_url=settings.alpaca_paper_url,
            headers=settings.alpaca_headers(),
            max_per_minute=150,
        )

    def close(self) -> None:
        """Close the connection pool."""
        self._client.close()

    def account(self) -> dict[str, Any]:
        """Equity, cash and status."""
        return self._client.get_json("/v2/account")

    def positions(self) -> list[dict[str, Any]]:
        """Open positions (symbol, qty, market_value, ...)."""
        return self._client.get_json("/v2/positions")

    def fractionable(self, symbol: str) -> bool:
        """Whether Alpaca allows fractional shares of ``symbol``."""
        return bool(self._client.get_json(f"/v2/assets/{symbol}").get("fractionable"))

    def cancel_open_orders(self) -> None:
        """Cancel every order not yet filled."""
        self._client.request("DELETE", "/v2/orders")

    def close_position(self, symbol: str) -> None:
        """Sell an entire position at market."""
        self._client.request("DELETE", f"/v2/positions/{symbol}")

    def submit(self, order: Order, client_id: str) -> dict[str, Any] | None:
        """Place a day market order; ``None`` if this client id was already placed."""
        body = {"symbol": order.symbol, "side": order.side, "type": "market",
                "time_in_force": "day", "client_order_id": client_id}  # fmt: skip
        if order.notional is not None:
            body["notional"] = f"{order.notional:.2f}"
        else:
            body["qty"] = f"{order.qty:g}"
        try:
            return self._client.request("POST", "/v2/orders", json=body).json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 422 and "client_order_id" in exc.response.text:
                log.info("order %s already placed", client_id)
                return None
            raise

    def orders(self, after: str) -> list[dict[str, Any]]:
        """Orders submitted after ``after`` (ISO time), any status."""
        return self._client.get_json(
            "/v2/orders", {"status": "all", "after": after, "limit": 500, "direction": "asc"}
        )
