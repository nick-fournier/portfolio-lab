from datetime import date
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.data.sources.alpaca import fetch_bars
from portfolio_lab.data.sources.fred import parse_dtb3


def _bar(t, c):
    return {"t": t, "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 1000, "vw": c, "n": 10}


def test_fetch_bars_paginates_and_uses_new_york_dates():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(urlparse(str(request.url)).query)
        requests.append(query)
        if "page_token" not in query:
            # EST: midnight New York = 05:00Z
            return httpx.Response(
                200,
                json={
                    "bars": {"AAA": [_bar("2024-03-08T05:00:00Z", 10.0)]},
                    "next_page_token": "p2",
                },
            )
        # EDT started 2024-03-10: midnight New York = 04:00Z
        return httpx.Response(
            200,
            json={"bars": {"AAA": [_bar("2024-03-11T04:00:00Z", 11.0)]}, "next_page_token": None},
        )

    client = RateLimitedClient(base_url="https://data.test", transport=httpx.MockTransport(handler))
    df = fetch_bars(client, ["AAA"], date(2024, 3, 8), date(2024, 3, 11), adjustment="all")

    assert df["date"].to_list() == [date(2024, 3, 8), date(2024, 3, 11)]
    assert df["close"].to_list() == [10.0, 11.0]
    assert requests[0]["adjustment"] == ["all"]
    assert requests[1]["page_token"] == ["p2"]


def test_fetch_bars_batches_symbols():
    batches = []

    def handler(request: httpx.Request) -> httpx.Response:
        symbols = parse_qs(urlparse(str(request.url)).query)["symbols"][0].split(",")
        batches.append(len(symbols))
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    client = RateLimitedClient(base_url="https://data.test", transport=httpx.MockTransport(handler))
    df = fetch_bars(client, [f"S{i:03d}" for i in range(250)], date(2024, 1, 2), date(2024, 1, 3))
    assert batches == [200, 50]
    assert df.is_empty()


def test_parse_dtb3_drops_holidays_and_converts_percent():
    df = parse_dtb3("observation_date,DTB3\n2024-01-02,5.25\n2024-01-15,.\n2024-01-16,5.20\n")
    assert df["date"].to_list() == [date(2024, 1, 2), date(2024, 1, 16)]
    assert df["rate"].to_list() == pytest.approx([0.0525, 0.052])
