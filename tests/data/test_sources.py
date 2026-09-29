from datetime import UTC, date, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.data.sources.alpaca import end_param, fetch_bars
from portfolio_lab.data.sources.fred import CONTEXT_SERIES, fetch_series, parse_dtb3


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


def test_end_param_caps_recent_data():
    now = datetime(2024, 3, 11, 22, 0, tzinfo=UTC)
    # Past day: end of that day in New York (EST, UTC-5).
    assert end_param(date(2024, 3, 8), now) == "2024-03-09T04:59:59Z"
    # Today: capped at now minus the 16-minute SIP delay.
    assert end_param(date(2024, 3, 11), now) == "2024-03-11T21:44:00Z"


def test_fetch_bars_keeps_decimals_after_whole_number_prices():
    # Regression: 120 whole-dollar closes followed by a fractional one used to be typed as
    # integers from the first rows, truncating 0.9909 to 0.
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(121)]
    closes = [2] * 120 + [0.9909]
    bars = [_bar(f"{d}T05:00:00Z", c) for d, c in zip(days, closes, strict=True)]
    payload = {"bars": {"AAA": bars}}
    client = RateLimitedClient(
        base_url="https://data.test",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
    )
    df = fetch_bars(client, ["AAA"], days[0], days[-1])
    assert df["close"][-1] == pytest.approx(0.9909)


def test_fetch_bars_skips_invalid_symbols():
    def handler(request: httpx.Request) -> httpx.Response:
        symbols = parse_qs(urlparse(str(request.url)).query)["symbols"][0].split(",")
        if "BAD-X" in symbols:
            return httpx.Response(400, json={"message": "invalid symbol: BAD-X"})
        return httpx.Response(
            200, json={"bars": {s: [_bar("2024-03-08T05:00:00Z", 10.0)] for s in symbols}}
        )

    client = RateLimitedClient(base_url="https://data.test", transport=httpx.MockTransport(handler))
    df = fetch_bars(client, ["AAA", "BAD-X", "BBB"], date(2024, 3, 8), date(2024, 3, 8))
    assert sorted(df["symbol"]) == ["AAA", "BBB"]


def test_fetch_bars_isolates_unnamed_bad_symbols():
    def handler(request: httpx.Request) -> httpx.Response:
        symbols = parse_qs(urlparse(str(request.url)).query)["symbols"][0].split(",")
        if "3UW:DU" in symbols:
            return httpx.Response(400, json={"message": "bad request"})
        return httpx.Response(
            200, json={"bars": {s: [_bar("2024-03-08T05:00:00Z", 10.0)] for s in symbols}}
        )

    client = RateLimitedClient(base_url="https://data.test", transport=httpx.MockTransport(handler))
    df = fetch_bars(
        client, ["AAA", "BBB", "3UW:DU", "CCC", "DDD"], date(2024, 3, 8), date(2024, 3, 8)
    )
    assert sorted(df["symbol"]) == ["AAA", "BBB", "CCC", "DDD"]


def test_fred_first_release_uses_publication_date_and_market_series_next_day():
    seen = []

    def handler(request):
        query = parse_qs(urlparse(str(request.url)).query)
        seen.append(query)
        obs = [
            {"date": "2024-01-01", "value": "3.7", "realtime_start": "2024-02-02"},
            {"date": "2024-02-01", "value": ".", "realtime_start": "2024-03-08"},
        ]
        return httpx.Response(200, json={"observations": obs})

    client = RateLimitedClient(transport=httpx.MockTransport(handler))
    unrate = fetch_series(client, "UNRATE", "k")
    assert seen[0]["output_type"] == ["4"]
    assert unrate.rows() == [("UNRATE", date(2024, 1, 1), 3.7, date(2024, 2, 2))]
    oil = fetch_series(client, "DCOILWTICO", "k")
    assert "output_type" not in seen[1]
    assert oil["available"].to_list() == [date(2024, 1, 2)]  # known the next day
    assert CONTEXT_SERIES["VIXCLS"].lag_days == 0
