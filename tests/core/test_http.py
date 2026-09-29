import httpx
import pytest

from portfolio_lab.core.http import RateLimitedClient


def _client(responses, sleeps):
    calls = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    return RateLimitedClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
        max_retries=2,
        sleep=sleeps.append,
    )


def test_retries_429_honoring_retry_after():
    sleeps = []
    client = _client(
        [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json={"ok": 1})],
        sleeps,
    )
    assert client.get_json("/x") == {"ok": 1}
    assert 7.0 in sleeps


def test_retries_transport_errors():
    sleeps = []
    client = _client([httpx.ConnectError("boom"), httpx.Response(200, text="hi")], sleeps)
    assert client.get_text("/x") == "hi"


def test_non_retryable_error_raises_immediately():
    sleeps = []
    client = _client([httpx.Response(404)], sleeps)
    with pytest.raises(httpx.HTTPStatusError):
        client.get("/x")
    assert sleeps == []


def test_gives_up_after_max_retries():
    sleeps = []
    client = _client([httpx.Response(503)] * 3, sleeps)
    with pytest.raises(httpx.HTTPStatusError):
        client.get("/x")
    assert len(sleeps) == 2


def test_rate_limit_spaces_requests():
    sleeps = []
    client = RateLimitedClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
        max_per_minute=60,
        sleep=sleeps.append,
        base_url="https://example.test",
    )
    client.get("/a")
    client.get("/b")
    assert sleeps and 0 < sleeps[-1] <= 1.0
