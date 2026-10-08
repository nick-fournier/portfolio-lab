import httpx

from portfolio_lab.core.config import Settings
from portfolio_lab.core.notify import URGENT, notify


def test_without_a_topic_nothing_is_sent(settings, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert not notify(settings, "title", "body")


def test_posts_title_priority_and_body_to_the_topic(tmp_path, monkeypatch):
    calls = []

    def post(url, content, timeout, headers):
        calls.append((url, content, headers))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    s = Settings(_env_file=None, PORTFOLIO_DATA_DIR=tmp_path, NTFY_TOPIC="t0p1c")
    assert notify(s, "portfolio: derive failed", "QualityError: dupes", URGENT)
    assert calls == [("https://ntfy.sh/t0p1c", b"QualityError: dupes",
                      {"Title": "portfolio: derive failed", "Priority": "5"})]  # fmt: skip


def test_a_failed_send_is_logged_not_raised(tmp_path, monkeypatch):
    def post(url, **_):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "post", post)
    s = Settings(_env_file=None, PORTFOLIO_DATA_DIR=tmp_path, NTFY_TOPIC="t0p1c")
    assert not notify(s, "title", "body")
