"""Push alerts to a phone through ntfy (https://ntfy.sh): one HTTP request per message.

Subscribe to the topic in the ntfy app. Without ``NTFY_TOPIC`` nothing is sent, and a
failed send is logged, never raised: an alert must not break the job it reports on.
"""

import logging

import httpx

from portfolio_lab.core.config import Settings

log = logging.getLogger(__name__)

#: ntfy's priorities: 1 (min) to 5 (urgent; overrides the phone's do-not-disturb).
LOW, DEFAULT, HIGH, URGENT = 2, 3, 4, 5


def notify(settings: Settings, title: str, message: str, priority: int = DEFAULT) -> bool:
    """Send one alert; return whether it was delivered."""
    if settings.ntfy_topic is None:
        return False
    url = f"{settings.ntfy_url.rstrip('/')}/{settings.ntfy_topic.get_secret_value()}"
    try:
        response = httpx.post(
            url, content=message.encode(), timeout=10,
            headers={"Title": title, "Priority": str(priority)},
        )  # fmt: skip
        response.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("alert not sent (%s): %s", type(exc).__name__, title)
        return False
    return True
