"""NYSE trading calendar: sessions and rebalance schedules.

Wraps ``exchange_calendars`` (XNYS) and returns plain ``datetime.date`` values, which is
what every dataset in the lab is keyed on.
"""

from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

Frequency = Literal["D", "W", "M", "Q"]

NEW_YORK = ZoneInfo("America/New_York")


@lru_cache
def _calendar() -> xcals.ExchangeCalendar:
    """Return the (cached) NYSE calendar."""
    return xcals.get_calendar("XNYS")


def sessions(start: date, end: date) -> list[date]:
    """Return the NYSE trading sessions from ``start`` to ``end``, inclusive."""
    days = _calendar().sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [d.date() for d in days]


def is_session(day: date) -> bool:
    """Return whether the NYSE is open on ``day``."""
    return bool(_calendar().is_session(pd.Timestamp(day)))


def previous_session(day: date) -> date:
    """Return the last NYSE session strictly before ``day``."""
    before = pd.Timestamp(day) - pd.Timedelta(days=1)
    return _calendar().date_to_session(before, direction="previous").date()


def last_complete_session(
    now: datetime | None = None, buffer: timedelta = timedelta(hours=1)
) -> date:
    """Return the most recent session whose daily bar is final.

    Today's session counts only once ``buffer`` has passed after its close (handling early
    closes), so a mid-day run never stores a partial bar.

    Args:
        now: Current time (timezone-aware); defaults to the system clock.
        buffer: How long after the close to wait for final prints.
    """
    now = now or datetime.now(UTC)
    today = now.astimezone(NEW_YORK).date()
    cal = _calendar()
    if cal.is_session(pd.Timestamp(today)):
        close = cal.session_close(pd.Timestamp(today)).to_pydatetime()
        if now >= close + buffer:
            return today
    return previous_session(today)


def sessions_back(day: date, n: int) -> date:
    """Return the session ``n`` sessions before ``day`` (``day`` itself counts if a session)."""
    anchor = pd.Timestamp(day)
    cal = _calendar()
    if not cal.is_session(anchor):
        anchor = cal.date_to_session(anchor, direction="previous")
    return cal.session_offset(anchor, -n).date()


def rebalance_dates(days: list[date], freq: Frequency) -> list[date]:
    """Pick rebalance dates from a list of sessions.

    Args:
        days: Trading sessions in ascending order.
        freq: ``"D"`` for every session, ``"W"`` for the last session of each ISO week,
            ``"M"`` for the last session of each month, ``"Q"`` of each calendar quarter.

    Returns:
        The subset of ``days`` on which to rebalance, in ascending order.
    """
    if freq == "D":
        return list(days)
    keys = {
        "W": lambda d: d.isocalendar()[:2],
        "M": lambda d: (d.year, d.month),
        "Q": lambda d: (d.year, (d.month - 1) // 3),
    }
    key = keys[freq]
    last: dict[tuple[int, int], date] = {}
    for d in days:
        last[key(d)] = d
    return sorted(last.values())
