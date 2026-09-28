from datetime import UTC, date, datetime

from portfolio_lab.core.calendar import (
    is_session,
    last_complete_session,
    previous_session,
    rebalance_dates,
    sessions,
    sessions_back,
)


def test_sessions_skip_weekends_and_holidays():
    days = sessions(date(2024, 7, 1), date(2024, 7, 8))
    assert date(2024, 7, 4) not in days  # Independence Day
    assert date(2024, 7, 6) not in days  # Saturday
    assert days == [
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
        date(2024, 7, 8),
    ]


def test_is_and_previous_session():
    assert is_session(date(2024, 7, 5))
    assert not is_session(date(2024, 7, 4))
    assert previous_session(date(2024, 7, 5)) == date(2024, 7, 3)
    assert previous_session(date(2024, 7, 7)) == date(2024, 7, 5)  # Sunday -> Friday


def test_sessions_back():
    assert sessions_back(date(2024, 7, 8), 2) == date(2024, 7, 3)
    assert sessions_back(date(2024, 7, 6), 1) == date(2024, 7, 3)  # Saturday anchors on Friday


def test_rebalance_dates():
    days = sessions(date(2024, 1, 1), date(2024, 3, 31))
    assert rebalance_dates(days, "M") == [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 28)]
    weekly = rebalance_dates(days, "W")
    assert weekly[0] == date(2024, 1, 5)
    assert date(2024, 3, 28) in weekly  # Good Friday week ends Thursday
    assert rebalance_dates(days, "D") == days


def test_last_complete_session():
    # 2024-07-08 (Mon, EDT): close 16:00 ET = 20:00 UTC; buffer 1h.
    assert last_complete_session(datetime(2024, 7, 8, 15, 0, tzinfo=UTC)) == date(2024, 7, 5)
    assert last_complete_session(datetime(2024, 7, 8, 20, 30, tzinfo=UTC)) == date(2024, 7, 5)
    assert last_complete_session(datetime(2024, 7, 8, 21, 0, tzinfo=UTC)) == date(2024, 7, 8)
    # 01:00 UTC Tuesday is still Monday evening in New York.
    assert last_complete_session(datetime(2024, 7, 9, 1, 0, tzinfo=UTC)) == date(2024, 7, 8)
    # Saturday -> Friday; early close on 2024-07-03 (13:00 ET = 17:00 UTC).
    assert last_complete_session(datetime(2024, 7, 6, 12, 0, tzinfo=UTC)) == date(2024, 7, 5)
    assert last_complete_session(datetime(2024, 7, 3, 18, 30, tzinfo=UTC)) == date(2024, 7, 3)
