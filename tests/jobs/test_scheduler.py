from datetime import UTC, datetime, timedelta

from portfolio_lab.jobs.scheduler import Job, is_due, load_state, run_once

# 2024-07-08 is a Monday; the close is 20:00 UTC and ingest waits until 00:00 UTC (20:00 ET).
MON_AFTERNOON = datetime(2024, 7, 8, 18, 0, tzinfo=UTC)
MON_NIGHT = datetime(2024, 7, 9, 0, 30, tzinfo=UTC)

DAILY = Job("daily", lambda s: None, "session")
WEEKLY = Job("weekly", lambda s: None, "weekly")


def test_session_job_due_once_per_session():
    assert is_due(DAILY, None, MON_AFTERNOON)
    covered_friday = {"last_session": "2024-07-05"}
    assert not is_due(DAILY, covered_friday, MON_AFTERNOON)  # Monday not final until 20:00 ET
    assert is_due(DAILY, covered_friday, MON_NIGHT)
    assert not is_due(DAILY, {"last_session": "2024-07-08"}, MON_NIGHT)


def test_session_job_catches_up_missed_sessions():
    assert is_due(DAILY, {"last_session": "2024-06-28"}, MON_AFTERNOON)


def test_weekly_job():
    assert is_due(WEEKLY, None, MON_NIGHT)
    recent = {"last_success": (MON_NIGHT - timedelta(days=3)).isoformat()}
    old = {"last_success": (MON_NIGHT - timedelta(days=8)).isoformat()}
    assert not is_due(WEEKLY, recent, MON_NIGHT)
    assert is_due(WEEKLY, old, MON_NIGHT)


def test_failed_job_waits_before_retry():
    failed = {"last_error": "boom", "last_attempt": (MON_NIGHT - timedelta(minutes=10)).isoformat()}
    assert not is_due(DAILY, failed, MON_NIGHT)
    failed["last_attempt"] = (MON_NIGHT - timedelta(minutes=45)).isoformat()
    assert is_due(DAILY, failed, MON_NIGHT)


def test_run_once_records_state_and_isolates_failures(settings):
    calls = []

    def boom(_settings):
        calls.append("boom")
        raise RuntimeError("provider down")

    jobs = (Job("fails", boom, "weekly"), Job("daily", lambda s: calls.append("daily"), "session"))
    assert run_once(settings, MON_NIGHT, jobs) == ["fails", "daily"]
    assert calls == ["boom", "daily"]

    state = load_state(settings)
    assert state["fails"]["last_error"] == "RuntimeError: provider down"
    assert state["daily"]["last_session"] == "2024-07-08"
    assert state["daily"]["last_error"] is None

    # Ten minutes later nothing is due: the failure is backing off, daily is done.
    assert run_once(settings, MON_NIGHT + timedelta(minutes=10), jobs) == []


def test_failures_alert_once_then_recovery_and_flags_alert(settings, monkeypatch):
    sent = []

    def record(_settings, title, _message, priority):
        sent.append((title, priority))

    monkeypatch.setattr("portfolio_lab.jobs.scheduler.notify", record)
    outcome = {"fail": True}

    def flaky(_settings):
        if outcome["fail"]:
            raise RuntimeError("provider down")
        return {"quality": {"flags": ["prices.rows: below its range"]}}

    jobs = (Job("flaky", flaky, "weekly"),)
    run_once(settings, MON_NIGHT, jobs)
    run_once(settings, MON_NIGHT + timedelta(minutes=45), jobs)  # same error again: no alert
    assert sent == [("portfolio: flaky failed", 4)]
    outcome["fail"] = False
    run_once(settings, MON_NIGHT + timedelta(minutes=90), jobs)
    assert [t for t, _ in sent[1:]] == ["portfolio: flaky recovered",
                                        "portfolio: 1 data quality flags"]  # fmt: skip
