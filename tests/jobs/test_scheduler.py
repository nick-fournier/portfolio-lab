import json
import threading
import time
from datetime import UTC, datetime, timedelta

from portfolio_lab.jobs import scheduler
from portfolio_lab.jobs.scheduler import (
    CHAINS,
    Chain,
    Job,
    daily_slot,
    load_state,
    next_daily,
    next_weekly,
    run_once,
    weekly_slot,
)

# 2024-07-08 is a Monday; the close is 20:00 UTC and the daily chain waits until 00:00 UTC.
MON_AFTERNOON = datetime(2024, 7, 8, 18, 0, tzinfo=UTC)
MON_NIGHT = datetime(2024, 7, 9, 0, 30, tzinfo=UTC)


def _utc(*args):
    return datetime(*args, tzinfo=UTC)


def test_daily_slot_is_the_latest_close_plus_the_wait():
    assert daily_slot(MON_AFTERNOON) == _utc(2024, 7, 6, 0, 0)  # Friday, 20:00 New York
    assert daily_slot(MON_NIGHT) == _utc(2024, 7, 9, 0, 0)  # Monday's
    assert next_daily(daily_slot(MON_AFTERNOON)) == _utc(2024, 7, 9, 0, 0)  # skips the weekend
    # Wednesday 2024-07-03 closed early (13:00 New York): its run is at 17:00.
    assert next_daily(_utc(2024, 7, 3, 0, 0)) == _utc(2024, 7, 3, 21, 0)


def test_weekly_slot_is_saturday_six_am_new_york():
    assert weekly_slot(MON_NIGHT) == _utc(2024, 7, 6, 10, 0)  # 06:00 EDT
    assert weekly_slot(_utc(2024, 7, 6, 9, 59)) == _utc(2024, 6, 29, 10, 0)  # a minute early
    assert weekly_slot(_utc(2024, 7, 6, 10, 0)) == _utc(2024, 7, 6, 10, 0)
    # Clocks go back on 2024-11-03: still 06:00 New York, now 11:00 UTC.
    assert next_weekly(_utc(2024, 11, 2, 10, 0)) == _utc(2024, 11, 9, 11, 0)


def test_the_real_chains_only_need_earlier_jobs_of_their_own():
    for chain in CHAINS:
        seen = set()
        for job in chain.jobs:
            assert set(job.needs) <= seen, (chain.name, job.name)  # listed after its needs
            seen.add(job.name)


def _chain(*jobs, name="test"):
    return Chain(name, daily_slot, next_daily, jobs)


def test_jobs_wait_for_their_needs_and_independent_ones_overlap(settings):
    lock, live, peak, order = threading.Lock(), {"heavy": 0, "light": 0}, {}, []
    both_fetching = threading.Barrier(2, timeout=5)  # breaks unless the fetches overlap

    def work(name, kind):
        def run(_settings):
            with lock:
                live[kind] += 1
                peak[kind] = max(peak.get(kind, 0), live[kind])
                order.append(name)
            if kind == "light":
                both_fetching.wait()
            else:
                time.sleep(0.02)  # long enough for a second heavy job to be caught overlapping
            with lock:
                live[kind] -= 1

        return run

    chain = _chain(
        Job("fetch_a", work("fetch_a", "light")),
        Job("fetch_b", work("fetch_b", "light")),
        Job("build", work("build", "heavy"), ("fetch_a", "fetch_b"), heavy=True),
        Job("model_a", work("model_a", "heavy"), ("build",), heavy=True),
        Job("model_b", work("model_b", "heavy"), ("build",), heavy=True),
    )
    assert sorted(run_once(settings, MON_NIGHT, (chain,))) == sorted(
        ["fetch_a", "fetch_b", "build", "model_a", "model_b"]
    )
    assert order.index("build") > max(order.index("fetch_a"), order.index("fetch_b"))
    assert order.index("build") < min(order.index("model_a"), order.index("model_b"))
    assert peak == {"light": 2, "heavy": 1}  # fetches side by side, heavy jobs one at a time
    assert run_once(settings, MON_NIGHT + timedelta(minutes=5), (chain,)) == []  # run complete


def test_a_failure_stops_only_what_needs_it_and_retries_later(settings):
    calls, outcome = [], {"fail": True}

    def flaky(_settings):
        calls.append("flaky")
        if outcome["fail"]:
            raise RuntimeError("provider down")

    chain = _chain(
        Job("flaky", flaky),
        Job("after", lambda s: calls.append("after"), ("flaky",)),
        Job("other", lambda s: calls.append("other")),
    )
    assert sorted(run_once(settings, MON_NIGHT, (chain,))) == ["flaky", "other"]
    entry = load_state(settings, (chain,))["test"]
    assert entry["jobs"]["flaky"]["last_error"] == "RuntimeError: provider down"
    assert "completed" not in entry
    assert run_once(settings, MON_NIGHT + timedelta(minutes=10), (chain,)) == []  # backing off
    outcome["fail"] = False
    assert run_once(settings, MON_NIGHT + timedelta(minutes=45), (chain,)) == ["flaky", "after"]
    assert calls == ["flaky", "other", "flaky", "after"]
    assert load_state(settings, (chain,))["test"]["completed"] == "2024-07-09T00:00:00+00:00"


def test_a_new_slot_runs_every_job_again(settings):
    calls = []
    chain = _chain(Job("a", lambda s: calls.append("a")))
    run_once(settings, MON_AFTERNOON, (chain,))  # Friday's run
    run_once(settings, MON_AFTERNOON + timedelta(hours=1), (chain,))  # still Friday's: nothing
    run_once(settings, MON_NIGHT, (chain,))  # Monday's
    assert calls == ["a", "a"]
    assert load_state(settings, (chain,))["test"]["next"] == "2024-07-10T00:00:00+00:00"


def test_state_from_before_chains_is_ignored(settings):
    path = settings.data_dir / "_status" / "scheduler.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"daily_ingest": {"last_session": "2024-07-05"}}))
    assert load_state(settings) == {}


def test_failures_alert_once_then_recovery_and_flags_alert(settings, monkeypatch):
    sent = []
    monkeypatch.setattr(scheduler, "notify", lambda _s, title, _m, priority: sent.append(
        (title, priority)))  # fmt: skip
    outcome = {"fail": True}

    def flaky(_settings):
        if outcome["fail"]:
            raise RuntimeError("provider down")
        return {"quality": {"flags": ["prices.rows: below its range"]}}

    chain = _chain(Job("flaky", flaky))
    run_once(settings, MON_NIGHT, (chain,))
    run_once(settings, MON_NIGHT + timedelta(minutes=45), (chain,))  # same error: no alert
    assert sent == [("portfolio: flaky failed", 4)]
    outcome["fail"] = False
    run_once(settings, MON_NIGHT + timedelta(minutes=90), (chain,))
    assert [t for t, _ in sent[1:]] == ["portfolio: flaky recovered",
                                        "portfolio: 1 data quality flags"]  # fmt: skip


def test_a_persistent_failure_repeats_its_alert_every_six_hours(settings, monkeypatch):
    sent = []
    monkeypatch.setattr(scheduler, "notify", lambda _s, title, message, _p: sent.append(
        (title, message)))  # fmt: skip

    def broken(_settings):
        raise RuntimeError("provider down")

    chain = _chain(Job("broken", broken))
    for minutes in range(0, 7 * 60, 30):  # a retry every 30 minutes for 7 hours
        run_once(settings, MON_NIGHT + timedelta(minutes=minutes), (chain,))
    assert [t for t, _ in sent] == ["portfolio: broken failed", "portfolio: broken still failing"]
    since = "since 2024-07-09 00:30 UTC, 12 retries: RuntimeError: provider down"
    assert sent[1][1].startswith(since)
