"""Scheduler: a long-running loop that runs each job when it is due.

State lives in ``_status/scheduler.json`` (last success, last attempt, last error, and for
daily jobs the last session covered), so a restart resumes where it left off and missed
sessions are caught up on the next tick. Jobs are idempotent, so a crash mid-job is
harmless; failures are logged, recorded and retried after :data:`RETRY_AFTER`.

The loop exits cleanly on SIGTERM/SIGINT (``docker stop``).
"""

import json
import logging
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from portfolio_lab.core.calendar import PRICE_UPDATE_DELAY, last_complete_session
from portfolio_lab.core.config import Settings
from portfolio_lab.core.notify import HIGH, LOW, URGENT, notify
from portfolio_lab.core.store import write_status
from portfolio_lab.data.quality import QualityError
from portfolio_lab.jobs import tasks

log = logging.getLogger(__name__)

STATE_JOB = "scheduler"
#: Daily ingest waits until 20:00 New York time (4h after the close) for late prints.
INGEST_BUFFER = PRICE_UPDATE_DELAY
WEEKLY = timedelta(days=7)
#: After a failure, wait this long before trying the job again.
RETRY_AFTER = timedelta(minutes=30)
#: While a job keeps failing with the same error, repeat its alert at most this often.
REALERT_AFTER = timedelta(hours=6)
#: How often the loop wakes up to check for due jobs.
TICK_SECONDS = 300


@dataclass(frozen=True)
class Job:
    """A named unit of work and how often it runs.

    Args:
        name: Job name, used as the key in the scheduler state.
        run: Function taking the settings and returning a summary dict.
        cadence: ``"session"`` (once per completed trading session) or ``"weekly"``.
    """

    name: str
    run: Callable[[Settings], Any]
    cadence: str


JOBS: tuple[Job, ...] = (
    # Fetch, then rewrite every source into the hive and rebuild the derived tables.
    Job("daily_ingest", tasks.daily_ingest_task, "session"),
    Job("sharadar_ingest", tasks.sharadar_ingest_task, "session"),
    Job("conform", tasks.conform_task, "session"),
    Job("derive", tasks.derive_task, "session"),
    # After the day's data: record the paper account, rebalance at month ends, and keep
    # the Paper page's backtest line current with the account.
    Job("paper", tasks.paper_task, "session"),
    Job("paper_backtest", tasks.paper_backtest_task, "session"),
    Job("verify_prices", tasks.verify_task, "weekly"),
    Job("fundamentals", tasks.fundamentals_task, "weekly"),
    Job("macro", tasks.macro_task, "weekly"),
    Job("delisted", tasks.delisted_task, "weekly"),
    Job("scheduled_backtests", tasks.scheduled_backtests_task, "weekly"),
    Job("make_vs_buy", tasks.make_vs_buy_task, "weekly"),
    Job("scoreboard", tasks.scoreboard_task, "weekly"),
    Job("context", tasks.context_task, "weekly"),
)


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def is_due(job: Job, state: dict[str, Any] | None, now: datetime) -> bool:
    """Decide whether ``job`` should run at ``now`` given its recorded state.

    Args:
        job: The job.
        state: The job's entry in the scheduler state (``None`` if it never ran).
        now: Current time, timezone-aware.
    """
    state = state or {}
    last_attempt = _parse(state.get("last_attempt"))
    if state.get("last_error") and last_attempt and now - last_attempt < RETRY_AFTER:
        return False
    if job.cadence == "session":
        target = last_complete_session(now, buffer=INGEST_BUFFER)
        done = state.get("last_session")
        return done is None or date.fromisoformat(done) < target
    last_success = _parse(state.get("last_success"))
    return last_success is None or now - last_success >= WEEKLY


def load_state(settings: Settings) -> dict[str, dict[str, Any]]:
    """Read the scheduler state (empty on first run)."""
    path = settings.data_dir / "_status" / f"{STATE_JOB}.json"
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text()).items() if isinstance(v, dict)}


def run_once(
    settings: Settings, now: datetime | None = None, jobs: tuple[Job, ...] = JOBS
) -> list[str]:
    """Run every due job once, in order; return the names of the jobs that ran.

    A failing job is recorded and does not stop later jobs.
    """
    now = now or datetime.now(UTC)
    state = load_state(settings)
    ran = []
    for job in jobs:
        entry = state.get(job.name, {})
        if not is_due(job, entry, now):
            continue
        previous = entry.get("last_error")
        entry = {**entry, "last_attempt": now.isoformat()}
        try:
            log.info("running %s", job.name)
            result = job.run(settings)
            entry |= {"last_success": now.isoformat(), "last_error": None,
                      "failing_since": None, "retries": 0, "last_alert": None}  # fmt: skip
            if job.cadence == "session":
                session = last_complete_session(now, buffer=INGEST_BUFFER)
                entry["last_session"] = session.isoformat()
            _alert_success(settings, job, result, previous)
        except Exception as exc:  # one failing job must not stop the loop
            log.exception("%s failed", job.name)
            entry["last_error"] = f"{type(exc).__name__}: {exc}"
            _alert_failure(settings, job, entry, previous, isinstance(exc, QualityError), now)
        state[job.name] = entry
        write_status(settings.data_dir, STATE_JOB, state)
        ran.append(job.name)
    return ran


def _alert_failure(
    settings: Settings,
    job: Job,
    entry: dict[str, Any],
    previous: str | None,
    urgent: bool,
    now: datetime,
) -> None:
    """Alert on a failure: a new error at once, a persisting one at most every REALERT_AFTER.

    A stuck job must not go quiet after its first alert, nor send one per retry.
    """
    new = entry["last_error"] != previous
    if new:
        entry["failing_since"], entry["retries"] = now.isoformat(), 0
    else:
        entry["retries"] = entry.get("retries", 0) + 1
    last_alert = _parse(entry.get("last_alert"))
    if not new and last_alert is not None and now - last_alert < REALERT_AFTER:
        return
    if new:
        title, body = f"portfolio: {job.name} failed", entry["last_error"]
    else:
        since = _parse(entry.get("failing_since")) or now
        title = f"portfolio: {job.name} still failing"
        retries = entry["retries"]
        body = f"since {since:%Y-%m-%d %H:%M} UTC, {retries} retries: {entry['last_error']}"
    notify(settings, title, body[:1000], URGENT if urgent else HIGH)
    entry["last_alert"] = now.isoformat()


def _alert_success(settings: Settings, job: Job, result: Any, previous: str | None) -> None:
    """A recovery after a failure, and any soft quality flags the job reports."""
    if previous:
        notify(settings, f"portfolio: {job.name} recovered", f"was: {previous[:500]}", LOW)
    flags = (result or {}).get("quality", {}).get("flags") if isinstance(result, dict) else None
    if flags:
        notify(settings, f"portfolio: {len(flags)} data quality flags", "\n".join(flags), LOW)


def run_forever(settings: Settings, tick_seconds: int = TICK_SECONDS) -> None:
    """Check for due jobs every ``tick_seconds`` until SIGTERM or SIGINT."""
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    log.info("scheduler started; checking every %ds", tick_seconds)
    while not stop.is_set():
        run_once(settings)
        stop.wait(tick_seconds)
    log.info("scheduler stopped")
