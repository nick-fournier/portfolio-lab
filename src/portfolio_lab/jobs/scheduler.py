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

from portfolio_lab.core.calendar import last_complete_session
from portfolio_lab.core.config import Settings
from portfolio_lab.core.store import write_status
from portfolio_lab.jobs import tasks

log = logging.getLogger(__name__)

STATE_JOB = "scheduler"
#: Daily ingest waits until 20:00 New York time (4h after the close) for late prints.
INGEST_BUFFER = timedelta(hours=4)
WEEKLY = timedelta(days=7)
#: After a failure, wait this long before trying the job again.
RETRY_AFTER = timedelta(minutes=30)
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
    Job("daily_ingest", tasks.daily_ingest_task, "session"),
    Job("verify_prices", tasks.verify_task, "weekly"),
    Job("fundamentals", tasks.fundamentals_task, "weekly"),
    Job("macro", tasks.macro_task, "weekly"),
    Job("features", tasks.features_task, "weekly"),
    Job("delisted", tasks.delisted_task, "weekly"),
    Job("scheduled_backtests", tasks.scheduled_backtests_task, "weekly"),
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
        entry = {**entry, "last_attempt": now.isoformat()}
        try:
            log.info("running %s", job.name)
            job.run(settings)
            entry |= {"last_success": now.isoformat(), "last_error": None}
            if job.cadence == "session":
                session = last_complete_session(now, buffer=INGEST_BUFFER)
                entry["last_session"] = session.isoformat()
        except Exception as exc:  # one failing job must not stop the loop
            log.exception("%s failed", job.name)
            entry["last_error"] = f"{type(exc).__name__}: {exc}"
        state[job.name] = entry
        write_status(settings.data_dir, STATE_JOB, state)
        ran.append(job.name)
    return ran


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
