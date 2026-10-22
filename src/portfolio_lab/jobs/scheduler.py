"""Scheduler: two chains of jobs at fixed New York times, each job after the jobs it needs.

- **Daily**, every trading day at the close plus :data:`INGEST_BUFFER` (20:00 New York,
  17:00 on early-close days): fetch the day's data, rebuild the hive, record the paper
  account and rerun its backtest.
- **Weekly**, Saturdays at :data:`WEEKLY_AT` New York: the slower fetches (EDGAR, FRED,
  delisted stocks, the price spot check), the hive again so they count, then the weekly
  models (the scheduled backtests, the scoreboard, market context, make versus buy).

A chain's run is named by its latest scheduled time, its *slot*. Within a run a job starts
once every job it needs has succeeded in that run, and jobs that don't depend on each other
run at the same time: at most :data:`HEAVY_AT_ONCE` heavy ones (anything that rewrites the
hive or loads the full price panel, several GB each) and :data:`LIGHT_AT_ONCE` light ones
(mostly waiting on the network). A failed job is retried after :data:`RETRY_AFTER` and only
the jobs that need it wait; the rest of the run finishes. A slot missed while the machine was
off runs at the next check, and only the latest slot of a chain is ever run. Chains run one
at a time, daily first.

State lives in ``_status/scheduler.json``: per chain, the last completed slot and each job's
last success, attempt and error. A restart resumes the run where it left off; jobs are
idempotent, so a crash mid-job is harmless.

The loop exits cleanly on SIGTERM/SIGINT (``docker stop``).
"""

import json
import logging
import signal
import threading
from collections.abc import Callable, Collection
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from portfolio_lab.core.calendar import (
    NEW_YORK,
    PRICE_UPDATE_DELAY,
    last_complete_session,
    next_session,
    session_close,
)
from portfolio_lab.core.config import Settings
from portfolio_lab.core.notify import HIGH, LOW, URGENT, notify
from portfolio_lab.core.store import write_status
from portfolio_lab.data.quality import QualityError
from portfolio_lab.jobs import tasks

log = logging.getLogger(__name__)

STATE_JOB = "scheduler"
#: The daily chain waits this long after the close (20:00 New York) for late prints.
INGEST_BUFFER = PRICE_UPDATE_DELAY
#: The weekly chain's day (Monday is 0) and time, New York.
WEEKLY_DAY, WEEKLY_AT = 5, time(6, 0)
#: Heavy jobs at once: each loads several GB, and the container has 10.
HEAVY_AT_ONCE = 1
#: Light jobs at once (fetches, mostly waiting on their providers' rate limits).
LIGHT_AT_ONCE = 4
#: After a failure, wait this long before trying the job again.
RETRY_AFTER = timedelta(minutes=30)
#: While a job keeps failing with the same error, repeat its alert at most this often.
REALERT_AFTER = timedelta(hours=6)
#: How often the loop wakes up to check for due runs.
TICK_SECONDS = 300


@dataclass(frozen=True)
class Job:
    """A named unit of work, the jobs it needs, and whether it is heavy.

    Args:
        name: Job name, the key in its chain's state.
        run: Function taking the settings and returning a summary dict.
        needs: Jobs of the same chain that must succeed first in the same run.
        heavy: Rewrites the hive or loads the full price panel (see :data:`HEAVY_AT_ONCE`).
    """

    name: str
    run: Callable[[Settings], Any]
    needs: tuple[str, ...] = ()
    heavy: bool = False


@dataclass(frozen=True)
class Chain:
    """Jobs run together at a fixed schedule.

    Args:
        name: Chain name, the key in the scheduler state.
        slot: The chain's latest scheduled time at or before a moment.
        following: The scheduled time after a slot.
        jobs: The jobs, in the order they start when several are ready.
    """

    name: str
    slot: Callable[[datetime], datetime]
    following: Callable[[datetime], datetime]
    jobs: tuple[Job, ...]


def daily_slot(now: datetime) -> datetime:
    """The latest trading day's close plus :data:`INGEST_BUFFER`, at or before ``now``."""
    return session_close(last_complete_session(now, buffer=INGEST_BUFFER)) + INGEST_BUFFER


def next_daily(slot: datetime) -> datetime:
    """The next trading day's close plus :data:`INGEST_BUFFER`."""
    return session_close(next_session(slot.astimezone(NEW_YORK).date())) + INGEST_BUFFER


def _weekly_on(day: date) -> datetime:
    return datetime.combine(day, WEEKLY_AT, NEW_YORK).astimezone(UTC)


def weekly_slot(now: datetime) -> datetime:
    """The latest :data:`WEEKLY_DAY` at :data:`WEEKLY_AT` New York, at or before ``now``."""
    today = now.astimezone(NEW_YORK).date()
    day = today - timedelta(days=(today.weekday() - WEEKLY_DAY) % 7)
    slot = _weekly_on(day)
    return slot if slot <= now else _weekly_on(day - timedelta(days=7))


def next_weekly(slot: datetime) -> datetime:
    """The same day and time a week later (New York wall time, across clock changes)."""
    return _weekly_on(slot.astimezone(NEW_YORK).date() + timedelta(days=7))


def _paper(settings: Settings) -> dict:
    """The paper account, without its own rebuild of the hive: the chain's derive just ran."""
    return tasks.paper_task(settings, refresh=False)


DAILY = Chain("daily", daily_slot, next_daily, (
    Job("daily_ingest", tasks.daily_ingest_task),
    Job("sharadar_ingest", tasks.sharadar_ingest_task),
    # Not needed by conform: a day-old T-bill rate is fine, and it is retried on its own.
    Job("rates", tasks.ingest_rates_task),
    Job("conform", tasks.conform_task, ("daily_ingest", "sharadar_ingest"), heavy=True),
    Job("derive", tasks.derive_task, ("conform",), heavy=True),
    Job("paper", _paper, ("derive",), heavy=True),
    Job("paper_backtest", tasks.paper_backtest_task, ("derive",), heavy=True),
))  # fmt: skip
WEEKLY = Chain("weekly", weekly_slot, next_weekly, (
    Job("fundamentals", tasks.fundamentals_task),
    Job("macro", tasks.macro_task),
    # Both write the price store, so one after the other.
    Job("delisted", tasks.delisted_task),
    Job("verify_prices", tasks.verify_task, ("delisted",)),
    Job("conform", tasks.conform_task, ("fundamentals", "macro", "verify_prices"), heavy=True),
    Job("derive", tasks.derive_task, ("conform",), heavy=True),
    Job("scheduled_backtests", tasks.scheduled_backtests_task, ("derive",), heavy=True),
    Job("scoreboard", tasks.scoreboard_task, ("derive",), heavy=True),
    Job("make_vs_buy", tasks.make_vs_buy_task, ("scheduled_backtests",)),
    Job("context", tasks.context_task, ("scoreboard",), heavy=True),
))  # fmt: skip
CHAINS: tuple[Chain, ...] = (DAILY, WEEKLY)


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def load_state(settings: Settings, chains: tuple[Chain, ...] = CHAINS) -> dict[str, dict[str, Any]]:
    """Read ``chains``' state (empty on first run); entries from before chains are dropped."""
    path = settings.data_dir / "_status" / f"{STATE_JOB}.json"
    if not path.exists():
        return {}
    state = json.loads(path.read_text())
    return {c.name: state[c.name] for c in chains if isinstance(state.get(c.name), dict)}


def _backing_off(entry: dict[str, Any], now: datetime) -> bool:
    attempt = _parse(entry.get("last_attempt"))
    return bool(entry.get("last_error")) and attempt is not None and now - attempt < RETRY_AFTER


def _full(busy: Collection[Job], job: Job) -> bool:
    """Whether ``job``'s kind (heavy or light) is at its limit with ``busy`` running."""
    heavy = sum(j.heavy for j in busy)
    return heavy >= HEAVY_AT_ONCE if job.heavy else len(busy) - heavy >= LIGHT_AT_ONCE


def run_chain(
    settings: Settings,
    chain: Chain,
    state: dict[str, dict[str, Any]],
    clock: Callable[[], datetime],
) -> list[str]:
    """Run whatever is left of ``chain``'s latest slot; return the jobs that ran (module docs).

    Updates ``state`` and writes it after every job.
    """
    now = clock()
    slot = chain.slot(now).isoformat()
    record = state.setdefault(chain.name, {})
    record["next"] = chain.following(chain.slot(now)).isoformat()
    if record.get("completed") == slot:
        return []
    entries: dict[str, dict[str, Any]] = record.setdefault("jobs", {})
    done = {j.name for j in chain.jobs if entries.get(j.name, {}).get("slot") == slot}
    blocked = {j.name for j in chain.jobs if j.name not in done
               and _backing_off(entries.get(j.name, {}), now)}  # fmt: skip
    running: dict[Future, Job] = {}
    ran: list[str] = []
    with ThreadPoolExecutor(max_workers=HEAVY_AT_ONCE + LIGHT_AT_ONCE) as pool:
        while True:
            for job in chain.jobs:
                if job.name in done | blocked or job in running.values():
                    continue
                if not all(n in done for n in job.needs) or _full(running.values(), job):
                    continue
                entry = entries.setdefault(job.name, {})
                entry["last_attempt"] = clock().isoformat()
                log.info("%s: running %s", chain.name, job.name)
                running[pool.submit(job.run, settings)] = job
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                job = running.pop(future)
                if _record(settings, job, entries, future, slot, clock()):
                    done.add(job.name)
                else:
                    blocked.add(job.name)
                ran.append(job.name)
                write_status(settings.data_dir, STATE_JOB, state)
    if done == {j.name for j in chain.jobs}:
        record["completed"] = slot
    write_status(settings.data_dir, STATE_JOB, state)
    return ran


def _record(
    settings: Settings,
    job: Job,
    entries: dict[str, dict[str, Any]],
    future: Future,
    slot: str,
    now: datetime,
) -> bool:
    """Save a finished job's outcome in its chain's entries and alert; return its success."""
    entry = entries[job.name]
    previous = entry.get("last_error")
    try:
        result = future.result()
    except Exception as exc:  # one failing job must only stop the jobs that need it
        log.error("%s failed", job.name, exc_info=exc)
        entry["last_error"] = f"{type(exc).__name__}: {exc}"
        _alert_failure(settings, job, entry, previous, isinstance(exc, QualityError), now)
        return False
    entry |= {"slot": slot, "last_success": now.isoformat(), "last_error": None,
              "failing_since": None, "retries": 0, "last_alert": None}  # fmt: skip
    _alert_success(settings, job, result, previous)
    return True


def run_once(
    settings: Settings, now: datetime | None = None, chains: tuple[Chain, ...] = CHAINS
) -> list[str]:
    """Run every chain's due work once, daily first; return the jobs that ran.

    Args:
        settings: Application settings.
        now: A fixed time for every step (tests); default the system clock.
        chains: The chains to consider.
    """
    clock = (lambda: now) if now is not None else (lambda: datetime.now(UTC))
    state = load_state(settings, chains)
    return [name for chain in chains for name in run_chain(settings, chain, state, clock)]


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
    """Check for due runs every ``tick_seconds`` until SIGTERM or SIGINT."""
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    log.info("scheduler started; checking every %ds", tick_seconds)
    while not stop.is_set():
        run_once(settings)
        stop.wait(tick_seconds)
    log.info("scheduler stopped")
