"""One-off maintenance jobs, run by hand (not scheduled)."""

import shutil
from datetime import date

from portfolio_lab.core.config import Settings
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data.conform import sharadar
from portfolio_lab.jobs.tasks import conform_task, derive_task


def rebuild_ids_task(settings: Settings) -> dict:
    """Rebuild the ids from scratch, then every table keyed by them (``data.ids``).

    For a burn-down, or the one-time move to permanent sids: the old ids are kept beside
    the new as ``ids.<date>``. Sharadar's securities get the same sids on every rebuild;
    the others are renumbered from the reserved range. Conform runs twice so listings
    numbered by one source on the first pass resolve in the others.
    """
    root = settings.data_dir
    paths = DataPaths(root)
    if paths.ids.exists():
        shutil.move(paths.ids, paths.ids.with_name(f"ids.{date.today():%Y%m%d}"))
    out = {"sharadar": sharadar.build(root)}
    conform_task(settings)
    out["conform"] = conform_task(settings)
    shutil.rmtree(paths.forecast_cache, ignore_errors=True)  # keyed by sid
    out["derive"] = derive_task(settings)
    return out
