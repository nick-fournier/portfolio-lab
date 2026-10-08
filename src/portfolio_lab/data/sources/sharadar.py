"""Sharadar's API (licensed): the nightly updates of its bulk tables.

Everything lands in the source's raw folder (``<root>/sharadar/raw``), so cancelling means
deleting one directory (``data.conform.sharadar``). Per night:

- ``tickers`` and ``actions``: the whole table (a few MB each), replacing the zip. The
  fundamentals' ticker -> permaticker map (``SF1`` rows) is kept per date in
  ``sf1-tickers/<YYYYMMDD>.parquet`` (written when it changes, starting with the bulk
  download's): fundamentals rows carry only a ticker, valid on the day they were pulled.
- ``fundamentals``: the rows updated since the last pull, as
  ``fundamentals-updates/<YYYYMMDD>.csv`` (the date is when they were pulled, which is
  also the date their share counts are split-adjusted to). A query returns at most
  :data:`MAX_ROWS` rows; a pull that hits the cap downloads the whole table instead and
  clears the updates.

Stock prices are not pulled: Alpaca carries them after the bulk download.
"""

import logging
import shutil
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

log = logging.getLogger(__name__)

URL = "https://api.sharadar.com/v1.0/data/{table}"
#: The API's cap on rows per query.
MAX_ROWS = 10_000
#: Days of overlap between pulls, so a late revision is never missed.
OVERLAP = timedelta(days=3)
UPDATES = "fundamentals-updates"
MAPS = "sf1-tickers"


def _get(key: str, table: str, params: dict, **kwargs) -> httpx.Response:
    response = httpx.get(URL.format(table=table), headers={"x-api-key": key}, params=params,
                         timeout=300, **kwargs)  # fmt: skip
    if response.status_code not in (200, 302):
        response.raise_for_status()
    return response


def download_table(key: str, table: str, raw: Path) -> Path:
    """The whole table as ``<raw>/<table>.zip`` (replaced atomically); its CSV is dropped."""
    response = _get(key, table, {"years": "full"}, follow_redirects=False)
    if response.status_code != 302:
        raise RuntimeError(f"sharadar: no bulk link for {table} ({response.status_code})")
    target, partial = raw / f"{table}.zip", raw / f"{table}.zip.partial"
    with httpx.stream("GET", response.headers["location"], timeout=None) as stream:
        stream.raise_for_status()
        with open(partial, "wb") as out:
            for chunk in stream.iter_bytes(1 << 20):
                out.write(chunk)
    partial.replace(target)
    (raw / f"{table}.csv").unlink(missing_ok=True)  # extracted by the conformer on demand
    return target


def updated_since(key: str, table: str, since: date) -> pl.DataFrame | None:
    """Rows of ``table`` updated on or after ``since``; None if the cap cut them short."""
    response = _get(key, table, {"lastupdated.gte": since.isoformat()})
    rows = pl.read_csv(response.content, infer_schema_length=0)
    return None if rows.height >= MAX_ROWS else rows


def last_pulled(raw: Path) -> date | None:
    """The date of the newest fundamentals update file, if any."""
    names = sorted(p.stem for p in (raw / UPDATES).glob("*.csv"))
    return datetime.strptime(names[-1], "%Y%m%d").date() if names else None


def keep_ticker_map(raw: Path, day: date) -> None:
    """Record the ``SF1`` ticker -> permaticker map of ``tickers.zip`` as of ``day``."""
    with zipfile.ZipFile(raw / "tickers.zip") as z:
        cols = ["table", "ticker", "permaticker"]
        rows = pl.read_csv(z.open(z.namelist()[0]).read(), columns=cols, infer_schema_length=0)
    new = rows.filter(pl.col("table") == "SF1").select("ticker", "permaticker").sort("ticker")
    folder = raw / MAPS
    folder.mkdir(exist_ok=True)
    kept = sorted(folder.glob("*.parquet"))
    if not kept or not pl.read_parquet(kept[-1]).equals(new):
        new.write_parquet(folder / f"{day:%Y%m%d}.parquet")


def fetch(key: str, raw: Path, today: date) -> dict:
    """Tonight's pulls (module docs); returns what was fetched."""
    if not any((raw / MAPS).glob("*.parquet")):  # the bulk download's map, before replacing it
        keep_ticker_map(raw, zip_asof(raw))
    for table in ("tickers", "actions"):
        download_table(key, table, raw)
    keep_ticker_map(raw, today)
    since = (last_pulled(raw) or zip_asof(raw)) - OVERLAP
    rows = updated_since(key, "fundamentals", since)
    if rows is None:
        log.warning("sharadar: fundamentals updates since %s hit the cap; full download", since)
        download_table(key, "fundamentals", raw)
        shutil.rmtree(raw / UPDATES, ignore_errors=True)
        return {"fundamentals": "full"}
    folder = raw / UPDATES
    folder.mkdir(exist_ok=True)
    rows.write_csv(folder / f"{today:%Y%m%d}.csv")
    return {"fundamentals_since": since.isoformat(), "fundamentals_rows": rows.height}


def zip_asof(raw: Path) -> date:
    """The date the bulk fundamentals file is current to: its newest ``lastupdated``."""
    with zipfile.ZipFile(raw / "fundamentals.zip") as z:
        dates = pl.read_csv(z.open(z.namelist()[0]).read(), columns=["lastupdated"])
    return dates["lastupdated"].cast(pl.Date).max()
