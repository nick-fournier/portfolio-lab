"""Sharadar's API (licensed): the nightly updates of its bulk tables.

Everything lands in the source's raw folder (``<root>/sharadar/raw``), so cancelling means
deleting one directory (``data.conform.sharadar``). Per night:

- ``tickers`` and ``actions``: the whole table (a few MB each), replacing the zip. The
  fundamentals' ticker -> permaticker map (``SF1`` rows) is kept per date in
  ``sf1-tickers/<YYYYMMDD>.parquet`` (written when it changes, starting with the bulk
  download's): fundamentals rows carry only a ticker, valid on the day they were pulled.
  The prices' map (``SEP`` and ``SFP`` rows) is kept the same way in ``price-tickers``.
- ``fundamentals``: the rows updated since the last pull, as
  ``fundamentals-updates/<YYYYMMDD>.csv`` (the date is when they were pulled, which is
  also the date their share counts are split-adjusted to). A query returns at most
  :data:`MAX_ROWS` rows; a pull that hits the cap downloads the whole table instead and
  clears the updates.
- ``stocks`` and ``funds`` (daily prices): the rows updated since the last pull, as
  ``<table>-updates/<YYYYMMDD>.parquet``. A dividend or split restates a security's whole
  adjusted history, and Sharadar sends those rows again with a new ``lastupdated``, so the
  updates carry restatements and corrections as well as the new days. A pull that hits
  the cap is split by date range until every part fits (:func:`updated_prices`).
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
#: Rows per query: the API's ``limit`` maximum (its default is 10,000).
MAX_ROWS = 100_000
#: Days of overlap between pulls, so a late revision is never missed.
OVERLAP = timedelta(days=3)
UPDATES = "fundamentals-updates"
MAPS = "sf1-tickers"
#: Where the price tables' ticker -> permaticker maps are kept.
PRICE_MAPS = "price-tickers"
#: The price tables pulled nightly, and the folder each one's updates land in.
PRICE_UPDATES = {"stocks": "stocks-updates", "funds": "funds-updates"}
#: The earliest price date a split query covers (Sharadar's prices start in 1998).
FIRST_PRICE = date(1990, 1, 1)


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
    response = _get(key, table, {"lastupdated.gte": since.isoformat(), "limit": MAX_ROWS})
    rows = pl.read_csv(response.content, infer_schema_length=0)
    return None if rows.height >= MAX_ROWS else rows


def updated_prices(key: str, table: str, since: date, today: date) -> pl.DataFrame:
    """Price rows of ``table`` updated on or after ``since``, every one of them.

    A query that returns :data:`MAX_ROWS` rows may have been cut short, so its date range
    is halved and each half asked again, until every part comes back under the cap.
    """
    parts, ranges = [], [(FIRST_PRICE, today)]
    while ranges:
        lo, hi = ranges.pop()
        params = {"lastupdated.gte": since.isoformat(), "date.gte": lo.isoformat(),
                  "date.lte": hi.isoformat(), "limit": MAX_ROWS}  # fmt: skip
        rows = pl.read_csv(_get(key, table, params).content, infer_schema_length=0)
        if rows.height < MAX_ROWS:
            parts.append(rows)
        elif lo == hi:
            raise RuntimeError(f"sharadar: {table} has {MAX_ROWS}+ rows on {lo} alone")
        else:
            mid = lo + timedelta(days=(hi - lo).days // 2)
            ranges += [(lo, mid), (mid + timedelta(days=1), hi)]
    return pl.concat(parts, how="diagonal") if parts else pl.DataFrame()


def last_pulled(raw: Path, folder: str = UPDATES) -> date | None:
    """The date of the newest update file in ``folder`` (fundamentals by default), if any."""
    names = sorted(p.stem for p in (raw / folder).glob("*.*"))
    return datetime.strptime(names[-1], "%Y%m%d").date() if names else None


def keep_ticker_map(
    raw: Path, day: date, tables: tuple[str, ...] = ("SF1",), name: str = MAPS
) -> None:
    """Record ``tables``' ticker -> permaticker map of ``tickers.zip`` as of ``day``."""
    with zipfile.ZipFile(raw / "tickers.zip") as z:
        cols = ["table", "ticker", "permaticker"]
        rows = pl.read_csv(z.open(z.namelist()[0]).read(), columns=cols, infer_schema_length=0)
    new = (
        rows.filter(pl.col("table").is_in(tables))
        .select("ticker", "permaticker")
        .unique()
        .sort("ticker", "permaticker")
    )
    folder = raw / name
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
    keep_ticker_map(raw, today, ("SEP", "SFP"), PRICE_MAPS)
    out = {}
    for table, name in PRICE_UPDATES.items():
        since = (last_pulled(raw, name) or zip_asof(raw)) - OVERLAP
        rows = updated_prices(key, table, since, today)
        (raw / name).mkdir(exist_ok=True)
        rows.write_parquet(raw / name / f"{today:%Y%m%d}.parquet")
        out[f"{table}_rows"] = rows.height
    since = (last_pulled(raw) or zip_asof(raw)) - OVERLAP
    rows = updated_since(key, "fundamentals", since)
    if rows is None:
        log.warning("sharadar: fundamentals updates since %s hit the cap; full download", since)
        download_table(key, "fundamentals", raw)
        shutil.rmtree(raw / UPDATES, ignore_errors=True)
        return out | {"fundamentals": "full"}
    folder = raw / UPDATES
    folder.mkdir(exist_ok=True)
    rows.write_csv(folder / f"{today:%Y%m%d}.csv")
    return out | {"fundamentals_since": since.isoformat(), "fundamentals_rows": rows.height}


def zip_asof(raw: Path) -> date:
    """The date the bulk fundamentals file is current to: its newest ``lastupdated``."""
    with zipfile.ZipFile(raw / "fundamentals.zip") as z:
        dates = pl.read_csv(z.open(z.namelist()[0]).read(), columns=["lastupdated"])
    return dates["lastupdated"].cast(pl.Date).max()
