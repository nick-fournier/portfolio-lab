"""Sharadar's bulk files as conformed tables, and the seed of the security ids.

The tables are ``tickers``, ``actions``, ``stocks``, ``funds`` and ``fundamentals``.

Prices: Sharadar's OHLC and volume are split-adjusted, ``closeunadj`` is the traded close
and ``closeadj`` adjusts for dividends too; raw levels undo the split factor
(``closeunadj / close``), keeping dollar volume unchanged. They come from the bulk table
plus the nightly updates (``sources.sharadar``), each ticker and day's latest version
winning; a dividend or split restates a security's whole history in one update, so every
security's adjusted close stays on one scale. Filings: the as-reported rows
(``ART`` = trailing twelve months, ``ARQ`` = the quarter), dated by the SEC filing date,
from the bulk table plus the nightly updates (``sources.sharadar``), each filing's latest
version winning. Sharadar adjusts share counts for every split after the filing up to
when the row was pulled; dividing by those splits' ratios restores the count as reported.
Everything under the license stays inside the source's folder, so cancelling means
deleting one directory.

:func:`build` writes every table from the bulk zips (prices included, about 3 minutes);
:func:`update` is the nightly step: ids, listings, actions, filings and prices from the
latest tickers, actions, fundamentals and price updates.
"""

import logging
import shutil
import zipfile
from datetime import date, datetime
from pathlib import Path

import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data import ids as ids_
from portfolio_lab.data import reader
from portfolio_lab.data.sources.sharadar import (
    MAPS,
    PRICE_MAPS,
    PRICE_UPDATES,
    UPDATES,
    zip_asof,
)

log = logging.getLogger(__name__)

SOURCE = "sharadar"
#: Sharadar fundamentals column -> concept (``schemas.CONCEPTS`` and ``OPTIONAL_CONCEPTS``).
CONCEPTS = {
    "netinc": "net_income", "ncfo": "cfo", "revenue": "revenue", "gp": "gross_profit",
    "cor": "cost_of_revenue", "opinc": "operating_income", "capex": "capex",
    "ncfdiv": "dividends", "shareswa": "shares_weighted", "assets": "assets",
    "assetsc": "assets_cur", "liabilitiesc": "liab_cur", "liabilities": "liabilities",
    "debtnc": "lt_debt", "debtc": "debt_cur", "equity": "equity", "cashneq": "cash",
    "sgna": "sga", "intexp": "interest_expense", "rnd": "rnd", "depamor": "depreciation",
    "ppnenet": "ppe", "inventory": "inventory", "receivables": "receivables",
    "taxexp": "tax", "ebit": "ebit", "intangibles": "intangibles", "payables": "payables",
    "investmentsc": "investments_cur", "investmentsnc": "investments_lt",
}  # fmt: skip
#: Cash outflows Sharadar reports as negatives; stored as positive amounts.
OUTFLOWS = ("capex", "dividends")
PERIODS = {"ART": "ttm", "ARQ": "q"}
#: Securities per pass over the price tables (bounds memory; see :func:`prices`).
CHUNKS = 6
#: Actions kept, with Sharadar's names.
ACTIONS = (
    "dividend", "split", "listed", "delisted", "acquisitionby", "mergerfrom",
    "acquisitioncash", "acquisitionstock", "acquisitionelectcash", "acquisitionelectstock",
    "spacmerger", "bankruptcyliquidation", "regulatorydelisting", "voluntarydelisting",
    "tickerchangefrom", "tickerchangeto",
)  # fmt: skip


def read(raw: Path, table: str, columns: list[str] | None = None) -> pl.DataFrame:
    """One bulk zip's CSV, all columns as strings."""
    with zipfile.ZipFile(raw / f"{table}.zip") as z:
        return pl.read_csv(z.open(z.namelist()[0]).read(), columns=columns, infer_schema_length=0)


def scan(raw: Path, table: str, dtypes: dict[str, pl.DataType] | None = None) -> pl.LazyFrame:
    """Lazily scan a large bulk table, extracting its CSV next to the zip once.

    Columns in ``dtypes`` are parsed as such (the rest as strings), which keeps the
    stocks table's 36 M rows from being read as text first.
    """
    csv = raw / f"{table}.csv"
    if not csv.exists():
        with zipfile.ZipFile(raw / f"{table}.zip") as z, open(csv, "wb") as f:
            shutil.copyfileobj(z.open(z.namelist()[0]), f)
    return pl.scan_csv(csv, infer_schema_length=0, schema_overrides=dtypes)


#: Price columns read from the bulk tables and the updates.
PRICE_COLS = {"ticker": pl.String, "date": pl.Date, "open": pl.Float64, "high": pl.Float64,
              "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
              "closeadj": pl.Float64, "closeunadj": pl.Float64, "lastupdated": pl.Date}  # fmt: skip


def pulls(raw: Path, table: str, bulk_asof: date) -> pl.LazyFrame:
    """Every pulled price row of ``table``, with the day its ticker was valid on.

    Adds ``asof`` (``bulk_asof`` for the bulk table, else the day the update was pulled)
    and ``_pull`` (-1 for the bulk table, then each update in order).
    """
    bulk = scan(raw, table, PRICE_COLS).select(list(PRICE_COLS))
    parts = [bulk.with_columns(pl.lit(bulk_asof).alias("asof"), pl.lit(-1).alias("_pull"))]
    for i, f in enumerate(sorted((raw / PRICE_UPDATES[table]).glob("*.parquet"))):
        pulled = datetime.strptime(f.stem, "%Y%m%d").date()
        part = pl.scan_parquet(f).select(  # pulled as text
            pl.col(c).str.to_date() if t == pl.Date else pl.col(c).cast(t)
            for c, t in PRICE_COLS.items()
        )
        parts.append(part.with_columns(pl.lit(pulled).alias("asof"), pl.lit(i).alias("_pull")))
    return pl.concat(parts)


def price_sids(rows: pl.LazyFrame, raw: Path, ids: ids_.Ids, sids: pl.DataFrame) -> pl.DataFrame:
    """ticker, asof, sid for every ticker of the pulled ``rows`` (:func:`pulls`).

    A row's ticker is valid on the day it was pulled: it resolves through Sharadar's own
    ticker -> permaticker map as of that day, the prices' (``PRICE_MAPS``) else the
    fundamentals' (``MAPS``; the bulk download's prices map was not kept), else today's
    tickers (``sids``), else the name in use that day in the ids (a security Sharadar
    has since dropped from its tickers). A security renamed or a ticker reused after a
    pull keeps the rows pulled under the old name with the security that had it.
    """
    keys = rows.select("ticker", "asof").unique().collect(engine="streaming")
    prices_map = _mapped(raw, ids, keys, PRICE_MAPS).rename({"sid": "_prices"})
    funds_map = _mapped(raw, ids, keys, MAPS).rename({"sid": "_fundamentals"})
    out = (
        keys.join(prices_map, on=["ticker", "asof"], how="left")
        .join(funds_map, on=["ticker", "asof"], how="left")
        .join(sids.rename({"sid": "_today"}), on="ticker", how="left")
        .select("ticker", "asof", pl.coalesce("_prices", "_fundamentals", "_today").alias("sid"))
    )
    left = out.filter(pl.col("sid").is_null()).drop("sid").rename({"asof": "date"})
    named = ids_.lookup(ids, left).rename({"date": "asof", "sid": "_named"})
    out = out.join(named, on=["ticker", "asof"], how="left").select(
        "ticker", "asof", pl.coalesce("sid", "_named").alias("sid")
    )
    lost = out.filter(pl.col("sid").is_null())
    if lost.height:
        names = lost["ticker"].unique().sort()
        log.warning("sharadar: %d price tickers resolve to no security, e.g. %s",
                    names.len(), names.head(10).to_list())  # fmt: skip
    return out.drop_nulls("sid")


def price_rows(rows: pl.LazyFrame, sids: pl.DataFrame) -> pl.LazyFrame:
    """The pulled ``rows`` (:func:`pulls`) of ``sids`` (ticker, asof, sid).

    One row per security and day: the latest ``lastupdated``, then the latest pull.
    """
    rows = rows.join(sids.lazy(), on=["ticker", "asof"])
    updates = (
        rows.filter(pl.col("_pull") >= 0)
        .sort("lastupdated", "_pull")
        .unique(["sid", "date"], keep="last")
    )
    bulk = rows.filter(pl.col("_pull") < 0)
    kept = bulk.join(updates.select("sid", "date"), on=["sid", "date"], how="anti")
    return pl.concat([kept, updates]).drop("ticker", "asof", "_pull", "lastupdated")


def prices(rows: pl.LazyFrame, sids: pl.DataFrame) -> pl.DataFrame:
    """Conformed price rows from the ``stocks`` or ``funds`` pulls (module docs).

    Args:
        rows: The table's pulled rows (:func:`pulls`).
        sids: ticker, asof, sid for the securities to keep (:func:`price_sids`; a few
            thousand at a time: the stocks table has 45 M rows).

    """
    frame = price_rows(rows, sids).collect(engine="streaming").sort("sid", "date")
    split = pl.col("closeunadj") / pl.col("close")
    prev_adj = pl.col("closeadj").shift(1).over("sid")
    rows = frame.select(
        "sid", "date",
        *[(pl.col(c) * split).alias(c) for c in ("open", "high", "low")],
        pl.col("closeunadj").alias("close"),
        (pl.col("volume") / split).alias("volume"),
        pl.col("closeadj").alias("close_adj"),
        (pl.col("closeadj") / prev_adj - 1).alias("ret_cc"),
        (pl.col("open") * pl.col("closeadj") / pl.col("close") / prev_adj - 1).alias("ret_co"),
    ).filter(pl.col("close") > 0)  # fmt: skip
    return rows


def listings(securities: pl.DataFrame) -> pl.DataFrame:
    """One listing episode per security, from its first to last price."""
    return securities.select("sid", pl.col("first").alias("from"), pl.col("last").alias("to"),
                             "exchange", "category")  # fmt: skip


def actions(raw: Path, sids: pl.DataFrame) -> pl.DataFrame:
    """Conformed actions (:data:`ACTIONS`), resolved through the current ticker."""
    rows = read(raw, "actions", ["date", "action", "ticker", "value"])
    return (
        rows.filter(pl.col("action").is_in(ACTIONS))
        .join(sids, on="ticker")
        .select("sid", pl.col("date").str.to_date(), "action",
                pl.col("value").cast(pl.Float64, strict=False))
    )  # fmt: skip


def fundamentals(raw: Path, columns: list[str]) -> pl.DataFrame:
    """The bulk fundamentals plus every nightly update, each filing's latest version.

    Adds ``asof``: the date the row's share counts are split-adjusted to (the bulk table's
    newest ``lastupdated``, or the day an update was pulled).
    """
    cols = [*columns, "lastupdated"]
    bulk = read(raw, "fundamentals", cols)
    parts = [bulk.with_columns(pl.col("lastupdated").str.to_date().max().alias("asof"))]
    for csv in sorted((raw / UPDATES).glob("*.csv")):
        pulled = datetime.strptime(csv.stem, "%Y%m%d").date()
        part = pl.read_csv(csv, columns=cols, infer_schema_length=0)
        parts.append(part.with_columns(pl.lit(pulled).alias("asof")))
    return (
        pl.concat(parts)
        .sort("lastupdated", "asof")
        .unique(["ticker", "dimension", "date", "reportperiod"], keep="last")
    )


def filings(raw: Path, ids: ids_.Ids, sids: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    """Conformed filings from the as-reported rows (module docs).

    A row's ticker is valid on the day it was pulled (``asof``): it resolves through
    Sharadar's ticker -> permaticker map for the fundamentals as of that day
    (``sources.sharadar.MAPS``), else today's (``sids``), else the dated names.

    Args:
        raw: The bulk zips and updates.
        ids: The id tables.
        sids: ticker, sid of today's ``SF1`` tickers.
        actions: The conformed actions; their splits restore reported share counts.
    """
    cols = ["ticker", "dimension", "date", "reportperiod", "fiscalperiod", "sharesbas",
            *CONCEPTS]  # fmt: skip
    rows = fundamentals(raw, cols).filter(pl.col("dimension").is_in(list(PERIODS)))
    keys = rows.select("ticker", "asof").unique()
    keys = keys.join(_mapped(raw, ids, keys), on=["ticker", "asof"], how="left")
    keys = keys.join(sids.rename({"sid": "_today"}), on="ticker", how="left")
    dated = ids_.lookup(ids, keys.select("ticker", pl.col("asof").alias("date")))
    keys = keys.join(dated.rename({"date": "asof", "sid": "_dated"}), on=["ticker", "asof"])
    keys = keys.select("ticker", "asof", pl.coalesce("sid", "_today", "_dated").alias("sid"))
    f = (
        rows.join(keys, on=["ticker", "asof"])
        .drop_nulls("sid")
        .with_columns(
            pl.col("date").str.to_date().alias("filed"),
            pl.col("reportperiod").str.to_date().alias("period_end"),
            pl.col("dimension").replace_strict(PERIODS).alias("period"),
            pl.when(pl.col("fiscalperiod").str.ends_with("Q4")).then(pl.lit("10-K"))
            .otherwise(pl.lit("10-Q")).alias("form"),
            *[pl.col(c).cast(pl.Float64, strict=False) for c in ("sharesbas", *CONCEPTS)],
        )
        .rename(CONCEPTS | {"sharesbas": "shares_out"})
        .with_columns(pl.col(c).abs() for c in OUTFLOWS)
        # a renamed company's filings come back under its new ticker: one version per sid
        .sort("lastupdated", "asof")
        .unique(["sid", "filed", "period_end", "period"], keep="last", maintain_order=True)
        .with_row_index("_r")
    )  # fmt: skip
    splits = actions.filter(pl.col("action") == "split").select(
        "sid", pl.col("date").alias("_split"), pl.col("value").alias("_ratio")
    )
    later = (
        f.select("_r", "sid", "filed", "asof")
        .join(splits, on="sid")
        .filter((pl.col("_split") > pl.col("filed")) & (pl.col("_split") <= pl.col("asof")))
        .group_by("_r")
        .agg(pl.col("_ratio").product().alias("_factor"))
    )
    f = f.join(later, on="_r", how="left").with_columns(
        pl.col("shares_out") / pl.col("_factor").fill_null(1.0)
    )
    return f.select("sid", "filed", "period_end", "period", "form", "shares_out",
                    *CONCEPTS.values())  # fmt: skip


def _mapped(raw: Path, ids: ids_.Ids, keys: pl.DataFrame, name: str = MAPS) -> pl.DataFrame:
    """ticker, asof, sid through the newest map kept in ``name`` dated on or before ``asof``."""
    maps = [
        pl.read_parquet(p).with_columns(
            pl.lit(datetime.strptime(p.stem, "%Y%m%d").date()).alias("_day"),
            pl.col("permaticker").cast(pl.Int64),
        )
        for p in sorted((raw / name).glob("*.parquet"))
    ]
    if not maps:
        return keys.with_columns(pl.lit(None, pl.Int64).alias("sid")).select(
            "ticker", "asof", "sid"
        )
    found = keys.sort("asof").join_asof(
        pl.concat(maps).sort("_day"), left_on="asof", right_on="_day", by="ticker",
        strategy="backward", check_sortedness=False,
    )  # fmt: skip
    return found.join(ids.sharadar, on="permaticker", how="left").select("ticker", "asof", "sid")


def _ids(paths: DataPaths, raw: Path) -> tuple[ids_.Ids, pl.DataFrame]:
    """The ids, seeded or refreshed, and (table, ticker, sid) from the tickers table.

    The tables are the prices' (``SEP``, ``SFP``) and the fundamentals' (``SF1``).
    """
    tickers = read(raw, "tickers")
    if (paths.ids / "securities.parquet").exists():
        ids = ids_.refresh(ids_.Ids.load(paths.ids), tickers, read(raw, "actions"))
    else:
        ids = ids_.from_sharadar(tickers, read(raw, "actions"))
    ids.save(paths.ids)
    current = (
        tickers.filter(pl.col("table").is_in(["SEP", "SFP", "SF1"]))
        .select("table", "ticker", pl.col("permaticker").cast(pl.Int64))
        .join(ids.sharadar, on="permaticker")
        .select("table", "ticker", "sid")
    )
    return ids, current


def _table(current: pl.DataFrame, *tables: str) -> pl.DataFrame:
    """ticker, sid for the given Sharadar tables, one row per ticker."""
    return current.filter(pl.col("table").is_in(tables)).drop("table").unique("ticker")


def _write_prices(root: Path, raw: Path, ids: ids_.Ids, current: pl.DataFrame) -> int:
    """Replace Sharadar's conformed prices from the bulk tables and updates; rows written."""
    reader.clear(root, SOURCE, "prices")
    written, bulk_asof = 0, zip_asof(raw)
    for table, name in (("SEP", "stocks"), ("SFP", "funds")):
        today = current.filter(pl.col("table") == table).drop("table").unique("ticker")
        rows = pulls(raw, name, bulk_asof)
        wanted = price_sids(rows, raw, ids, today)
        parts = wanted.with_columns((pl.col("sid") % CHUNKS).alias("_k")).partition_by("_k")
        for k, chunk in enumerate(parts):
            part = prices(rows, chunk.drop("_k"))
            written += reader.write(root, SOURCE, "prices", part, f"{name}-{k}")
            del part
    return written


def update(root: Path) -> dict:
    """The nightly step: ids, listings, actions, filings and prices from the latest tables."""
    paths = DataPaths(root)
    raw = paths.raw(SOURCE)
    ids, current = _ids(paths, raw)
    written = _write_prices(root, raw, ids, current)
    sf1 = _table(current, "SF1")
    current = _table(current, "SEP", "SFP")
    acts = actions(raw, current)
    return {
        "securities": ids.securities.height,
        "prices": written,
        "listings": reader.write(root, SOURCE, "listings", listings(ids.securities)),
        "actions": reader.write(root, SOURCE, "actions", acts),
        "filings": reader.write(root, SOURCE, "filings", filings(raw, ids, sf1, acts)),
    }


def build(root: Path, raw: Path | None = None) -> dict:
    """Seed (or refresh) the ids and write every conformed table from the bulk zips.

    Existing ids are refreshed, never re-seeded, so no sid changes (``ids.refresh``).

    Args:
        root: The data directory; the zips are read from ``<root>/sharadar/raw`` unless
            ``raw`` is given, and the tables written under ``<root>/sharadar/conformed``.
        raw: Folder with the bulk zips, if elsewhere.
    """
    paths = DataPaths(root)
    raw = raw or paths.raw(SOURCE)
    ids, current = _ids(paths, raw)
    counts = {"prices": _write_prices(root, raw, ids, current)}
    sf1, current = _table(current, "SF1"), _table(current, "SEP", "SFP")
    acts = actions(raw, current)
    counts["listings"] = reader.write(root, SOURCE, "listings", listings(ids.securities))
    counts["actions"] = reader.write(root, SOURCE, "actions", acts)
    counts["filings"] = reader.write(root, SOURCE, "filings", filings(raw, ids, sf1, acts))
    counts["securities"] = ids.securities.height
    return counts
