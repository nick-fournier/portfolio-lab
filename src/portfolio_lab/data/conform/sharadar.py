"""Sharadar's bulk files as conformed tables, and the seed of the security ids.

The tables are ``tickers``, ``actions``, ``stocks``, ``funds`` and ``fundamentals``.

Prices: Sharadar's OHLC and volume are split-adjusted, ``closeunadj`` is the traded close
and ``closeadj`` adjusts for dividends too; raw levels undo the split factor
(``closeunadj / close``), keeping dollar volume unchanged. Filings: the as-reported rows
(``ART`` = trailing twelve months, ``ARQ`` = the quarter), dated by the SEC filing date,
from the bulk table plus the nightly updates (``sources.sharadar``), each filing's latest
version winning. Sharadar adjusts share counts for every split after the filing up to
when the row was pulled; dividing by those splits' ratios restores the count as reported.
Everything under the license stays inside the source's folder, so cancelling means
deleting one directory.

:func:`build` writes every table from the bulk zips (prices included, about 3 minutes);
:func:`update` is the nightly step: ids, listings, actions and filings from the latest
tickers, actions and fundamentals, leaving prices as they are.
"""

import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data import ids as ids_
from portfolio_lab.data import reader
from portfolio_lab.data.sources.sharadar import MAPS, UPDATES

log = logging.getLogger(__name__)

SOURCE = "sharadar"
#: Sharadar fundamentals column -> concept (``schemas.CONCEPTS``).
CONCEPTS = {
    "netinc": "net_income", "ncfo": "cfo", "revenue": "revenue", "gp": "gross_profit",
    "cor": "cost_of_revenue", "opinc": "operating_income", "capex": "capex",
    "ncfdiv": "dividends", "shareswa": "shares_weighted", "assets": "assets",
    "assetsc": "assets_cur", "liabilitiesc": "liab_cur", "liabilities": "liabilities",
    "debtnc": "lt_debt", "debtc": "debt_cur", "equity": "equity", "cashneq": "cash",
    "sgna": "sga", "intexp": "interest_expense", "rnd": "rnd", "depamor": "depreciation",
    "ppnenet": "ppe", "inventory": "inventory", "receivables": "receivables",
    "taxexp": "tax", "ebit": "ebit", "intangibles": "intangibles", "payables": "payables",
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


def prices(raw: Path, table: str, sids: pl.DataFrame) -> pl.DataFrame:
    """Conformed price rows from the ``stocks`` or ``funds`` table (module docs).

    Args:
        raw: The bulk zips.
        table: ``stocks`` or ``funds``.
        sids: ticker, sid for the securities to keep (a few thousand at a time: the
            stocks table has 45 M rows).

    """
    cols = ["ticker", "date", "open", "high", "low", "close", "volume", "closeadj", "closeunadj"]
    dtypes = {"date": pl.Date} | dict.fromkeys(cols[2:], pl.Float64)
    frame = (
        scan(raw, table, dtypes)
        .select(cols)
        .join(sids.lazy(), on="ticker")
        .drop("ticker")
        .collect(engine="streaming")
        .sort("sid", "date")
    )
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


def _mapped(raw: Path, ids: ids_.Ids, keys: pl.DataFrame) -> pl.DataFrame:
    """ticker, asof, sid through the newest kept ``SF1`` map dated on or before ``asof``."""
    maps = [
        pl.read_parquet(p).with_columns(
            pl.lit(datetime.strptime(p.stem, "%Y%m%d").date()).alias("_day"),
            pl.col("permaticker").cast(pl.Int64),
        )
        for p in sorted((raw / MAPS).glob("*.parquet"))
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


def update(root: Path) -> dict:
    """The nightly step: ids, listings, actions and filings from the latest tables.

    Prices are left as written by :func:`build`.
    """
    paths = DataPaths(root)
    raw = paths.raw(SOURCE)
    ids, current = _ids(paths, raw)
    sf1 = _table(current, "SF1")
    current = _table(current, "SEP", "SFP")
    acts = actions(raw, current)
    return {
        "securities": ids.securities.height,
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
    reader.clear(root, SOURCE, "prices")
    counts = {"prices": 0}
    for table, name in (("SEP", "stocks"), ("SFP", "funds")):
        wanted = current.filter(pl.col("table") == table).drop("table")
        parts = wanted.with_columns((pl.col("sid") % CHUNKS).alias("_k")).partition_by("_k")
        for k, chunk in enumerate(parts):
            rows = prices(raw, name, chunk.drop("_k"))
            counts["prices"] += reader.write(root, SOURCE, "prices", rows, f"{name}-{k}")
            del rows
    sf1, current = _table(current, "SF1"), _table(current, "SEP", "SFP")
    acts = actions(raw, current)
    counts["listings"] = reader.write(root, SOURCE, "listings", listings(ids.securities))
    counts["actions"] = reader.write(root, SOURCE, "actions", acts)
    counts["filings"] = reader.write(root, SOURCE, "filings", filings(raw, ids, sf1, acts))
    counts["securities"] = ids.securities.height
    return counts
