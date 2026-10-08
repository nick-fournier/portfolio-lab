"""Sharadar's bulk files as conformed tables, and the seed of the security ids.

The tables are ``tickers``, ``actions``, ``stocks``, ``funds`` and ``fundamentals``.

Prices: Sharadar's OHLC and volume are split-adjusted, ``closeunadj`` is the traded close
and ``closeadj`` adjusts for dividends too; raw levels undo the split factor
(``closeunadj / close``), keeping dollar volume unchanged. Filings: the as-reported rows
(``ART`` = trailing twelve months, ``ARQ`` = the quarter), dated by the SEC filing date;
share counts, which Sharadar adjusts for every later split, are divided by the split
factor on the filing date to restore the count as reported. Everything under the license
stays inside the source's folder, so cancelling means deleting one directory.
"""

import logging
import shutil
import zipfile
from pathlib import Path

import polars as pl

from portfolio_lab.data import ids as ids_
from portfolio_lab.data import reader

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


def prices(raw: Path, table: str, sids: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Conformed price rows from the ``stocks`` or ``funds`` table (module docs).

    Args:
        raw: The bulk zips.
        table: ``stocks`` or ``funds``.
        sids: ticker, sid for the securities to keep (a few thousand at a time: the
            stocks table has 45 M rows).

    Returns:
        The rows, and ``splits``: sid, date, split (the cumulative split factor) on the
        days it changed, for the filings' share counts.
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
        split.round(6).alias("split"),
    ).filter(pl.col("close") > 0)  # fmt: skip
    changed = pl.col("split") != pl.col("split").shift(1).over("sid")
    splits = rows.filter(changed.fill_null(True)).select("sid", "date", "split")
    return rows.drop("split"), splits


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


def filings(raw: Path, sids: pl.DataFrame, splits: pl.DataFrame) -> pl.DataFrame:
    """Conformed filings from the as-reported rows (module docs).

    Args:
        raw: The bulk zips.
        sids: ticker, sid.
        splits: sid, date, split (from :func:`prices`), to restore reported share counts.
    """
    cols = ["ticker", "dimension", "date", "reportperiod", "fiscalperiod", "sharesbas",
            *CONCEPTS]  # fmt: skip
    f = (
        read(raw, "fundamentals", cols)
        .filter(pl.col("dimension").is_in(list(PERIODS)))
        .join(sids, on="ticker")
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
    )  # fmt: skip
    f = (
        f.sort("filed")
        .join_asof(
            splits.select("sid", pl.col("date").alias("filed"), "split").sort("filed"),
            on="filed", by="sid", strategy="backward", check_sortedness=False,
        )
        .with_columns(pl.col("shares_out") / pl.col("split").fill_null(1.0))
    )  # fmt: skip
    return f.select("sid", "filed", "period_end", "period", "form", "shares_out",
                    *CONCEPTS.values())  # fmt: skip


def build(root: Path, raw: Path | None = None) -> dict:
    """Seed (or refresh) the ids and write every conformed table from the bulk zips.

    Existing ids are refreshed, never re-seeded, so no sid changes (``ids.refresh``).

    Args:
        root: The data directory; the zips are read from ``<root>/sharadar/raw`` unless
            ``raw`` is given, and the tables written under ``<root>/sharadar/conformed``.
        raw: Folder with the bulk zips, if elsewhere.
    """
    from portfolio_lab.core.paths import DataPaths  # noqa: PLC0415 - avoid a cycle

    paths = DataPaths(root)
    raw = raw or paths.raw(SOURCE)
    tickers = read(raw, "tickers")
    if (paths.ids / "securities.parquet").exists():
        ids = ids_.refresh(ids_.Ids.load(paths.ids), tickers, read(raw, "actions"))
    else:
        ids = ids_.from_sharadar(tickers, read(raw, "actions"))
    ids.save(paths.ids)
    current = (
        tickers.filter(pl.col("table").is_in(["SEP", "SFP"]))
        .select("table", "ticker", pl.col("permaticker").cast(pl.Int64))
        .join(ids.sharadar, on="permaticker")
        .select("table", "ticker", "sid")
    )
    reader.clear(root, SOURCE, "prices")
    counts, splits = {"prices": 0}, []
    for table, name in (("SEP", "stocks"), ("SFP", "funds")):
        wanted = current.filter(pl.col("table") == table).drop("table")
        parts = wanted.with_columns((pl.col("sid") % CHUNKS).alias("_k")).partition_by("_k")
        for k, chunk in enumerate(parts):
            rows, split = prices(raw, name, chunk.drop("_k"))
            counts["prices"] += reader.write(root, SOURCE, "prices", rows, f"{name}-{k}")
            splits.append(split)
            del rows
    splits = pl.concat(splits)
    current = current.drop("table").unique("ticker")
    counts["listings"] = reader.write(root, SOURCE, "listings", listings(ids.securities))
    counts["actions"] = reader.write(root, SOURCE, "actions", actions(raw, current))
    counts["filings"] = reader.write(root, SOURCE, "filings", filings(raw, current, splits))
    counts["securities"] = ids.securities.height
    return counts
