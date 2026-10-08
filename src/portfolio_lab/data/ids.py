"""Security ids: one integer ``sid`` per security, and how each source's names map to it.

Tickers are reused (CADE was Cadence Bancorp until 2021, Cadence Bank after) and renamed
(VRX became BHC), so a ticker is only a name for a security over a date range. Every
conformed table keys on ``sid`` instead, and sources resolve their tickers through
:func:`lookup` with the dates they saw the ticker trade.

Three tables, in ``DataPaths.ids``:

- ``securities``: sid, name, exchange, category (``common``, ``adr``, ``preferred``,
  ``etf``, ``cef``, ...), cik, cusip, figi, sic, first and last trading dates (``last`` null
  while alive).
- ``tickers``: sid, ticker, from, to (null while current), ``dated`` (from a recorded
  change, or inferred from a source's list of related names).
- ``sharadar``: sid, permaticker.

A sid is permanent: assigned once, never renumbered, whatever order data arrives in.

- A security Sharadar knows has its ``permaticker`` (Sharadar's permanent id, never
  reused) as its sid (:func:`from_sharadar`); rebuilding from scratch gives the same sids.
- Any other security (a new listing Sharadar doesn't have yet, or anything once Sharadar
  is gone) gets the next sid from :data:`RESERVED` up (:func:`extend`), recorded in the
  tables and never reassigned.
- A newer Sharadar tickers table is merged in by :func:`refresh`, which adds securities
  and updates names and delisting dates but never changes a sid: a security numbered from
  the reserved range before Sharadar listed it keeps that sid.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from portfolio_lab.core.store import write_parquet_atomic

log = logging.getLogger(__name__)

SECURITIES = {
    "sid": pl.Int64, "name": pl.String, "exchange": pl.String, "category": pl.String,
    "cik": pl.Int64, "cusip": pl.String, "figi": pl.String, "sic": pl.Int64, "first": pl.Date,
    "last": pl.Date,
}  # fmt: skip
TICKERS = {"sid": pl.Int64, "ticker": pl.String, "from": pl.Date, "to": pl.Date,
           "dated": pl.Boolean}  # fmt: skip
#: Sharadar's category strings -> ours (first match wins, checked in order).
CATEGORIES = (
    ("Preferred", "preferred"), ("ADR", "adr"), ("Common", "common"), ("ETF", "etf"),
    ("ETMF", "etf"), ("CEF", "cef"), ("ETN", "etn"), ("ETD", "etd"), ("UNIT", "unit"),
)  # fmt: skip
#: Sids of securities Sharadar doesn't know start above this (permatickers are 7 digits).
RESERVED = 100_000_000
#: A new Sharadar security takes over a reserved sid whose ticker started within this.
SAME_LISTING = timedelta(days=10)


@dataclass(frozen=True)
class Ids:
    """The three id tables (module docs)."""

    securities: pl.DataFrame
    tickers: pl.DataFrame
    sharadar: pl.DataFrame

    @classmethod
    def load(cls, folder: Path) -> "Ids":
        """Read the tables from ``folder`` (``DataPaths.ids``)."""
        return cls(*(pl.read_parquet(folder / f"{n}.parquet") for n in _FILES))

    def save(self, folder: Path) -> None:
        """Write the tables to ``folder``."""
        for name in _FILES:
            write_parquet_atomic(getattr(self, name), folder / f"{name}.parquet")

    def extend(self, new: pl.DataFrame) -> "Ids":
        """Ids with new securities appended, numbered from :data:`RESERVED` up.

        Args:
            new: ticker, name, exchange, category, first (and optionally cik); one row per
                new security, none of whose tickers may resolve through :func:`lookup`
                for its dates.
        """
        if new.is_empty():
            return self
        taken = self.securities.filter(pl.col("sid") > RESERVED)["sid"].max()
        start = int(taken) + 1 if taken is not None else RESERVED + 1
        sids = pl.int_range(start, start + new.height, eager=True).alias("sid")
        securities = _frame(new.with_columns(sids), SECURITIES)
        tickers = _frame(
            new.with_columns(sids, pl.col("first").alias("from"), pl.lit(True).alias("dated")),
            TICKERS,
        )
        return Ids(
            pl.concat([self.securities, securities]),
            pl.concat([self.tickers, tickers]),
            self.sharadar,
        )


_FILES = ("securities", "tickers", "sharadar")


def _frame(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """``frame`` in ``schema``: present columns cast, missing ones null."""
    cols = [pl.col(c).cast(t) if c in frame.columns else pl.lit(None, dtype=t).alias(c)
            for c, t in schema.items()]  # fmt: skip
    return frame.select(cols)


def category(sharadar: str | None) -> str:
    """Our category for one of Sharadar's."""
    for needle, ours in CATEGORIES:
        if sharadar and needle in sharadar:
            return ours
    return "other"


def _segments(tickers: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    """Dated ticker ranges per security from Sharadar's recorded ticker changes.

    A change on day ``D`` is a ``tickerchangefrom`` row (``contraticker`` = the old name)
    and a ``tickerchangeto`` row (the new name), both filed under the security's current
    ticker. The old name's range ends the day before ``D``.
    """
    changes = (
        actions.filter(pl.col("action") == "tickerchangefrom")
        .select("ticker", pl.col("date").str.to_date(), pl.col("contraticker").alias("old"))
        .join(
            actions.filter(pl.col("action") == "tickerchangeto").select(
                "ticker", pl.col("date").str.to_date(), pl.col("contraticker").alias("new")
            ),
            on=["ticker", "date"],
        )
        .join(tickers.select("sid", "ticker", "first", "last"), on="ticker")
        .sort("sid", "date")
    )
    # Each change closes the previous name; the first name starts at the first price.
    changes = changes.with_columns(
        pl.col("date").shift(1).over("sid").fill_null(pl.col("first")).alias("from"),
        (pl.col("date") - timedelta(days=1)).alias("to"),
    )
    old = changes.select("sid", pl.col("old").alias("ticker"), "from", "to")
    # The current name runs from the last change (or the first price) to the last price.
    last_change = changes.group_by("sid").agg(pl.col("date").max().alias("from"))
    current = (
        tickers.select("sid", "ticker", "first", "last")
        .join(last_change, on="sid", how="left")
        .select("sid", "ticker", pl.coalesce("from", "first").alias("from"),
                pl.col("last").alias("to"))
    )  # fmt: skip
    # a name changed on its first day (e.g. a when-issued ticker) covers no dates
    named = pl.concat([old, current]).filter(
        pl.col("to").is_null() | (pl.col("to") >= pl.col("from"))
    )
    return named.with_columns(pl.lit(True).alias("dated"))


def from_sharadar(
    tickers: pl.DataFrame, actions: pl.DataFrame, known: pl.DataFrame | None = None
) -> Ids:
    """Ids from Sharadar's ``tickers`` and ``actions`` tables (as read, all strings).

    Every row of the ``SEP`` (stocks) and ``SFP`` (funds) tables becomes a security whose
    sid is its permaticker, or the sid ``known`` (sid, permaticker) already gives it; the
    CIK comes from the SEC filings link, the names from recorded ticker changes
    (:func:`_segments`) plus Sharadar's ``relatedtickers`` (undated).
    """
    rows = (
        tickers.filter(pl.col("table").is_in(["SEP", "SFP"]))
        .unique("permaticker", keep="first")
        .with_columns(pl.col("permaticker").cast(pl.Int64))
        .sort("permaticker")
    )
    if known is not None:
        rows = rows.join(known.rename({"sid": "_known"}), on="permaticker", how="left")
    else:
        rows = rows.with_columns(pl.lit(None, pl.Int64).alias("_known"))
    rows = (
        rows.with_columns(pl.coalesce("_known", "permaticker").alias("sid"))
        .drop("_known")
        .with_columns(
            pl.col("firstpricedate").str.to_date().alias("first"),
            pl.when(pl.col("isdelisted") == "Y")
            .then(pl.col("lastpricedate").str.to_date())
            .alias("last"),
            pl.col("secfilings").str.extract(r"CIK=(\d+)", 1).cast(pl.Int64).alias("cik"),
            pl.col("category").map_elements(category, return_dtype=pl.String),
            pl.col("cusips").str.split(" ").list.first().alias("cusip"),
            pl.col("siccode").cast(pl.Int64, strict=False).alias("sic"),
        )
    )
    securities = _frame(rows, SECURITIES)
    dated = _segments(rows, actions)
    related = (
        rows.select("sid", pl.col("relatedtickers").fill_null("").str.split(" ").alias("ticker"),
                    "first", "last")
        .explode("ticker")
        .filter(pl.col("ticker").is_not_null() & (pl.col("ticker") != ""))
        .join(dated.select("sid", "ticker"), on=["sid", "ticker"], how="anti")
        .select("sid", "ticker", pl.col("first").alias("from"), pl.col("last").alias("to"),
                pl.lit(False).alias("dated"))
    )  # fmt: skip
    names = pl.concat([dated, related]).unique(["sid", "ticker", "from"]).sort("sid", "from")
    return Ids(securities, _frame(names, TICKERS), rows.select("sid", "permaticker"))


def refresh(ids: Ids, tickers: pl.DataFrame, actions: pl.DataFrame) -> Ids:
    """``ids`` updated from a newer Sharadar tickers table, without changing any sid.

    Securities already known keep their sids and take the new names and dates. A security
    Sharadar lists for the first time takes over the reserved sid of the same listing
    (same ticker, first trading days within :data:`SAME_LISTING`) if another source
    numbered it first, else its permaticker. Securities the new table lacks are kept.
    """
    fresh = from_sharadar(tickers, actions)
    new = fresh.securities.join(ids.sharadar, left_on="sid", right_on="permaticker",
                                how="anti").select("sid", "first")  # fmt: skip
    reserved = ids.tickers.filter(pl.col("sid") > RESERVED).join(
        ids.securities.select("sid", pl.col("first").alias("_first")), on="sid"
    )
    adopted = (
        new.join(fresh.tickers.filter(pl.col("to").is_null()).select("sid", "ticker"), on="sid")
        .join(reserved.select(pl.col("sid").alias("_sid"), "ticker", "_first"), on="ticker")
        .filter((pl.col("first") - pl.col("_first")).abs() <= SAME_LISTING)
        .unique("sid", keep="first")
        .select(pl.col("_sid").alias("sid"), pl.col("sid").alias("permaticker"))
    )
    known = pl.concat([ids.sharadar, adopted])
    fresh = from_sharadar(tickers, actions, known)
    kept = ~pl.col("sid").is_in(fresh.securities["sid"].implode())
    merged = Ids(
        pl.concat([fresh.securities, ids.securities.filter(kept)]).sort("sid"),
        pl.concat([fresh.tickers, ids.tickers.filter(kept)]).sort("sid", "from"),
        pl.concat([fresh.sharadar, known.join(fresh.sharadar, on="permaticker", how="anti")]),
    )
    before = dict(ids.sharadar.select("permaticker", "sid").rows())
    after = dict(merged.sharadar.select("permaticker", "sid").rows())
    changed = [p for p, s in before.items() if after.get(p) != s]
    if changed:  # the one rule ids must never break
        raise ValueError(f"ids: refresh would change {len(changed)} sids, e.g. {changed[:5]}")
    return merged


def lookup(ids: Ids, rows: pl.DataFrame) -> pl.DataFrame:
    """Resolve each row's ticker to the security it named on that date.

    Args:
        ids: The id tables.
        rows: ticker, date, and anything else; one row per observation.

    Returns:
        ``rows`` with ``sid``: null where no name covers the date, or two inferred ones
        do (a dated name beats an inferred one).
    """
    far = date(2999, 12, 31)
    names = ids.tickers.with_columns(pl.col("to").fill_null(far))
    covering = (
        rows.select("ticker", "date").unique()
        .join(names, on="ticker")
        .filter((pl.col("date") >= pl.col("from")) & (pl.col("date") <= pl.col("to")))
        .sort("dated", descending=True)
        .group_by("ticker", "date")
        .agg(pl.col("sid").first(), pl.col("dated").first(),
             pl.col("sid").n_unique().alias("_n"), pl.col("dated").sum().alias("_dated"))
    )  # fmt: skip
    # one dated name, or one inferred name and no dated one: resolved; otherwise not
    one_dated, one_inferred = pl.col("_dated") == 1, (pl.col("_dated") == 0) & (pl.col("_n") == 1)
    resolved = covering.filter(one_dated | one_inferred)
    out = rows.join(resolved.select("ticker", "date", "sid"), on=["ticker", "date"], how="left")
    missing = out.filter(pl.col("sid").is_null())["ticker"].n_unique()
    if missing:
        log.info("ids: %d tickers have dates no name covers", missing)
    return out
