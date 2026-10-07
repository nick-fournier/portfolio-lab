"""The conformed tables: one schema per table, shared by every source.

Each source writes the tables it has, in these columns, keyed on our security id ``sid``
(``data.ids``). The reader (``data.reader``) unions the sources' copies and keeps one row
per key, so nothing downstream knows a source exists. ``REQUIRED`` names the columns that
make a row complete; the reader prefers the more complete row before applying the source
priority.
"""

import polars as pl

SID = pl.Int64

#: Daily bars: raw (as traded) levels and returns from adjusted closes. ``volume`` is
#: optional so a fund's NAV series fits.
PRICES = {
    "sid": SID, "date": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
    "close": pl.Float64, "volume": pl.Float64, "close_adj": pl.Float64,
    "ret_cc": pl.Float64, "ret_co": pl.Float64,
}  # fmt: skip
#: Listing episodes: where a security traded and what it was, as date ranges.
LISTINGS = {
    "sid": SID, "from": pl.Date, "to": pl.Date, "exchange": pl.String, "category": pl.String,
}  # fmt: skip
#: Corporate actions and status changes (dividends, splits, listings, delistings, buyouts).
ACTIONS = {"sid": SID, "date": pl.Date, "action": pl.String, "value": pl.Float64}
#: One row per filing: what the company reported, as of the filing date.
#: ``period`` is ``ttm`` (trailing twelve months) or ``q`` (the quarter alone).
CONCEPTS = (
    "net_income", "cfo", "revenue", "gross_profit", "cost_of_revenue", "operating_income",
    "capex", "dividends", "shares_weighted", "assets", "assets_cur", "liab_cur",
    "liabilities", "lt_debt", "debt_cur", "equity", "cash", "sga", "interest_expense",
    "rnd", "depreciation", "ppe", "inventory", "receivables", "tax", "ebit", "intangibles",
    "payables",
)  # fmt: skip
FILINGS = {
    "sid": SID, "filed": pl.Date, "period_end": pl.Date, "period": pl.String,
    "form": pl.String, "shares_out": pl.Float64, **dict.fromkeys(CONCEPTS, pl.Float64),
}  # fmt: skip
#: Point-in-time macro series: the value and the date it became known.
SERIES = {"series": pl.String, "date": pl.Date, "available": pl.Date, "value": pl.Float64}

TABLES: dict[str, dict[str, pl.DataType]] = {
    "prices": PRICES, "listings": LISTINGS, "actions": ACTIONS, "filings": FILINGS,
    "series": SERIES,
}  # fmt: skip
KEYS: dict[str, tuple[str, ...]] = {
    "prices": ("sid", "date"), "listings": ("sid", "from"), "actions": ("sid", "date", "action"),
    "filings": ("sid", "filed", "period_end", "period"), "series": ("series", "date"),
}  # fmt: skip
#: Columns a complete row has (beyond the key); the reader counts them per row.
REQUIRED: dict[str, tuple[str, ...]] = {
    "prices": ("open", "high", "low", "close", "ret_cc", "ret_co"),
    "listings": ("exchange", "category"),
    "actions": (),
    "filings": CONCEPTS,
    "series": ("available", "value"),
}
#: Tables stored in ``year=YYYY`` partitions of their date column.
PARTITIONED: dict[str, str] = {"prices": "date"}


def conform(frame: pl.DataFrame, table: str) -> pl.DataFrame:
    """``frame`` in ``table``'s schema: columns cast, missing ones null, extras dropped."""
    schema = TABLES[table]
    present = [pl.col(c).cast(t) for c, t in schema.items() if c in frame.columns]
    absent = [pl.lit(None, dtype=t).alias(c) for c, t in schema.items() if c not in frame.columns]
    return frame.select(*present, *absent).select(list(schema))
