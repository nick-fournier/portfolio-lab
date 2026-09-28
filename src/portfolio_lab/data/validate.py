"""Ingest invariants for price rows: drop impossible bars, flag suspicious returns.

Validation never raises on bad provider data. It removes rows that cannot be right and
returns human-readable issues, which ingest jobs log and record in their status file.
"""

import polars as pl

from portfolio_lab.core.calendar import sessions

#: Tolerance for OHLC ordering checks (prices are floats from the provider).
_EPS = 1e-6
#: Returns outside this range are kept but reported for review.
RETURN_BOUNDS = (-0.95, 10.0)


def validate_price_rows(rows: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Check price rows and drop the ones that violate hard invariants.

    Hard invariants (rows dropped): one row per (symbol, date), a trading-session date,
    positive close, and ``low <= open, close <= high``. Soft check (reported only): close-to-
    close returns within :data:`RETURN_BOUNDS`.

    Args:
        rows: Price rows with at least symbol, date, open, high, low, close, ret_cc.

    Returns:
        The rows that passed, and a list of issue descriptions.
    """
    issues: list[str] = []
    if rows.is_empty():
        return rows, issues

    dupes = rows.filter(pl.struct("symbol", "date").is_duplicated())
    if dupes.height:
        issues.append(f"{dupes.height} duplicate (symbol, date) rows; kept the last")
        rows = rows.unique(subset=["symbol", "date"], keep="last", maintain_order=True)

    valid_days = sessions(rows["date"].min(), rows["date"].max())
    off_calendar = rows.filter(~pl.col("date").is_in(valid_days))
    if off_calendar.height:
        issues.append(f"{off_calendar.height} rows on non-session dates dropped")

    lo, hi = pl.min_horizontal("open", "close"), pl.max_horizontal("open", "close")
    impossible = (pl.col("close") <= 0) | (pl.col("low") > lo + _EPS) | (pl.col("high") < hi - _EPS)
    bad = rows.filter(impossible)
    if bad.height:
        examples = bad.select("symbol", "date").head(3).rows()
        issues.append(f"{bad.height} rows with impossible OHLC dropped, e.g. {examples}")

    kept = rows.filter(pl.col("date").is_in(valid_days) & ~impossible)

    low, high = RETURN_BOUNDS
    outliers = kept.filter((pl.col("ret_cc") < low) | (pl.col("ret_cc") > high))
    if outliers.height:
        examples = outliers.select("symbol", "date", "ret_cc").head(3).rows()
        issues.append(f"{outliers.height} extreme returns kept for review, e.g. {examples}")
    return kept, issues
