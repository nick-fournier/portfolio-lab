"""FRED: the 3-month Treasury bill rate (series ``DTB3``), used as the risk-free rate."""

import io

import polars as pl

from portfolio_lab.core.http import RateLimitedClient

DTB3_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def parse_dtb3(text: str) -> pl.DataFrame:
    """Parse FRED's DTB3 CSV into ``date`` and annualized ``rate`` (a fraction, not percent).

    FRED marks holidays with ``.``; those rows are dropped.
    """
    df = pl.read_csv(io.StringIO(text), null_values=".", infer_schema=False)
    return (
        df.rename({df.columns[0]: "date", df.columns[1]: "rate"})
        .drop_nulls("rate")
        .with_columns(pl.col("date").str.to_date(), (pl.col("rate").cast(pl.Float64) / 100))
        .sort("date")
    )


def fetch_dtb3(client: RateLimitedClient) -> pl.DataFrame:
    """Download the full DTB3 history (no API key required)."""
    return parse_dtb3(client.get_text(DTB3_URL, {"id": "DTB3"}))
