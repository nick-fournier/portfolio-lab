"""U.S. Treasury daily bill rates: the 3-month (13-week) bill, a fallback for FRED's DTB3.

Treasury's own closing quotes, on the same bank-discount basis as DTB3 (the Fed's H.15
secondary-market rate); the two usually differ by a few basis points. One file per year.
"""

import io

import polars as pl

from portfolio_lab.core.http import RateLimitedClient

BILL_RATES_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
)
#: The 3-month bill on a bank-discount basis, as DTB3.
COLUMN = "13 WEEKS BANK DISCOUNT"


def parse_bill_rates(text: str) -> pl.DataFrame:
    """Parse a year of Treasury bill rates into ``date`` and ``rate`` (a fraction, not percent).

    Raises:
        ValueError: If the file has no 3-month bank-discount column.
    """
    df = pl.read_csv(io.StringIO(text), infer_schema=False)
    if COLUMN not in df.columns:
        raise ValueError(f"Treasury bill rates: no {COLUMN!r} column in {df.columns[:3]}")
    return (
        df.select(
            pl.col("Date").str.to_date("%m/%d/%Y").alias("date"),
            (pl.col(COLUMN).cast(pl.Float64) / 100).alias("rate"),
        )
        .drop_nulls("rate")
        .sort("date")
    )


def fetch_bill_rates(client: RateLimitedClient, year: int) -> pl.DataFrame:
    """Download one calendar year of daily bill rates (no API key required)."""
    params = {"type": "daily_treasury_bill_rates", "field_tdr_date_value": year,
              "_format": "csv"}  # fmt: skip
    return parse_bill_rates(client.get_text(BILL_RATES_URL.format(year=year), params))
