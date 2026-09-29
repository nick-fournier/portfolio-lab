"""FRED: the risk-free rate (``DTB3``) and the market and economic series for M9 context.

The T-bill rate comes from the keyless CSV endpoint. Context series use the FRED API
(``FRED_API_KEY``). Market-priced series are never revised and count as known the day
after their date (yields and spot prices publish the next business day). Economic series
are revised for months, so their *first release* is used, each value dated by when it
was published (FRED's ``output_type=4``), which keeps backtests free of later revisions.
"""

import io
from dataclasses import dataclass
from datetime import date, timedelta

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


API_URL = "https://api.stlouisfed.org/fred/series/observations"
#: History fetched for context series (enough for long percentiles and 12-month changes).
CONTEXT_START = date(2000, 1, 1)


@dataclass(frozen=True)
class Series:
    """A FRED context series.

    Args:
        name: Our column name.
        kind: ``log`` (prices: log changes), ``diff`` (rates, spreads, levels: differences)
            or ``yoy`` (indexes: year-on-year % change as the level).
        first_release: Use the first published value, dated by publication.
        lag_days: For market series, days after the observation date it becomes known.
    """

    name: str
    kind: str
    first_release: bool = False
    lag_days: int = 1


#: FRED series id -> how we use it.
CONTEXT_SERIES: dict[str, Series] = {
    "DCOILWTICO": Series("oil", "log"),
    "DHHNGSP": Series("natural_gas", "log"),
    "PCOPPUSDM": Series("copper", "log", lag_days=45),  # monthly average, published later
    "DFF": Series("fed_funds", "diff"),
    "DGS3MO": Series("yield_3m", "diff"),
    "DGS2": Series("yield_2y", "diff"),
    "DGS10": Series("yield_10y", "diff"),
    "DGS30": Series("yield_30y", "diff"),
    "T10YIE": Series("breakeven_10y", "diff"),
    "DFII10": Series("real_yield_10y", "diff"),
    "BAA10Y": Series("baa_spread", "diff"),
    "VIXCLS": Series("vix", "diff", lag_days=0),  # closes the same day
    "DTWEXBGS": Series("dollar", "log"),
    "NFCI": Series("financial_conditions", "diff", first_release=True),
    "UNRATE": Series("unemployment", "diff", first_release=True),
    "PAYEMS": Series("payrolls", "yoy", first_release=True),
    "ICSA": Series("jobless_claims", "log", first_release=True),
    "CPIAUCSL": Series("cpi", "yoy", first_release=True),
    "CPILFESL": Series("core_cpi", "yoy", first_release=True),
    "INDPRO": Series("industrial_production", "yoy", first_release=True),
    "RSAFS": Series("retail_sales", "yoy", first_release=True),
    "HOUST": Series("housing_starts", "log", first_release=True),
    "UMCSENT": Series("consumer_sentiment", "diff", first_release=True),
}

OBSERVATION_SCHEMA = {"series": pl.String, "date": pl.Date, "value": pl.Float64,
                      "available": pl.Date}  # fmt: skip


def parse_observations(payload: dict, series_id: str, spec: Series) -> pl.DataFrame:
    """FRED API observations -> series, date, value, available (missing values dropped)."""
    rows = []
    for obs in payload.get("observations", []):
        if obs["value"] in (".", ""):
            continue
        day = date.fromisoformat(obs["date"])
        available = (
            date.fromisoformat(obs["realtime_start"])
            if spec.first_release
            else day + timedelta(days=spec.lag_days)
        )
        rows.append((series_id, day, float(obs["value"]), available))
    return pl.DataFrame(rows, schema=OBSERVATION_SCHEMA, orient="row")


def fetch_series(client: RateLimitedClient, series_id: str, api_key: str) -> pl.DataFrame:
    """Download one context series since :data:`CONTEXT_START` (see :data:`CONTEXT_SERIES`)."""
    spec = CONTEXT_SERIES[series_id]
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": CONTEXT_START.isoformat(),
    }
    if spec.first_release:  # every observation's first published value and its date
        params |= {"realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
                   "output_type": 4}  # fmt: skip
    return parse_observations(client.get_json(API_URL, params), series_id, spec)
