"""Tiingo's public ticker list: every stock it has carried, including dead ones.

``supported_tickers.zip`` (free, no API key) lists each ticker with its asset type, final
exchange and first/last dates of data. Stocks that stopped trading appear under their
*final* ticker (e.g. SVB as ``SIVBQ`` after it fell to OTC), and Alpaca serves their
exchange-traded history under that same ticker. Together they let us add companies that
were delisted since 2016, which a universe built from today's listings misses
(survivorship bias).
"""

import csv
import io
import re
import zipfile
from datetime import date

import polars as pl

from portfolio_lab.core.http import RateLimitedClient

URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"

#: Final venues that are exchanges. A dead stock whose final venue is anything else
#: (PINK, OTCMKTS, EXPM, ...) fell off its exchange before it stopped trading.
EXCHANGES = frozenset({"NYSE", "NASDAQ", "NYSE MKT", "AMEX", "NYSE ARCA", "BATS"})

# Tiingo marks non-common securities with suffixes, e.g. A-P-DM (preferred), AA-W
# (warrant), AAC-U (unit). Checked on the raw Tiingo ticker, in order.
_EXCLUDE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("preferred", re.compile(r"^-|-P(-|$)")),
    ("when_issued", re.compile(r"-W-?I$")),
    ("warrant", re.compile(r"-W(S|T)?$")),
    ("unit", re.compile(r"-U(N)?$")),
    ("right", re.compile(r"-R(T)?$")),
]
# NASDAQ five-letter symbols encode the issue type in the fifth letter.
_NASDAQ_FIFTH_LETTER = {"W": "warrant", "U": "unit", "R": "right"}

SCHEMA = {
    "symbol": pl.String,
    "tiingo_ticker": pl.String,
    "exchange": pl.String,
    "asset_type": pl.String,
    "currency": pl.String,
    "start": pl.Date,
    "end": pl.Date,
}


def normalize(ticker: str) -> str:
    """Tiingo ticker -> our (and Alpaca's) symbol format: upper case, ``.`` for classes."""
    return ticker.strip().upper().replace("-", ".")


def parse_supported_tickers(zip_bytes: bytes) -> pl.DataFrame:
    """Parse ``supported_tickers.zip`` into :data:`SCHEMA` rows (rows without dates dropped)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        text = archive.read(archive.namelist()[0]).decode()
    rows = [
        (
            normalize(r["ticker"]),
            r["ticker"],
            r["exchange"],
            r["assetType"],
            r["priceCurrency"],
            date.fromisoformat(r["startDate"]),
            date.fromisoformat(r["endDate"]),
        )
        for r in csv.DictReader(io.StringIO(text))
        if r["startDate"] and r["endDate"]
    ]
    return pl.DataFrame(rows, schema=SCHEMA, orient="row")


def exclude_reason(tiingo_ticker: str) -> str | None:
    """Why a ticker is not a common stock, from Tiingo's suffixes (``None`` to keep it)."""
    for reason, pattern in _EXCLUDE_RULES:
        if pattern.search(tiingo_ticker.upper()):
            return reason
    base = tiingo_ticker.upper()
    if len(base) == 5 and base.isalpha() and base[-1] in _NASDAQ_FIFTH_LETTER:
        return _NASDAQ_FIFTH_LETTER[base[-1]]
    return None


def dead_stocks(
    tickers: pl.DataFrame, current: set[str], since: date, before: date
) -> pl.DataFrame:
    """Stocks not listed today that left their exchange on or after ``since``.

    A stock whose final venue is OTC counts as dead even if it still trades there (SVB
    still quotes as ``SIVBQ``); one whose final venue is an exchange must have stopped
    trading before ``before``, so a listing missing from today's directory by lag is not
    mistaken for a dead one.

    Args:
        tickers: Parsed Tiingo list.
        current: Symbols in today's directory (these are handled by the daily universe).
        since: Earliest last-trading date of interest (the start of our price history).
        before: Exchange-listed tickers trading on or after this date are treated as alive.

    Returns:
        symbol, exchange (final venue), start, end, ``fell_to_otc``, ``exclude_reason`` and
        ``included`` for each candidate.
    """
    candidates = tickers.filter(
        (pl.col("asset_type") == "Stock")
        # US symbols only: letters/digits/dots starting with a letter (drops foreign
        # listings such as "3UW:DU" or "900904").
        & pl.col("symbol").str.contains(r"^[A-Z][A-Z0-9.]*$")
        & (pl.col("currency") == "USD")
        & (pl.col("end") >= since)
        & ((pl.col("end") < before) | ~pl.col("exchange").is_in(list(EXCHANGES)))
        & ~pl.col("symbol").is_in(list(current))
    ).unique(subset="symbol", keep="last")
    reasons = [exclude_reason(t) for t in candidates["tiingo_ticker"]]
    return (
        candidates.with_columns(
            (~pl.col("exchange").is_in(list(EXCHANGES))).alias("fell_to_otc"),
            pl.Series("exclude_reason", reasons, dtype=pl.String),
        )
        .with_columns(pl.col("exclude_reason").is_null().alias("included"))
        .select("symbol", "exchange", "start", "end", "fell_to_otc", "exclude_reason", "included")
        .sort("symbol")
    )


def fetch_supported_tickers(client: RateLimitedClient) -> pl.DataFrame:
    """Download and parse Tiingo's public ticker list (about 1 MB)."""
    return parse_supported_tickers(client.get(URL).content)
