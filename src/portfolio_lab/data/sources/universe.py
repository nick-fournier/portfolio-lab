"""NASDAQ Trader symbol directory: the listed-security universe and its classification.

Two pipe-delimited files, refreshed daily by NASDAQ, list every security on NASDAQ
(``nasdaqlisted.txt``) and on the other US exchanges (``otherlisted.txt``). We keep common
stocks (and ADRs) on NASDAQ, NYSE and NYSE American, and record an explicit
``exclude_reason`` for everything else (ETFs, funds, warrants, units, rights, preferreds,
debt, test issues, other exchanges).
"""

import io
import re

import polars as pl

from portfolio_lab.core.http import RateLimitedClient

BASE_URL = "https://www.nasdaqtrader.com/dynamic/SymDir"

#: Exchange codes kept: NASDAQ (our own code "Q"), NYSE ("N"), NYSE American ("A").
#: Dropped: NYSE Arca ("P"), Cboe BZX ("Z") and IEX ("V"), which list almost only ETFs.
KEPT_EXCHANGES = frozenset({"Q", "N", "A"})

# Name patterns, checked case-insensitively after the flag and symbol rules.
_NAME_RULES: list[tuple[str, re.Pattern[str]]] = [
    # A coupon rate in the name marks notes, bonds, preferred units and structured products.
    ("fixed_income", re.compile(r"%")),
    (
        "preferred",
        re.compile(
            r"\bpreferred (stock|shares?|securities)\b|\bpreference shares?\b|-\s*preferred\b", re.I
        ),
    ),
    ("warrant", re.compile(r"\bwarrants?\b", re.I)),
    ("unit", re.compile(r"\bunits?,? each\b|\bunits?$", re.I)),
    # Plural "Rights" or a name ending in "Right"; not "the right to receive" in ADR names.
    ("right", re.compile(r"\brights\b|\bright$", re.I)),
    ("debt", re.compile(r"\bnotes? due\b|\bdebentures?\b|\bsenior notes\b", re.I)),
    ("fund", re.compile(r"\bfund\b|\bbond\b.*\btrust\b", re.I)),
]

# Symbol suffixes: ACT symbols use "$" for preferreds and ".W"/".U"/".R" for derivatives.
_SUFFIX_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("preferred", re.compile(r"\$")),
    ("warrant", re.compile(r"\.WS?$")),
    ("unit", re.compile(r"\.U$")),
    ("right", re.compile(r"\.R$")),
]

# NASDAQ five-letter symbols encode the issue type in the fifth letter.
_NASDAQ_FIFTH_LETTER = {"W": "warrant", "U": "unit", "R": "right"}

SCHEMA = {
    "symbol": pl.String,
    "name": pl.String,
    "exchange": pl.String,
    "etf": pl.Boolean,
    "test_issue": pl.Boolean,
}


def normalize_symbol(symbol: str) -> str:
    """Return the canonical (Alpaca-compatible) form of a directory symbol.

    Directory symbols already use ``.`` for share classes (``BRK.B``), which is what
    Alpaca expects, so this only trims and upper-cases.
    """
    return symbol.strip().upper()


def _read_pipe_file(text: str) -> pl.DataFrame:
    """Parse a symbol directory file, dropping its ``File Creation Time`` footer row."""
    body = "\n".join(line for line in text.splitlines() if not line.startswith("File Creation"))
    return pl.read_csv(io.StringIO(body), separator="|", infer_schema=False)


def parse_nasdaq_listed(text: str) -> pl.DataFrame:
    """Parse ``nasdaqlisted.txt`` into the common directory schema."""
    df = _read_pipe_file(text)
    return df.select(
        pl.col("Symbol").alias("symbol"),
        pl.col("Security Name").str.strip_chars().alias("name"),
        pl.lit("Q").alias("exchange"),
        (pl.col("ETF") == "Y").alias("etf"),
        ((pl.col("Test Issue") == "Y") | (pl.col("NextShares") == "Y")).alias("test_issue"),
    )


def parse_other_listed(text: str) -> pl.DataFrame:
    """Parse ``otherlisted.txt`` into the common directory schema."""
    df = _read_pipe_file(text)
    return df.select(
        pl.col("ACT Symbol").alias("symbol"),
        pl.col("Security Name").str.strip_chars().alias("name"),
        pl.col("Exchange").alias("exchange"),
        (pl.col("ETF") == "Y").alias("etf"),
        (pl.col("Test Issue") == "Y").alias("test_issue"),
    )


def exclude_reason(
    symbol: str, name: str, exchange: str, etf: bool, test_issue: bool
) -> str | None:
    """Return why a security is excluded from the universe, or ``None`` to keep it.

    Rules are applied in order: directory flags, exchange, symbol suffix, NASDAQ
    fifth-letter code, then security-name patterns.
    """
    flags = [("etf", etf), ("test_issue", test_issue), ("exchange", exchange not in KEPT_EXCHANGES)]
    if hit := next((reason for reason, flagged in flags if flagged), None):
        return hit
    if hit := next((reason for reason, pattern in _SUFFIX_RULES if pattern.search(symbol)), None):
        return hit
    if exchange == "Q" and len(symbol) == 5 and symbol[-1] in _NASDAQ_FIFTH_LETTER:
        return _NASDAQ_FIFTH_LETTER[symbol[-1]]
    return next((reason for reason, pattern in _NAME_RULES if pattern.search(name or "")), None)


def classify(directory: pl.DataFrame) -> pl.DataFrame:
    """Add ``exclude_reason`` and ``included`` columns to a parsed directory.

    Args:
        directory: Rows in the common directory schema (see :data:`SCHEMA`).

    Returns:
        The directory with normalized symbols, one row per symbol, plus classification.
    """
    # Vectorized equivalent of normalize_symbol.
    rows = directory.with_columns(pl.col("symbol").str.strip_chars().str.to_uppercase())
    reasons = [
        exclude_reason(r["symbol"], r["name"], r["exchange"], r["etf"], r["test_issue"])
        for r in rows.iter_rows(named=True)
    ]
    return (
        rows.with_columns(pl.Series("exclude_reason", reasons, dtype=pl.String))
        .with_columns(pl.col("exclude_reason").is_null().alias("included"))
        .unique(subset="symbol", keep="first", maintain_order=True)
        .sort("symbol")
    )


def fetch_directory(client: RateLimitedClient) -> pl.DataFrame:
    """Download both directory files and return the classified universe.

    Args:
        client: HTTP client (no authentication needed).
    """
    nasdaq = parse_nasdaq_listed(client.get_text(f"{BASE_URL}/nasdaqlisted.txt"))
    other = parse_other_listed(client.get_text(f"{BASE_URL}/otherlisted.txt"))
    return classify(pl.concat([nasdaq, other]))
