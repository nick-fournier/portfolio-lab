import io
import zipfile
from datetime import date

import polars as pl
import pytest

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data.ingest import delisted, prices
from portfolio_lab.data.ingest.delisted import ingest_delisted
from portfolio_lab.data.sources.tiingo import dead_stocks, exclude_reason, parse_supported_tickers

from .test_prices import DAYS, FakeProvider

HEADER = "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
ROWS = [
    "sivbq,PINK,Stock,USD,1990-01-02,2024-02-15",  # fell to OTC
    "BRK-A,NYSE,Stock,USD,1990-01-02,2024-02-15",  # class share, acquired-style exit
    "AAC-U,NYSE,Stock,USD,2021-01-02,2024-02-15",  # unit
    "ABCDW,NASDAQ,Stock,USD,2021-01-02,2024-02-15",  # NASDAQ warrant
    "3UW:DU,DUSE,Stock,EUR,2021-01-02,2024-02-15",  # foreign
    "SPYX,NYSE ARCA,ETF,USD,2016-01-02,2024-02-15",  # not a stock
    "OLD,NYSE,Stock,USD,2001-01-02,2012-05-01",  # died before our history
    "ALIVE,NASDAQ,Stock,USD,2001-01-02,2024-03-28",  # still trading
    "FRCB,PINK,Stock,USD,1990-01-02,2024-03-28",  # bankrupt, still quoted OTC
    "LISTED,NASDAQ,Stock,USD,2001-01-02,2024-02-15",  # in today's directory
    "NODATE,NASDAQ,Stock,USD,,",
]


def _zip(rows=ROWS):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("supported_tickers.csv", HEADER + "\n".join(rows) + "\n")
    return buf.getvalue()


def test_parse_normalizes_symbols_and_drops_undated_rows():
    df = parse_supported_tickers(_zip())
    assert df.height == len(ROWS) - 1
    assert df.filter(pl.col("tiingo_ticker") == "BRK-A")["symbol"].item() == "BRK.A"
    assert "SIVBQ" in df["symbol"].to_list()


@pytest.mark.parametrize(
    ("ticker", "reason"),
    [
        ("A-P-DM", "preferred"),
        ("-P-A", "preferred"),
        ("AA-W", "warrant"),
        ("AA-WS", "warrant"),
        ("AA-WI", "when_issued"),
        ("AAC-U", "unit"),
        ("AB-RT", "right"),
        ("ABCDW", "warrant"),
        ("ABCDU", "unit"),
        ("BRK-A", None),
        ("SIVBQ", None),
        ("GOOGL", None),
    ],
)
def test_exclude_reason(ticker, reason):
    assert exclude_reason(ticker) == reason


def test_dead_stocks_filters_and_flags():
    dead = dead_stocks(
        parse_supported_tickers(_zip()), {"LISTED"}, date(2016, 1, 4), date(2024, 3, 14)
    )
    rows = {r["symbol"]: r for r in dead.iter_rows(named=True)}
    assert set(rows) == {"SIVBQ", "BRK.A", "AAC.U", "ABCDW", "FRCB"}
    assert rows["FRCB"]["fell_to_otc"]
    assert rows["SIVBQ"]["fell_to_otc"] and rows["SIVBQ"]["included"]
    assert not rows["BRK.A"]["fell_to_otc"] and rows["BRK.A"]["included"]
    assert (rows["AAC.U"]["exclude_reason"], rows["ABCDW"]["exclude_reason"]) == ("unit", "warrant")


def test_ingest_backfills_once(settings, monkeypatch):
    fake = FakeProvider(["SIVBQ"])  # Alpaca has SIVBQ but not BRK.A
    monkeypatch.setattr(prices, "fetch_bars", fake.fetch_bars)
    monkeypatch.setattr(
        delisted, "fetch_supported_tickers", lambda client: parse_supported_tickers(_zip())
    )

    first = ingest_delisted(settings, None, None, today=DAYS[-1])
    # No directory snapshot stored, so LISTED counts as dead too.
    assert (first["included"], first["fetched_now"], first["with_prices"]) == (4, 4, 1)
    assert first["fell_to_otc"] == 1
    table = pl.read_parquet(DataPaths(settings.data_dir).universe_delisted)
    assert table.filter("has_prices")["symbol"].to_list() == ["SIVBQ"]

    second = ingest_delisted(settings, None, None, today=DAYS[-1])
    assert second["fetched_now"] == 0  # each dead stock is tried once
