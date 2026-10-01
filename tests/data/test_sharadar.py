import zipfile
from datetime import date

import polars as pl
import pytest

from portfolio_lab.data.ingest.sharadar import price_rows, universe


def _zip(folder, table, frame):
    csv = folder / f"{table}.csv.tmp"
    frame.write_csv(csv)
    with zipfile.ZipFile(folder / f"{table}.zip", "w") as z:
        z.write(csv, f"{table}.csv")
    csv.unlink()


def test_universe_marks_dead_stocks_not_bought_out_as_fell_to_otc(tmp_path):
    tickers = pl.DataFrame(
        {"table": ["SEP"] * 4, "ticker": ["LIVE", "BOUGHT", "BUST", "ADR"],
         "name": ["a", "b", "c", "d"], "exchange": ["NYSE", "NASDAQ", "NYSE", "NYSE"],
         "permaticker": ["1", "2", "3", "4"], "siccode": ["3571", "2834", "6022", "1311"],
         "sicindustry": ["x"] * 4, "isdelisted": ["N", "Y", "Y", "N"],
         "category": ["Domestic Common Stock", "Domestic Common Stock",
                      "Domestic Common Stock Primary Class", "ADR Common Stock"]}
    )  # fmt: skip
    actions = pl.DataFrame({"ticker": ["BOUGHT", "BUST"], "action": ["acquisitionby", "delisted"]})
    _zip(tmp_path, "tickers", tickers)
    _zip(tmp_path, "actions", actions)
    symbols, dead = universe(tmp_path)
    assert set(symbols["symbol"]) == {"LIVE", "BOUGHT", "BUST"}  # no ADRs
    assert dict(dead.select("symbol", "fell_to_otc").iter_rows()) == {"BOUGHT": False, "BUST": True}


def test_price_rows_undo_later_splits_and_use_adjusted_returns(tmp_path):
    # A 2-for-1 split after these days: Sharadar's close is halved, closeunadj is traded.
    stocks = pl.DataFrame(
        {"ticker": ["A", "A"], "date": ["2020-01-02", "2020-01-03"],
         "open": [49.0, 50.0], "high": [51.0, 52.0], "low": [48.0, 49.0],
         "close": [50.0, 51.0], "volume": [2000.0, 2000.0],
         "closeadj": [49.0, 51.0], "closeunadj": [100.0, 102.0]}
    )  # fmt: skip
    _zip(tmp_path, "stocks", stocks)
    rows = price_rows(tmp_path, "stocks", {"A"}).sort("date")
    first, second = rows.row(0, named=True), rows.row(1, named=True)
    assert first["date"] == date(2020, 1, 2)
    assert first["close"] == 100.0 and first["volume"] == 1000.0  # raw levels
    assert first["open"] == pytest.approx(98.0)
    assert second["ret_cc"] == pytest.approx(51.0 / 49.0 - 1)  # dividend-adjusted
    assert second["ret_co"] == pytest.approx(50.0 * 51.0 / 51.0 / 49.0 - 1)
