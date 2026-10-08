"""Sharadar's conformer: nightly updates, split-restored share counts, ids kept."""

import io
import zipfile
from datetime import date

import polars as pl
import pytest

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data import ids as ids_
from portfolio_lab.data import reader
from portfolio_lab.data.conform import sharadar
from portfolio_lab.data.sources.sharadar import MAPS, UPDATES, keep_ticker_map
from tests.data.test_hive import ACTIONS, TICKERS


def _zip(path, frame):
    buffer = io.StringIO()
    frame.write_csv(buffer)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(path.stem + ".csv", buffer.getvalue())


def _fund(ticker, filed, period_end, shares, net_income, lastupdated, quarter="Q4"):
    row = dict.fromkeys(sharadar.CONCEPTS, "")
    return row | {"ticker": ticker, "dimension": "ART", "date": filed, "reportperiod": period_end,
                  "fiscalperiod": f"2021{quarter}", "sharesbas": str(shares),
                  "netinc": str(net_income), "lastupdated": lastupdated}  # fmt: skip


@pytest.fixture
def root(tmp_path):
    raw = DataPaths(tmp_path).raw(sharadar.SOURCE)
    raw.mkdir(parents=True)
    _zip(raw / "tickers.zip", TICKERS)
    # NEW split 2:1 on 2022-06-01 (Sharadar's share counts are adjusted for it)
    split = pl.DataFrame({"date": ["2022-06-01"], "action": ["split"], "ticker": ["NEW"],
                          "contraticker": [None], "value": ["2.0"]})  # fmt: skip
    actions = ACTIONS.with_columns(pl.lit(None, pl.String).alias("value"))
    _zip(raw / "actions.zip", pl.concat([actions, split]))
    _zip(raw / "fundamentals.zip", pl.DataFrame([
        _fund("NEW", "2022-02-15", "2021-12-31", 200, 10, "2022-07-01"),  # filed before the split
        _fund("NEW", "2022-08-10", "2022-06-30", 200, 12, "2022-08-10", "Q2"),  # after it
    ]))  # fmt: skip
    return tmp_path


def test_update_restores_share_counts_and_takes_the_latest_version(root):
    raw = DataPaths(root).raw(sharadar.SOURCE)
    sharadar.update(root)
    f = reader.read(root, "filings").sort("filed")
    assert f["sid"].to_list() == [200, 200]
    assert f["shares_out"].to_list() == [100.0, 200.0]  # undone for the later split only
    # a nightly update restates the 2021 filing; pulled before another 2:1 split
    (raw / UPDATES).mkdir()
    pl.DataFrame([_fund("NEW", "2022-02-15", "2021-12-31", 200, 11, "2026-10-08")]).write_csv(
        raw / UPDATES / "20261008.csv"
    )
    sharadar.update(root)
    f = reader.read(root, "filings").sort("filed")
    assert f["net_income"].to_list() == [11.0, 12.0] and f["shares_out"][0] == 100.0


def test_update_refreshes_ids_without_renumbering(root):
    sharadar.update(root)
    first = ids_.Ids.load(DataPaths(root).ids).securities.sort("sid")
    sharadar.update(root)
    assert ids_.Ids.load(DataPaths(root).ids).securities.sort("sid").equals(first)
    assert reader.read(root, "listings")["sid"].unique().sort().to_list() == [100, 200, 300]


def test_a_ticker_renamed_after_the_bulk_download_keeps_its_history(root):
    """The bulk rows say MN; tonight's tickers call that company MN1 and MN is reused."""
    raw = DataPaths(root).raw(sharadar.SOURCE)
    folder = raw / MAPS
    folder.mkdir()
    pl.DataFrame({"ticker": ["MN"], "permaticker": ["200"]}).write_parquet(
        folder / "20220701.parquet"
    )
    _zip(raw / "fundamentals.zip", pl.DataFrame(
        [_fund("MN", "2022-02-15", "2021-12-31", 50, 10, "2022-07-01")]))  # fmt: skip
    sharadar.update(root)
    assert reader.read(root, "filings")["sid"].to_list() == [200]


def test_the_ticker_map_is_kept_only_when_it_changes(root):
    raw = DataPaths(root).raw(sharadar.SOURCE)
    keep_ticker_map(raw, date(2026, 10, 1))
    keep_ticker_map(raw, date(2026, 10, 2))
    assert [p.name for p in (raw / MAPS).glob("*.parquet")] == ["20261001.parquet"]
