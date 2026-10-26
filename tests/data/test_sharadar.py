"""Sharadar's conformer: nightly updates, split-restored share counts, ids kept."""

import io
import zipfile
from datetime import date

import polars as pl
import pytest

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data import ids as ids_
from portfolio_lab.data import quality, reader, schemas
from portfolio_lab.data.conform import sharadar
from portfolio_lab.data.sources import sharadar as api
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


def _bar(ticker, day, close, closeadj, lastupdated):
    return {"ticker": ticker, "date": day, "open": close, "high": close, "low": close,
            "close": close, "volume": 100.0, "closeadj": closeadj, "closeunadj": close,
            "lastupdated": lastupdated}  # fmt: skip


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
    # prices as of the bulk download (2022-08-10): NEW for three days, FUND for one
    _zip(raw / "stocks.zip", pl.DataFrame([
        _bar("NEW", "2022-08-08", 10.0, 9.0, "2022-08-08"),
        _bar("NEW", "2022-08-09", 11.0, 10.0, "2022-08-09"),
        _bar("NEW", "2022-08-10", 12.0, 11.0, "2022-08-10"),
    ]))  # fmt: skip
    _zip(raw / "funds.zip", pl.DataFrame([_bar("FUND", "2022-08-10", 50.0, 50.0, "2022-08-10")]))
    return tmp_path


def _pulled(raw, table, day, rows):
    folder = raw / api.PRICE_UPDATES[table]
    folder.mkdir(exist_ok=True)
    pl.DataFrame(rows).cast(pl.String).write_parquet(folder / f"{day}.parquet")


def test_price_updates_restate_the_history_and_add_the_new_days(root):
    raw = DataPaths(root).raw(sharadar.SOURCE)
    sharadar.update(root)
    p = reader.read(root, "prices").sort("sid", "date")
    assert p.select("sid", "close", "close_adj").rows() == [
        (200, 10.0, 9.0), (200, 11.0, 10.0), (200, 12.0, 11.0), (300, 50.0, 50.0)]  # fmt: skip
    # a dividend on 8/15: sharadar restates NEW's history and sends the new day
    _pulled(raw, "stocks", "20220815", [
        _bar("NEW", "2022-08-08", 10.0, 8.9, "2022-08-15"),
        _bar("NEW", "2022-08-09", 11.0, 9.9, "2022-08-15"),
        _bar("NEW", "2022-08-10", 12.0, 10.9, "2022-08-15"),
        _bar("NEW", "2022-08-15", 13.0, 11.9, "2022-08-15"),
    ])  # fmt: skip
    # the next pull overlaps it: the same day again, corrected
    _pulled(raw, "stocks", "20220816", [_bar("NEW", "2022-08-15", 13.0, 12.0, "2022-08-16")])
    sharadar.update(root)
    p = reader.read(root, "prices").filter(pl.col("sid") == 200).sort("date")
    assert p["close"].to_list() == [10.0, 11.0, 12.0, 13.0]
    assert p["close_adj"].to_list() == [8.9, 9.9, 10.9, 12.0]
    assert p["ret_cc"][1] == pytest.approx(9.9 / 8.9 - 1)  # one scale across the restatement
    assert quality.invariants(root) == []


def test_price_rows_pulled_under_an_old_name_stay_with_the_security(root):
    """A pull from when NEW was called OLD resolves through Sharadar's map kept that day."""
    raw = DataPaths(root).raw(sharadar.SOURCE)
    (raw / api.PRICE_MAPS).mkdir()
    pl.DataFrame({"ticker": ["OLD"], "permaticker": ["200"]}).write_parquet(
        raw / api.PRICE_MAPS / "20200102.parquet"
    )
    _pulled(raw, "stocks", "20200102", [_bar("OLD", "2020-01-02", 5.0, 4.0, "2020-01-02")])
    sharadar.update(root)
    p = reader.read(root, "prices").filter(pl.col("sid") == 200).sort("date")
    assert p["date"].to_list()[0] == date(2020, 1, 2) and p.height == 4


def test_a_capped_price_pull_is_split_until_every_part_fits(tmp_path, monkeypatch):
    days = [date(2026, 10, d) for d in range(1, 31)]
    table = pl.DataFrame([_bar(t, d.isoformat(), 1.0, 1.0, "2026-10-30")
                          for d in days for t in ("A", "B", "C")])  # fmt: skip
    calls = []

    def fake_get(key, name, params, **kwargs):
        calls.append(params)
        on = table.filter(pl.col("date").is_between(pl.lit(params["date.gte"]),
                                                    pl.lit(params["date.lte"])))  # fmt: skip
        cut = on.head(params["limit"])
        return type("Response", (), {"content": cut.write_csv().encode()})()

    monkeypatch.setattr(api, "_get", fake_get)
    monkeypatch.setattr(api, "MAX_ROWS", 10)
    rows = api.updated_prices("key", "stocks", date(2026, 10, 1), date(2026, 10, 30))
    assert rows.sort("date", "ticker").equals(table.sort("date", "ticker").cast(pl.String))
    assert len(calls) > 1 and all(c["limit"] == 10 for c in calls)


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


def test_filings_carry_investments_without_changing_which_source_wins(tmp_path, root):
    raw = DataPaths(root).raw(sharadar.SOURCE)
    rows = [_fund("NEW", "2022-02-15", "2021-12-31", 200, 10, "2022-07-01")
            | {"investmentsc": "5", "investmentsnc": "7"}]  # fmt: skip
    _zip(raw / "fundamentals.zip", pl.DataFrame(rows))
    sharadar.update(root)
    f = reader.read(root, "filings")
    assert f.select("investments_cur", "investments_lt").row(0) == (5.0, 7.0)
    assert not set(schemas.OPTIONAL_CONCEPTS) & set(schemas.REQUIRED["filings"])


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


def test_a_renamed_ticker_restating_a_filing_does_not_duplicate_it(root):
    """The bulk rows say NEW; after a rename, the nightly update sends the same filing as NEWER."""
    raw = DataPaths(root).raw(sharadar.SOURCE)
    (raw / MAPS).mkdir()
    pl.DataFrame({"ticker": ["NEWER"], "permaticker": ["200"]}).write_parquet(
        raw / MAPS / "20261008.parquet"
    )
    (raw / UPDATES).mkdir()
    pl.DataFrame([_fund("NEWER", "2022-02-15", "2021-12-31", 200, 11, "2026-10-08")]).write_csv(
        raw / UPDATES / "20261008.csv"
    )
    sharadar.update(root)
    f = reader.read(root, "filings").sort("filed")
    assert f["sid"].to_list() == [200, 200]
    assert f["net_income"].to_list() == [11.0, 12.0]  # the newer version wins
    assert quality.invariants(root) == []  # no duplicate keys in the stored table


def test_the_ticker_map_is_kept_only_when_it_changes(root):
    raw = DataPaths(root).raw(sharadar.SOURCE)
    keep_ticker_map(raw, date(2026, 10, 1))
    keep_ticker_map(raw, date(2026, 10, 2))
    assert [p.name for p in (raw / MAPS).glob("*.parquet")] == ["20261001.parquet"]
