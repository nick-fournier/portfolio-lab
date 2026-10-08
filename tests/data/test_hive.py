from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.backtest.costs import CostModel
from portfolio_lab.backtest.engine import BacktestConfig, run
from portfolio_lab.core.calendar import sessions as _sessions
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data import ids as ids_
from portfolio_lab.data import quality, reader, schemas
from portfolio_lab.data.conform import fred, nasdaq, tiingo
from portfolio_lab.data.conform.alpaca import _one_per_day
from portfolio_lab.data.conform.edgar import primary_sids
from portfolio_lab.data.derived import daily
from portfolio_lab.research.panel import EligibilityRules, Panel
from portfolio_lab.strategies.base import create

TICKERS = pl.DataFrame(
    {"table": ["SEP", "SEP", "SFP", "SF1"], "permaticker": ["200", "100", "300", "100"],
     "ticker": ["NEW", "ACME", "FUND", "ACME"], "name": ["New Co", "Acme", "A Fund", "Acme"],
     "exchange": ["NYSE"] * 4, "isdelisted": ["N", "Y", "N", "Y"],
     "category": ["Domestic Common Stock", "Domestic Common Stock Primary Class", "ETF",
                  "Domestic Common Stock Primary Class"],
     "cusips": ["111 222", "333", None, "333"], "figi": ["BBG1", None, None, None],
     "siccode": ["3571", "2834", None, "2834"],
     "secfilings": ["https://sec.gov/x?action=getcompany&CIK=0000000077&y=1", None, None, None],
     "firstpricedate": ["2015-01-02", "2000-01-03", "2010-01-04", "2000-01-03"],
     "lastpricedate": ["2026-01-30", "2014-06-30", "2026-01-30", "2014-06-30"],
     "relatedtickers": ["ACME", None, None, None]}
)  # fmt: skip
# NEW was called OLD until 2020-03-02 (Sharadar files both rows under the current ticker).
ACTIONS = pl.DataFrame(
    {"date": ["2020-03-02", "2020-03-02", "2021-05-05"],
     "action": ["tickerchangefrom", "tickerchangeto", "dividend"],
     "ticker": ["NEW", "NEW", "NEW"], "contraticker": ["OLD", "NEW", None]}
)  # fmt: skip


@pytest.fixture
def ids():
    return ids_.from_sharadar(TICKERS, ACTIONS)


def test_ids_seed_one_security_per_permaticker_with_its_names(ids):
    s = ids.securities.sort("sid")
    assert s["sid"].to_list() == [100, 200, 300]  # each security's permaticker
    assert s["category"].to_list() == ["common", "common", "etf"]
    assert s.filter(pl.col("sid") == 200).row(0, named=True) | {} == pytest.approx(
        {"sid": 200, "name": "New Co", "exchange": "NYSE", "category": "common", "cik": 77,
         "cusip": "111", "figi": "BBG1", "sic": 3571, "first": date(2015, 1, 2), "last": None}
    )  # fmt: skip
    assert s.filter(pl.col("sid") == 100)["last"][0] == date(2014, 6, 30)  # dead
    names = ids.tickers.filter(pl.col("sid") == 200).sort("from", "ticker")
    assert names.select("ticker", "from", "to", "dated").rows() == [
        ("ACME", date(2015, 1, 2), None, False),  # from relatedtickers: undated
        ("OLD", date(2015, 1, 2), date(2020, 3, 1), True),
        ("NEW", date(2020, 3, 2), None, True),
    ]
    assert dict(ids.sharadar.rows()) == {100: 100, 200: 200, 300: 300}


def test_lookup_resolves_a_ticker_by_the_date_it_was_used(ids):
    rows = pl.DataFrame(
        {"ticker": ["OLD", "NEW", "OLD", "ACME", "ACME", "NONE"],
         "date": [date(2016, 1, 4), date(2021, 1, 4), date(2021, 1, 4), date(2010, 1, 4),
                  date(2016, 1, 4), date(2016, 1, 4)]}
    )  # fmt: skip
    out = ids_.lookup(ids, rows)
    # OLD after the rename and an unknown name resolve to nothing; ACME was Acme's dated
    # name until 2014 and New Co's inferred name after, so each date picks its company.
    assert out["sid"].to_list() == [200, 200, None, 100, 200, None]


def test_extend_numbers_new_securities_from_the_reserved_range(ids):
    new = pl.DataFrame({"ticker": ["IPO"], "name": ["Ipo Inc"], "exchange": ["NASDAQ"],
                        "category": ["common"], "first": [date(2026, 2, 2)]})  # fmt: skip
    more = ids.extend(new)
    assert more.securities["sid"].max() == ids_.RESERVED + 1
    assert ids_.lookup(more, pl.DataFrame({"ticker": ["IPO"], "date": [date(2026, 3, 1)]}))[
        "sid"
    ].to_list() == [ids_.RESERVED + 1]
    assert (
        more.extend(new.with_columns(pl.lit("IPO2").alias("ticker"))).securities["sid"].max()
        == ids_.RESERVED + 2
    )


def test_reader_keeps_the_most_complete_row_then_the_priority_source(tmp_path):
    day = date(2020, 1, 2)
    full = {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "ret_cc": 0.0, "ret_co": 0.0}
    # sid 1: both complete -> sharadar wins; sid 2: alpaca complete, sharadar not -> alpaca.
    reader.write(tmp_path, "sharadar", "prices", pl.DataFrame(
        [{"sid": 1, "date": day, "volume": 10.0, **full},
         {"sid": 2, "date": day, "volume": 10.0, **full, "ret_cc": None}]))  # fmt: skip
    reader.write(tmp_path, "alpaca", "prices", pl.DataFrame(
        [{"sid": 1, "date": day, "volume": 20.0, **full},
         {"sid": 2, "date": day, "volume": 20.0, **full},
         {"sid": 3, "date": date(2021, 1, 4), "volume": 30.0, **full}]))  # fmt: skip
    out = reader.read(tmp_path, "prices", columns=["volume"])
    assert out.columns == ["sid", "date", "volume"]
    assert dict(zip(out["sid"], out["volume"], strict=True)) == {1: 10.0, 2: 20.0, 3: 30.0}
    assert reader.read(tmp_path, "prices", start=date(2021, 1, 1))["sid"].to_list() == [3]
    assert reader.read(tmp_path, "prices", sids=[2])["sid"].to_list() == [2]
    assert reader.sources(tmp_path, "prices") == ["alpaca", "sharadar"]
    assert reader.read(tmp_path, "filings").is_empty()


def test_conform_casts_fills_missing_and_drops_extra_columns():
    frame = pl.DataFrame({"sid": [1], "date": [date(2020, 1, 2)], "close": [1], "junk": [0]})
    out = schemas.conform(frame, "prices")
    assert out.columns == list(schemas.PRICES)
    assert out.schema["close"] == pl.Float64 and out["volume"][0] is None


def _old_store(tmp_path):
    """A data directory in the old layout with one listed stock, one dead one, and rates."""
    old = DataPaths(tmp_path / "old")
    pl.DataFrame({"symbol": ["NEW", "IPO"], "name": ["New Co", "Ipo Inc"], "exchange": ["N", "Q"],
                  "etf": [False, False], "test_issue": [False, False],
                  "exclude_reason": [None, None], "included": [True, True],
                  "first_seen": [date(2026, 9, 1), date(2026, 9, 20)],
                  "last_seen": [date(2026, 10, 1), date(2026, 10, 1)]}).write_parquet(
        _mk(old.universe_symbols))  # fmt: skip
    pl.DataFrame({"symbol": ["ACME"], "exchange": ["PINK"], "start": [date(2010, 1, 4)],
                  "end": [date(2014, 6, 30)], "fell_to_otc": [True], "exclude_reason": [None],
                  "included": [True], "has_prices": [True]}).write_parquet(
        _mk(old.universe_delisted))  # fmt: skip
    pl.DataFrame({"date": [date(2020, 1, 2)], "rate": [0.015]}).write_parquet(_mk(old.rates))
    return old.root


def _mk(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_store_conformers_resolve_ids_add_new_listings_and_write_series(tmp_path, ids):
    ids.save(tmp_path / "ids")
    store = _old_store(tmp_path)
    assert nasdaq.build(tmp_path, store) == {"listings": 2, "new_ids": 1}
    listings = reader.read(tmp_path, "listings")
    assert listings.select("sid", "exchange", "category").rows() == [
        (200, "NYSE", "common"), (ids_.RESERVED + 1, "NASDAQ", "common")]  # fmt: skip
    new_sid = pl.col("sid") == ids_.RESERVED + 1
    assert ids_.Ids.load(tmp_path / "ids").securities.filter(new_sid)["name"][0] == ("Ipo Inc")
    assert tiingo.build(tmp_path, store) == {"actions": 2}
    actions = reader.read(tmp_path, "actions")
    assert actions.select("sid", "date", "action").rows() == [
        (100, date(2014, 6, 30), "delisted"), (100, date(2014, 6, 30), "otc")]  # fmt: skip
    assert fred.build(tmp_path, store) == {"series": 1}
    assert reader.read(tmp_path, "series").row(0) == ("DTB3", date(2020, 1, 2),
                                                      date(2020, 1, 2), 0.015)  # fmt: skip


def test_edgar_attaches_a_cik_to_its_longest_listed_common_stock():
    securities = pl.DataFrame(
        {"sid": [1, 2, 3, 4], "cik": [9, 9, 9, None],
         "category": ["preferred", "common", "common", "common"],
         "first": [date(2000, 1, 3)] * 4,
         "last": [None, date(2005, 1, 3), None, None]}
    )  # fmt: skip
    assert primary_sids(securities).rows() == [(9, 3)]


#: SPY's sid in the test hive: the first reserved one (Sharadar's test table lacks it).
SPY = ids_.RESERVED + 1


def _hive(tmp_path, ids):
    """A hive with two common stocks and SPY priced over 300 sessions."""
    ids = ids.extend(pl.DataFrame({"ticker": ["SPY"], "name": ["S&P 500"], "exchange": ["NYSEARCA"],
                                   "category": ["etf"], "first": [date(2000, 1, 3)]}))  # fmt: skip
    ids.save(tmp_path / "ids")
    days = _sessions(date(2019, 1, 2), date(2020, 3, 31))
    rows = []
    for sid, price, volume in ((200, 50.0, 1e5), (300, 20.0, 2e5), (SPY, 300.0, 1e6)):
        for k, d in enumerate(days):
            close = price * (1 + 0.001 * k)
            rows.append({"sid": sid, "date": d, "open": close, "high": close, "low": close,
                         "close": close, "volume": volume, "ret_cc": 0.001 if k else None,
                         "ret_co": 0.0})  # fmt: skip
    reader.write(tmp_path, "sharadar", "prices", pl.DataFrame(rows))
    reader.write(tmp_path, "sharadar", "listings", pl.DataFrame(
        {"sid": [200, 300, SPY], "from": [date(2015, 1, 2)] * 3,
         "to": [None, date(2019, 6, 28), None],
         "exchange": ["NYSE"] * 3, "category": ["common", "common", "etf"]}))  # fmt: skip
    reader.write(
        tmp_path,
        "sharadar",
        "actions",
        pl.DataFrame(
            {
                "sid": [300],
                "date": [date(2019, 6, 28)],
                "action": ["acquisitionby"],
                "value": [None],
            }
        ),
    )
    reader.write(tmp_path, "fred", "series", pl.DataFrame(
        {"series": ["DTB3"], "date": [date(2019, 1, 2)], "available": [date(2019, 1, 2)],
         "value": [0.0252]}))  # fmt: skip
    return days


def test_daily_table_and_panel_from_the_hive(tmp_path, ids):
    days = _hive(tmp_path, ids)
    assert daily.build(tmp_path)["rows"] == 3 * len(days)
    table = pl.read_parquet(daily.folder(tmp_path) / "year=*" / "*.parquet").sort("sid", "date")
    sid3 = table.filter(pl.col("sid") == 300)
    assert sid3["bars_seen"].to_list() == list(range(1, len(days) + 1))
    assert sid3["common"].to_list() == [d <= date(2019, 6, 28) for d in days]
    assert table.filter(pl.col("sid") == SPY)["common"].any() is False  # an ETF
    assert table["adv"].drop_nulls().min() == pytest.approx(50.0 * 1e5, rel=0.2)

    p = daily.panel(tmp_path, rules=EligibilityRules(5.0, 1e6, 60, 252))
    assert p.symbols == ["200", "300", str(SPY)] and p.market == 2
    assert p.field("close").dtype == np.float32
    assert p.universe == {"200", "300"} and not p.eligible[:, 2].any()
    # sid 200: $5 M/day, eligible once 252 bars are seen; 300 never (dead before 252 bars)
    assert p.eligible[:, 0].sum() == len(days) - 251 and not p.eligible[:, 1].any()
    assert p.fell_to_otc.tolist() == [False, False, False]  # sid 300 was bought out
    assert p.rf_daily[0] == pytest.approx(0.0252 / 252)


def test_hive_panel_names_tickers_and_backtests_report_them(tmp_path, ids):
    days = _hive(tmp_path, ids)
    daily.build(tmp_path)
    p = Panel.load(tmp_path, rules=EligibilityRules(5.0, 1e4, 60, 252))  # delegates to the hive
    assert p.names == {"200": "NEW", "300": "FUND", str(SPY): "SPY"}
    assert p.resolve("SPY") == str(SPY)
    config = BacktestConfig(start=days[260], end=days[-1], costs=CostModel(notional=1e5))
    result = run(create("buy_hold"), p, config)
    assert set(result.weights["symbol"]) == {"SPY"}
    assert result.daily["benchmark_ret"].drop_nulls().len() > 0


def test_alpaca_keeps_one_row_per_security_day_preferring_the_name_used_that_day(ids):
    # sid 200 was OLD until 2020-03-01 and NEW after; Alpaca has both tickers on both days
    rows = pl.DataFrame({
        "sid": [200, 200, 200, 200], "ticker": ["OLD", "NEW", "OLD", "NEW"],
        "date": [date(2019, 6, 3), date(2019, 6, 3), date(2021, 6, 1), date(2021, 6, 1)],
        "close": [1.0, 1.1, 2.0, 2.2],
    })  # fmt: skip
    out = _one_per_day(ids, rows).sort("date")
    assert out.select("date", "ticker").rows() == [(date(2019, 6, 3), "OLD"),
                                                   (date(2021, 6, 1), "NEW")]  # fmt: skip


def test_quality_invariants_catch_impossible_rows_and_duplicates(tmp_path):
    day = date(2020, 1, 2)
    good = {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0,
            "ret_cc": 0.01, "ret_co": 0.0}  # fmt: skip
    reader.write(tmp_path, "alpaca", "prices", pl.DataFrame(
        [{"sid": 1, "date": day, **good}, {"sid": 2, "date": day, **good}]))  # fmt: skip
    assert quality.invariants(tmp_path) == []
    reader.write(tmp_path, "alpaca", "prices", pl.DataFrame(
        [{"sid": 1, "date": day, **good}, {"sid": 1, "date": day, **good},
         {"sid": 2, "date": day, **good, "close": 0.0, "ret_cc": -1.0}]))  # fmt: skip
    broken = quality.invariants(tmp_path)
    assert any("duplicate keys" in b for b in broken)
    assert any("close <= 0" in b for b in broken) and any("-100%" in b for b in broken)


def test_measured_checks_flag_only_what_leaves_its_own_history():
    history = [100.0 + (k % 5) for k in range(40)]
    assert quality._judge("m", history, 102.0)["flag"] is None
    assert "outside its history" in quality._judge("m", history, 150.0)["flag"]
    assert quality._judge("m", history, 106.0)["flag"] is None  # a record by a normal step
    assert quality._judge("m", history[:5], 150.0)["flag"] is None  # too little history to judge


def test_ids_are_permanent_whatever_the_order_or_how_often_they_are_built():
    shuffled = TICKERS.reverse()
    assert (
        ids_.from_sharadar(shuffled, ACTIONS)
        .securities.sort("sid")
        .equals(ids_.from_sharadar(TICKERS, ACTIONS).securities.sort("sid"))
    )


def test_refresh_adds_and_updates_but_never_renumbers(ids):
    # IPO is numbered from the reserved range first (NASDAQ), then Sharadar lists it as 400.
    listing = {"ticker": ["IPO"], "name": ["Ipo Inc"], "exchange": ["NASDAQ"],
               "category": ["common"], "first": [date(2026, 2, 2)]}  # fmt: skip
    ids = ids.extend(pl.DataFrame(listing))
    ipo = TICKERS.head(1).with_columns(
        pl.lit("400").alias("permaticker"), pl.lit("IPO").alias("ticker"),
        pl.lit("2026-02-03").alias("firstpricedate"),
        pl.lit(None, pl.String).alias("relatedtickers"),
    )  # fmt: skip
    newer = pl.concat([TICKERS.with_columns(  # NEW is now delisted
        pl.when(pl.col("ticker") == "NEW").then(pl.lit("Y")).otherwise("isdelisted")
        .alias("isdelisted")), ipo])  # fmt: skip
    out = ids_.refresh(ids, newer, ACTIONS)
    s = out.securities.sort("sid")
    assert s["sid"].to_list() == [100, 200, 300, ids_.RESERVED + 1]  # IPO kept its sid
    assert s.filter(pl.col("sid") == 200)["last"][0] == date(2026, 1, 30)
    assert dict(out.sharadar.rows())[ids_.RESERVED + 1] == 400
    assert ids_.refresh(out, newer, ACTIONS).securities.sort("sid").equals(s)  # idempotent


def test_a_name_changed_on_its_first_day_is_dropped_and_the_current_name_shown():
    """Sandisk listed as SNDKV and became SNDK the same day: SNDK is its name."""
    tickers = TICKERS.head(1).with_columns(
        pl.lit("SNDK").alias("ticker"), pl.lit("2025-02-24").alias("firstpricedate"),
        pl.lit(None, pl.String).alias("relatedtickers"),
    )  # fmt: skip
    actions = pl.DataFrame(
        {"date": ["2025-02-24", "2025-02-24"], "action": ["tickerchangefrom", "tickerchangeto"],
         "ticker": ["SNDK", "SNDK"], "contraticker": ["SNDKV", "SNDK"]}
    )  # fmt: skip
    ids = ids_.from_sharadar(tickers, actions)
    assert ids.tickers["ticker"].to_list() == ["SNDK"]
    assert daily._names(ids, [200]) == {"200": "SNDK"}
