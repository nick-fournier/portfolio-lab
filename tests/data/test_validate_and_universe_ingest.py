from datetime import date

import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.data.ingest.universe import current_symbols, merge_symbols
from portfolio_lab.data.validate import validate_price_rows


def _row(symbol, day, o, h, l, c, ret=0.01):  # noqa: E741 (OHLC naming)
    return {
        "symbol": symbol,
        "date": day,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "ret_cc": ret,
    }


def test_validate_drops_impossible_and_off_calendar_rows():
    rows = pl.DataFrame(
        [
            _row("A", date(2024, 1, 2), 10, 11, 9, 10.5),
            _row("A", date(2024, 1, 3), 10, 9, 8, 10.5),  # high below close
            _row("A", date(2024, 1, 6), 10, 11, 9, 10.5),  # Saturday
            _row("B", date(2024, 1, 2), 10, 11, 9, 0),  # non-positive close
            _row("C", date(2024, 1, 2), 10, 60, 9, 55, ret=12.0),  # extreme but possible
        ]
    )
    kept, issues = validate_price_rows(rows)
    assert kept.select("symbol", "date").rows() == [
        ("A", date(2024, 1, 2)),
        ("C", date(2024, 1, 2)),
    ]
    assert any("impossible OHLC" in i for i in issues)
    assert any("non-session" in i for i in issues)
    assert any("extreme returns" in i for i in issues)


def _directory(symbols, included=True):
    return pl.DataFrame(
        {
            "symbol": symbols,
            "name": [f"{s} Common Stock" for s in symbols],
            "exchange": ["Q"] * len(symbols),
            "etf": [False] * len(symbols),
            "test_issue": [False] * len(symbols),
            "exclude_reason": [None if included else "etf"] * len(symbols),
            "included": [included] * len(symbols),
        },
        schema_overrides={"exclude_reason": pl.String},
    )


def test_merge_symbols_tracks_first_and_last_seen():
    day1, day2 = date(2024, 1, 2), date(2024, 1, 3)
    master = merge_symbols(None, _directory(["AAA", "OLD"]), day1)
    master = merge_symbols(master, _directory(["AAA", "NEW"]), day2)
    rows = {r["symbol"]: r for r in master.iter_rows(named=True)}
    assert (rows["AAA"]["first_seen"], rows["AAA"]["last_seen"]) == (day1, day2)
    assert (rows["OLD"]["first_seen"], rows["OLD"]["last_seen"]) == (day1, day1)  # delisted, kept
    assert (rows["NEW"]["first_seen"], rows["NEW"]["last_seen"]) == (day2, day2)


def test_current_symbols_uses_latest_snapshot_and_included_only(tmp_path):
    paths = DataPaths(tmp_path)
    assert current_symbols(paths) == []
    master = merge_symbols(None, _directory(["AAA", "OLD"]), date(2024, 1, 2))
    master = merge_symbols(
        master,
        pl.concat([_directory(["AAA"]), _directory(["ETFX"], included=False)]),
        date(2024, 1, 3),
    )
    write_parquet_atomic(master, paths.universe_symbols)
    assert current_symbols(paths) == ["AAA"]


def test_validate_nulls_impossible_returns():
    rows = pl.DataFrame(
        [
            _row("A", date(2024, 1, 2), 10, 11, 9, 10.5, ret=-1.0),
            _row("A", date(2024, 1, 3), 10, 11, 9, 10.5, ret=float("inf")),
            _row("A", date(2024, 1, 4), 10, 11, 9, 10.5, ret=0.02),
        ]
    ).with_columns(pl.lit(0.0).alias("ret_co"))
    kept, issues = validate_price_rows(rows)
    assert kept["ret_cc"].to_list() == [None, None, 0.02]
    assert kept.height == 3
    assert any("impossible returns nulled" in i for i in issues)
