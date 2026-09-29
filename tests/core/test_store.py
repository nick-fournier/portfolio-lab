from datetime import date

import polars as pl

from portfolio_lab.core.store import (
    read_status,
    scan,
    upsert_parquet,
    write_parquet_atomic,
    write_status,
)


def _rows(values):
    return pl.DataFrame(
        {
            "symbol": [s for s, _, _ in values],
            "date": [d for _, d, _ in values],
            "x": [x for *_, x in values],
        }
    )


def test_atomic_write_leaves_no_temp_files(tmp_path):
    path = tmp_path / "a" / "data.parquet"
    write_parquet_atomic(_rows([("A", date(2024, 1, 2), 1.0)]), path)
    assert pl.read_parquet(path).height == 1
    assert [p.name for p in path.parent.iterdir()] == ["data.parquet"]


def test_upsert_new_rows_win_and_sorted(tmp_path):
    path = tmp_path / "data.parquet"
    upsert_parquet(
        _rows([("B", date(2024, 1, 2), 1.0), ("A", date(2024, 1, 2), 1.0)]),
        path,
        ["symbol", "date"],
    )
    upsert_parquet(
        _rows([("A", date(2024, 1, 2), 9.0), ("A", date(2024, 1, 3), 2.0)]),
        path,
        ["symbol", "date"],
    )
    out = pl.read_parquet(path)
    assert out.rows() == [
        ("A", date(2024, 1, 2), 9.0),
        ("A", date(2024, 1, 3), 2.0),
        ("B", date(2024, 1, 2), 1.0),
    ]


def test_upsert_is_idempotent(tmp_path):
    path = tmp_path / "data.parquet"
    rows = _rows([("A", date(2024, 1, 2), 1.0), ("B", date(2024, 1, 3), 2.0)])
    upsert_parquet(rows, path, ["symbol", "date"])
    first = pl.read_parquet(path)
    upsert_parquet(rows, path, ["symbol", "date"])
    assert pl.read_parquet(path).equals(first)


def test_scan_missing_and_present(tmp_path):
    assert scan(tmp_path / "nothing") is None
    write_parquet_atomic(
        _rows([("A", date(2024, 1, 2), 1.0)]), tmp_path / "ds" / "year=2024" / "data.parquet"
    )
    assert scan(tmp_path / "ds", "year=*/data.parquet").collect().height == 1


def test_status_roundtrip(tmp_path):
    assert read_status(tmp_path, "job") is None
    write_status(tmp_path, "job", {"max_date": date(2024, 1, 2), "rows": 3})
    status = read_status(tmp_path, "job")
    assert status["max_date"] == "2024-01-02"
    assert status["rows"] == 3
    assert "finished_at" in status
