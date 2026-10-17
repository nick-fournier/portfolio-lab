"""The universe floors a walk can ask for (research.optimizer.data ``months`` docs, 2026-10-16)."""

import polars as pl

#: (min price, min daily $ volume, no blank checks).
Floors = tuple[float, float, bool]
NO_FLOORS: Floors = (0.0, 0.0, False)
#: Blank-check shells' SIC code in the security master.
BLANK_CHECK = 6770


def blank_checks(data_dir) -> frozenset[str]:
    """Symbols the security master files as blank-check shells (SIC 6770) today."""
    from portfolio_lab.core.paths import DataPaths  # noqa: PLC0415

    ids = pl.read_parquet(DataPaths(data_dir).ids / "securities.parquet")
    return frozenset(ids.filter(pl.col("sic") == BLANK_CHECK)["sid"].cast(pl.String).to_list())
