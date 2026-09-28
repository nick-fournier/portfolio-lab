"""Dataset locations under the data directory, defined once so every module agrees."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Paths of every dataset, relative to a data directory root.

    Args:
        root: The data directory (``Settings.data_dir``).
    """

    root: Path

    @property
    def prices_daily(self) -> Path:
        """Daily stock bars and returns, partitioned ``year=YYYY/data.parquet``."""
        return self.root / "prices" / "daily"

    @property
    def prices_benchmarks(self) -> Path:
        """Daily bars for benchmark ETFs (excluded from the stock universe)."""
        return self.root / "prices" / "benchmarks"

    @property
    def universe_snapshots(self) -> Path:
        """One classified symbol-directory snapshot per day, ``date=YYYY-MM-DD/``."""
        return self.root / "universe" / "snapshots"

    @property
    def universe_symbols(self) -> Path:
        """Master symbol table with first/last seen dates and classification."""
        return self.root / "universe" / "symbols.parquet"

    @property
    def rates(self) -> Path:
        """Risk-free rate history (FRED DTB3)."""
        return self.root / "rates" / "dtb3.parquet"

    @staticmethod
    def year_partition(dataset: Path, year: int) -> Path:
        """Return the parquet file for ``year`` within a year-partitioned dataset."""
        return dataset / f"year={year}" / "data.parquet"
