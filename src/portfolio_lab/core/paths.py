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
    def universe_delisted(self) -> Path:
        """Stocks that stopped trading since the history start, from Tiingo's ticker list."""
        return self.root / "universe" / "delisted.parquet"

    @property
    def rates(self) -> Path:
        """Risk-free rate history (FRED DTB3)."""
        return self.root / "rates" / "dtb3.parquet"

    @property
    def edgar_bulk(self) -> Path:
        """Downloaded SEC ``companyfacts.zip`` (about 1.4 GB, replaced when it changes)."""
        return self.root / "raw" / "edgar" / "companyfacts.zip"

    @property
    def fundamentals_facts(self) -> Path:
        """10-K and 10-Q facts for the universe's companies, with filing dates."""
        return self.root / "fundamentals" / "company_facts.parquet"

    @property
    def fundamentals_companies(self) -> Path:
        """Company profiles from the SEC: name, SIC industry code, fiscal year end."""
        return self.root / "fundamentals" / "companies.parquet"

    @property
    def fundamentals_tickers(self) -> Path:
        """Ticker -> CIK map from the SEC, as of the last ingest."""
        return self.root / "fundamentals" / "tickers.parquet"

    @property
    def fscores(self) -> Path:
        """Point-in-time Piotroski F-scores by symbol and filing date."""
        return self.root / "fundamentals" / "fscores.parquet"

    @property
    def forecast_cache(self) -> Path:
        """Cached return forecasts, one folder per model configuration."""
        return self.root / "cache" / "forecasts"

    @property
    def scoreboard(self) -> Path:
        """Signal scoreboard: per-date rank IC and quintile returns for every signal."""
        return self.root / "results" / "scoreboard.parquet"

    @staticmethod
    def year_partition(dataset: Path, year: int) -> Path:
        """Return the parquet file for ``year`` within a year-partitioned dataset."""
        return dataset / f"year={year}" / "data.parquet"
