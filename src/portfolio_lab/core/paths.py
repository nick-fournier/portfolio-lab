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
    def macro(self) -> Path:
        """FRED context series: observations with the date each became known."""
        return self.root / "macro" / "observations.parquet"

    @property
    def environment(self) -> Path:
        """Monthly point-in-time market environment (``research.context``)."""
        return self.root / "macro" / "environment.parquet"

    @property
    def context_conditions(self) -> Path:
        """Trait payoffs (IC) by market condition (``research.conditions``)."""
        return self.root / "results" / "context_conditions.parquet"

    @property
    def context_dial(self) -> Path:
        """Forward market return and risk by market condition (the caution dial)."""
        return self.root / "results" / "context_dial.parquet"

    @property
    def models(self) -> Path:
        """Model results: predictions, summary, calibration and importance tables."""
        return self.root / "results" / "models"

    @property
    def fund_prices(self) -> Path:
        """Daily prices adjusted for distributions for the make-vs-buy funds."""
        return self.root / "prices" / "funds.parquet"

    @property
    def make_vs_buy(self) -> Path:
        """Make-vs-buy results: ``summary.parquet`` and ``growth.parquet``."""
        return self.root / "results" / "make_vs_buy"

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
    def fundamentals_states(self) -> Path:
        """Point-in-time state of every company as of each of its filings."""
        return self.root / "fundamentals" / "states.parquet"

    @property
    def insider_trades(self) -> Path:
        """Open-market insider purchases and sales by symbol and filing date (Form 4)."""
        return self.root / "fundamentals" / "insider_trades.parquet"

    @property
    def institutional_holders(self) -> Path:
        """Number of 13F institutional holders by symbol and quarter, with availability date."""
        return self.root / "fundamentals" / "institutional_holders.parquet"

    @property
    def features(self) -> Path:
        """Monthly point-in-time feature panel (``research.features``)."""
        return self.root / "features" / "monthly.parquet"

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
