"""The research panel: wide, read-only (date x symbol) arrays built once per backtest.

The panel holds every field a strategy or the engine needs, aligned on NYSE sessions:

- ``close``: raw close (what traded), for price filters.
- ``ret_cc`` / ``ret_co``: adjusted close-to-close and close-to-open returns.
- ``adv``: trailing median dollar volume (raw close x volume) over the eligibility window.
- ``eligible``: whether each stock may be held on each date (see :class:`EligibilityRules`).

All trailing statistics use data up to and including each row only, so row ``i`` of every
field is knowable at the close of session ``i``. Arrays are marked read-only; strategies
never receive the panel itself, only a :class:`~portfolio_lab.research.dataview.DataView`
bounded at a decision date.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from portfolio_lab.core.calendar import sessions
from portfolio_lab.core.paths import DataPaths

#: Fields stored as float arrays in the panel.
FLOAT_FIELDS = ("close", "ret_cc", "ret_co", "adv")
TRADING_DAYS = 252


@dataclass(frozen=True)
class EligibilityRules:
    """Point-in-time rules for which stocks may be held on a date.

    Args:
        min_price: Minimum raw close, in dollars (excludes penny stocks).
        min_dollar_volume: Minimum trailing median daily dollar volume.
        adv_window: Sessions in the trailing dollar-volume median.
        min_history: Minimum number of bars observed so far (excludes fresh listings).
    """

    min_price: float = 5.0
    min_dollar_volume: float = 1e6
    adv_window: int = 60
    min_history: int = 252


class Panel:
    """Wide read-only arrays of prices, returns, liquidity and eligibility.

    Build with :meth:`load` (from the data directory) or :meth:`from_long` (from a long
    frame, used by tests). Symbols not in ``universe`` (e.g. benchmark ETFs) are present
    for pricing but never eligible.

    Args:
        dates: Sessions, ascending; row ``i`` of every array is session ``dates[i]``.
        symbols: Column labels, ascending.
        fields: Float arrays keyed by name (see :data:`FLOAT_FIELDS`), shape (dates, symbols).
        eligible: Boolean array, same shape.
        rf_daily: Daily risk-free rate per session (annual rate / 252).
        universe: Symbols that may ever be eligible; the rest (benchmarks) are price-only.
        fundamentals: Optional point-in-time scores by symbol and filing date (symbol,
            filed, fscore, n_signals), from ``research.piotroski``.
        fell_to_otc: Dead stocks whose final venue was OTC (fell off their exchange),
            which the backtest exits with a delisting return rather than the last price.
        traded: Boolean array, same shape: the bar had volume. Halted and dead stocks
            often keep printing zero-volume bars at a frozen price, which are not trades.
            Defaults to "has a close".
    """

    def __init__(  # noqa: PLR0913 - optional extras are keyword-only
        self,
        dates: list[date],
        symbols: list[str],
        fields: dict[str, np.ndarray],
        eligible: np.ndarray,
        rf_daily: np.ndarray,
        universe: Iterable[str] = (),
        *,
        fundamentals: pl.DataFrame | None = None,
        fell_to_otc: Iterable[str] = (),
        traded: np.ndarray | None = None,
    ):
        self.dates = list(dates)
        self.universe = frozenset(universe)
        self.fundamentals = fundamentals.sort("filed") if fundamentals is not None else None
        self.n_delisted = 0  # dead stocks added to the universe (set by ``load``)
        self.features: pl.DataFrame | None = None  # monthly feature panel (set by ``load``)
        self.environment: pl.DataFrame | None = None  # monthly environment (set by ``load``)
        self.symbols = list(symbols)
        self.date_index = {d: i for i, d in enumerate(self.dates)}
        self.symbol_index = {s: j for j, s in enumerate(self.symbols)}
        self.fields = dict(fields)
        self.eligible = eligible
        self.rf_daily = rf_daily
        otc = set(fell_to_otc)
        self.fell_to_otc = np.array([sym in otc for sym in self.symbols], dtype=bool)
        self.traded = np.isfinite(self.fields["close"]) if traded is None else traded
        # Row of each symbol's last trade (-1 if none): later rows mean it never trades again.
        self.last_bar = np.where(
            self.traded.any(axis=0),
            len(self.dates) - 1 - np.argmax(self.traded[::-1], axis=0),
            -1,
        )
        for array in (
            *self.fields.values(),
            self.eligible,
            self.rf_daily,
            self.traded,
            self.fell_to_otc,
            self.last_bar,
        ):
            array.flags.writeable = False

    def __repr__(self) -> str:
        span = f"{self.dates[0]}..{self.dates[-1]}" if self.dates else "empty"
        return f"Panel({len(self.dates)} sessions {span}, {len(self.symbols)} symbols)"

    def field(self, name: str) -> np.ndarray:
        """Return the read-only array for ``name`` (one of :data:`FLOAT_FIELDS`)."""
        return self.fields[name]

    @classmethod
    def from_long(
        cls,
        prices: pl.DataFrame,
        universe: Iterable[str],
        rates: pl.DataFrame | None = None,
        rules: EligibilityRules | None = None,
        fell_to_otc: Iterable[str] = (),
    ) -> "Panel":
        """Build a panel from long price rows.

        Args:
            prices: Rows with symbol, date, close, volume, ret_cc and ret_co.
            universe: Symbols that may ever be eligible (others are price-only).
            rates: Optional ``date``/``rate`` (annual fraction) risk-free history.
            rules: Eligibility rules; defaults to :class:`EligibilityRules`.
            fell_to_otc: Dead stocks whose final venue was OTC (see the class docs).
        """
        rules = rules or EligibilityRules()
        universe = set(universe)
        prices = prices.sort("symbol", "date").with_columns(
            (pl.col("close") * pl.col("volume"))
            .rolling_median(rules.adv_window, min_samples=rules.adv_window // 2)
            .over("symbol")
            .alias("adv"),
            pl.int_range(1, pl.len() + 1).over("symbol").alias("bars_seen"),
        )
        prices = prices.with_columns(
            (
                pl.col("symbol").is_in(list(universe))
                & (pl.col("volume") > 0)
                & (pl.col("close") > rules.min_price)
                & (pl.col("adv") > rules.min_dollar_volume)
                & (pl.col("bars_seen") >= rules.min_history)
            )
            .fill_null(False)
            .alias("eligible")
        )

        # Scatter long rows into wide arrays by (row, col) index. This avoids a pivot,
        # which is minutes slower with thousands of symbol columns.
        dates = sessions(prices["date"].min(), prices["date"].max())
        symbols = sorted(prices["symbol"].unique())
        date_pos = pl.DataFrame({"date": dates, "_row": range(len(dates))})
        sym_pos = pl.DataFrame({"symbol": symbols, "_col": range(len(symbols))})
        prices = prices.join(date_pos, on="date").join(sym_pos, on="symbol")
        rows, cols = prices["_row"].to_numpy(), prices["_col"].to_numpy()

        shape = (len(dates), len(symbols))
        fields = {}
        for name in FLOAT_FIELDS:
            array = np.full(shape, np.nan)
            array[rows, cols] = prices[name].cast(pl.Float64).fill_null(np.nan).to_numpy()
            fields[name] = array
        eligible = np.zeros(shape, dtype=bool)
        eligible[rows, cols] = prices["eligible"].to_numpy()
        traded = np.zeros(shape, dtype=bool)
        traded[rows, cols] = (prices["volume"] > 0).fill_null(False).to_numpy()
        rf = _daily_rates(dates, rates)
        return cls(
            dates, symbols, fields, eligible, rf, universe, fell_to_otc=fell_to_otc, traded=traded
        )

    @classmethod
    def load(
        cls,
        data_dir: Path,
        start: date | None = None,
        end: date | None = None,
        rules: EligibilityRules | None = None,
    ) -> "Panel":
        """Load stocks, benchmarks and rates from the data directory into a panel.

        The universe is the set of symbols currently marked ``included`` plus the stocks
        delisted since 2016 (``universe/delisted.parquet``), so backtests are not limited to
        today's survivors.

        Args:
            data_dir: The data directory (``Settings.data_dir``).
            start: First session to include (default: all history).
            end: Last session to include (default: all history).
            rules: Eligibility rules.
        """
        paths = DataPaths(data_dir)
        columns = ["symbol", "date", "close", "volume", "ret_cc", "ret_co"]
        frames = [
            pl.scan_parquet(dataset / "year=*" / "data.parquet").select(columns)
            for dataset in (paths.prices_daily, paths.prices_benchmarks)
            if any(dataset.glob("year=*/data.parquet"))
        ]
        if not frames:
            raise FileNotFoundError(f"no price data under {data_dir}; run `plab ingest` first")
        lf = pl.concat(frames)
        if start:
            lf = lf.filter(pl.col("date") >= start)
        if end:
            lf = lf.filter(pl.col("date") <= end)
        symbols = pl.read_parquet(paths.universe_symbols)
        universe = symbols.filter("included")["symbol"].to_list()
        dead = pl.DataFrame(schema={"symbol": pl.String, "fell_to_otc": pl.Boolean})
        if paths.universe_delisted.exists():
            dead = pl.read_parquet(paths.universe_delisted).filter("included", "has_prices")
        universe += dead["symbol"].to_list()
        fell_to_otc = dead.filter("fell_to_otc")["symbol"].to_list()
        rates = pl.read_parquet(paths.rates) if paths.rates.exists() else None
        panel = cls.from_long(lf.collect(), universe, rates, rules, fell_to_otc)
        if paths.fscores.exists():
            panel.fundamentals = pl.read_parquet(paths.fscores).sort("filed")
        if paths.features.exists():
            panel.features = pl.read_parquet(paths.features).sort("date")
        if paths.environment.exists():
            panel.environment = pl.read_parquet(paths.environment).sort("date")
        panel.n_delisted = dead.height
        return panel


def _daily_rates(dates: list[date], rates: pl.DataFrame | None) -> np.ndarray:
    """Align an annual rate history to ``dates`` (as-of, forward-filled) as daily rates."""
    if rates is None or rates.is_empty():
        return np.zeros(len(dates))
    aligned = (
        pl.DataFrame({"date": dates})
        .join_asof(rates.sort("date"), on="date", strategy="backward")
        .with_columns(pl.col("rate").fill_null(strategy="forward").fill_null(0.0))
    )
    return aligned["rate"].to_numpy() / TRADING_DAYS
