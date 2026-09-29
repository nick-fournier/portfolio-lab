"""Point-in-time view of the panel: everything a strategy may see at a decision date.

A :class:`DataView` is bounded at panel row ``index`` (the decision session's close).
Every accessor slices rows ``<= index``; there is no method that takes a later date or
index, so a strategy cannot look ahead even by mistake. The underlying arrays are
read-only, so a strategy also cannot corrupt data for later decisions.
"""

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

from portfolio_lab.research.panel import TRADING_DAYS, Panel

#: Scores from filings older than this (about 18 months) are treated as stale.
MAX_FILING_AGE_DAYS = 550


class DataView:
    """Read-only access to panel data up to and including one session.

    Args:
        panel: The panel to view.
        index: Row of the decision session; nothing after it is visible.
    """

    def __init__(self, panel: Panel, index: int):
        if not 0 <= index < len(panel.dates):
            raise IndexError(f"index {index} outside panel of {len(panel.dates)} sessions")
        self._panel = panel
        self._index = index

    @property
    def asof(self) -> date:
        """The decision session (data through its close is visible)."""
        return self._panel.dates[self._index]

    @property
    def index(self) -> int:
        """Row of the decision session in the panel."""
        return self._index

    def _columns(self, symbols: Sequence[str] | None) -> tuple[list[str], list[int]]:
        """Resolve symbols (default: all) to panel column positions, skipping unknown ones."""
        if symbols is None:
            return list(self._panel.symbols), list(range(len(self._panel.symbols)))
        known = [s for s in symbols if s in self._panel.symbol_index]
        return known, [self._panel.symbol_index[s] for s in known]

    def _window(self, lookback: int) -> slice:
        """Rows of the trailing window of ``lookback`` sessions ending at the decision date."""
        return slice(max(0, self._index - lookback + 1), self._index + 1)

    def eligible(self) -> list[str]:
        """Symbols that may be held as of the decision date."""
        mask = self._panel.eligible[self._index]
        return [self._panel.symbols[j] for j in np.flatnonzero(mask)]

    def top_liquid(self, n: int, among: Sequence[str] | None = None) -> list[str]:
        """The ``n`` most liquid symbols by trailing median dollar volume.

        Args:
            n: How many to return.
            among: Candidate symbols; defaults to :meth:`eligible`.
        """
        names, cols = self._columns(self.eligible() if among is None else among)
        adv = self._panel.field("adv")[self._index, cols]
        order = np.argsort(-np.nan_to_num(adv, nan=-np.inf), kind="stable")
        return [names[k] for k in order[:n] if np.isfinite(adv[k])]

    def returns(
        self, lookback: int, symbols: Sequence[str] | None = None, kind: str = "cc"
    ) -> pd.DataFrame:
        """Trailing adjusted returns as a (dates x symbols) frame; NaN where no bar.

        Args:
            lookback: Number of sessions, ending at the decision date.
            symbols: Columns to include (default: all).
            kind: ``"cc"`` for close-to-close, ``"co"`` for close-to-open.
        """
        names, cols = self._columns(symbols)
        rows = self._window(lookback)
        data = self._panel.field(f"ret_{kind}")[rows][:, cols]
        return pd.DataFrame(data, index=pd.DatetimeIndex(self._panel.dates[rows]), columns=names)

    def prices(self, lookback: int, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        """Adjusted price index rebuilt from returns, starting at 1.0; NaN before first bar.

        Missing bars inside a symbol's history carry the price forward (zero return).
        """
        rets = self.returns(lookback, symbols)
        started = rets.notna().cumsum() > 0
        index = (1 + rets.fillna(0)).cumprod()
        return index.where(started)

    def close(self, symbols: Sequence[str] | None = None) -> pd.Series:
        """Raw closes on the decision date."""
        names, cols = self._columns(symbols)
        return pd.Series(self._panel.field("close")[self._index, cols], index=names)

    def fscores(
        self,
        symbols: Sequence[str] | None = None,
        min_signals: int = 8,
        max_age_days: int = MAX_FILING_AGE_DAYS,
    ) -> dict[str, int]:
        """Latest Piotroski F-score per symbol from filings made strictly before ``asof``.

        A filing dated ``asof`` is not visible yet (it may arrive after the close), and a
        score older than ``max_age_days`` (a company that stopped filing) is ignored.

        Args:
            symbols: Symbols to return (default: all with a score).
            min_signals: Minimum number of the nine signals that must be computable.
            max_age_days: Maximum age of the filing behind a score.
        """
        table = self._panel.fundamentals
        if table is None:
            return {}
        oldest = self.asof - timedelta(days=max_age_days)
        latest = (
            table.filter((pl.col("filed") < self.asof) & (pl.col("filed") >= oldest))
            .group_by("symbol")
            .last()
            .filter(pl.col("n_signals") >= min_signals)
        )
        if symbols is not None:
            latest = latest.filter(pl.col("symbol").is_in(list(symbols)))
        return dict(zip(latest["symbol"], latest["fscore"].cast(int), strict=True))

    def risk_free(self) -> float:
        """Annual risk-free rate as of the decision date (fraction, e.g. 0.05)."""
        return float(self._panel.rf_daily[self._index] * TRADING_DAYS)
