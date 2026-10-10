"""Class 2's expected returns: the nine-term forecasts in Grinold's form.

**Expected returns** (:meth:`NineInputs.expected`). The forecast for each candidate
(``forecasters.nine``, next month over the T-bill) is put in Grinold's form,
``T-bill + sqrt(12) x IC x volatility x z``: ``z`` is the forecast standardized across the
candidates, volatility the stock's annual volatility over the price window, and IC the
forecast's skill, its average monthly rank correlation with outcomes over every month whose
outcome was known by the decision date (at least :data:`MIN_IC_MONTHS`). These are honest
expected returns; how hard to bet on them is the optimizer's risk aversion
(``risk_aversion``), not a scale on the forecasts. Candidates without a forecast are left out.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import polars as pl

from portfolio_lab.forecasters import grade, nine

TRADING_DAYS = 252
MIN_IC_MONTHS = nine.MIN_IC_MONTHS
#: Forecasts this old or newer count for a rebalance.
MAX_AGE = timedelta(days=7)


@dataclass
class NineInputs:
    """The forecasts in ``folder`` (``DataPaths.forecaster``) and their track record, loaded once.

    Args:
        folder: Where ``nine.FILE`` is.
    """

    folder: Path
    _forecasts: pl.DataFrame = field(init=False, repr=False)
    _dates: list[date] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._forecasts = f = pl.read_parquet(self.folder / nine.FILE)
        self._dates = sorted(f["date"].unique().to_list())
        # month m's IC is known once its outcome is, at the next month's forecast
        self._ic = grade.grade_months(f.select("date", "symbol", "forecast", "actual"))

    def first(self) -> date:
        """The first month with forecasts."""
        return self._dates[0]

    def _made(self, asof: date) -> date | None:
        made = [d for d in self._dates if d <= asof and asof - d <= MAX_AGE]
        return max(made) if made else None

    def forecasts(self, asof: date) -> dict[str, float]:
        """The forecasts made for this rebalance, by symbol (empty if none)."""
        made = self._made(asof)
        if made is None:
            return {}
        month = self._forecasts.filter(pl.col("date") == made)
        return dict(zip(month["symbol"], month["forecast"], strict=True))

    def expected(self, asof: date, prices: pd.DataFrame, risk_free: float) -> pd.Series:
        """Annual expected returns by Grinold's rule (module docs).

        Args:
            asof: The decision date.
            prices: The candidates' price window.
            risk_free: Annual T-bill rate.
        """
        forecast, made = self.forecasts(asof), self._made(asof)
        names = [s for s in prices.columns if s in forecast]
        past = self._ic.filter(pl.col("date") < made)["ic"] if made else pl.Series([])
        if len(names) < 2 or past.len() < MIN_IC_MONTHS:
            return pd.Series(dtype=float)
        f = pd.Series({s: forecast[s] for s in names}, dtype=float)
        if f.std() == 0:
            return pd.Series(dtype=float)
        vol = prices[names].pct_change().std() * TRADING_DAYS**0.5
        return risk_free + 12**0.5 * float(past.mean()) * vol * (f - f.mean()) / f.std()
