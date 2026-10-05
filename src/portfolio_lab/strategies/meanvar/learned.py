"""The next-month forecaster's saved forecasts as mean-variance's expected returns.

``research.forecaster`` writes walk-forward forecasts (each month's made only from earlier
months) to ``<data>/results/forecaster/``. :class:`LearnedForecasts` serves them by month
end and turns them into the annual expected returns the optimizer needs:

- ``forecaster``: the forecast is a return relative to the average stock, so the level comes
  from production's own forecast: the candidates' average AR(1) expected return, plus 12 times
  each stock's monthly forecast.
- ``forecaster_excess``: the forecast is a return over the T-bill (``--excess``), so the
  expected return is the T-bill rate plus 12 times the forecast, with no level added.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import polars as pl

EXPECTED = ("model", "forecaster", "forecaster_excess")
#: How stale a forecast may be (a rebalance date a few days after the forecast's month end).
MAX_AGE = timedelta(days=7)


class LearnedForecasts:
    """Monthly forecasts by month end (loaded on first use).

    Args:
        folder: The forecaster's results folder.
        excess: Read the forecasts over the T-bill instead of relative to the average stock.
    """

    def __init__(self, folder: Path, excess: bool = False):
        self.path = folder / ("forecasts_excess.parquet" if excess else "forecasts.parquet")
        self.excess = excess
        self._by_date: dict[date, dict[str, float]] | None = None

    def _load(self) -> dict[date, dict[str, float]]:
        from portfolio_lab.research.forecaster import walk  # noqa: PLC0415 - only when used

        combined = walk.combine(pl.read_parquet(self.path), center=not self.excess)
        out: dict[date, dict[str, float]] = {}
        for (day,), month in combined.group_by("date"):
            out[day] = dict(
                zip(month["symbol"].to_list(), month["forecast"].to_list(), strict=True)
            )
        return out

    def at(self, asof: date) -> dict[str, float]:
        """The latest forecasts made at or before ``asof`` (within :data:`MAX_AGE`), by symbol."""
        if self._by_date is None:
            self._by_date = self._load()
        made = [d for d in self._by_date if d <= asof and asof - d <= MAX_AGE]
        return self._by_date[max(made)] if made else {}


def expected_returns(forecast: dict[str, float], model_mu: pd.Series, risk_free: float,
                     excess: bool) -> pd.Series:  # fmt: skip
    """Annual expected returns for the candidates that have a forecast (module docs)."""
    names = [s for s in model_mu.index if s in forecast]
    f = pd.Series({s: forecast[s] for s in names}, dtype=float)
    level = risk_free if excess else float(model_mu.reindex(names).mean())
    return level + 12 * f
