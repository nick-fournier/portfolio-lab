"""Class 2's inputs to the optimizer: the nine-term forecasts and the covariance built on them.

**Expected returns** (:meth:`NineInputs.expected`). The forecast for each candidate
(``research.forecaster.nine``, next month over the T-bill) is put in Grinold's form,
``T-bill + k x volatility x z``: ``z`` is the forecast standardized across the candidates,
volatility the stock's annual volatility over the price window, and ``k`` makes the
spread of the result match the spread of an AR(1) model's expected returns over the same
window (:func:`ar1_annual`), so the optimizer sees returns of a familiar size. Candidates
without a forecast, or whose AR(1) fails, are left out.

**Covariance** (:meth:`NineInputs.covariance`): the average of three estimates, each
annual:

- *price*: the candidates' daily log returns over the window, shrunk (Ledoit-Wolf);
- *factor*: ``X F X' + D``, with ``X`` each stock's nine terms this month (interactions
  centered as in ``nine.slopes``, on a full-sample mean, as in research), ``F`` the
  covariance of the terms' monthly payoffs (``nine.slopes``) over every earlier month, and
  ``D`` each stock's forecast-error variance over its last :data:`ERROR_MONTHS` months;
- *residual*: the forecast errors' own covariance over the last :data:`RESIDUAL_MONTHS`
  months, shrunk (Ledoit-Wolf).

A stock with too little error history gets :data:`IDIOSYNCRATIC` of its price variance
instead. Only months whose outcome was known by the decision date are used.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.covariance import LedoitWolf

from portfolio_lab.research.forecaster import nine
from portfolio_lab.research.forecaster.nine import ar1_annual

TRADING_DAYS = 252
#: Forecasts this old or newer count for a rebalance.
MAX_AGE = timedelta(days=7)
#: Months of payoffs needed for the factor covariance (else price only).
MIN_SLOPE_MONTHS = 24
#: Months of errors behind a stock's own variance, and the fewest it needs.
ERROR_MONTHS, MIN_ERROR_MONTHS = 36, 12
#: Months of errors behind the residual covariance, the fewest it needs, and how many a
#: stock may miss.
RESIDUAL_MONTHS, MIN_RESIDUAL_MONTHS, MAX_MISSING = 60, 24, 12
#: Share of price variance assumed idiosyncratic when a stock has no error history.
IDIOSYNCRATIC = 0.85**2


def _shrunk(returns: np.ndarray) -> np.ndarray:
    return LedoitWolf().fit(returns).covariance_


@dataclass
class NineInputs:
    """Forecasts, payoffs and errors from ``folder`` (``DataPaths.forecaster``), loaded once.

    Args:
        folder: Where ``nine.FILE`` and ``nine.SLOPES`` are.
    """

    folder: Path
    _forecasts: pl.DataFrame = field(init=False, repr=False)
    _slopes: pl.DataFrame = field(init=False, repr=False)
    _dates: list[date] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        f = pl.read_parquet(self.folder / nine.FILE)
        self._forecasts = f.with_columns((pl.col("actual") - pl.col("forecast")).alias("err"))
        self._slopes = pl.read_parquet(self.folder / nine.SLOPES).sort("date")
        self._dates = sorted(f["date"].unique().to_list())

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
        """Annual expected returns in Grinold's form (module docs).

        Args:
            asof: The decision date.
            prices: The candidates' price window.
            risk_free: Annual T-bill rate.
        """
        forecast = self.forecasts(asof)
        reference = ar1_annual(prices)
        names = [s for s in reference.index if s in forecast]
        f = pd.Series({s: forecast[s] for s in names}, dtype=float)
        if len(f) < 2 or f.std() == 0:
            return pd.Series(dtype=float)
        vol = prices[names].pct_change().std() * TRADING_DAYS**0.5
        raw = (f - f.mean()) / f.std() * vol.reindex(names).fillna(0.0)
        k = float(reference.reindex(names).std() / raw.std()) if raw.std() > 0 else 1.0
        return risk_free + k * raw

    def covariance(self, asof: date, prices: pd.DataFrame) -> pd.DataFrame:
        """The average of the price, factor and residual covariances (module docs)."""
        names = list(prices.columns)
        n = len(names)
        returns = np.log(prices).diff().dropna().to_numpy()
        price = _shrunk(returns) * TRADING_DAYS
        fallback = IDIOSYNCRATIC * np.diag(price)
        payoffs = self._slopes.filter(pl.col("date") < asof).select(nine.TERMS).to_numpy()
        made = self._made(asof)
        if len(payoffs) < MIN_SLOPE_MONTHS or made is None:
            return pd.DataFrame(price, index=names, columns=names)
        f = np.cov(payoffs.T) * 12
        center = float(self._slopes["center"][0])
        month = self._forecasts.filter(pl.col("date") == made)
        terms = nine.centered(month, center).select("symbol", *nine.TERMS)
        exposure = dict(zip(terms["symbol"], terms.select(nine.TERMS).to_numpy(), strict=True))
        x = np.array([np.nan_to_num(exposure.get(s, np.zeros(len(nine.TERMS)))) for s in names])
        errors = self._forecasts.filter(
            (pl.col("date") < asof) & pl.col("symbol").is_in(names) & pl.col("err").is_not_null()
        ).select("date", "symbol", "err")
        months = sorted(errors["date"].unique().to_list())
        recent = errors.filter(pl.col("date").is_in(months[-ERROR_MONTHS:]))
        own = recent.group_by("symbol").agg(pl.col("err").var().alias("v"), pl.len().alias("k"))
        variance = {s: 12 * v for s, v, k in own.iter_rows() if k >= MIN_ERROR_MONTHS and v}
        d = np.array([variance.get(s, fallback[i]) for i, s in enumerate(names)])
        factor = x @ f @ x.T + np.diag(d)
        residual = np.diag(fallback)
        window = errors.filter(pl.col("date").is_in(months[-RESIDUAL_MONTHS:]))
        wide = window.pivot(on="symbol", index="date", values="err").sort("date")
        cols = [s for s in names if s in wide.columns and wide[s].null_count() <= MAX_MISSING]
        if len(cols) >= 2 and wide.height >= MIN_RESIDUAL_MONTHS:
            filled = wide.select(cols).with_columns(pl.all().fill_null(strategy="mean"))
            idx = [names.index(s) for s in cols]
            residual[np.ix_(idx, idx)] = _shrunk(filled.to_numpy()) * 12
        out = (price + factor + residual) / 3
        out = 0.5 * (out + out.T) + 1e-8 * np.eye(n)
        return pd.DataFrame(out, index=names, columns=names)
