"""Better expected returns for meanvar than each stock's own trailing return (issue #39).

Production's forecast is a separate AR(1) per stock, which with daily returns collapses to
roughly the stock's trailing-year average: one noisy number per stock, nothing shared.
Two alternatives (``MeanVar.forecast``):

- **Shrinkage** (:func:`james_stein`): pull each trailing return toward the candidates'
  average, the noisier stocks (higher volatility) harder. The amount comes from the
  James-Stein formula, so there is nothing to tune.
- **Pooled linear model** (:class:`PooledModel`): one set of coefficients for all stocks,
  learned from every past month: each month's cross-section of next-month returns of the
  :data:`POOL` most liquid stocks is regressed on their inputs (as cross-sectional ranks),
  and the monthly coefficients are averaged (Fama-MacBeth). Past returns enter as inputs,
  so the momentum part is learned across stocks rather than per stock (a panel
  autoregression). Walk-forward: on each date only months whose next month has ended are
  used, so nothing looks ahead.

:class:`ForecastCheck` grades a forecast after the fact: at each rebalance, how well the
previous rebalance's forecasts ranked the candidates by the returns that followed (rank
IC) and whether their size matched (the slope of realized on forecast returns).
"""

from datetime import date, timedelta
from itertools import pairwise

import numpy as np
import pandas as pd
import polars as pl

from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.piotroski import PIOTROSKI, health_scores

#: Stocks per month in the pooled model's regressions (production's candidate pool).
POOL = 500
#: Months of history before the pooled model is used (before that: the AR(1) forecast).
MIN_MONTHS = 36
MONTHS_PER_YEAR = 12
#: Past-return inputs: the 12-month return excluding the last month, and the last month.
MOMENTUM = ("mom_12_1", "ret_1m")
HEALTH = "health"
#: Input sets for the pooled model.
INPUTS: dict[str, tuple[str, ...]] = {
    "pooled_health": (*MOMENTUM, HEALTH),
    "pooled_nine": (*MOMENTUM, *PIOTROSKI),
    "pooled_top": (*MOMENTUM, "cfo_to_assets", "gross_profitability", "share_issuance"),
}
FORECASTS = ("shrink", *INPUTS)


def james_stein(mu: pd.Series, vol: pd.Series, years: float = 1.0) -> pd.Series:
    """Shrink annual expected returns toward their mean (positive-part James-Stein).

    Each estimate's sampling variance is its annual volatility squared over the years of
    data behind it; stock ``i`` keeps ``1 - min(1, (k - 3) * var_i / sum((mu - mean)^2))``
    of its distance from the mean.
    """
    k = len(mu)
    if k < 4:
        return mu
    mean = mu.mean()
    spread = float(((mu - mean) ** 2).sum())
    if spread <= 0:
        return mu
    variance = (vol.reindex(mu.index).fillna(vol.mean()) ** 2) / years
    shrink = ((k - 3) * variance / spread).clip(upper=1.0)
    return mean + (1 - shrink) * (mu - mean)


def _ranks(frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Cross-sectional ranks scaled to [-0.5, 0.5]; missing values are 0 (neutral)."""
    out = []
    for c in columns:
        values = frame[c].cast(pl.Float64)
        ranked = values.rank("average") / values.count() - 0.5 - 0.5 / max(values.count(), 1)
        out.append(ranked.fill_null(0.0).fill_nan(0.0).to_numpy())
    return np.column_stack(out) if out else np.zeros((frame.height, 0))


class PooledModel:
    """Walk-forward Fama-MacBeth regression of next-month returns (see module docs).

    Args:
        inputs: Key of :data:`INPUTS`.
    """

    def __init__(self, inputs: str):
        self.columns = INPUTS[inputs]
        self._raw = tuple(c for c in self.columns if c != HEALTH)
        self._months: dict[date, tuple[np.ndarray, float]] = {}  # month -> (slopes, mean)

    def _cross_section(self, rows: pl.DataFrame, include: list[str] = ()) -> pl.DataFrame:
        """One month's :data:`POOL` most liquid stocks (plus ``include``), with the inputs."""
        top = rows.drop_nulls("log_adv").sort("log_adv", descending=True).head(POOL)
        extra = rows.filter(
            pl.col("symbol").is_in(list(include)) & ~pl.col("symbol").is_in(top["symbol"].to_list())
        )
        pool = pl.concat([top, extra])
        if HEALTH in self.columns:
            scores = health_scores(pool.select("symbol", *PIOTROSKI))
            pool = pool.with_columns(
                pl.col("symbol").replace_strict(scores, default=None, return_dtype=pl.Float64)
                .alias(HEALTH)
            )  # fmt: skip
        return pool

    def _needed(self) -> list[str]:
        """Feature columns to read: the raw inputs, health's nine metrics, liquidity."""
        return sorted({*self._raw, *(PIOTROSKI if HEALTH in self.columns else ()), "log_adv"})

    def update(self, view: DataView) -> None:
        """Fit every month whose next month has ended by ``view.asof`` and isn't fit yet."""
        history = view.feature_history(self._needed())
        months = sorted(history["date"].unique().to_list())
        for start, end in pairwise(months):
            if start in self._months or end > view.asof:
                continue
            pool = self._cross_section(history.filter(pl.col("date") == start))
            if pool.height < 50:
                continue
            y = view.period_returns(pool["symbol"].to_list(), start, end)
            x = _ranks(pool, self.columns)
            design = np.column_stack([np.ones(len(y)), x])
            coef, *_ = np.linalg.lstsq(design, y.to_numpy(), rcond=None)
            self._months[start] = (coef[1:], float(y.mean()))

    def predict(self, view: DataView, symbols: list[str]) -> pd.Series | None:
        """Annual expected returns for ``symbols`` (None until :data:`MIN_MONTHS` are fit).

        Inputs are ranked within the latest month's pool, as in training; the level is the
        average month's pool return, so only the spread between stocks is learned.
        """
        self.update(view)
        if len(self._months) < MIN_MONTHS:
            return None
        slopes = np.mean([s for s, _ in self._months.values()], axis=0)
        level = float(np.mean([m for _, m in self._months.values()]))
        latest = view.feature_history(self._needed(), since=view.asof - timedelta(days=45))
        if latest.is_empty():
            return None
        latest = latest.filter(pl.col("date") == latest["date"].max())
        pool = self._cross_section(latest, include=symbols)
        monthly = level + _ranks(pool, self.columns) @ slopes
        forecast = pd.Series((1 + monthly) ** MONTHS_PER_YEAR - 1, index=pool["symbol"].to_list())
        return forecast.reindex(symbols)


class ForecastCheck:
    """Grades each rebalance's forecasts once the next month has played out (see docs)."""

    def __init__(self) -> None:
        self._last: tuple[date, pd.Series] | None = None
        self.log: list[dict] = []

    def record(self, view: DataView, mu: pd.Series) -> None:
        """Grade the previous forecasts against returns since, then keep these ones.

        Only the first call per date counts (``soften="average"`` weighs several sets).
        """
        if self._last is not None and self._last[0] == view.asof:
            return
        if self._last is not None:
            when, before = self._last
            realized = view.period_returns(list(before.index), when, view.asof)
            both = pd.DataFrame({"f": before, "r": realized}).dropna()
            if len(both) >= 10 and both["f"].std() > 0:
                demeaned = both - both.mean()
                monthly_f = demeaned["f"] / MONTHS_PER_YEAR
                slope = float((monthly_f * demeaned["r"]).sum() / (monthly_f**2).sum())
                ic = float(both["f"].rank().corr(both["r"].rank()))
                self.log.append({"date": when.isoformat(), "ic": ic, "slope": slope})
        self._last = (view.asof, mu.copy())
