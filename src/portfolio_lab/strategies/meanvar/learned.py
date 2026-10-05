"""The next-month forecaster's saved forecasts as mean-variance's expected returns.

``research.forecaster`` writes walk-forward forecasts (each month's made only from earlier
months) to ``<data>/results/forecaster/``. :class:`LearnedForecasts` serves them by month
end and turns them into the annual expected returns the optimizer needs:

- ``forecaster``: the forecast is a return relative to the average stock, so the level comes
  from production's own forecast: the candidates' average AR(1) expected return, plus 12 times
  each stock's monthly forecast.
- ``forecaster_excess``: the forecast is a return over the T-bill (``--excess``), so the
  expected return is the T-bill rate plus 12 times the forecast, with no level added.
- ``black_litterman``: :func:`black_litterman`: start from the returns that make the
  candidates' market-value weights optimal, and move toward the ``forecaster_excess``
  expected returns as far as the forecasts' measured skill allows (``confidence``).
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import polars as pl

EXPECTED = ("model", "forecaster", "forecaster_excess", "black_litterman")
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
        self._skill: dict[date, float] = {}

    def _load(self) -> dict[date, dict[str, float]]:
        from portfolio_lab.research.forecaster import walk  # noqa: PLC0415 - only when used

        combined = walk.combine(pl.read_parquet(self.path), center=not self.excess)
        # skill: slope of outcome on forecast over earlier months (Grinold's scale)
        scale = walk.calibrate(combined).group_by("date").agg(pl.col("scale").first())
        self._skill = dict(scale.iter_rows())
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

    def confidence(self, asof: date) -> float:
        """The forecasts' measured skill at ``asof``, between 0 and 1 (module docs)."""
        if self._by_date is None:
            self._by_date = self._load()
        made = [d for d in self._skill if d <= asof and asof - d <= MAX_AGE]
        return float(min(max(self._skill[max(made)], 0.0), 1.0)) if made else 0.0


def expected_returns(forecast: dict[str, float], model_mu: pd.Series, risk_free: float,
                     excess: bool, match_spread: bool = False) -> pd.Series:  # fmt: skip
    """Annual expected returns for the candidates that have a forecast (module docs).

    ``match_spread`` stretches the forecasts (keeping their order) so they vary across the
    candidates as much as ``model_mu`` does: a check on whether the forecasts' small size,
    rather than their ranking, decides the portfolio.
    """
    names = [s for s in model_mu.index if s in forecast]
    f = 12 * pd.Series({s: forecast[s] for s in names}, dtype=float)
    if match_spread and f.std() > 0:
        f = (f - f.mean()) * model_mu.reindex(names).std() / f.std() + f.mean()
    level = risk_free if excess else float(model_mu.reindex(names).mean())
    return level + f


def black_litterman(
    views: pd.Series, cov: pd.DataFrame, market_value: pd.Series, delta: float, risk_free: float,
    confidence: float,
) -> pd.Series:  # fmt: skip
    """Expected returns blending the market's implied returns with ``views`` (module docs).

    Args:
        views: Annual expected return per stock from the forecasts.
        cov: Annual covariance of the candidates.
        market_value: Each candidate's market value (the neutral weights).
        delta: The market's risk aversion (expected excess return over variance).
        risk_free: Annual T-bill rate.
        confidence: How far to move toward each view, 0 to 1 (Idzorek's method).
    """
    from pypfopt import black_litterman as bl  # noqa: PLC0415

    names = [s for s in views.index if s in cov.index and market_value.get(s, 0) > 0]
    sub = cov.loc[names, names]
    prior = bl.market_implied_prior_returns(market_value[names], delta, sub, risk_free)
    if confidence <= 0:
        return prior
    model = bl.BlackLittermanModel(
        sub, pi=prior, absolute_views=views[names].to_dict(), omega="idzorek",
        view_confidences=[confidence] * len(names),
    )  # fmt: skip
    return model.bl_returns()


def black_litterman_at(
    view, views: pd.Series, prices: pd.DataFrame, confidence: float
) -> pd.Series:
    """:func:`black_litterman` at a rebalance.

    Covariance from ``prices``, market values from the features, and the market's risk
    aversion from SPY's history up to the rebalance.
    """
    from pypfopt.black_litterman import market_implied_risk_aversion  # noqa: PLC0415
    from pypfopt.risk_models import CovarianceShrinkage  # noqa: PLC0415

    names = [s for s in views.index if s in prices.columns]
    cov = CovarianceShrinkage(prices[names], frequency=252).ledoit_wolf()
    mv = view.features(names, ["market_value"])
    caps = pd.Series(dict(zip(mv["symbol"], mv["market_value"], strict=True)), dtype=float)
    spy = view.prices(view.index + 1, ["SPY"]).dropna()  # all history up to now
    delta = market_implied_risk_aversion(spy["SPY"], 252, view.risk_free())
    return black_litterman(views[names], cov, caps.fillna(0.0), max(float(delta), 0.5),
                           view.risk_free(), confidence)  # fmt: skip
