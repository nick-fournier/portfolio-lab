"""The mean-variance strategy: forecast returns, then optimize on the efficient frontier."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pandas as pd

from portfolio_lab.core.calendar import Frequency
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.piotroski import PIOTROSKI, health_scores
from portfolio_lab.strategies import explain
from portfolio_lab.strategies.base import Weights, register
from portfolio_lab.strategies.meanvar.forecast import (
    TRADING_DAYS,
    Forecaster,
    ForecastSpec,
    price_windows,
)
from portfolio_lab.strategies.meanvar.optimize import optimize

#: What each forecast model really computes, for the run explanation.
_MODEL_TEXT = {
    "historical_mean": (
        "Each stock's average annual return over the past {span}. This is extrapolation, not "
        "a prediction: it assumes last year's return continues."
    ),
    "ar1_logret": (
        "An AR(1) model fitted to each stock's past {span} of daily returns. AR(1) means "
        "autoregressive with one lag: the next return is predicted from the latest return "
        "times a fitted coefficient, plus an average. Daily returns have almost no "
        "day-to-day memory, so the coefficient is near zero and the forecast collapses to "
        "roughly the trailing {adj} average: extrapolation, not a prediction."
    ),
    "arima320_price": (
        "The original optimizer's ARIMA(3,2,0) on price levels. ARIMA(p, d, q) reads: p = 3, "
        "it looks back at the 3 most recent values; d = 2, it first takes differences twice, "
        "so it models the change in the price's day-to-day change (the acceleration); q = 0, "
        "no moving-average terms. So it extrapolates the recent price trend and its curvature."
    ),
}


@register("meanvar")
@dataclass
class MeanVar:
    """Mean-variance optimization: forecast returns, then maximize return per unit of risk.

    Each month it takes the most liquid eligible stocks, forecasts their returns with the
    chosen model, estimates how they move together from a year of daily prices, and solves
    for the maximum-Sharpe portfolio (no stock above 10%). This rebuilds the original
    optimizer's core, with its units and data handling fixed. With trailing-return
    forecasts it concentrates in recent winners (a momentum tilt), the style most
    flattered by survivorship bias in free data, so treat its results as an upper bound.

    Args:
        model: Return model: ``ar1_logret`` (default), ``arima320_price`` (the legacy
            model) or ``historical_mean`` (no forecast).
        objective: ``max_sharpe``, ``min_volatility`` or ``max_quadratic_utility``.
        top_n: Candidates: the most liquid eligible stocks.
        lookback: Sessions of history for forecasts and covariance.
        horizon: Forecast horizon in sessions.
        max_weight: Cap on any single weight.
        min_fscore: If set, only stocks with a Piotroski F-score at least this high are
            candidates (the original optimizer's filter), before taking the most liquid.
        healthy_share: If set, only this share of eligible stocks with the highest
            continuous F-score (``research.piotroski.health_scores``: the nine Piotroski
            measures as percentiles, averaged) are candidates, before taking the most liquid.
        health_schedule: How often the healthy set is recomputed: ``Q`` at the first
            rebalance of each quarter (fundamentals change with quarterly filings), or
            ``None`` at every rebalance. Between updates the last set is reused, limited to
            stocks still eligible.
        schedule: Rebalance frequency.
        cache_dir: Forecast cache root, set by the runner (not a strategy parameter).
        workers: Processes used for model fits (not a strategy parameter).
    """

    model: str = "ar1_logret"
    objective: str = "max_sharpe"
    top_n: int = 100
    lookback: int = 252
    horizon: int = 21
    max_weight: float = 0.10
    min_fscore: int | None = None
    healthy_share: float | None = None
    health_schedule: str | None = None
    schedule: Frequency = "M"
    name: str = "meanvar"
    cache_dir: Path | None = field(default=None, repr=False, metadata={"param": False})
    workers: int = field(default=1, metadata={"param": False})

    def __post_init__(self) -> None:
        self.top_n, self.lookback, self.horizon = (
            int(self.top_n),
            int(self.lookback),
            int(self.horizon),
        )
        self.max_weight = float(self.max_weight)
        self.last_signals: dict[str, dict[str, float]] = {}
        self.min_fscore = int(self.min_fscore) if self.min_fscore is not None else None
        self.healthy_share = float(self.healthy_share) if self.healthy_share is not None else None
        self._forecaster: Forecaster | None = None
        self._healthy: tuple[tuple[int, int], list[str]] | None = None  # (quarter, set)

    example_columns: ClassVar[dict[str, str]] = {
        "Expected return (annual)": "pct",
        "Volatility (annual)": "pct",
    }

    def explain(self) -> dict[str, str]:
        """Plain-language description of a run (see ``strategies.explain``)."""
        years = round(self.lookback / 252)
        span = "year" if years <= 1 else f"{years} years"
        adj = "one-year" if years <= 1 else f"{years}-year"
        screen = (
            f" Only companies with a Piotroski F-score of {self.min_fscore}+ from their latest "
            "annual report are considered, before taking the most liquid."
            if self.min_fscore is not None
            else ""
        )
        if self.healthy_share is not None:
            refresh = (" The healthy set is recomputed at the first rebalance of each quarter."
                       if self.health_schedule == "Q" else "")  # fmt: skip
            screen += (
                f" Only the healthiest {self.healthy_share:.0%} of stocks are considered, "
                "before taking the most liquid: health is a continuous Piotroski F-score, each "
                "of its nine measures (profitability, cash flow, improving returns, cash-backed "
                "earnings, falling debt, rising liquidity, no dilution, improving margins and "
                "asset turnover) as a percentile among eligible stocks, averaged."
            ) + refresh
        objective = {
            "max_sharpe": "the best expected return per unit of risk (maximum Sharpe ratio "
            "against the T-bill rate)",
            "min_volatility": "the lowest risk (minimum volatility), ignoring the forecasts",
            "max_quadratic_utility": "the best trade-off of expected return against risk",
        }[self.objective]
        signal = (
            "the legacy ARIMA price trend"
            if self.model == "arima320_price"
            else f"each stock's trailing {adj} return"
        )
        return {
            "summary": (
                f"A mean-variance portfolio of the {self.top_n} most liquid stocks, using "
                f"{signal} as its expected return."
            ),
            "candidates": explain.candidates(self.top_n)
            + f" Stocks missing more than 5% of the past {span}'s prices are skipped."
            + screen,
            "signal": _MODEL_TEXT[self.model].format(span=span, adj=adj)
            + " Forecasts are capped at +500% a year. The Signals page shows how well this "
            "ranking has actually predicted returns.",
            "construction": (
                f"Estimates how the candidates move together from the past {span} of daily prices "
                f"(a shrunk covariance), then picks the long-only weights with {objective}. No "
                f"stock above {self.max_weight:.0%}, fully invested; if no mix beats cash it "
                "falls back to the lowest-risk mix. In practice it favors stocks with high "
                "trailing returns and low volatility, spread across stocks that don't move "
                "together."
            ),
            "drivers": (
                "A risk-controlled tilt toward last year's winners: the return comes from "
                "winners continuing to win (strongly so for large caps in 2017-2026), and the "
                "optimizer mainly lowers risk compared with plain momentum."
            ),
            "related": (
                "Same candidates as momentum (pool=100): momentum ranks by the 12-month return "
                "excluding the last month and equal-weights the top 20; meanvar uses the full "
                f"trailing {span} and weights by expected return, volatility and co-movement "
                f"({self.max_weight:.0%} cap). The AR(1) and trailing-average runs hold almost "
                "identical portfolios."
            ),
        }

    def _get_forecaster(self) -> Forecaster:
        """Create the forecaster lazily, once the runner has set ``cache_dir``/``workers``."""
        if self._forecaster is None:
            spec = ForecastSpec(self.model, self.horizon, self.lookback)
            self._forecaster = Forecaster(spec, self.cache_dir, self.workers)
        return self._forecaster

    def _healthy_set(self, view: DataView, among: list[str] | None) -> list[str]:
        """The healthiest ``healthy_share`` of candidates, refreshed per ``health_schedule``."""
        quarter = (view.asof.year, (view.asof.month - 1) // 3)
        if self.health_schedule == "Q" and self._healthy and self._healthy[0] == quarter:
            eligible = set(among if among is not None else view.eligible())
            return [s for s in self._healthy[1] if s in eligible]
        pool = among if among is not None else view.eligible()
        health = health_scores(view.features(pool, list(PIOTROSKI)))
        ranked = sorted(health, key=lambda s: -health[s])
        chosen = ranked[: round(len(ranked) * self.healthy_share)]
        self._healthy = (quarter, chosen)
        return chosen

    def target_weights(self, view: DataView) -> Weights:
        """Forecast the most liquid eligible stocks and optimize their weights."""
        among = None
        if self.min_fscore is not None:
            among = [s for s, f in view.fscores(view.eligible()).items() if f >= self.min_fscore]
        if self.healthy_share is not None:
            among = self._healthy_set(view, among)
        prices = price_windows(view, view.top_liquid(self.top_n, among=among), self.lookback)
        if prices.shape[1] < 2 or len(prices) < 30:
            return {}
        windows = {s: prices[s].to_numpy() for s in prices.columns}
        mu = pd.Series(self._get_forecaster().forecast(view.asof, windows), dtype=float)
        weights = optimize(mu, prices, view.risk_free(), self.objective, self.max_weight)
        vol = prices.pct_change().std() * TRADING_DAYS**0.5
        exp_col, vol_col = self.example_columns
        self.last_signals = {s: {exp_col: float(mu[s]), vol_col: float(vol[s])} for s in weights}
        return weights

    def close(self) -> None:
        """Release the forecaster's worker pool."""
        if self._forecaster is not None:
            self._forecaster.close()
