"""Stress gauge: a self-calibrating sense of "thin ice" before drawdowns and momentum crashes.

Built from the research on what precedes them. Four inputs, each read at every rebalance:

- ``own_downside_vol``: downside volatility of the portfolio held over the past three
  months. Volatility-managed momentum cut momentum's crashes (Barroso & Santa-Clara 2015);
  downside volatility works better out of sample than total volatility.
- ``panic``: the market's drawdown from its two-year high together with its six-month
  volatility: the "panic states" after which momentum crashed (Daniel & Moskowitz 2016).
- ``absorption``: share of the candidates' return variance explained by their top fifth
  of principal components; markets moving as one are fragile (Kritzman et al. 2011).
- ``nfci``: the Chicago Fed's National Financial Conditions Index, where available.

Each input becomes a percentile of **its own history up to that date**, the percentiles are
averaged, and the average is again ranked against its own history, so the final ``stress``
(0 to 1) needs no hand-set thresholds and adapts as markets change. Nothing is fitted to
returns. Until an input has :data:`MIN_HISTORY` readings it is left out; until the composite
has, stress is neutral (0.5).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_HISTORY = 24
YEAR = 252


def percentile(history: list[float], value: float) -> float:
    """Share of ``history`` at or below ``value`` (``history`` includes ``value``)."""
    return float(np.mean(np.asarray(history) <= value))


def downside_vol(returns: pd.Series) -> float | None:
    """Annualized downside deviation (root mean square of negative daily returns)."""
    r = returns.dropna().to_numpy()
    return float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2) * YEAR)) if len(r) > 20 else None


def panic(market: pd.Series) -> float | None:
    """Drawdown from the two-year high times six-month volatility (both positive)."""
    r = market.dropna()
    if len(r) < YEAR:
        return None
    level = (1 + r).cumprod()
    drawdown = 1 - level.iloc[-1] / level.iloc[-2 * YEAR :].max()
    vol = r.iloc[-YEAR // 2 :].std() * np.sqrt(YEAR)
    return float(drawdown * vol)


def absorption(returns: pd.DataFrame) -> float | None:
    """Fraction of variance absorbed by the top fifth of eigenvectors of the correlations."""
    clean = returns.dropna(axis=1, thresh=int(0.9 * len(returns))).fillna(0.0)
    if clean.shape[1] < 10:
        return None
    eig = np.sort(np.linalg.eigvalsh(np.corrcoef(clean.to_numpy(), rowvar=False)))[::-1]
    return float(eig[: max(1, len(eig) // 5)].sum() / eig.sum())


@dataclass
class StressGauge:
    """Running histories of the inputs and the composite (see module docs)."""

    histories: dict[str, list[float]] = field(default_factory=dict)
    composite: list[float] = field(default_factory=list)

    def prewarm(self, name: str, values: list[float]) -> None:
        """Seed an input's history with readings from before the first rebalance."""
        self.histories.setdefault(name, []).extend(v for v in values if v is not None)

    def update(self, readings: dict[str, float | None]) -> tuple[float, dict[str, float]]:
        """Add this rebalance's readings; return (stress, each input's percentile)."""
        pcts = {}
        for name, value in readings.items():
            if value is None or not np.isfinite(value):
                continue
            history = self.histories.setdefault(name, [])
            history.append(value)
            if len(history) >= MIN_HISTORY:
                pcts[name] = percentile(history, value)
        if not pcts:
            return 0.5, pcts
        self.composite.append(float(np.mean(list(pcts.values()))))
        if len(self.composite) < MIN_HISTORY:
            return 0.5, pcts
        return percentile(self.composite, self.composite[-1]), pcts


def tilt(stress: float, low: float, high: float) -> float:
    """How far to move from max Sharpe toward min variance: 0 below ``low``, 1 above ``high``."""
    return float(np.clip((stress - low) / (high - low), 0.0, 1.0))
