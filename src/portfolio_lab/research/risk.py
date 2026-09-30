"""Factor risk model: how stocks move together, estimated through shared drivers.

Estimating every pair's co-movement directly needs far more history than a couple of
years provides once there are thousands of stocks. A factor model assumes stocks move
together *because* they share drivers, and estimates only those:

- **Exposures** (known on the rebalance date): the market (every stock 1), its sector (one
  of :data:`SECTORS`, from the SIC code), and five styles as cross-sectional z-scores
  (winsorized at +/-3, missing = 0): size, value (earnings yield and book-to-market),
  momentum, volatility and profitability (return on assets).
- **Factor returns**: each past day's stock returns regressed on those exposures (static
  exposures from the rebalance date, applied to the trailing window of past returns only).
- **Covariance**: ``X F X' + D``, with ``F`` the exponentially weighted covariance of the
  factor returns and ``D`` each stock's leftover (specific) variance, shrunk halfway toward
  the typical stock's. Annualized.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from portfolio_lab.research.dataview import DataView

WINDOW = 504
HALF_LIFE = 126
TRADING_DAYS = 252
#: Share of the window a stock needs returns for to be in the model.
MIN_COVERAGE = 0.8
#: Specific variance is shrunk this far toward the cross-sectional median.
SPECIFIC_SHRINK = 0.5
STYLES = {
    "size": ("log_size",),
    "value": ("earnings_yield", "book_to_market"),
    "momentum": ("mom_12_1",),
    "volatility": ("volatility",),
    "profitability": ("roa",),
}
#: Sector -> SIC 2-digit major groups (anything else, or no SIC, is "unclassified").
SECTORS: dict[str, tuple[int, ...]] = {
    "resources": (*range(1, 18),),  # agriculture, mining, construction
    "chemicals_pharma": (28,),
    "technology": (35, 36, 38),
    "manufacturing": tuple(g for g in range(20, 40) if g not in (28, 35, 36, 38)),
    "transport": (*range(40, 48),),
    "communications": (48,),
    "utilities": (49,),
    "trade": (*range(50, 60),),
    "finance": (*range(60, 68),),
    "business_services": (73,),
    "other_services": tuple(g for g in range(70, 90) if g != 73),
}
_SECTOR_OF = {g: s for s, groups in SECTORS.items() for g in groups}
SECTOR_NAMES = (*SECTORS, "unclassified")


def sector(sic2: int | None) -> str:
    """The sector of a SIC major group (``unclassified`` if unknown)."""
    return _SECTOR_OF.get(sic2, "unclassified") if sic2 is not None else "unclassified"


def _zscore(values: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score, winsorized at +/-3, missing values at 0 (average)."""
    ok = np.isfinite(values)
    out = np.zeros(len(values))
    if ok.sum() > 2 and np.std(values[ok]) > 0:
        z = (values[ok] - values[ok].mean()) / values[ok].std()
        out[ok] = np.clip(z, -3, 3)
    return out


@dataclass(frozen=True)
class RiskModel:
    """A fitted factor risk model for ``symbols`` on one date (annualized)."""

    symbols: list[str]
    exposures: np.ndarray  # stocks x factors
    factors: list[str]
    factor_cov: np.ndarray  # factors x factors
    specific_var: np.ndarray  # stocks
    sectors: list[str]

    def cov(self, symbols: list[str]) -> np.ndarray:
        """Covariance matrix of ``symbols`` (all must be in the model)."""
        index = {s: k for k, s in enumerate(self.symbols)}
        rows = [index[s] for s in symbols]
        x = self.exposures[rows]
        return x @ self.factor_cov @ x.T + np.diag(self.specific_var[rows])

    def volatility(self, symbols: list[str]) -> np.ndarray:
        """Each symbol's total annualized volatility."""
        return np.sqrt(np.diag(self.cov(symbols)))


def exposures(features: pl.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """Exposure matrix for the rows of ``features`` (one date), factor names, sectors."""
    sectors = [sector(g) for g in features["sic2"].to_list()] if "sic2" in features.columns \
        else ["unclassified"] * features.height  # fmt: skip
    columns = [np.ones(features.height)]
    names = ["market"]
    for name in SECTOR_NAMES:
        dummy = np.array([s == name for s in sectors], dtype=float)
        if dummy.sum() > 0:
            columns.append(dummy)
            names.append(f"sector: {name}")
    for style, inputs in STYLES.items():
        parts = [
            _zscore(features[c].cast(pl.Float64).fill_null(np.nan).to_numpy())
            for c in inputs if c in features.columns
        ]  # fmt: skip
        if parts:
            columns.append(_zscore(np.mean(parts, axis=0)))
            names.append(style)
    return np.column_stack(columns), names, sectors


def _ewma_weights(n: int, half_life: int) -> np.ndarray:
    w = 0.5 ** (np.arange(n)[::-1] / half_life)
    return w / w.sum()


def fit(view: DataView, symbols: list[str]) -> RiskModel | None:
    """Fit the risk model for ``symbols`` as of the view's date (see module docs).

    Returns ``None`` when fewer than 20 symbols have enough history.
    """
    feats = view.features(symbols, [c for c in _feature_columns()])
    if feats.is_empty():
        return None
    returns = view.returns(WINDOW, feats["symbol"].to_list())
    keep = (returns.notna().mean() >= MIN_COVERAGE).to_numpy()
    if keep.sum() < 20:
        return None
    kept = [s for s, k in zip(returns.columns, keep, strict=True) if k]
    feats = feats.filter(pl.col("symbol").is_in(kept)).sort(
        pl.col("symbol").replace_strict({s: i for i, s in enumerate(kept)})
    )
    x, names, sectors = exposures(feats)
    r = returns[kept].fillna(0.0).to_numpy()  # days x stocks
    # Factor returns by least squares (a small ridge keeps sector dummies well-posed).
    xtx = x.T @ x + 1e-6 * np.eye(x.shape[1])
    f = np.linalg.solve(xtx, x.T @ r.T).T  # days x factors
    resid = r - f @ x.T
    w = _ewma_weights(len(r), HALF_LIFE)
    fc = f - w @ f
    factor_cov = (fc * w[:, None]).T @ fc * TRADING_DAYS
    specific = (w @ (resid - w @ resid) ** 2) * TRADING_DAYS
    specific = (1 - SPECIFIC_SHRINK) * specific + SPECIFIC_SHRINK * np.median(specific)
    return RiskModel(kept, x, names, factor_cov, specific, sectors)


def _feature_columns() -> list[str]:
    return ["sic2", *{c for inputs in STYLES.values() for c in inputs}]
