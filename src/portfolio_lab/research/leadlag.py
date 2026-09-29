"""Lead-lag networks: which stocks' returns tend to follow which others' a period later?

For a window of returns over periods of ``horizon`` sessions (1 = daily, 5 = weekly), each
stock's series is standardized and the lagged correlation matrix is computed,

    C[i, j] = corr(r_i(t - 1), r_j(t)),

so row ``i`` holds how much stock ``i``'s return predicts every stock's next-period return:
``i`` leads ``j``. With ~3,000 stocks that is ~9 million pairs from a year of data, almost
all noise, so each follower keeps only its ``k`` strongest leaders (by absolute
correlation, excluding itself). A follower's prediction is the correlation-weighted sum of
its leaders' latest standardized returns.

``market`` mode replaces the pairwise network with one leader for everyone: the equal-weight
return of the most liquid stocks (the classic "big stocks lead small ones" effect).
"""

from dataclasses import dataclass
from datetime import date

import numpy as np

from portfolio_lab.research.dataview import DataView

#: Number of most liquid stocks averaged into the ``market`` leader.
MARKET_LEADERS = 50
#: Share of the window's periods a stock needs returns for.
MIN_COVERAGE = 0.9


@dataclass(frozen=True)
class Network:
    """A lead-lag model estimated at one date.

    Args:
        asof: Last session of the estimation window.
        symbols: Stocks in the model (the followers and potential leaders).
        leaders: ``(followers x k)`` indices into ``symbols`` of each follower's leaders.
        weights: ``(followers x k)`` lagged correlations for those leaders.
        mean: Per-symbol mean period return over the window (to standardize new returns).
        std: Per-symbol standard deviation over the window.
    """

    asof: date
    symbols: list[str]
    leaders: np.ndarray
    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def period_returns(view: DataView, symbols: list[str], periods: int, horizon: int) -> np.ndarray:
    """Compounded returns over consecutive non-overlapping blocks of ``horizon`` sessions.

    The last block ends at the view's decision date. Missing bars count as zero return.

    Returns:
        A ``(periods x symbols)`` array, oldest first; NaN where a symbol had no bar at all
        in a block.
    """
    daily = view.returns(periods * horizon, symbols).to_numpy()
    usable = len(daily) // horizon * horizon
    blocks = daily[len(daily) - usable :].reshape(-1, horizon, len(symbols))
    seen = np.isfinite(blocks).any(axis=1)
    compounded = np.prod(1 + np.nan_to_num(blocks), axis=1) - 1
    return np.where(seen, compounded, np.nan)


def lagged_correlation(z: np.ndarray) -> np.ndarray:
    """``C[i, j] = corr(z_i(t - 1), z_j(t))`` for standardized, gap-free columns ``z``."""
    return z[:-1].T @ z[1:] / (len(z) - 1)


def estimate(
    view: DataView, horizon: int = 1, window: int = 252, k: int = 10, mode: str = "pairs"
) -> Network | None:
    """Estimate a lead-lag network from data up to the view's decision date.

    Args:
        view: Point-in-time data; the window ends at its decision date.
        horizon: Sessions per period (1 daily, 5 weekly).
        window: Periods of history.
        k: Leaders kept per follower (``pairs`` mode).
        mode: ``pairs`` (stock-to-stock network) or ``market`` (one shared leader).

    Returns:
        The network, or ``None`` with too little data.

    Raises:
        ValueError: For an unknown mode.
    """
    if mode not in ("pairs", "market"):
        raise ValueError(f"unknown lead-lag mode {mode!r}")
    symbols = view.eligible()
    returns = period_returns(view, symbols, window, horizon)
    keep = np.isfinite(returns).mean(axis=0) >= MIN_COVERAGE
    if len(returns) < window // 2 or keep.sum() < 2:
        return None
    symbols = [s for s, ok in zip(symbols, keep, strict=True) if ok]
    returns = np.nan_to_num(returns[:, keep])
    mean, std = returns.mean(axis=0), returns.std(axis=0)
    std = np.where(std > 0, std, np.inf)  # constant series standardize to zero
    z = (returns - mean) / std

    if mode == "market":
        big = set(view.top_liquid(MARKET_LEADERS, among=symbols))
        market = z[:, [j for j, s in enumerate(symbols) if s in big]].mean(axis=1)
        market = (market - market.mean()) / market.std()
        beta = market[:-1] @ z[1:] / (len(z) - 1)
        # One leader column per follower: the market series, stored as index -1.
        leaders = np.full((len(symbols), 1), -1)
        weights = beta[:, None]
        return Network(view.asof, symbols, leaders, weights, mean, std)

    corr = lagged_correlation(z)
    np.fill_diagonal(corr, 0.0)  # own lag is short-term reversal, measured separately
    k = min(k, len(symbols) - 1)
    # For follower j (column), the k leaders i with the largest |C[i, j]|.
    leaders = np.argpartition(-np.abs(corr), k - 1, axis=0)[:k].T
    weights = np.take_along_axis(corr.T, leaders, axis=1)
    return Network(view.asof, symbols, leaders, weights, mean, std)


def predict(network: Network, view: DataView, horizon: int) -> dict[str, float]:
    """Each follower's score from its leaders' latest standardized period return.

    The latest period is the ``horizon`` sessions ending at the view's decision date, which
    may be after the network's estimation date.
    """
    latest = period_returns(view, network.symbols, 1, horizon)
    if not len(latest):
        return {}
    z = np.nan_to_num((latest[-1] - network.mean) / network.std)
    if network.leaders.shape[1] == 1 and network.leaders[0, 0] == -1:  # market mode
        big = set(view.top_liquid(MARKET_LEADERS, among=network.symbols))
        leader_z = z[[j for j, s in enumerate(network.symbols) if s in big]].mean()
        scores = network.weights[:, 0] * leader_z
    else:
        scores = (network.weights * z[network.leaders]).sum(axis=1)
    return dict(zip(network.symbols, scores.tolist(), strict=True))
