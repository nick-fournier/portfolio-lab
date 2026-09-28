"""Performance metrics computed from a backtest's daily returns."""

import numpy as np

TRADING_DAYS = 252


def _max_drawdown(nav: np.ndarray) -> tuple[float, int]:
    """Return the deepest peak-to-trough decline and the longest time under water (days)."""
    peaks = np.maximum.accumulate(nav)
    drawdown = nav / peaks - 1
    longest = current = 0
    for underwater in drawdown < 0:
        current = current + 1 if underwater else 0
        longest = max(longest, current)
    return float(drawdown.min()), longest


def compute(
    returns: np.ndarray,
    rf_daily: np.ndarray,
    benchmark: np.ndarray | None = None,
    turnover: np.ndarray | None = None,
    holdings: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize daily strategy returns.

    Args:
        returns: Daily portfolio returns (after costs).
        rf_daily: Daily risk-free rate for the same days.
        benchmark: Daily benchmark returns, for beta, alpha and information ratio.
        turnover: Daily one-way turnover (sum of absolute weight changes).
        holdings: Number of positions held each day.

    Returns:
        A flat dict of metrics; ratios are annualized with 252 trading days.
    """
    n = len(returns)
    nav = np.cumprod(1 + returns)
    years = n / TRADING_DAYS
    excess = returns - rf_daily
    vol = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    downside = excess[excess < 0]
    downside_dev = (
        float(np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS)) if downside.size else 0.0
    )
    max_dd, dd_days = _max_drawdown(np.r_[1.0, nav])

    metrics = {
        "days": float(n),
        "total_return": float(nav[-1] - 1) if n else 0.0,
        "cagr": float(nav[-1] ** (1 / years) - 1) if n else 0.0,
        "volatility": vol,
        "sharpe": float(np.mean(excess) * TRADING_DAYS / vol) if vol > 0 else 0.0,
        "sortino": float(np.mean(excess) * TRADING_DAYS / downside_dev)
        if downside_dev > 0
        else 0.0,
        "max_drawdown": max_dd,
        "max_drawdown_days": float(dd_days),
    }
    if turnover is not None:
        metrics["turnover_annual"] = float(np.sum(turnover) / years) if n else 0.0
    if holdings is not None:
        metrics["avg_holdings"] = float(np.mean(holdings)) if n else 0.0
    if benchmark is not None and n > 1:
        bench_excess = benchmark - rf_daily
        var = float(np.var(bench_excess, ddof=1))
        beta = float(np.cov(excess, bench_excess, ddof=1)[0, 1] / var) if var > 0 else 0.0
        active = returns - benchmark
        tracking = float(np.std(active, ddof=1) * np.sqrt(TRADING_DAYS))
        metrics |= {
            "beta": beta,
            "alpha": float((np.mean(excess) - beta * np.mean(bench_excess)) * TRADING_DAYS),
            "tracking_error": tracking,
            "information_ratio": float(np.mean(active) * TRADING_DAYS / tracking)
            if tracking > 0
            else 0.0,
        }
    return metrics
