"""Make versus buy: our strategies against funds anyone can buy.

Fund returns come from prices adjusted for distributions, so they are after the fund's
own fees and include reinvested dividends; our strategies' returns are after modeled
trading costs but pay no management fee. Everyone is measured with the same statistics
(``backtest.metrics.compute``) over two windows: each series' full history since the
start date, and the common period in which every series exists.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from portfolio_lab.backtest.metrics import compute


@dataclass(frozen=True)
class Fund:
    """A fund to compare against."""

    symbol: str
    name: str
    category: str
    source: str = "alpaca"  # "tiingo" for open-end mutual funds (not exchange-traded)


CATEGORIES = {
    "ours": "Our strategies",
    "passive": "Passive, broad market",
    "equal": "Passive, equal weight",
    "factor": "Passive factor funds",
    "active_etf": "Active ETFs",
    "closed_end": "Closed-end funds",
    "mutual": "Active mutual funds",
    "holding": "Buy a great manager",
}

FUNDS = (
    Fund("SPY", "S&P 500 (SPDR)", "passive"),
    Fund("VTI", "Total US market (Vanguard)", "passive"),
    Fund("QQQ", "Nasdaq 100 (Invesco)", "passive"),
    Fund("IWM", "Russell 2000 small caps (iShares)", "passive"),
    Fund("RSP", "S&P 500 equal weight (Invesco)", "equal"),
    Fund("QUAL", "US quality (iShares)", "factor"),
    Fund("VTV", "US value (Vanguard)", "factor"),
    Fund("MTUM", "US momentum (iShares)", "factor"),
    Fund("USMV", "US minimum volatility (iShares)", "factor"),
    Fund("ARKK", "ARK Innovation", "active_etf"),
    Fund("AVUV", "Avantis US small-cap value", "active_etf"),
    Fund("DFAC", "Dimensional US core equity 2", "active_etf"),
    Fund("ADX", "Adams Diversified Equity", "closed_end"),
    Fund("TY", "Tri-Continental", "closed_end"),
    Fund("USA", "Liberty All-Star Equity", "closed_end"),
    Fund("GAM", "General American Investors", "closed_end"),
    Fund("FCNTX", "Fidelity Contrafund", "mutual", "tiingo"),
    Fund("VPMCX", "Vanguard PRIMECAP", "mutual", "tiingo"),
    Fund("AGTHX", "American Funds Growth Fund of America", "mutual", "tiingo"),
    Fund("DODGX", "Dodge & Cox Stock", "mutual", "tiingo"),
    Fund("TRBCX", "T. Rowe Price Blue Chip Growth", "mutual", "tiingo"),
    Fund("BRK.B", "Berkshire Hathaway", "holding"),
)

#: Our strategy (run label) -> the fund it most directly competes with.
PAIRS = {
    "equal_weight": "RSP",
    "equal_weight (top_n=100)": "RSP",
    "momentum": "MTUM",
    "meanvar": "USMV",
    "meanvar (top_n=500)": "USMV",
    "piotroski (pool=100)": "QUAL",
    "meanvar (min_fscore=7)": "QUAL",
    "meanvar (healthy_share=0.27)": "QUAL",
    "meanvar (health_rank_pool=400, soften=taper, bear_defense=True, rebound=equal)": "QUAL",
}

METRICS = ("cagr", "volatility", "sharpe", "max_drawdown", "beta", "alpha")


def _daily_rf(dates: list[date], rates: pl.DataFrame | None) -> np.ndarray:
    """Daily risk-free rate for each date (annual T-bill rate / 252, as of that date)."""
    if rates is None or rates.is_empty():
        return np.zeros(len(dates))
    joined = pl.DataFrame({"date": dates}).join_asof(rates.sort("date"), on="date")
    return (joined["rate"].fill_null(0.0) / 252).to_numpy()


def _stats(returns: pl.DataFrame, name: str, start: date, rates: pl.DataFrame | None) -> dict:
    """Metrics of one series (date, ret) from ``start`` against SPY (column ``spy``)."""
    rows = returns.filter((pl.col("date") >= start) & pl.col(name).is_not_null())
    dates = rows["date"].to_list()
    metrics = compute(
        rows[name].to_numpy(), _daily_rf(dates, rates), rows["spy"].fill_null(0.0).to_numpy()
    )
    return {"start": dates[0], "end": dates[-1], **{m: metrics.get(m) for m in METRICS}}


#: Series starting more than this long after the comparison start are "young": they don't
#: push back the common period (they are measured from their own start in it instead).
YOUNG_AFTER = timedelta(days=365)


def compare(
    series: dict[str, tuple[str, str, pl.DataFrame]],
    rates: pl.DataFrame | None,
    start: date,
    ours: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Statistics for every series over its full history and over the common period.

    The common period starts when every series that existed within :data:`YOUNG_AFTER` of
    ``start`` has data; younger funds don't shorten it for everyone.

    Args:
        series: Key -> (display name, category, frame of date and daily ``ret``). Must
            include ``SPY`` (the market, for beta and alpha).
        rates: Risk-free history (date, annual rate).
        start: Earliest date considered.
        ours: Key of one of our series: it is also measured over each other series' own
            history (period ``ours_since:<key>``), for head-to-heads with young funds.

    Returns:
        (summary, growth): summary has key, name, category, period (``full``, ``common``
        or ``ours_since:<key>``), start, end and :data:`METRICS`; growth has date, key and
        the growth of $1 over the common period.
    """
    wide = None
    for key, (_, _, frame) in series.items():
        part = frame.select("date", pl.col("ret").alias(key))
        wide = part if wide is None else wide.join(part, on="date", how="full", coalesce=True)
    wide = (
        wide.filter(pl.col("date") >= start).sort("date").with_columns(pl.col("SPY").alias("spy"))
    )
    firsts = {k: wide.filter(pl.col(k).is_not_null())["date"].min() for k in series}
    first = min(d for d in firsts.values() if d is not None)
    common = max(d for d in firsts.values() if d is not None and d <= first + YOUNG_AFTER)
    rows = []
    for key, (name, category, _) in series.items():
        if firsts[key] is None:
            continue
        for period, since in (("full", firsts[key]), ("common", max(common, firsts[key]))):
            rows.append({"key": key, "name": name, "category": category, "period": period,
                         **_stats(wide, key, since, rates)})  # fmt: skip
        if ours and key != ours and firsts.get(ours) and firsts[key] >= firsts[ours]:
            name_ours, category_ours, _ = series[ours]
            rows.append({"key": ours, "name": name_ours, "category": category_ours,
                         "period": f"ours_since:{key}",
                         **_stats(wide, ours, firsts[key], rates)})  # fmt: skip
    growth = (
        wide.filter(pl.col("date") >= common)
        .unpivot(index="date", on=list(series), variable_name="key", value_name="ret")
        .with_columns((1 + pl.col("ret").fill_null(0.0)).cum_prod().over("key").alias("growth"))
        .select("date", "key", "growth")
    )
    return pl.DataFrame(rows), growth
