"""Plain-language explanations of how a strategy's backtest run works.

Every strategy provides ``explain() -> dict`` with the keys in :data:`KEYS`, written for
someone who hasn't read the code. The schedule-and-execution step is the same machinery
for every strategy, so it is generated here from the backtest settings. Strategies can also
record ``last_signals`` (per held symbol, the values it ranked by) at each rebalance, with
``example_columns`` naming and formatting them; the engine saves the latest ones as the
run's worked example.
"""

from portfolio_lab.core.calendar import Frequency

#: Keys every ``explain()`` returns.
KEYS = ("summary", "candidates", "signal", "construction", "drivers", "related")

ELIGIBLE = (
    "Eligible means it closed above $5 that day, trades over $1M a day (median over the past "
    "60 trading days), has at least a year of price history, and actually traded that day."
)

_SCHEDULES = {
    "D": "every trading day",
    "W": "on the last trading day of each week",
    "M": "on the last trading day of each month",
    "Q": "on the last trading day of each quarter",
}


def candidates(pool: int | None) -> str:
    """The standard candidate-selection sentence for a pool of the most liquid stocks."""
    if pool is None:
        return f"Every eligible stock on the rebalance date. {ELIGIBLE}"
    return (
        f"The {pool} most liquid eligible stocks on the rebalance date, ranked by median daily "
        f"dollar volume over the past 60 trading days. {ELIGIBLE}"
    )


def execution(
    schedule: Frequency, half_spread_bps: float, notional: float, max_missing_days: int,
    delisting_return: float,
) -> str:  # fmt: skip
    """The schedule-and-execution step, from the run's actual settings."""
    return (
        f"Rebalances {_SCHEDULES.get(schedule, schedule)}. It decides with data up to that "
        "day's close and trades at the next day's open, paying half the bid-ask spread "
        f"({half_spread_bps:g} bps) plus a market-impact cost that grows with the trade's size "
        f"relative to the stock's volume (at a ${notional:,.0f} portfolio). Uninvested cash earns "
        f"the 3-month T-bill rate. A holding with no trades for {max_missing_days} sessions is "
        f"sold; if it fell off its exchange, at a {-delisting_return:.0%} loss."
    )
