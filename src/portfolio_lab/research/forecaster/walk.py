"""Walk forward: refit both parts every month on all earlier months, forecast the month.

For month end ``t`` the training months are every month before ``t`` (their targets end
by ``t``). Part 1 (``linear``) is fitted, then part 2 (``trees``) on part 1's residual
over the same months; both forecast ``t``'s stocks. The trees' output is kept on its own
(``correction``) so its strength is chosen separately by :func:`combine`.

:func:`combine`: forecast = part 1 + k · part 2, with k from :data:`STRENGTHS` chosen each
month as the one with the lowest squared error (each part demeaned within the month,
against ``actual``) summed over all earlier forecast months.
"""

import logging
import time
from collections.abc import Callable
from datetime import date

import numpy as np
import polars as pl

from portfolio_lab.research.forecaster import linear
from portfolio_lab.research.forecaster.dataset import INPUTS
from portfolio_lab.research.forecaster.trees import TreesPart

log = logging.getLogger(__name__)

#: First month forecast (earlier months are training only).
FIRST_FORECAST = date(2004, 1, 1)
#: First year graded (the strength needs earlier forecasts to be chosen from).
FIRST_GRADED = 2009
STRENGTHS = (0.0, 0.25, 0.5, 1.0)


def run(
    stocks: pl.DataFrame,
    market: pl.DataFrame,
    *,
    start: date = FIRST_FORECAST,
    components: int | None = None,
    part2=None,
    skip: set[date] | None = None,
    save: Callable[[pl.DataFrame], None] | None = None,
) -> pl.DataFrame:
    """Forecasts for every month end from ``start`` (module docs).

    Args:
        stocks: From ``dataset.stocks`` (sorted by date).
        market: From ``dataset.market``.
        start: First month end to forecast.
        components: Part 1's component count fixed instead of chosen by k-fold; 0 = no
            part 1 (trees alone, on the target itself).
        part2: What fits part 1's residual each month: ``trees.TreesPart`` (default, on
            the CPU) or ``nets.NetsPart``; anything with ``dispersion``, ``arrays``,
            ``fit``, ``predict`` and ``release``.
        skip: Month ends already forecast (left out).
        save: Called with each month's forecasts as soon as it is done.

    Returns:
        date, symbol, actual, size, linear, correction and components (part 1's count).
    """
    stocks = stocks.sort("date", "symbol")
    dates = stocks["date"].to_numpy()
    months = sorted(set(stocks["date"].to_list()))
    bounds = np.searchsorted(dates, np.array(months, dtype=dates.dtype))
    span = {
        d: slice(int(a), int(b))
        for d, a, b in zip(months, bounds, [*bounds[1:], len(dates)], strict=True)
    }
    labeled = set(stocks.filter(pl.col("y").is_not_null())["date"].unique().to_list())
    conditions = stocks.select("date").join(market, on="date", how="left")
    raw = stocks.select(INPUTS).to_numpy()
    x = linear.design(raw, np.nan_to_num(conditions["vix"].to_numpy()),
                      np.nan_to_num(conditions["dispersion"].to_numpy()))  # fmt: skip
    y = stocks["y"].to_numpy()
    sums = {d: linear.month_sums(x[span[d]], y[span[d]]) for d in months if d in labeled}
    part2 = part2 or TreesPart()
    market_cols = [c for c in market.columns if c.startswith(("env_", "mkt_"))]
    if part2.dispersion:
        market_cols.append("dispersion")
    market_cols = getattr(part2, "columns", None) or market_cols
    xt = np.hstack([raw, conditions.select(market_cols).to_numpy()]).astype(np.float32)
    del raw
    years = stocks["date"].dt.year().to_numpy()
    if hasattr(part2, "row_months"):  # month index of every row, for month-wise batches
        part2.row_months = np.searchsorted(np.array(months, dtype=dates.dtype), dates)
    fit_x, to_device = part2.arrays(xt)
    out = []
    for t in months:
        if t < start or t in (skip or set()):
            continue
        train = [d for d in months if d < t and d in labeled]
        rows = slice(0, span[t].start)  # every earlier month (sorted by date)
        tic = time.monotonic()
        coef, k = linear.fit(train, sums, components)
        residual = to_device(y[rows] - x[rows] @ coef)
        part2.now = t
        part2.fit(fit_x[rows], residual, years[rows])
        here = span[t]
        month = (
            stocks[here]
            .select("date", "symbol", "actual", "size")
            .with_columns(
                pl.Series("linear", x[here] @ coef),
                pl.Series("correction", part2.predict(fit_x[here])),
                pl.lit(k).alias("components"),
                pl.lit(getattr(part2, "held_err", None), dtype=pl.Float64).alias("held_err"),
            )
        )
        log.info("forecast %s: %d stocks, %d components, %.0fs", t, month.height, k,
                 time.monotonic() - tic)  # fmt: skip
        part2.release()
        out.append(month)
        if save is not None:
            save(month)
    return pl.concat(out) if out else pl.DataFrame()


def combine(forecasts: pl.DataFrame) -> pl.DataFrame:
    """``forecasts`` with ``strength`` and ``forecast`` (module docs).

    With no part 1 (``components`` 0 throughout), the trees are the forecast as they are.
    """
    if "components" in forecasts.columns and (forecasts["components"] == 0).all():
        return forecasts.drop_nulls("correction").with_columns(
            pl.lit(1.0).alias("strength"), pl.col("correction").alias("forecast")
        )
    f = forecasts.drop_nulls(["linear", "correction"]).with_columns(
        (pl.col("linear") - pl.col("linear").mean().over("date")).alias("_b"),
        (pl.col("correction") - pl.col("correction").mean().over("date")).alias("_c"),
    )
    err = (
        f.drop_nulls("actual")
        .group_by("date")
        .agg(
            *[
                ((pl.col("actual") - pl.col("_b") - k * pl.col("_c")) ** 2).sum().alias(str(k))
                for k in STRENGTHS
            ]
        )
        .sort("date")
    )
    months = sorted(f["date"].unique().to_list())
    past = err["date"].to_numpy()
    cumulative = np.vstack(
        [
            np.zeros(len(STRENGTHS)),
            np.cumsum(err.select([str(k) for k in STRENGTHS]).to_numpy(), axis=0),
        ]
    )
    chosen = [STRENGTHS[int(np.argmin(cumulative[np.searchsorted(past, np.datetime64(d))]))]
              for d in months]  # fmt: skip
    strength = pl.DataFrame({"date": months, "strength": chosen})
    return (
        f.drop("_b", "_c")
        .join(strength, on="date")
        .with_columns(
            (pl.col("linear") + pl.col("strength") * pl.col("correction")).alias("forecast")
        )
    )
