"""Market context: the prevailing environment, market valuation, and each stock's sensitivity.

Context is the same for every stock on a date, so on its own it can't rank stocks. It
enters two ways, both point-in-time:

- **Environment** (:func:`environment`, one row per date): each FRED series as known on
  that date (``fred.CONTEXT_SERIES``), its 3- and 12-month change, and its percentile within
  its own history up to that date; yield-curve slopes; and **market valuation** from our
  own feature panel (the market's total earnings and book value over its total market
  value, and the earnings yield's gap over the 10-year Treasury yield). This describes
  conditions; it is never used to time the market.
- **Sensitivities** (:func:`sensitivities`, per stock): a trailing regression of each
  stock's weekly returns on the market and on weekly moves in oil, the 10-year yield, the
  dollar and the Baa credit spread, so each factor's beta is net of the market. A
  **tailwind** is a sensitivity times that factor's move over the past three months: is
  the environment currently moving in this stock's favor?
"""

from datetime import date, timedelta

import numpy as np
import polars as pl

from portfolio_lab.data.sources.fred import CONTEXT_SERIES
from portfolio_lab.research.panel import Panel

#: Days back for the 3- and 12-month changes.
CHANGES = {"chg3m": 91, "chg12m": 365}
#: Factors for stock sensitivities: (environment column, transform of weekly moves).
FACTORS = {"oil": "log", "yield_10y": "diff", "dollar": "log", "baa_spread": "diff"}
SENSITIVITY_WEEKS = 104
WEEK = 5
#: Share of weeks a stock needs returns for to get sensitivities.
MIN_COVERAGE = 0.8
#: Per-stock context features added to the monthly panel.
STOCK_FEATURES = (
    *(f"{f}_beta" for f in FACTORS), *(f"{f}_tailwind" for f in FACTORS), "macro_tailwind",
)  # fmt: skip


def _prepared(observations: pl.DataFrame) -> pl.DataFrame:
    """Observations renamed to our names, with year-on-year series converted to % changes.

    Year-on-year changes compare first-release values twelve observations apart (monthly
    series), dated by the later value's publication.
    """
    names = pl.DataFrame(
        [(sid, s.name, s.kind) for sid, s in CONTEXT_SERIES.items()],
        schema=["series", "name", "kind"],
        orient="row",
    )
    obs = observations.join(names, on="series").sort("name", "date")
    yoy = pl.col("value") / pl.col("value").shift(12).over("name") - 1
    return obs.with_columns(
        pl.when(pl.col("kind") == "yoy").then(yoy).otherwise(pl.col("value")).alias("value")
    ).drop_nulls("value")


def _asof(obs: pl.DataFrame, dates: list[date], name: str) -> np.ndarray:
    """The latest value of series ``name`` known on each date (NaN before any)."""
    series = obs.filter(pl.col("name") == name).sort("available", "date")
    if series.is_empty():
        return np.full(len(dates), np.nan)
    frame = pl.DataFrame({"date": dates}).sort("date")
    joined = frame.join_asof(
        series.select("available", "value"), left_on="date", right_on="available",
        strategy="backward",
    )  # fmt: skip
    return joined["value"].fill_null(np.nan).to_numpy()


def _percentile(obs: pl.DataFrame, dates: list[date], name: str) -> np.ndarray:
    """Percentile of each date's value among all values of ``name`` known by that date."""
    series = obs.filter(pl.col("name") == name).sort("available")
    known = series["available"].to_numpy().astype("datetime64[D]")
    values = series["value"].to_numpy()
    counts = np.searchsorted(known, np.array(dates, dtype="datetime64[D]"), side="right")
    out = np.full(len(dates), np.nan)
    for k, n in enumerate(counts):
        if n >= 20:  # the latest known value's rank among everything known so far
            out[k] = (values[:n] < values[n - 1]).mean()
    return out


def environment(
    observations: pl.DataFrame, dates: list[date], features: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Point-in-time environment on each of ``dates`` (see module docs).

    Args:
        observations: FRED observations (``fred.OBSERVATION_SCHEMA``).
        dates: Dates to describe (e.g. month ends).
        features: The monthly feature panel, for market valuation (needs ``market_value``,
            ``earnings_yield`` and ``book_to_market``).

    Returns:
        date, then per series its level, ``_chg3m``, ``_chg12m`` and ``_pct``; the curve
        slopes; and market valuation columns when ``features`` is given.
    """
    obs = _prepared(observations)
    kinds = {s.name: s.kind for s in CONTEXT_SERIES.values()}
    columns: dict[str, np.ndarray] = {}
    for name, kind in kinds.items():
        now = _asof(obs, dates, name)
        columns[name] = now
        for label, days in CHANGES.items():
            before = _asof(obs, [d - timedelta(days=days) for d in dates], name)
            with np.errstate(invalid="ignore", divide="ignore"):
                change = np.log(now / before) if kind == "log" else now - before
            columns[f"{name}_{label}"] = change
        columns[f"{name}_pct"] = _percentile(obs, dates, name)
    for short in ("2y", "3m"):
        columns[f"curve_10y_{short}"] = columns["yield_10y"] - columns[f"yield_{short}"]
        for label in CHANGES:
            columns[f"curve_10y_{short}_{label}"] = (
                columns[f"yield_10y_{label}"] - columns[f"yield_{short}_{label}"]
            )
    env = pl.DataFrame({"date": dates, **columns}).fill_nan(None)
    if features is not None:
        env = env.join(market_valuation(features), on="date", how="left").with_columns(
            (pl.col("market_earnings_yield") - pl.col("yield_10y") / 100).alias(
                "equity_risk_premium"
            )
        )
    return env.sort("date")


def market_valuation(features: pl.DataFrame) -> pl.DataFrame:
    """The market's earnings and book yields per date, and their percentile to that date.

    Aggregates (total earnings over total market value) rather than averages, so large
    companies count by size and a few extreme ratios can't dominate.
    """
    has = pl.col("market_value").is_not_null()
    earnings = (pl.col("earnings_yield") * pl.col("market_value")).filter(
        has & pl.col("earnings_yield").is_not_null()
    )
    book = (pl.col("book_to_market") * pl.col("market_value")).filter(
        has & pl.col("book_to_market").is_not_null()
    )
    value_e = pl.col("market_value").filter(has & pl.col("earnings_yield").is_not_null())
    value_b = pl.col("market_value").filter(has & pl.col("book_to_market").is_not_null())
    agg = (
        features.group_by("date")
        .agg(
            (earnings.sum() / value_e.sum()).alias("market_earnings_yield"),
            (book.sum() / value_b.sum()).alias("market_book_to_market"),
        )
        .sort("date")
    )
    # Expensive = low earnings yield: percentile of the yield among dates so far.
    ey = agg["market_earnings_yield"].to_numpy()
    pct = [np.nan if k < 12 else float((ey[: k + 1] < ey[k]).mean()) for k in range(len(ey))]
    return agg.with_columns(pl.Series("market_earnings_yield_pct", pct).fill_nan(None))


def _weekly_factor_moves(obs: pl.DataFrame, ends: list[date]) -> np.ndarray:
    """Moves of each factor between consecutive week ends ((weeks-1) x factors)."""
    moves = []
    for name, kind in FACTORS.items():
        level = _asof(obs, ends, name)
        with np.errstate(invalid="ignore", divide="ignore"):
            moves.append(np.diff(np.log(level)) if kind == "log" else np.diff(level))
    return np.column_stack(moves)


def sensitivities(panel: Panel, observations: pl.DataFrame, dates: list[date]) -> pl.DataFrame:
    """Per-stock factor betas (net of the market) on each of ``dates``, from past weeks only.

    Returns:
        date, symbol and ``{factor}_beta`` for each factor in :data:`FACTORS`.
    """
    obs = _prepared(observations)
    ret = panel.field("ret_cc")
    spy = panel.market
    frames = []
    for day in dates:
        i = panel.date_index[day]
        span = SENSITIVITY_WEEKS * WEEK
        if i < span:
            continue
        ends = [panel.dates[j] for j in range(i - span, i + 1, WEEK)]
        factors = _weekly_factor_moves(obs, ends)
        cols = np.flatnonzero(panel.eligible[i])
        daily = ret[i - span + 1 : i + 1]
        blocks = daily.reshape(SENSITIVITY_WEEKS, WEEK, -1)
        weekly = np.prod(1 + np.nan_to_num(blocks), axis=1) - 1
        seen = np.isfinite(blocks).any(axis=1)
        market = weekly[:, spy]
        # Factors without data (e.g. a series not yet published) are left out; their
        # betas are reported as missing rather than dropping the whole regression.
        usable = np.isfinite(factors).mean(axis=0) >= MIN_COVERAGE
        x = np.column_stack([np.ones(SENSITIVITY_WEEKS), market, factors[:, usable]])
        ok_rows = np.isfinite(x).all(axis=1)
        if ok_rows.sum() < SENSITIVITY_WEEKS // 2:
            continue
        keep = cols[seen[:, cols].mean(axis=0) >= MIN_COVERAGE]
        coef, *_ = np.linalg.lstsq(x[ok_rows], weekly[ok_rows][:, keep], rcond=None)
        betas = iter(coef[2:])
        frames.append(
            pl.DataFrame(
                {
                    "date": [day] * len(keep),
                    "symbol": [panel.symbols[j] for j in keep],
                    **{
                        f"{f}_beta": next(betas) if ok else np.full(len(keep), np.nan)
                        for f, ok in zip(FACTORS, usable, strict=True)
                    },
                }
            ).fill_nan(None)
        )
    return pl.concat(frames) if frames else pl.DataFrame()


def tailwinds(sens: pl.DataFrame, env: pl.DataFrame) -> pl.DataFrame:
    """Sensitivity times each factor's 3-month move, and their sum (``macro_tailwind``)."""
    moves = env.select("date", *[f"{f}_chg3m" for f in FACTORS])
    joined = sens.join(moves, on="date", how="left")
    parts = [(pl.col(f"{f}_beta") * pl.col(f"{f}_chg3m")).alias(f"{f}_tailwind") for f in FACTORS]
    out = joined.with_columns(parts).drop([f"{f}_chg3m" for f in FACTORS])
    return out.with_columns(
        pl.sum_horizontal([f"{f}_tailwind" for f in FACTORS]).alias("macro_tailwind")
    )
