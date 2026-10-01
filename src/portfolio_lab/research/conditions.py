"""How prevailing conditions relate to which traits pay off, and to market risk.

Descriptive measurements for the Market context page; nothing here trades.

- :func:`conditional_ic`: a trait's monthly rank IC (from the scoreboard) split by the
  environment on each date. Conditions are bucketed by where the series stood within its
  own history *up to that date* (low, middle or high third), so buckets use no future data.
- :func:`caution_dial`: the market's (SPY) return, volatility and worst drawdown over the
  following months, split the same way by valuation, credit spreads, VIX and the equity
  risk premium. With data since 2017 this is a short sample; treat it as a description,
  not a forecast.
"""

import numpy as np
import polars as pl

#: Traits whose payoff is split by conditions (scoreboard signal names).
TRAITS = (
    "cfo_to_assets", "roa", "earnings_yield", "book_to_market", "gross_profitability",
    "share_issuance", "mom_12_1", "volatility", "fscore",
)  # fmt: skip
#: Condition -> (environment column, bucket edges, bucket labels). Thirds are relative to the
#: series' own history to that date (FRED series since 2000; stock valuation, the
#: universe's earnings yield, since 2017).
CONDITIONS: dict[str, tuple[str, tuple[float, ...], tuple[str, ...]]] = {
    "Credit spreads": ("baa_spread_pct", (1 / 3, 2 / 3), ("tight", "middle", "wide")),
    "VIX": ("vix_pct", (1 / 3, 2 / 3), ("calm", "middle", "stressed")),
    "Yield curve (10y minus 2y)": ("curve_10y_2y", (0.0, 1.0), ("inverted", "flat", "steep")),
    "10-year yield": ("yield_10y_pct", (1 / 3, 2 / 3), ("low", "middle", "high")),
    "Stock valuation": (
        "market_earnings_yield_pct", (1 / 3, 2 / 3), ("expensive", "middle", "cheap"),
    ),
    "Equity risk premium": (
        "equity_risk_premium", (0.0, 0.02), ("below bonds", "0-2%", "above 2%"),
    ),
}  # fmt: skip
#: Months ahead for the caution dial.
DIAL_MONTHS = (3, 12)
TRADING_MONTH = 21


def _bucket(values: pl.Expr, edges: tuple[float, ...], labels: tuple[str, ...]) -> pl.Expr:
    """Label each value by the bucket its edges put it in (null stays null)."""
    expr = pl.when(values < edges[0]).then(pl.lit(labels[0]))
    for edge, label in zip(edges[1:], labels[1:-1], strict=True):
        expr = expr.when(values < edge).then(pl.lit(label))
    return expr.when(values.is_not_null()).then(pl.lit(labels[-1]))


def _with_buckets(frame: pl.DataFrame, env: pl.DataFrame) -> pl.DataFrame:
    """Join each date's condition buckets (one column per condition)."""
    buckets = env.select(
        "date", *[_bucket(pl.col(col), e, lab).alias(name) for name, (col, e, lab) in
                  CONDITIONS.items() if col in env.columns]
    )  # fmt: skip
    return frame.join(buckets, on="date", how="inner")


def conditional_ic(
    scores: pl.DataFrame, env: pl.DataFrame, pool: str = "top500", horizon: int = 21
) -> pl.DataFrame:
    """Mean monthly IC of each trait in each condition bucket.

    Args:
        scores: Scoreboard rows (signal, pool, horizon, date, ic, ...).
        env: Environment from ``research.context.environment``.
        pool: Scoreboard pool.
        horizon: Scoreboard horizon.

    Returns:
        trait, condition, bucket, months, mean_ic, t (mean over standard error).
    """
    rows = scores.filter(
        (pl.col("pool") == pool) & (pl.col("horizon") == horizon)
        & pl.col("signal").is_in(list(TRAITS))
    ).select("signal", "date", "ic")  # fmt: skip
    rows = _with_buckets(rows, env)
    out = []
    for condition, (_, _, labels) in CONDITIONS.items():
        if condition not in rows.columns:
            continue
        stats = (
            rows.drop_nulls(condition)
            .group_by("signal", condition)
            .agg(
                pl.len().alias("months"),
                pl.col("ic").mean().alias("mean_ic"),
                (pl.col("ic").mean() / pl.col("ic").std() * pl.len().sqrt()).alias("t"),
            )
            .rename({"signal": "trait", condition: "bucket"})
            .with_columns(pl.lit(condition).alias("condition"))
        )
        order = pl.col("bucket").replace_strict({b: k for k, b in enumerate(labels)})
        out.append(stats.sort("trait", order))
    return pl.concat(out).select("trait", "condition", "bucket", "months", "mean_ic", "t")


def _forward(market: np.ndarray, start: int, months: int) -> tuple[float, float, float]:
    """Return, annualized volatility and worst drawdown of daily ``market`` returns ahead."""
    window = market[start + 1 : start + 1 + months * TRADING_MONTH]
    growth = np.cumprod(1 + window)
    drawdown = float((growth / np.maximum.accumulate(np.r_[1.0, growth])[1:] - 1).min())
    return float(growth[-1] - 1), float(window.std() * np.sqrt(252)), drawdown


def caution_dial(
    env: pl.DataFrame, dates: list, market: np.ndarray, date_index: dict
) -> pl.DataFrame:
    """Forward market return, volatility and drawdown by condition bucket.

    Args:
        env: Environment rows (month ends).
        dates: Session dates of ``market``.
        market: Daily market (SPY) returns aligned with ``dates``.
        date_index: Session date -> position in ``dates``.

    Returns:
        condition, bucket, months_ahead, samples, mean_return, share_positive,
        mean_volatility, mean_drawdown.
    """
    rows = []
    for day in env["date"]:
        i = date_index.get(day)
        for months in DIAL_MONTHS:
            if i is None or i + months * TRADING_MONTH >= len(dates):
                continue
            ret, vol, dd = _forward(market, i, months)
            rows.append({"date": day, "months_ahead": months, "ret": ret, "vol": vol, "dd": dd})
    if not rows:
        return pl.DataFrame()
    frame = _with_buckets(pl.DataFrame(rows), env)
    out = []
    for condition, (_, _, labels) in CONDITIONS.items():
        if condition not in frame.columns:
            continue
        order = pl.col("bucket").replace_strict({b: k for k, b in enumerate(labels)})
        out.append(
            frame.drop_nulls(condition)
            .group_by(condition, "months_ahead")
            .agg(
                pl.len().alias("samples"),
                pl.col("ret").mean().alias("mean_return"),
                (pl.col("ret") > 0).mean().alias("share_positive"),
                pl.col("vol").mean().alias("mean_volatility"),
                pl.col("dd").mean().alias("mean_drawdown"),
            )
            .rename({condition: "bucket"})
            .with_columns(pl.lit(condition).alias("condition"))
            .sort("months_ahead", order)
        )
    return pl.concat(out).select(
        "condition", "bucket", "months_ahead", "samples", "mean_return", "share_positive",
        "mean_volatility", "mean_drawdown",
    )  # fmt: skip


def current_conditions(env: pl.DataFrame) -> list[dict]:
    """Each condition's bucket and value on the latest date."""
    latest = _with_buckets(env.tail(1).select("date"), env.tail(1)).row(0, named=True)
    last = env.tail(1).row(0, named=True)
    return [
        {"condition": name, "bucket": latest.get(name), "value": last.get(col)}
        for name, (col, _, _) in CONDITIONS.items()
        if col in env.columns
    ]
