"""Pick test: do the model's picks beat the 100 most liquid stocks the following month?

A plain check of the model as a selector, before any optimizer. Each month end, hold the
model's best-rated 20 or 100 stocks from every eligible stock, in equal weights, and
compare the next month's return with three bars over the same sessions: the 100 most
liquid stocks in equal weights (the pool meanvar picks from, which tracks SPY), SPY, and
meanvar's actual return. The trailing 12-month return picked the same way is the reference
(meanvar's own signal, without its optimizer), and so is every stock with an F-score of 7
or more. Returns are before trading costs.
"""

import numpy as np
import polars as pl

from portfolio_lab.research.panel import Panel
from portfolio_lab.research.scoreboard import forward_returns

HOLDS = (50, 100, 200)
POOL = 100
#: Reference selection: every stock with at least this Piotroski F-score, equally weighted.
FSCORE_MIN = 7
MONTH = 21


def _meanvar_months(daily: pl.DataFrame, panel: Panel, dates: list) -> dict:
    """Meanvar's compounded return over the ``MONTH`` sessions after each date."""
    ret = dict(zip(daily["date"], daily["ret"], strict=True))
    out = {}
    for day in dates:
        i = panel.date_index[day]
        window = panel.dates[i + 1 : i + MONTH + 1]
        if len(window) == MONTH and all(d in ret for d in window):
            out[day] = float(np.prod([1 + ret[d] for d in window]) - 1)
    return out


def _summary(months: pl.DataFrame) -> pl.DataFrame:
    """Per selection: share of months beating each bar, average margin, and its t-stat."""
    diff = pl.col("picks") - pl.col("pool100")
    return (
        months.group_by("selection")
        .agg(
            pl.len().alias("months"),
            (diff > 0).mean().alias("beat_pool100"),
            diff.mean().alias("vs_pool100"),
            (diff.mean() / diff.std() * pl.len().sqrt()).alias("t_pool100"),
            (pl.col("picks") > pl.col("spy")).mean().alias("beat_spy"),
            (pl.col("picks") - pl.col("spy")).mean().alias("vs_spy"),
            (pl.col("picks") > pl.col("meanvar")).mean().alias("beat_meanvar"),
            (pl.col("picks") - pl.col("meanvar")).mean().alias("vs_meanvar"),
        )
        .sort("vs_pool100", descending=True)
    )


def run(
    scores: dict[str, pl.DataFrame],
    data: pl.DataFrame,
    panel: Panel,
    meanvar_daily: pl.DataFrame | None = None,
) -> dict[str, pl.DataFrame]:
    """Monthly pick returns against the bars, and the summary.

    Args:
        scores: Model name -> (date, symbol, trees) out-of-sample scores.
        data: The modeling dataset before ranking (for liquidity and trailing returns).
        panel: Prices, for forward returns.
        meanvar_daily: The meanvar run's daily returns (date, ret); without it the meanvar
            columns are null.

    Returns:
        ``months`` (one row per month and selection), ``summary``, and ``yearly`` (each
        selection's compounded return minus the pool's, per year).
    """
    base = data.select(
        "date", "symbol",
        pl.col("log_adv").rank("ordinal", descending=True).over("date").alias("liquid_rank"),
        ((1 + pl.col("mom_12_1")) * (1 + pl.col("ret_1m")) - 1).alias("trailing_12m"),
        pl.col("fscore") if "fscore" in data.columns else pl.lit(None, pl.Float64).alias("fscore"),
    )  # fmt: skip
    for name, frame in scores.items():
        base = base.join(frame.select("date", "symbol", pl.col("trees").alias(name)),
                         on=["date", "symbol"], how="left")  # fmt: skip
    dates = sorted(set.intersection(*[set(f["date"].unique()) for f in scores.values()]))
    dates = [d for d in dates if panel.date_index[d] + MONTH < len(panel.dates)]
    meanvar = _meanvar_months(meanvar_daily, panel, dates) if meanvar_daily is not None else {}
    spy = panel.symbol_index["SPY"]
    rows = []
    for day in dates:
        if meanvar_daily is not None and day not in meanvar:
            continue
        month = base.filter(pl.col("date") == day)
        fwd = forward_returns(panel, panel.date_index[day], MONTH)
        pool = [
            panel.symbol_index[s] for s in month.filter(pl.col("liquid_rank") <= POOL)["symbol"]
        ]
        bars = {"pool100": float(np.nanmean(fwd[pool])), "spy": float(fwd[spy]),
                "meanvar": meanvar.get(day)}  # fmt: skip
        healthy = month.filter(pl.col("fscore") >= FSCORE_MIN)["symbol"]
        if healthy.len():
            idx = [panel.symbol_index[s] for s in healthy]
            rows.append({"date": day, "selection": f"fscore >= {FSCORE_MIN} (all)",
                         "picks": float(np.nanmean(fwd[idx])), **bars})  # fmt: skip
        for score in [*scores, "trailing_12m"]:
            ranked = month.drop_nulls(score).sort(score, descending=True)["symbol"]
            for hold in HOLDS:
                idx = [panel.symbol_index[s] for s in ranked.head(hold)]
                rows.append({"date": day, "selection": f"{score} top {hold}",
                             "picks": float(np.nanmean(fwd[idx])), **bars})  # fmt: skip
    months = pl.DataFrame(rows)
    yearly = (
        months.group_by("selection", pl.col("date").dt.year().alias("year"))
        .agg(
            ((1 + pl.col("picks")).product() - (1 + pl.col("pool100")).product()).alias(
                "vs_pool100"
            )
        )
        .pivot(on="selection", index="year", values="vs_pool100")
        .sort("year")
    )
    return {"months": months, "summary": _summary(months), "yearly": yearly}
