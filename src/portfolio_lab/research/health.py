"""Trash risk: a learned score for which companies to eliminate before optimizing.

Fundamentals don't say where money will flow next, but they do say which companies are
unhealthy. This model does the Piotroski F-score's job with learned rather than fixed
thresholds: it estimates the probability that a stock **blows up** over the next six
months, meaning it lands in the worst tenth of all eligible stocks, loses more than 40%, or
is delisted in distress (which the forward return already charges as a 30% loss).

Inputs are the business features (``features.FUNDAMENTAL``: valuation, quality, their
changes, the F-score and ownership), optionally with price volatility. Walk-forward like
``research.forecasts``: trained each January on earlier months whose outcomes were known.

:func:`compare` judges the score against F-score >= 7 **at the same strictness**: each
month it rejects as many stocks as the F-score filter does, and reports how the rejected
and kept stocks did and what share of the actual blow-ups each method caught.
"""

import numpy as np
import polars as pl

from portfolio_lab.research.dataset import month_weights
from portfolio_lab.research.features import FUNDAMENTAL
from portfolio_lab.research.forecasts import _fit, _test_periods

HORIZON = 126
#: A loss this large counts as a blow-up, as does landing in the worst tenth that month.
CRASH = -0.40
WORST_SHARE = 0.10
FSCORE_MIN = 7
VOLATILITY = ("volatility", "idio_vol", "beta")


def blowups(data: pl.DataFrame, horizon: int = HORIZON) -> pl.DataFrame:
    """Add ``blowup`` (1, 0, or null while the outcome is unknown)."""
    fwd = pl.col(f"fwd_{horizon}")
    worst = fwd.rank("ordinal").over("date") <= WORST_SHARE * fwd.count().over("date")
    return data.with_columns(((fwd < CRASH) | worst).cast(pl.Int8).alias("blowup"))


def walk_forward(
    data: pl.DataFrame, horizon: int = HORIZON, volatility: bool = False
) -> pl.DataFrame:
    """Out-of-sample blow-up probability per stock-month.

    Args:
        data: Prepared modeling data (``models.prepare``) with labels.
        horizon: Outcome window in sessions.
        volatility: Also use price volatility and beta as inputs.

    Returns:
        date, symbol, risk.
    """
    data = blowups(data, horizon)
    labeled = data.filter(pl.col("blowup").is_not_null())
    columns = [c for c in (*FUNDAMENTAL, *(VOLATILITY if volatility else ())) if c in data.columns]
    out = []
    for test, cutoff in _test_periods(data, labeled, "yearly"):
        if test.is_empty():
            continue
        train = labeled.filter(
            (pl.col("date") < cutoff) & (pl.col(f"label_end_{horizon}") < test["session"].min())
        )
        if train["date"].n_unique() < 36:
            continue
        x = train.select(columns).to_numpy().astype(float)
        seen = np.isfinite(x).any(axis=0)
        w = month_weights(train["date"])
        model = _fit(True, x[:, seen], train["blowup"].to_numpy(), w / w.mean())
        risk = model.predict_proba(test.select(columns).to_numpy().astype(float)[:, seen])[:, 1]
        out.append(test.select("date", "symbol").with_columns(pl.Series("risk", risk)))
    return pl.concat(out) if out else pl.DataFrame()


def compare(
    raw: pl.DataFrame, scores: dict[str, pl.DataFrame], pool: int | None = None
) -> pl.DataFrame:
    """Each filter's rejected vs kept stocks, at the F-score filter's strictness.

    Args:
        raw: The modeling dataset before ranking (for F-scores, liquidity and outcomes).
        scores: Name -> (date, symbol, risk).
        pool: Judge only among the ``pool`` most liquid stocks (``None``: all eligible).

    Returns:
        One row per filter: months, rejected share, 1- and 6-month excess return of kept
        and rejected stocks (vs the pool average, per year), blow-ups caught.
    """
    data = blowups(raw).filter(pl.col("fwd_21").is_not_null() & pl.col("fscore").is_not_null())
    if pool:
        liquid = pl.col("log_adv").rank("ordinal", descending=True).over("date") <= pool
        data = data.filter(liquid)
    for name, frame in scores.items():
        data = data.join(frame.select("date", "symbol", pl.col("risk").alias(name)),
                         on=["date", "symbol"], how="inner")  # fmt: skip
    data = data.with_columns(
        (pl.col("fscore") < FSCORE_MIN).alias("reject_fscore"),
        *[(pl.col(f"fwd_{h}") - pl.col(f"fwd_{h}").mean().over("date")).alias(f"ex_{h}")
          for h in (21, HORIZON)],
    )  # fmt: skip
    n_reject = pl.col("reject_fscore").sum().over("date")
    data = data.with_columns(
        (pl.col(name).rank("ordinal", descending=True).over("date") <= n_reject).alias(
            f"reject_{name}"
        )
        for name in scores
    )
    rows = []
    for name in ("fscore", *scores):
        flag = pl.col(f"reject_{name}")
        monthly = data.group_by("date").agg(
            *[pl.col(f"ex_{h}").filter(flag).mean().alias(f"rej_{h}") for h in (21, HORIZON)],
            *[pl.col(f"ex_{h}").filter(~flag).mean().alias(f"keep_{h}") for h in (21, HORIZON)],
            (pl.col("blowup").filter(flag).sum() / pl.col("blowup").sum()).alias("caught"),
            flag.mean().alias("rejected"),
        )
        diff = (monthly["keep_21"] - monthly["rej_21"]).drop_nulls().to_numpy()
        rows.append({
            "filter": name, "months": monthly.height, "rejected": monthly["rejected"].mean(),
            "kept_1m": monthly["keep_21"].mean() * 12, "rejected_1m": monthly["rej_21"].mean() * 12,
            "t_kept_minus_rejected": diff.mean() / diff.std() * np.sqrt(len(diff)),
            "kept_6m": monthly[f"keep_{HORIZON}"].mean() * 2,
            "rejected_6m": monthly[f"rej_{HORIZON}"].mean() * 2,
            "blowups_caught": monthly["caught"].fill_nan(None).mean(),
        })  # fmt: skip
    return pl.DataFrame(rows)
