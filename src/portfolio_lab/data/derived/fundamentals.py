"""The fundamentals table: each trailing-twelve-month filing with its prior year and F-score.

From the reader's ``filings`` (``period = "ttm"``): every concept as filed, the same
concepts from the filing about one year earlier (``prior_<concept>``, matched on the
period end within :data:`TOLERANCE` and filed no later than this one, so the pair was
knowable on the filing date), and the Piotroski F-score on annual filings
(``research.piotroski``). One row per (sid, filed, period_end).
"""

from datetime import timedelta
from pathlib import Path

import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.data import reader, schemas
from portfolio_lab.research.fundamentals import CONCEPTS as PRIOR
from portfolio_lab.research.piotroski import fscores_from_states

TOLERANCE = "20d"
YEAR = timedelta(days=365)


def with_prior(filings: pl.DataFrame, key: str = "sid") -> pl.DataFrame:
    """``filings`` with ``prior_<concept>`` from the filing a year earlier (module docs).

    Args:
        filings: Rows with ``key``, filed, period_end and the concepts.
        key: The company column.
    """
    concepts = [c for c in PRIOR if c in filings.columns]
    earlier = filings.select(
        key, pl.col("filed").alias("_prior_filed"),
        (pl.col("period_end") + YEAR).alias("_match"),
        *[pl.col(c).alias(f"prior_{c}") for c in concepts],
    ).sort("_match")  # fmt: skip
    out = filings.sort("period_end").join_asof(
        earlier, left_on="period_end", right_on="_match", by=key, strategy="nearest",
        tolerance=TOLERANCE, check_sortedness=False,
    )  # fmt: skip
    known = pl.col("_prior_filed") <= pl.col("filed")
    return out.with_columns(
        pl.when(known).then(pl.col(f"prior_{c}")).alias(f"prior_{c}") for c in concepts
    ).drop("_prior_filed", "_match")


def build(root: Path) -> dict:
    """Rebuild the table from the reader's filings."""
    filings = reader.read(root, "filings").filter(pl.col("period") == "ttm")
    rows = with_prior(filings.drop("period"))
    scores = fscores_from_states(rows, key="sid").select(
        "sid", "filed", pl.col("fiscal_end").alias("period_end"), "fscore", "n_signals"
    )
    rows = rows.join(scores, on=["sid", "filed", "period_end"], how="left")
    lead = ["sid", "filed", "period_end", "form", "shares_out"]
    rows = rows.select(*lead, *schemas.CONCEPTS, *[f"prior_{c}" for c in PRIOR],
                       "fscore", "n_signals").sort("sid", "filed", "period_end")  # fmt: skip
    write_parquet_atomic(rows, DataPaths(root).fundamentals)
    return {"filings": rows.height, "scored": int(rows["fscore"].is_not_null().sum())}
