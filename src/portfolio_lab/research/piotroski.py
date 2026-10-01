"""Point-in-time Piotroski F-scores from annual 10-K facts.

For every 10-K filing, the score is computed from the values **known on its filing date**:
for each (concept, period end), the latest value filed on or before that date. A later
restatement therefore changes later scores but never earlier ones, which is what an
investor could actually have seen. Fiscal years are paired by period-end dates (about a
year apart), not by row order.

The nine binary signals (Piotroski, 2000), for fiscal year t versus t-1:

1. ROA > 0 (net income / assets at the start of the year)
2. Cash flow from operations > 0
3. ROA improved
4. Cash flow from operations / starting assets > ROA (earnings backed by cash)
5. Long-term debt / average assets fell (or stayed at zero)
6. Current ratio improved
7. No increase in weighted shares outstanding
8. Gross margin improved
9. Asset turnover (revenue / starting assets) improved

A signal is missing when an input is; banks (no current assets or gross profit) and
IFRS filers (no US-GAAP data) drop out naturally. Scores keep the count of available
signals so users can require, say, at least 8 of 9.
"""

from datetime import date, timedelta

import polars as pl

from portfolio_lab.research.dataset import rank_features

#: Each concept's tags in priority order; per filing and period the first one reported wins.
CONCEPT_TAGS: dict[str, tuple[str, ...]] = {
    "net_income": (
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ),
    "cfo": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "gross_profit": ("GrossProfit",),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "shares": ("WeightedAverageNumberOfSharesOutstandingBasic",),
    "assets": ("Assets",),
    "assets_cur": ("AssetsCurrent",),
    "liab_cur": ("LiabilitiesCurrent",),
    "liabilities": ("Liabilities",),
    "lt_debt": ("LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"),
}

#: Period ends within this many days of "one year earlier" count as the prior fiscal year.
YEAR_TOLERANCE = timedelta(days=35)
SIGNALS = [f"f{i}" for i in range(1, 10)]


def concept_facts(facts: pl.DataFrame) -> pl.DataFrame:
    """Map raw tags to concepts, keeping the highest-priority tag per filing and period.

    Args:
        facts: Rows with cik, tag, end, value, filed, accn (see ``edgar.FACT_SCHEMA``).

    Returns:
        cik, concept, end, filed, accn, value.
    """
    mapping = pl.DataFrame(
        [
            (tag, concept, rank)
            for concept, tags in CONCEPT_TAGS.items()
            for rank, tag in enumerate(tags)
        ],
        schema=["tag", "concept", "rank"],
        orient="row",
    )
    return (
        facts.join(mapping, on="tag")
        .sort("rank")
        .unique(subset=["cik", "concept", "end", "accn"], keep="first")
        .select("cik", "concept", "end", "filed", "accn", "value")
    )


def _prior_end(filings: pl.DataFrame, ends: pl.DataFrame, source: str, target: str) -> pl.DataFrame:
    """Add ``target``: the company's period end nearest to one year before ``source``."""
    lookup = filings.with_columns((pl.col(source) - pl.duration(days=365)).alias("_want")).sort(
        "_want"
    )
    matched = lookup.join_asof(
        ends.rename({"end": target}).sort(target),
        left_on="_want",
        right_on=target,
        by="cik",
        strategy="nearest",
        tolerance=YEAR_TOLERANCE,
        coalesce=False,
        check_sortedness=False,  # sorted above; polars can't verify it with `by` groups
    )
    return matched.drop("_want")


def _known_value(filings: pl.DataFrame, cf: pl.DataFrame, concept: str, end_col: str) -> pl.Series:
    """Value of ``concept`` for period ``end_col`` as known on each filing's date."""
    right = (
        cf.filter(pl.col("concept") == concept)
        .select("cik", pl.col("end").alias(end_col), "filed", "accn", "value")
        .sort("filed", "accn")
    )
    left = filings.select("_row", "cik", "filed", end_col).sort("filed")
    joined = left.join_asof(
        right, on="filed", by=["cik", end_col], strategy="backward", check_sortedness=False
    )
    return joined.sort("_row")["value"]


def build_fscores(facts: pl.DataFrame) -> pl.DataFrame:
    """Compute a point-in-time F-score for every 10-K filing.

    Args:
        facts: Annual facts (see ``edgar.FACT_SCHEMA``).

    Returns:
        One row per filing: cik, accn, filed, fiscal_end, fscore, n_signals and the nine
        signals ``f1``..``f9`` (1, 0 or null when an input is missing).
    """
    cf = concept_facts(facts)
    filings = (
        cf.group_by("cik", "accn")
        .agg(pl.col("filed").max(), pl.col("end").max().alias("end_t"))
        .sort("cik", "filed", "accn")
    )
    ends = cf.select("cik", "end").unique()
    filings = _prior_end(filings, ends, "end_t", "end_t1")
    filings = _prior_end(filings, ends, "end_t1", "end_t2")
    filings = filings.with_row_index("_row")

    needed = {
        "t": ("net_income", "cfo", "revenue", "gross_profit", "cost_of_revenue", "shares",
              "assets", "assets_cur", "liab_cur", "liabilities", "lt_debt"),
        "t1": ("net_income", "revenue", "gross_profit", "cost_of_revenue", "shares",
               "assets", "assets_cur", "liab_cur", "liabilities", "lt_debt"),
        "t2": ("net_income", "assets"),
    }  # fmt: skip
    values = {
        f"{concept}_{period}": _known_value(filings, cf, concept, f"end_{period}")
        for period, concepts in needed.items()
        for concept in concepts
    }
    frame = filings.with_columns(**values)
    return (
        frame.with_columns(_signals())
        .with_columns(
            pl.sum_horizontal(SIGNALS).alias("fscore"),
            pl.sum_horizontal(pl.col(s).is_not_null() for s in SIGNALS).alias("n_signals"),
        )
        .select(
            "cik",
            "accn",
            "filed",
            pl.col("end_t").alias("fiscal_end"),
            "fscore",
            "n_signals",
            *SIGNALS,
        )
    )


def _signals() -> list[pl.Expr]:
    """The nine F-score signals as 0/1 expressions (null when an input is missing)."""
    c = pl.col

    def gross_profit(p: str) -> pl.Expr:
        return pl.coalesce(c(f"gross_profit_{p}"), c(f"revenue_{p}") - c(f"cost_of_revenue_{p}"))

    def lt_debt(p: str) -> pl.Expr:
        # A company reporting total liabilities but no long-term debt tag has none.
        return (
            pl.when(c(f"lt_debt_{p}").is_null() & c(f"liabilities_{p}").is_not_null())
            .then(0.0)
            .otherwise(c(f"lt_debt_{p}"))
        )

    roa_t = c("net_income_t") / c("assets_t1")
    roa_t1 = c("net_income_t1") / c("assets_t2")
    lever_t = lt_debt("t") / ((c("assets_t") + c("assets_t1")) / 2)
    lever_t1 = lt_debt("t1") / ((c("assets_t1") + c("assets_t2")) / 2)
    current_t = c("assets_cur_t") / c("liab_cur_t")
    current_t1 = c("assets_cur_t1") / c("liab_cur_t1")
    margin_t = gross_profit("t") / c("revenue_t")
    margin_t1 = gross_profit("t1") / c("revenue_t1")
    turnover_t = c("revenue_t") / c("assets_t1")
    turnover_t1 = c("revenue_t1") / c("assets_t2")

    tests = [
        roa_t > 0,
        c("cfo_t") > 0,
        roa_t > roa_t1,
        c("cfo_t") / c("assets_t1") > roa_t,
        (lever_t < lever_t1) | ((lever_t == 0) & (lever_t1 == 0)),
        current_t > current_t1,
        c("shares_t") <= c("shares_t1"),
        margin_t > margin_t1,
        turnover_t > turnover_t1,
    ]
    return [test.cast(pl.Int8).alias(name) for name, test in zip(SIGNALS, tests, strict=True)]


def fscores_by_symbol(fscores: pl.DataFrame, tickers: pl.DataFrame) -> pl.DataFrame:
    """Attach ticker symbols (one CIK can have several, e.g. BRK.A and BRK.B).

    Returns:
        symbol, filed, fiscal_end, fscore, n_signals, sorted by symbol and filing date.
    """
    return (
        fscores.join(tickers, on="cik")
        .select("symbol", "filed", "fiscal_end", "fscore", "n_signals")
        .sort("symbol", "filed")
    )


def fscores_from_states(states: pl.DataFrame) -> pl.DataFrame:
    """F-scores from annual filing states (``research.fundamentals`` layout, 10-K rows).

    For data sources that deliver fiscal-year values with the prior year already paired
    (the Sharadar history). ROA and asset turnover use year-end assets for both years,
    since assets two years back are not in a state; otherwise as in the module docs.

    Returns:
        cik, filed, fiscal_end, fscore, n_signals.
    """
    c = pl.col

    def ratio(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return pl.when(b > 0).then(a / b)

    gross, gross_py = c("revenue") - c("cost_of_revenue"), c("revenue_py") - c("cost_of_revenue_py")
    gross = pl.coalesce(c("gross_profit"), gross)
    gross_py = pl.coalesce(c("gross_profit_py"), gross_py)
    lt, lt_py = c("lt_debt").fill_null(0.0), c("lt_debt_py").fill_null(0.0)
    signals = [
        ratio(c("net_income"), c("assets")) > 0,
        c("cfo") > 0,
        ratio(c("net_income"), c("assets")) > ratio(c("net_income_py"), c("assets_py")),
        c("cfo") > c("net_income"),
        ratio(lt, c("assets")) <= ratio(lt_py, c("assets_py")),
        ratio(c("assets_cur"), c("liab_cur")) > ratio(c("assets_cur_py"), c("liab_cur_py")),
        c("shares_weighted") <= c("shares_weighted_py"),
        ratio(gross, c("revenue")) > ratio(gross_py, c("revenue_py")),
        ratio(c("revenue"), c("assets")) > ratio(c("revenue_py"), c("assets_py")),
    ]
    scored = states.filter(c("form") == "10-K").with_columns(
        s.cast(pl.Int8).alias(name) for s, name in zip(signals, SIGNALS, strict=True)
    )
    return scored.select(
        "cik", "filed", pl.col("period_end").alias("fiscal_end"),
        pl.sum_horizontal(SIGNALS).alias("fscore"),
        pl.sum_horizontal(c(s).is_not_null() for s in SIGNALS).alias("n_signals"),
    )  # fmt: skip


# Continuous F-score: each of the nine signals as a percentile instead of a 0/1 step.

#: The nine Piotroski metrics (as monthly feature percentiles) and their good direction.
PIOTROSKI = {
    "roa": 1, "cfo_to_assets": 1, "d_roa": 1, "accruals": -1, "d_lt_debt": -1,
    "d_current_ratio": 1, "share_issuance": -1, "d_gross_margin": 1, "d_asset_turnover": 1,
}  # fmt: skip
MIN_METRICS = 6


def continuous(data: pl.DataFrame, metrics: dict[str, int] = PIOTROSKI) -> pl.DataFrame:
    """The metrics' percentiles (flipped where lower is better), averaged.

    Args:
        data: Prepared data (features as monthly percentiles, ``models.prepare``).
        metrics: Metric -> direction (default: the nine Piotroski metrics).

    Returns:
        date, symbol, score (null with fewer than :data:`MIN_METRICS` metrics).
    """
    metrics = {c: d for c, d in metrics.items() if c in data.columns}
    oriented = [pl.col(c) if d > 0 else 1 - pl.col(c) for c, d in metrics.items()]
    count = pl.sum_horizontal(pl.col(c).is_not_null() for c in metrics)
    return data.select(
        "date", "symbol",
        pl.when(count >= MIN_METRICS).then(pl.mean_horizontal(oriented)).alias("score"),
    )  # fmt: skip


def health_scores(features: pl.DataFrame) -> dict[str, float]:
    """Continuous F-score for one date's raw feature rows (symbol + the nine metrics).

    Each metric becomes a percentile among the given stocks, as ``models.prepare`` does for
    the whole panel, then :func:`continuous` averages them. Stocks with too few metrics
    get no score.
    """
    if features.is_empty():
        return {}
    ranked = rank_features(features.with_columns(pl.lit(date(2000, 1, 1)).alias("date")),
                           [c for c in PIOTROSKI if c in features.columns])  # fmt: skip
    scored = continuous(ranked).drop_nulls("score")
    return dict(zip(scored["symbol"], scored["score"], strict=True))
