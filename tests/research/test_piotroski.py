from datetime import date

import polars as pl

from portfolio_lab.data.sources.edgar import FACT_SCHEMA
from portfolio_lab.research.piotroski import build_fscores, concept_facts, fscores_by_symbol

FY0, FY1, FY2 = date(2020, 12, 31), date(2021, 12, 31), date(2022, 12, 31)
FILED_A, FILED_B, FILED_C = date(2022, 2, 15), date(2023, 2, 15), date(2023, 6, 1)

# fiscal year -> concept values (duration facts for flows, instants for balances)
YEARS = {
    FY0: {"Assets": 1000},
    FY1: {
        "NetIncomeLoss": 50, "NetCashProvidedByUsedInOperatingActivities": 70, "Revenues": 500,
        "GrossProfit": 200, "WeightedAverageNumberOfSharesOutstandingBasic": 100,
        "Assets": 1100, "AssetsCurrent": 300, "LiabilitiesCurrent": 200, "LongTermDebt": 400,
    },
    FY2: {
        "NetIncomeLoss": 80, "NetCashProvidedByUsedInOperatingActivities": 100, "Revenues": 600,
        "GrossProfit": 270, "WeightedAverageNumberOfSharesOutstandingBasic": 100,
        "Assets": 1200, "AssetsCurrent": 360, "LiabilitiesCurrent": 200, "LongTermDebt": 380,
    },
}  # fmt: skip


def _facts(cik=1, drop=()):
    """Filing A (FY1 10-K), B (FY2 10-K), C (FY2 10-K/A restating FY2 net income to 40)."""
    rows = []
    for accn, filed, form, years in [
        ("A", FILED_A, "10-K", (FY1, FY0)),
        ("B", FILED_B, "10-K", (FY2, FY1)),
    ]:
        for end in years:
            rows += [
                (cik, tag, end, float(v), filed, accn, form, None)
                for tag, v in YEARS[end].items()
                if tag not in drop
            ]
    rows.append((cik, "NetIncomeLoss", FY2, 40.0, FILED_C, "C", "10-K/A", None))
    return pl.DataFrame(rows, schema=FACT_SCHEMA, orient="row")


def _score(scores, accn):
    return scores.filter(pl.col("accn") == accn).row(0, named=True)


def test_hand_computed_score_and_restatement_is_point_in_time():
    scores = build_fscores(_facts())
    b = _score(scores, "B")
    # Every signal improves from FY1 to FY2 (see YEARS): a perfect 9.
    assert (b["fscore"], b["n_signals"], b["fiscal_end"]) == (9, 9, FY2)
    # The amendment restates FY2 net income 80 -> 40: ROA 0.036 no longer beats 0.05.
    c = _score(scores, "C")
    assert (c["fscore"], c["n_signals"], c["f3"]) == (8, 9, 0)
    # ...but the score as known on filing B's date is unchanged.
    assert _score(build_fscores(_facts()), "B")["fscore"] == 9


def test_first_filing_lacks_history():
    a = _score(build_fscores(_facts()), "A")
    assert a["fiscal_end"] == FY1
    assert a["n_signals"] < 8  # no FY0 flows, no FY-1 balance sheet


def test_bank_without_current_assets_or_gross_profit():
    b = _score(build_fscores(_facts(drop=("AssetsCurrent", "GrossProfit"))), "B")
    assert b["n_signals"] == 7
    assert b["f6"] is None and b["f8"] is None


def test_missing_long_term_debt_counts_as_zero_when_liabilities_reported():
    facts = _facts(drop=("LongTermDebt",))
    extra = pl.DataFrame(
        [(1, "Liabilities", end, 500.0, FILED_B, "B", "10-K", None) for end in (FY2, FY1)]
        + [(1, "Liabilities", FY0, 500.0, FILED_A, "A", "10-K", None)],
        schema=FACT_SCHEMA,
        orient="row",
    )
    assert _score(build_fscores(pl.concat([facts, extra])), "B")["f5"] == 1  # zero both years


def test_tag_priority_within_a_filing():
    facts = pl.concat(
        [
            _facts(),
            pl.DataFrame(
                [(1, "ProfitLoss", FY2, 999.0, FILED_B, "B", "10-K", None)],
                schema=FACT_SCHEMA,
                orient="row",
            ),
        ]
    )
    cf = concept_facts(facts).filter((pl.col("concept") == "net_income") & (pl.col("accn") == "B"))
    assert cf.filter(pl.col("end") == FY2)["value"].to_list() == [80.0]  # NetIncomeLoss wins


def test_prior_year_matched_by_period_end_not_order():
    # A 52/53-week fiscal year: prior year ends 2021-12-25, not 2021-12-31.
    shifted = _facts().with_columns(
        pl.when(pl.col("end") == FY1).then(date(2021, 12, 25)).otherwise(pl.col("end")).alias("end")
    )
    assert _score(build_fscores(shifted), "B")["n_signals"] == 9


def test_symbols_attached_per_cik():
    scores = build_fscores(_facts(cik=7))
    tickers = pl.DataFrame({"symbol": ["BRK.A", "BRK.B", "OTHER"], "cik": [7, 7, 8]})
    out = fscores_by_symbol(scores, tickers)
    assert set(out["symbol"]) == {"BRK.A", "BRK.B"}
    assert out.height == 2 * scores.height


def test_output_columns():
    cols = build_fscores(_facts()).columns
    assert cols[:6] == ["cik", "accn", "filed", "fiscal_end", "fscore", "n_signals"]
    assert cols[6:] == [f"f{i}" for i in range(1, 10)]
