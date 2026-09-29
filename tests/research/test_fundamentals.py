from datetime import date

import polars as pl
import pytest

from portfolio_lab.data.sources.edgar import FACT_SCHEMA
from portfolio_lab.research.fundamentals import filing_states

D = date


def _facts():
    """A calendar-year company: three 2022 10-Qs, the 2022 10-K, a 2023 Q1 10-Q, then a
    10-K/A restating 2022 net income from 100 to 90."""
    rows = []

    def add(tag, end, value, filed, accn, form, start=None):
        rows.append((1, tag, end, float(value), filed, accn, form, start))

    quarters = [
        (D(2022, 3, 31), 20, D(2022, 5, 5), "q1"),
        (D(2022, 6, 30), 25, D(2022, 8, 5), "q2"),
        (D(2022, 9, 30), 30, D(2022, 11, 5), "q3"),
    ]
    starts = {D(2022, 3, 31): D(2022, 1, 1), D(2022, 6, 30): D(2022, 4, 1),
              D(2022, 9, 30): D(2022, 7, 1)}  # fmt: skip
    for end, value, filed, accn in quarters:
        add("NetIncomeLoss", end, value, filed, accn, "10-Q", starts[end])
        add("Assets", end, 1000, filed, accn, "10-Q")
    add("NetIncomeLoss", D(2022, 9, 30), 75, D(2022, 11, 5), "q3", "10-Q", D(2022, 1, 1))  # 9M
    add("NetIncomeLoss", D(2022, 12, 31), 100, D(2023, 2, 15), "k", "10-K", D(2022, 1, 1))
    add("Assets", D(2022, 12, 31), 1100, D(2023, 2, 15), "k", "10-K")
    add("Assets", D(2021, 12, 31), 900, D(2023, 2, 15), "k", "10-K")
    add("EntityCommonStockSharesOutstanding", D(2023, 2, 1), 105, D(2023, 2, 15), "k", "10-K")
    add("NetIncomeLoss", D(2023, 3, 31), 30, D(2023, 5, 5), "q1b", "10-Q", D(2023, 1, 1))
    add("NetIncomeLoss", D(2022, 12, 31), 90, D(2023, 6, 1), "ka", "10-K/A", D(2022, 1, 1))
    return pl.DataFrame(rows, schema=FACT_SCHEMA, orient="row")


@pytest.fixture
def states():
    rows = filing_states(_facts(), workers=1)
    return {r["accn"]: r for r in rows.iter_rows(named=True)}


def test_fourth_quarter_derived_and_ttm_from_quarters(states):
    k = states["k"]
    assert k["period_end"] == D(2022, 12, 31)
    assert k["net_income"] == pytest.approx(100)  # 20 + 25 + 30 + (100 - 75)
    q1b = states["q1b"]
    assert q1b["net_income"] == pytest.approx(25 + 30 + 25 + 30)


def test_point_in_time_restatement(states):
    assert states["q1b"]["net_income"] == pytest.approx(110)  # before the amendment
    assert states["ka"]["net_income"] == pytest.approx(90)  # the amended year
    assert states["k"]["net_income"] == pytest.approx(100)  # unchanged


def test_unknown_history_is_null_and_balances_have_prior_year(states):
    assert states["q3"]["net_income"] is None  # no fourth quarter of 2021 yet
    k = states["k"]
    assert (k["assets"], k["assets_py"]) == (1100, 900)
    assert k["net_income_py"] is None
    assert k["shares_out"] == 105
    assert states["q1"]["shares_out"] is None


def test_year_fallback_when_quarters_are_missing():
    facts = pl.DataFrame(
        [(2, "Revenues", D(2022, 12, 31), 500.0, D(2023, 3, 1), "a", "10-K", D(2022, 1, 1))],
        schema=FACT_SCHEMA,
        orient="row",
    )
    assert filing_states(facts, workers=1)["revenue"].to_list() == [500.0]


def test_cash_flow_quarters_from_year_to_date_values():
    """Cash flow is reported year-to-date: 3, 6, 9 months and the year (10, 25, 45, 70)."""
    ytd = [(D(2022, 3, 31), 10), (D(2022, 6, 30), 25), (D(2022, 9, 30), 45), (D(2022, 12, 31), 70)]
    rows = [
        (3, "NetCashProvidedByUsedInOperatingActivities", end, float(v), D(2023, 3, 1), "a",
         "10-K", D(2022, 1, 1))
        for end, v in ytd
    ]  # fmt: skip
    state = filing_states(pl.DataFrame(rows, schema=FACT_SCHEMA, orient="row"), workers=1)
    assert state["cfo"].to_list() == [pytest.approx(70.0)]  # 10 + 15 + 20 + 25
