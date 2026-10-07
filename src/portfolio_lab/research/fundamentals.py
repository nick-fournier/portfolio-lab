"""Point-in-time quarterly fundamentals: each company's state as of every filing it made.

For every 10-K and 10-Q filing, :func:`filing_states` computes what was known on its filing
date about the fiscal period it reports (``period_end``) and the same period a year
earlier (columns suffixed ``_py``):

- **Flows** (net income, cash flow, revenue, ...) as trailing-twelve-month (TTM) sums of the
  last four fiscal quarters. Quarters are reported directly or follow from two year-to-date
  values with the same start (six months minus three, nine minus six, the year minus nine):
  cash-flow statements are only reported year-to-date, and the fourth quarter never is.
  When the four quarters can't be assembled, a fiscal-year value ending on the period end
  is used.
- **Balances** (assets, equity, debt, ...) at the period end.
- **Shares outstanding**: the latest cover-page count known at filing.

Only facts filed on or before each filing date are used, and for each period the latest
value filed so far wins, so a restatement changes later states but never earlier ones.
"""

import multiprocessing
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from itertools import groupby, pairwise

import polars as pl

from portfolio_lab.data.sources.edgar import (
    HALF_YEAR_DAYS,
    NINE_MONTH_DAYS,
    QUARTER_DAYS,
    YEAR_DAYS,
)

#: Each concept's tags in priority order (first reported wins, per filing and period).
FLOW_TAGS: dict[str, tuple[str, ...]] = {
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
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
                        "CostOfServices"),
    "operating_income": ("OperatingIncomeLoss",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
    "dividends": ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock",
                  "PaymentsOfOrdinaryDividends"),
    "sga": ("SellingGeneralAndAdministrativeExpense",),
    "interest_expense": ("InterestExpense", "InterestExpenseDebt"),
    "rnd": ("ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"),
    "depreciation": ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                     "DepreciationAmortizationAndAccretionNet"),
    "tax": ("IncomeTaxExpenseBenefit",),
}  # fmt: skip
#: Share counts averaged over a period: the latest quarter's value, not a sum.
AVERAGE_TAGS: dict[str, tuple[str, ...]] = {
    "shares_weighted": ("WeightedAverageNumberOfSharesOutstandingBasic",),
}
BALANCE_TAGS: dict[str, tuple[str, ...]] = {
    "assets": ("Assets",),
    "assets_cur": ("AssetsCurrent",),
    "liab_cur": ("LiabilitiesCurrent",),
    "liabilities": ("Liabilities",),
    "lt_debt": ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
                "LongTermDebt"),
    "debt_cur": ("DebtCurrent", "LongTermDebtCurrent",
                 "LongTermDebtAndCapitalLeaseObligationsCurrent", "ShortTermBorrowings"),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"),
    "ppe": ("PropertyPlantAndEquipmentNet",),
    "inventory": ("InventoryNet",),
    "receivables": ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent"),
    "payables": ("AccountsPayableCurrent",),
    "intangibles": ("IntangibleAssetsNetIncludingGoodwill",),
}  # fmt: skip
SHARES_TAGS = ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding")
CONCEPTS = [*FLOW_TAGS, *AVERAGE_TAGS, *BALANCE_TAGS]
#: Two period ends within this many days are the same period.
SAME_DAY = timedelta(days=7)
WORKERS = 4


def _kind(days: int | None) -> str | None:
    """Classify a duration by length: q, 6m, 9m or fy (None for instants and others)."""
    if days is None:
        return None
    spans = (
        ("q", QUARTER_DAYS),
        ("6m", HALF_YEAR_DAYS),
        ("9m", NINE_MONTH_DAYS),
        ("fy", YEAR_DAYS),
    )
    for name, (lo, hi) in spans:
        if lo <= days <= hi:
            return name
    return None


def _concept_rows(facts: pl.DataFrame) -> pl.DataFrame:
    """Map tags to concepts, keeping the highest-priority tag per filing and period."""
    groups = {**FLOW_TAGS, **AVERAGE_TAGS, **BALANCE_TAGS, "shares_out": SHARES_TAGS}
    mapping = pl.DataFrame(
        [(t, c, r) for c, tags in groups.items() for r, t in enumerate(tags)],
        schema=["tag", "concept", "rank"],
        orient="row",
    )
    return (
        facts.join(mapping, on="tag")
        .sort("rank")
        .unique(subset=["cik", "concept", "start", "end", "accn"], keep="first")
        .select("cik", "concept", "start", "end", "value", "filed", "accn", "form")
        .sort("filed", "accn")
    )


class _Known:
    """What was known so far about one company: the latest value per concept and period."""

    def __init__(self) -> None:
        self.durations: dict[str, dict[tuple[date, date], float]] = {}
        self.instants: dict[str, dict[date, float]] = {}
        self._quarters: dict[str, dict[date, float]] = {}

    def add(self, concept: str, start: date | None, end: date, value: float) -> None:
        if start is None:
            self.instants.setdefault(concept, {})[end] = value
        else:
            self.durations.setdefault(concept, {})[(start, end)] = value
            self._quarters.pop(concept, None)

    def quarters(self, concept: str) -> dict[date, float]:
        """Discrete quarter values by period end, derived from year-to-date values if needed.

        A quarter ending at ``e2`` is ``ytd(e2) - ytd(e1)`` for two year-to-date values with
        the same start whose ends are a quarter apart (e.g. six months minus three). A fourth
        quarter that is never reported on its own (10-Ks give only the year) is the year
        minus the three quarters inside it.
        """
        if concept in self._quarters:
            return self._quarters[concept]
        quarters: dict[date, float] = {}
        by_start: dict[date, list[tuple[date, float]]] = {}
        for (start, end), value in self.durations.get(concept, {}).items():
            kind = _kind((end - start).days)
            if kind == "q":
                quarters[end] = value
            if kind is not None:
                by_start.setdefault(start, []).append((end, value))
        for periods in by_start.values():
            periods.sort()
            for (e1, v1), (e2, v2) in pairwise(periods):
                gap = (e2 - e1).days
                if QUARTER_DAYS[0] <= gap <= QUARTER_DAYS[1] and _near(quarters, e2) is None:
                    quarters[e2] = v2 - v1
        for (start, end), value in self.durations.get(concept, {}).items():
            if _kind((end - start).days) != "fy" or _near(quarters, end) is not None:
                continue
            inside = [e for e in quarters if start + QUARTER_GAP[0] <= e < end - SAME_DAY]
            if len(inside) == 3:
                quarters[end] = value - sum(quarters[e] for e in inside)
        self._quarters[concept] = quarters
        return quarters

    def year(self, concept: str, end: date) -> float | None:
        """A fiscal-year value ending at ``end``."""
        for (start, e), value in self.durations.get(concept, {}).items():
            if abs(e - end) <= SAME_DAY and _kind((e - start).days) == "fy":
                return value
        return None

    def ttm(self, concept: str, end: date) -> float | None:
        """Sum of the four quarters ending at ``end``, else a fiscal year ending there."""
        quarters = self.quarters(concept)
        match = _near(quarters, end)
        total = 0.0
        for _ in range(4):
            if match is None:
                return self.year(concept, end)
            total += quarters[match]
            match = _previous(quarters, match)
        return total

    def latest_quarter(self, concept: str, end: date) -> float | None:
        """The quarter (or, failing that, year) value ending at ``end``."""
        quarters = self.quarters(concept)
        match = _near(quarters, end)
        return quarters[match] if match is not None else self.year(concept, end)

    def balance(self, concept: str, end: date) -> float | None:
        """The instant value at ``end``."""
        values = self.instants.get(concept, {})
        match = _near(values, end)
        return values[match] if match is not None else None

    def shares(self) -> float | None:
        """The most recent shares-outstanding count (latest date)."""
        values = self.instants.get("shares_out")
        return values[max(values)] if values else None


#: Days between consecutive quarter ends: 12 to 16 weeks, with slack.
QUARTER_GAP = (timedelta(days=QUARTER_DAYS[0] - 7), timedelta(days=QUARTER_DAYS[1] + 7))


def _previous(quarters: dict[date, float], end: date) -> date | None:
    """The quarter end before ``end``: the latest one 11 to 18 weeks earlier."""
    earlier = [e for e in quarters if QUARTER_GAP[0] <= end - e <= QUARTER_GAP[1]]
    return max(earlier) if earlier else None


def _near(values: dict[date, float], when: date) -> date | None:
    """The key of ``values`` within :data:`SAME_DAY` of ``when`` (closest), if any."""
    if when in values:
        return when
    best = None
    for key in values:
        gap = abs(key - when)
        if gap <= SAME_DAY and (best is None or gap < abs(best - when)):
            best = key
    return best


def _company_states(rows: pl.DataFrame) -> list[dict]:
    """Walk one company's filings in date order, recording the known state after each."""
    known = _Known()
    states = []
    records = rows.select("cik", "accn", "form", "filed", "concept", "start", "end", "value")
    for accn, group in groupby(records.iter_rows(), key=lambda r: r[1]):
        filing = list(group)
        for _, _, _, _, concept, start, end, value in filing:
            known.add(concept, start, end, value)
        ends = [r[6] for r in filing if r[5] is not None and r[4] != "shares_out"]
        if not ends:
            continue
        end = max(ends)
        prior = end - timedelta(days=365)
        state = {
            "cik": filing[0][0], "accn": accn, "form": filing[0][2],
            "filed": max(r[3] for r in filing), "period_end": end, "shares_out": known.shares(),
        }  # fmt: skip
        for when, suffix in ((end, ""), (prior, "_py")):
            for c in FLOW_TAGS:
                state[c + suffix] = known.ttm(c, when)
            for c in AVERAGE_TAGS:
                state[c + suffix] = known.latest_quarter(c, when)
            for c in BALANCE_TAGS:
                state[c + suffix] = known.balance(c, when)
        states.append(state)
    return states


def _companies_states(parts: list[pl.DataFrame]) -> list[dict]:
    """Pool worker: states for several companies."""
    return [state for part in parts for state in _company_states(part)]


def _chunks(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def filing_states(facts: pl.DataFrame, workers: int = WORKERS) -> pl.DataFrame:
    """Point-in-time state per filing (see module docs).

    Args:
        facts: Company facts (``edgar.FACT_SCHEMA``), 10-K and 10-Q.
        workers: Processes; 1 runs inline.

    Returns:
        One row per filing: cik, accn, form, filed, period_end, shares_out, and each concept
        in :data:`CONCEPTS` with its prior-year value (``_py``); nulls where unknown.
    """
    parts = _concept_rows(facts).partition_by("cik", maintain_order=True)
    if workers > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            chunks = pool.map(_companies_states, list(_chunks(parts, 50)))
            states = [s for chunk in chunks for s in chunk]
    else:
        states = _companies_states(parts)
    schema = {"cik": pl.Int64, "accn": pl.String, "form": pl.String, "filed": pl.Date,
              "period_end": pl.Date, "shares_out": pl.Float64}  # fmt: skip
    schema |= {c + s: pl.Float64 for s in ("", "_py") for c in CONCEPTS}
    return pl.DataFrame(states, schema=schema).sort("cik", "filed", "accn")
