"""SEC EDGAR: company fundamentals from XBRL filings, and the ticker-to-CIK map.

The bulk ``companyfacts.zip`` (about 1.4 GB, 20k companies, refreshed nightly by the SEC)
holds every XBRL fact each company has filed. We keep the facts from 10-K and 10-Q
filings for the tags in :data:`TAGS` and :data:`DEI_TAGS`, each with its ``filed`` date,
so research code can reconstruct what was known on any past date. Flows are kept for
quarters, six- and nine-month year-to-date periods and fiscal years: cash-flow statements
are reported year-to-date only, so quarters are differences of consecutive year-to-date
values. The Piotroski F-score uses only the annual facts (:func:`annual`).

The SEC requires a descriptive User-Agent with contact details on every request.
"""

import json
import logging
import zipfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from portfolio_lab.core.http import RateLimitedClient

log = logging.getLogger(__name__)

BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

#: us-gaap tags extracted, with whether each is a flow over the fiscal year ("duration") or
#: a balance at year end ("instant"). Several tags can feed one concept; see
#: ``research.piotroski.CONCEPT_TAGS`` for the priority order.
TAGS: dict[str, str] = {
    "NetIncomeLoss": "duration",
    "ProfitLoss": "duration",
    "NetIncomeLossAvailableToCommonStockholdersBasic": "duration",
    "NetCashProvidedByUsedInOperatingActivities": "duration",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": "duration",
    "Revenues": "duration",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "duration",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "duration",
    "SalesRevenueNet": "duration",
    "GrossProfit": "duration",
    "CostOfRevenue": "duration",
    "CostOfGoodsAndServicesSold": "duration",
    "CostOfGoodsSold": "duration",
    "CostOfServices": "duration",
    "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization": "duration",
    "PolicyholderBenefitsAndClaimsIncurredNet": "duration",
    "WeightedAverageNumberOfSharesOutstandingBasic": "duration",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted": "duration",
    "Assets": "instant",
    "AssetsCurrent": "instant",
    "LiabilitiesCurrent": "instant",
    "Liabilities": "instant",
    "LongTermDebtNoncurrent": "instant",
    "LongTermDebt": "instant",
    "OperatingLeaseLiabilityNoncurrent": "instant",
    "LongTermDebtAndCapitalLeaseObligations": "instant",
    # Valuation, cash and payout.
    "StockholdersEquity": "instant",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "instant",
    "CommonStockSharesOutstanding": "instant",
    "CashAndCashEquivalentsAtCarryingValue": "instant",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": "instant",
    "Cash": "instant",
    "DebtCurrent": "instant",
    "LongTermDebtCurrent": "instant",
    "LongTermDebtAndCapitalLeaseObligationsCurrent": "instant",
    "ShortTermBorrowings": "instant",
    "OperatingIncomeLoss": "duration",
    "PaymentsToAcquirePropertyPlantAndEquipment": "duration",
    "PaymentsToAcquireProductiveAssets": "duration",
    "ProceedsFromSaleOfPropertyPlantAndEquipment": "duration",
    "ProceedsFromSaleOfProductiveAssets": "duration",
    "PaymentsOfDividends": "duration",
    "PaymentsOfDividendsCommonStock": "duration",
    "PaymentsOfOrdinaryDividends": "duration",
    # Operating detail (R&D, SG&A, interest, depreciation, tax, working capital).
    "SellingGeneralAndAdministrativeExpense": "duration",
    "InterestExpense": "duration",
    "InterestExpenseDebt": "duration",
    "ResearchAndDevelopmentExpense": "duration",
    "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost": "duration",
    "DepreciationDepletionAndAmortization": "duration",
    "DepreciationAndAmortization": "duration",
    "DepreciationAmortizationAndAccretionNet": "duration",
    "IncomeTaxExpenseBenefit": "duration",
    "PropertyPlantAndEquipmentNet": "instant",
    "InventoryNet": "instant",
    "AccountsReceivableNetCurrent": "instant",
    "ReceivablesNetCurrent": "instant",
    "AccountsPayableCurrent": "instant",
    "IntangibleAssetsNetIncludingGoodwill": "instant",
}
#: Cover-page (``dei``) tags: shares outstanding as of a date close to the filing.
DEI_TAGS: dict[str, str] = {"EntityCommonStockSharesOutstanding": "instant"}
ANNUAL_FORMS = frozenset({"10-K", "10-K/A"})
FORMS = ANNUAL_FORMS | {"10-Q", "10-Q/A"}
#: Duration facts kept, by length in days: quarters, six- and nine-month year-to-date, years.
#: Wide enough for 52/53-week calendars, whose quarters run 12 to 16 weeks (Costco,
#: AutoZone): a 16-week quarter is 112 days, 24 weeks to date 168, 36 weeks 252.
QUARTER_DAYS = (77, 118)
HALF_YEAR_DAYS = (160, 200)
NINE_MONTH_DAYS = (245, 290)
YEAR_DAYS = (350, 380)

FACT_SCHEMA = {
    "cik": pl.Int64,
    "tag": pl.String,
    "end": pl.Date,
    "value": pl.Float64,
    "filed": pl.Date,
    "accn": pl.String,
    "form": pl.String,
    "start": pl.Date,
}


def _kept_span(days: int) -> bool:
    """Whether a duration of ``days`` is a quarter, six or nine months, or a fiscal year."""
    spans = (QUARTER_DAYS, HALF_YEAR_DAYS, NINE_MONTH_DAYS, YEAR_DAYS)
    return any(lo <= days <= hi for lo, hi in spans)


def extract_facts(payload: dict[str, Any], cik: int | None = None) -> list[tuple]:
    """10-K and 10-Q facts for :data:`TAGS` and :data:`DEI_TAGS` from one company's JSON.

    Args:
        payload: The parsed JSON of one ``CIK##########.json`` member.
        cik: The company's CIK, used when the payload omits it (some files do).

    Returns:
        Rows in :data:`FACT_SCHEMA` order (``start`` is null for instant facts). Duration
        facts are kept for quarters, six and nine months, and fiscal years; values are in
        USD, or in shares for share counts.
    """
    cik = int(payload.get("cik") or cik or 0)
    if not cik:
        return []
    facts = payload.get("facts", {})
    rows = []
    for namespace, tags in (("us-gaap", TAGS), ("dei", DEI_TAGS)):
        section = facts.get(namespace, {})
        for tag, kind in tags.items():
            units = section.get(tag, {}).get("units", {})
            for fact in units.get("USD", []) + units.get("shares", []):
                if fact.get("form") not in FORMS:
                    continue
                end = date.fromisoformat(fact["end"])
                start = None
                if kind == "duration":
                    if "start" not in fact:
                        continue
                    start = date.fromisoformat(fact["start"])
                    if not _kept_span((end - start).days):
                        continue
                rows.append(
                    (cik, tag, end, float(fact["val"]), date.fromisoformat(fact["filed"]),
                     fact["accn"], fact["form"], start)
                )  # fmt: skip
    return rows


def annual(facts: pl.DataFrame) -> pl.DataFrame:
    """The 10-K facts that describe fiscal years: instants and year-long durations."""
    days = (pl.col("end") - pl.col("start")).dt.total_days()
    return facts.filter(
        pl.col("form").is_in(list(ANNUAL_FORMS))
        & (pl.col("start").is_null() | days.is_between(*YEAR_DAYS))
    )


def extract_members(zip_path: Path, members: list[str]) -> list[tuple]:
    """Extract several companies from the bulk zip, opening it once (runs in workers)."""
    with zipfile.ZipFile(zip_path) as archive:
        return [
            row
            for m in members
            for row in extract_facts(json.loads(archive.read(m)), cik=int(m[3:13]))
        ]


def member_name(cik: int) -> str:
    """Zip member holding a company's facts, e.g. ``CIK0000320193.json``."""
    return f"CIK{cik:010d}.json"


def download_bulk(client: RateLimitedClient, dest: Path, etag: str | None) -> str | None:
    """Download ``companyfacts.zip`` unless it is unchanged since ``etag``.

    Args:
        client: HTTP client sending the SEC User-Agent.
        dest: Where to write the zip.
        etag: ETag of the copy already on disk, if any.

    Returns:
        The new ETag if a new file was downloaded, else ``None``.
    """
    head = client.head(BULK_URL)
    new_etag = head.headers.get("ETag")
    if etag and new_etag == etag and dest.exists():
        log.info("edgar: companyfacts.zip unchanged (%s)", etag)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    client.download(BULK_URL, tmp)
    tmp.replace(dest)
    log.info("edgar: downloaded companyfacts.zip (%.0f MB)", dest.stat().st_size / 1e6)
    return new_etag


def fetch_ticker_map(client: RateLimitedClient) -> pl.DataFrame:
    """Current ticker -> CIK map, with share classes written as ``BRK.B`` (our format)."""
    rows = client.get_json(TICKERS_URL).values()
    return pl.DataFrame(
        {
            "symbol": [r["ticker"].replace("-", ".").upper() for r in rows],
            "cik": [int(r["cik_str"]) for r in rows],
        }
    ).unique(subset="symbol", keep="first")


PROFILE_SCHEMA = {
    "cik": pl.Int64,
    "name": pl.String,
    "sic": pl.Int64,
    "sic_description": pl.String,
    "fiscal_year_end": pl.String,
}


def fetch_profiles(client: RateLimitedClient, ciks: Sequence[int]) -> pl.DataFrame:
    """Company name, SIC industry code and fiscal year end from EDGAR's submissions API.

    One request per company; companies the SEC has no record for are skipped.
    """
    rows = []
    for n, cik in enumerate(ciks, 1):
        if n % 500 == 0:
            log.info("edgar: %d/%d company profiles", n, len(ciks))
        try:
            d = client.get_json(SUBMISSIONS_URL.format(cik=cik))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            continue
        sic = d.get("sic")
        rows.append(
            (cik, d.get("name"), int(sic) if sic else None, d.get("sicDescription") or None,
             d.get("fiscalYearEnd") or None)
        )  # fmt: skip
    return pl.DataFrame(rows, schema=PROFILE_SCHEMA, orient="row")
