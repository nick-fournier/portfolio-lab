"""SEC EDGAR: company fundamentals from XBRL filings, and the ticker-to-CIK map.

The bulk ``companyfacts.zip`` (about 1.4 GB, 20k companies, refreshed nightly by the SEC)
holds every XBRL fact each company has filed. We keep only **annual** facts from 10-K
filings for the tags the Piotroski F-score needs, each with its ``filed`` date, so
research code can reconstruct what was known on any past date.

The SEC requires a descriptive User-Agent with contact details on every request.
"""

import json
import logging
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from portfolio_lab.core.http import RateLimitedClient

log = logging.getLogger(__name__)

BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
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
    "WeightedAverageNumberOfSharesOutstandingBasic": "duration",
    "Assets": "instant",
    "AssetsCurrent": "instant",
    "LiabilitiesCurrent": "instant",
    "Liabilities": "instant",
    "LongTermDebtNoncurrent": "instant",
    "LongTermDebt": "instant",
    "LongTermDebtAndCapitalLeaseObligations": "instant",
}
ANNUAL_FORMS = frozenset({"10-K", "10-K/A"})
#: A duration fact counts as a fiscal year if it spans this many days.
YEAR_DAYS = (350, 380)

FACT_SCHEMA = {
    "cik": pl.Int64,
    "tag": pl.String,
    "end": pl.Date,
    "value": pl.Float64,
    "filed": pl.Date,
    "accn": pl.String,
    "form": pl.String,
}


def extract_annual_facts(payload: dict[str, Any], cik: int | None = None) -> list[tuple]:
    """Annual 10-K facts for :data:`TAGS` from one company's ``companyfacts`` JSON.

    Args:
        payload: The parsed JSON of one ``CIK##########.json`` member.
        cik: The company's CIK, used when the payload omits it (some files do).

    Returns:
        Rows in :data:`FACT_SCHEMA` order. Duration facts are kept only when they span a
        fiscal year; values are in USD, or in shares for share counts.
    """
    cik = int(payload.get("cik") or cik or 0)
    if not cik:
        return []
    gaap = payload.get("facts", {}).get("us-gaap", {})
    rows = []
    for tag, kind in TAGS.items():
        units = gaap.get(tag, {}).get("units", {})
        for fact in units.get("USD", []) + units.get("shares", []):
            if fact.get("form") not in ANNUAL_FORMS:
                continue
            end = date.fromisoformat(fact["end"])
            if kind == "duration":
                if "start" not in fact:
                    continue
                days = (end - date.fromisoformat(fact["start"])).days
                if not YEAR_DAYS[0] <= days <= YEAR_DAYS[1]:
                    continue
            rows.append(
                (cik, tag, end, float(fact["val"]), date.fromisoformat(fact["filed"]),
                 fact["accn"], fact["form"])
            )  # fmt: skip
    return rows


def extract_members(zip_path: Path, members: list[str]) -> list[tuple]:
    """Extract several companies from the bulk zip, opening it once (runs in workers)."""
    with zipfile.ZipFile(zip_path) as archive:
        return [
            row
            for m in members
            for row in extract_annual_facts(json.loads(archive.read(m)), cik=int(m[3:13]))
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
