from datetime import date

import httpx
import polars as pl

from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.data.sources.edgar import (
    BULK_URL,
    FACT_SCHEMA,
    download_bulk,
    extract_annual_facts,
    fetch_ticker_map,
    member_name,
)


def _fact(end, val, form="10-K", start=None, filed="2023-02-15"):
    fact = {"end": end, "val": val, "accn": "0001-23-000001", "form": form, "filed": filed}
    return fact | ({"start": start} if start else {})


PAYLOAD = {
    "cik": 320193,
    "facts": {
        "us-gaap": {
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        _fact("2022-12-31", 80, start="2022-01-01"),  # fiscal year: kept
                        _fact("2022-12-31", 20, start="2022-10-01"),  # a quarter: dropped
                        _fact("2022-12-31", 80, form="10-Q", start="2022-01-01"),  # 10-Q: dropped
                    ]
                }
            },
            "Assets": {"units": {"USD": [_fact("2022-12-31", 1200)]}},  # instant: kept
            "WeightedAverageNumberOfSharesOutstandingBasic": {
                "units": {"shares": [_fact("2022-12-31", 100, start="2022-01-01")]}
            },
            "SomethingElse": {"units": {"USD": [_fact("2022-12-31", 1)]}},  # not a used tag
        }
    },
}


def test_extract_keeps_annual_10k_facts_only():
    rows = extract_annual_facts(PAYLOAD)
    df = pl.DataFrame(rows, schema=FACT_SCHEMA, orient="row").sort("tag")
    assert df.select("tag", "value").rows() == [
        ("Assets", 1200.0),
        ("NetIncomeLoss", 80.0),
        ("WeightedAverageNumberOfSharesOutstandingBasic", 100.0),
    ]
    assert df["filed"].to_list() == [date(2023, 2, 15)] * 3
    assert set(df["cik"]) == {320193}


def test_extract_uses_fallback_cik_and_member_names():
    rows = extract_annual_facts({**PAYLOAD, "cik": None}, cik=42)
    assert {r[0] for r in rows} == {42}
    assert extract_annual_facts({"facts": {}}) == []
    assert member_name(320193) == "CIK0000320193.json"


def test_ticker_map_uses_dot_share_classes():
    payload = {
        "0": {"cik_str": 1067983, "ticker": "BRK-B"},
        "1": {"cik_str": 320193, "ticker": "AAPL"},
    }
    client = RateLimitedClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    )
    tickers = fetch_ticker_map(client)
    assert dict(tickers.iter_rows()) == {"BRK.B": 1067983, "AAPL": 320193}


def test_bulk_download_skipped_when_etag_unchanged(tmp_path):
    requests = []

    def handler(request):
        requests.append(request.method)
        return httpx.Response(200, headers={"ETag": '"v2"'}, content=b"zipbytes")

    client = RateLimitedClient(transport=httpx.MockTransport(handler))
    dest = tmp_path / "companyfacts.zip"
    assert download_bulk(client, dest, etag=None) == '"v2"'
    assert dest.read_bytes() == b"zipbytes"
    assert download_bulk(client, dest, etag='"v2"') is None  # unchanged: HEAD only
    assert requests == ["HEAD", "GET", "HEAD"]
    assert str(BULK_URL).startswith("https://www.sec.gov/")
