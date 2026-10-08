import httpx
import polars as pl

from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.data.conform.edgar import sharadar_definitions
from portfolio_lab.data.sources.edgar import (
    BULK_URL,
    FACT_SCHEMA,
    annual,
    download_bulk,
    extract_facts,
    fetch_profiles,
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
                        _fact("2022-12-31", 80, start="2022-01-01"),  # fiscal year
                        _fact("2022-12-31", 20, start="2022-10-01"),  # fourth quarter
                        _fact("2022-09-30", 60, form="10-Q", start="2022-01-01"),  # nine months
                        _fact("2022-06-30", 40, form="10-Q", start="2022-01-01"),  # six months
                        _fact("2022-04-30", 30, form="10-Q", start="2022-01-01"),  # four: dropped
                        _fact("2022-12-31", 80, form="8-K", start="2022-01-01"),  # 8-K: dropped
                    ]
                }
            },
            "Assets": {"units": {"USD": [_fact("2022-12-31", 1200)]}},  # instant: kept
            "WeightedAverageNumberOfSharesOutstandingBasic": {
                "units": {"shares": [_fact("2022-12-31", 100, start="2022-01-01")]}
            },
            "SomethingElse": {"units": {"USD": [_fact("2022-12-31", 1)]}},  # not a used tag
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {"shares": [_fact("2023-02-01", 105, form="10-K")]}
            }
        },
    },
}


def test_extract_keeps_quarters_nine_months_years_and_cover_shares():
    rows = extract_facts(PAYLOAD)
    df = pl.DataFrame(rows, schema=FACT_SCHEMA, orient="row").sort("tag", "value")
    assert df.select("tag", "value").rows() == [
        ("Assets", 1200.0),
        ("EntityCommonStockSharesOutstanding", 105.0),
        ("NetIncomeLoss", 20.0),
        ("NetIncomeLoss", 40.0),
        ("NetIncomeLoss", 60.0),
        ("NetIncomeLoss", 80.0),
        ("WeightedAverageNumberOfSharesOutstandingBasic", 100.0),
    ]
    assert df.filter(pl.col("tag") == "Assets")["start"].to_list() == [None]
    assert set(df["cik"]) == {320193}


def test_annual_keeps_10k_years_and_instants():
    df = pl.DataFrame(extract_facts(PAYLOAD), schema=FACT_SCHEMA, orient="row")
    kept = annual(df).sort("tag", "value")
    assert kept.select("tag", "value").rows() == [
        ("Assets", 1200.0),
        ("EntityCommonStockSharesOutstanding", 105.0),
        ("NetIncomeLoss", 80.0),
        ("WeightedAverageNumberOfSharesOutstandingBasic", 100.0),
    ]


def test_extract_uses_fallback_cik_and_member_names():
    rows = extract_facts({**PAYLOAD, "cik": None}, cik=42)
    assert {r[0] for r in rows} == {42}
    assert extract_facts({"facts": {}}) == []
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


def test_fetch_profiles_reads_industry_and_skips_unknown():
    def handler(request):
        if "CIK0000000002" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(
            200,
            json={"name": "SVB", "sic": "6022", "sicDescription": "Banks", "fiscalYearEnd": "1231"},
        )

    client = RateLimitedClient(transport=httpx.MockTransport(handler))
    profiles = fetch_profiles(client, [1, 2])
    assert profiles.rows() == [(1, "SVB", 6022, "Banks", "1231")]


def test_debt_and_capex_as_sharadar_defines_them():
    """Debt includes operating leases; capex is net of asset sales."""
    states = pl.DataFrame({
        "lt_debt": [100.0, None, 50.0], "op_lease_nc": [20.0, 30.0, None],
        "capex": [40.0, 10.0, None], "capex_proceeds": [5.0, None, 3.0],
    })  # fmt: skip
    out = sharadar_definitions(states)
    assert out["lt_debt"].to_list() == [120.0, 30.0, 50.0]
    assert out["capex"].to_list() == [35.0, 10.0, None]
