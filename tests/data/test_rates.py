from datetime import date

import httpx
import polars as pl
import pytest

from portfolio_lab.core.http import RateLimitedClient
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import read_status
from portfolio_lab.data.ingest.rates import ingest_rates
from portfolio_lab.data.sources.fred import FredUnavailableError

from .test_sources import MAINTENANCE_PAGE

DTB3 = "observation_date,DTB3\n2026-10-19,4.10\n2026-10-20,4.08\n"
TREASURY = {
    "2025": 'Date,"13 WEEKS BANK DISCOUNT"\n12/31/2025,4.20\n12/30/2025,4.21\n',
    "2026": 'Date,"13 WEEKS BANK DISCOUNT"\n01/02/2026,4.18\n10/21/2026,4.07\n',
}


def _client(fred: str | None, treasury: bool = True) -> RateLimitedClient:
    """FRED answering ``fred`` (``None``: a 500), Treasury answering per year or a 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "fred.stlouisfed.org":
            return httpx.Response(200, text=fred) if fred else httpx.Response(500)
        year = request.url.path.split("/")[-2]
        return httpx.Response(200, text=TREASURY[year]) if treasury else httpx.Response(404)

    return RateLimitedClient(transport=httpx.MockTransport(handler), sleep=lambda s: None)


def _store(settings, days):
    path = DataPaths(settings.data_dir).rates
    path.parent.mkdir(parents=True)
    pl.DataFrame({"date": days, "rate": [0.042] * len(days)}).write_parquet(path)
    return path


def test_ingest_rates_from_fred(settings):
    summary = ingest_rates(settings, _client(DTB3))
    assert summary["source"] == "fred" and summary["max_date"] == date(2026, 10, 20)
    assert read_status(settings.data_dir, "rates")["source"] == "fred"


@pytest.mark.parametrize("fred", [MAINTENANCE_PAGE, None])
def test_ingest_rates_extends_from_treasury_when_fred_is_down(settings, fred):
    path = _store(settings, [date(2025, 12, 30)])
    summary = ingest_rates(settings, _client(fred), today=date(2026, 10, 22))
    rates = pl.read_parquet(path)
    # Only the days after the stored end, across the new year.
    assert rates["date"].to_list() == [date(2025, 12, 30), date(2025, 12, 31),
                                       date(2026, 1, 2), date(2026, 10, 21)]  # fmt: skip
    assert rates["rate"][-1] == pytest.approx(0.0407)
    assert summary["source"] == "treasury" and summary["max_date"] == date(2026, 10, 21)


def test_ingest_rates_keeps_the_stored_file_when_both_fail(settings):
    path = _store(settings, [date(2026, 10, 20)])
    before = path.read_bytes()
    with pytest.raises(httpx.HTTPStatusError):
        ingest_rates(settings, _client(MAINTENANCE_PAGE, treasury=False), date(2026, 10, 22))
    assert path.read_bytes() == before


def test_ingest_rates_needs_fred_without_a_stored_history(settings):
    with pytest.raises(FredUnavailableError):
        ingest_rates(settings, _client(MAINTENANCE_PAGE))
