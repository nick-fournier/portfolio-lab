from datetime import UTC, date, datetime

import polars as pl
import pytest
from fastapi.testclient import TestClient

from portfolio_lab.backtest.results import RunResult, save_run
from portfolio_lab.core.store import write_status
from portfolio_lab.web.app import create_app
from portfolio_lab.web.routes.status import data_freshness


def _run(strategy, dates):
    daily = pl.DataFrame(
        {
            "date": dates,
            "nav": [1.01, 1.03],
            "ret": [0.01, 0.0198],
            "turnover": [1.0, 0.0],
            "cost": [0.0005, 0.0],
            "cash": [0.0, 0.0],
            "holdings": [2, 2],
            "benchmark_ret": [0.005, 0.01],
        }
    )
    weights = pl.DataFrame({"date": [dates[0]] * 2, "symbol": ["AAA", "BBB"], "weight": [0.6, 0.4]})
    meta = {
        "strategy": strategy,
        "params": {"top_n": None},
        "schedule": "M",
        "start": dates[0],
        "end": dates[-1],
        "benchmark": "SPY",
        "costs": {"half_spread_bps": 5.0, "notional": 100000.0},
        "caveats": ["Survivorship bias: test caveat"],
    }
    metrics = {"cagr": 0.12, "sharpe": 1.1, "max_drawdown": -0.2, "alpha": 0.01}
    return RunResult(meta, metrics, daily, weights)


@pytest.fixture
def client(tmp_path):
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    ids = [
        save_run(_run("equal_weight", days), tmp_path),
        save_run(_run("buy_hold", days), tmp_path),
    ]
    write_status(
        tmp_path, "prices", {"max_date": date(2024, 1, 3), "rows_written": 10, "issues": ["x"]}
    )
    write_status(
        tmp_path,
        "scheduler",
        {
            "daily_ingest": {
                "last_success": "2024-01-03T01:00:00+00:00",
                "last_session": "2024-01-02",
                "last_error": None,
            }
        },
    )
    test_client = TestClient(create_app(tmp_path))
    test_client.run_ids = ids
    return test_client


def test_overview_compares_strategies(client):
    page = client.get("/")
    assert page.status_code == 200
    for expected in ("equal_weight", "buy_hold", "Growth of $1", "#holdings", "How it works"):
        assert expected in page.text


def test_about(client):
    page = client.get("/about")
    assert page.status_code == 200
    assert "Survivorship bias" in page.text and "Lead-lag" in page.text


def test_runs_list_filters_and_history(client, tmp_path):
    page = client.get("/runs")
    assert page.status_code == 200
    assert "equal_weight" in page.text and "buy_hold" in page.text
    # A second run of the same configuration is hidden unless history is requested.
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    save_run(_run("buy_hold", days), tmp_path)
    latest = client.get("/runs?strategy=buy_hold", headers={"HX-Request": "true"})
    everything = client.get("/runs?strategy=buy_hold&history=true", headers={"HX-Request": "true"})
    assert latest.text.count("/runs/") == 1
    assert everything.text.count("/runs/") == 2
    partial = client.get("/runs?strategy=buy_hold", headers={"HX-Request": "true"})
    assert "<html" not in partial.text  # HTMX gets just the table
    assert "buy_hold" in partial.text and "equal_weight" not in partial.text


def test_run_detail(client):
    page = client.get(f"/runs/{client.run_ids[0]}")
    assert page.status_code == 200
    for expected in (
        "Growth of $1",
        "Survivorship bias: test caveat",
        "AAA",
        "60.00%",
        "/vendor/plotly.min.js",
    ):
        assert expected in page.text


@pytest.mark.parametrize("run_id", ["nope", "%2E%2E", "a%2F..%2Fb"])
def test_run_detail_404(client, run_id):
    assert client.get(f"/runs/{run_id}").status_code == 404


def test_status_and_health(client):
    page = client.get("/status")
    assert page.status_code == 200
    assert "rows_written" in page.text and "daily_ingest" in page.text
    assert client.get("/healthz").text == "ok"
    assert client.get("/vendor/plotly.min.js").status_code == 200


def test_data_freshness():
    monday_evening = datetime(2024, 7, 8, 23, 0, tzinfo=UTC)
    assert data_freshness({"max_date": "2024-07-08"}, monday_evening) == "fresh"
    assert data_freshness({"max_date": "2024-07-05"}, monday_evening) == "stale"
    assert data_freshness(None, monday_evening) == "missing"


def test_signals_page(client, tmp_path):
    assert "has not run yet" in client.get("/signals").text
    rows = [
        (signal, pool, 21, date(2024, m, 28), 100, ic, 0.02, 0.01)
        for signal, ic in (("momentum", 0.03), ("fscore", -0.01))
        for pool in ("top100", "all")
        for m in (1, 2, 3)
    ]
    schema = ["signal", "pool", "horizon", "date", "n", "ic", "top", "bottom"]
    path = tmp_path / "results" / "scoreboard.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=schema, orient="row").write_parquet(path)
    page = client.get("/signals")
    assert page.status_code == 200
    for expected in ("What is IC?", "100 most liquid stocks", "momentum", "+0.030"):
        assert expected in page.text


def test_overview_drops_benchmark_line_duplicated_by_buy_hold(client):
    page = client.get("/").text
    assert "buy_hold (SPY)" in page
    assert "SPY (benchmark)" not in page  # buy_hold already is the SPY line


def test_run_page_explains_the_run(tmp_path):
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    result = _run("momentum", days)
    result.meta["explain"] = {
        "summary": "Equal-weights last year's winners.",
        "candidates": "The 100 most liquid stocks.",
        "signal": "Return from 12 months ago to 1 month ago.",
        "construction": "Top 20, equal weights.",
        "execution": "Rebalances monthly.",
        "drivers": "Whether winners keep winning.",
        "related": "Same candidates as meanvar.",
    }
    result.meta["example"] = {
        "date": days[0],
        "holdings": 20,
        "columns": {"Return": "pct"},
        "rows": [{"symbol": "AAA", "weight": 0.05, "Return": 1.5}],
    }
    run_id = save_run(result, tmp_path)
    page = TestClient(create_app(tmp_path)).get(f"/runs/{run_id}").text
    for expected in ("How this run works", "Portfolio construction", "Worked example",
                     "largest 1 shown", "150.0%", "Compared with related strategies"):  # fmt: skip
        assert expected in page, expected
    assert "Equal-weights last year" in TestClient(create_app(tmp_path)).get("/").text
