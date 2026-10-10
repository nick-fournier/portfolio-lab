import json
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

from portfolio_lab.backtest.results import RunResult, save_run
from portfolio_lab.core.store import write_status
from portfolio_lab.jobs.taxes import publish_runs
from portfolio_lab.web import series
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
    for expected in ("equal_weight", "SPY", "Growth of $1", "Returns", "How it works"):
        assert expected in page.text


def test_how_it_works(client):
    page = client.get("/how-it-works")
    assert page.status_code == 200
    for expected in ("Piotroski health", "Forecaster", "Grinold", "Sharadar", "paper account"):
        assert expected in page.text, expected
    old = client.get("/about", follow_redirects=False)
    assert old.status_code == 301 and old.headers["location"] == "/how-it-works"


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
    assert "Prices through Wed Jan 3" in page.text and "update" in page.text
    assert client.get("/healthz").text == "ok"
    assert client.get("/vendor/plotly.min.js").status_code == 200


def test_data_freshness():
    monday_evening = datetime(2024, 7, 8, 23, 0, tzinfo=UTC)  # 7 pm New York
    # Friday's prices on Monday evening: Monday's update is due at 8 pm, not yet late.
    friday = data_freshness({"max_date": "2024-07-05"}, monday_evening)
    assert friday["state"] == "current" and friday["next"].hour == 20
    assert friday["next"].date() == date(2024, 7, 8)
    tuesday_morning = datetime(2024, 7, 9, 13, 0, tzinfo=UTC)
    assert data_freshness({"max_date": "2024-07-05"}, tuesday_morning)["state"] == "late"
    assert data_freshness({"max_date": "2024-07-08"}, tuesday_morning)["state"] == "current"
    assert data_freshness(None, monday_evening) == {"state": "missing"}


def test_signals_page(client, tmp_path):
    assert "has not run yet" in client.get("/signals").text
    rows = [
        (signal, pool, horizon, date(2024, m, 28), 100, ic, 0.02, 0.01)
        for signal, ic in (("momentum", 0.03), ("accruals", -0.01))
        for pool in ("top500", "all")  # only production's pool is shown
        for horizon in (21, 63)
        for m in (1, 2, 3)
    ]
    schema = ["signal", "pool", "horizon", "date", "n", "ic", "top", "bottom"]
    path = tmp_path / "results" / "scoreboard.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=schema, orient="row").write_parquet(path)
    page = client.get("/signals")
    assert page.status_code == 200
    for expected in ("next month's return", "momentum", "+0.030", "Lower is better",
                     "Reliability", "High 20%"):  # fmt: skip
        assert expected in page.text, expected
    assert page.text.count("<tbody>") == 1  # next month only, production's pool only
    assert "Next quarter" not in page.text
    # accruals: lower is better, IC negative every month, so it was right 100% of the time.
    assert ">100%<" in page.text


def test_overview_lists_the_market_once(client):
    page = client.get("/").text
    assert page.count(">SPY<") == 1  # buy-and-hold SPY is the market row, not a strategy


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


def test_context_page(client, tmp_path):
    assert "No market context yet" in client.get("/context").text
    day = date(2024, 1, 31)
    (tmp_path / "derived").mkdir(exist_ok=True)
    pl.DataFrame(
        {"date": [day], "oil": [80.0], "oil_chg3m": [0.1], "oil_pct": [0.6], "vix": [15.0],
         "vix_pct": [0.2], "equity_risk_premium": [-0.01]}
    ).write_parquet(tmp_path / "derived" / "environment.parquet")  # fmt: skip
    pl.DataFrame(
        {"trait": ["roa"], "condition": ["VIX"], "bucket": ["calm"],
         "months": [30], "mean_ic": [0.04], "t": [2.5]}
    ).write_parquet(tmp_path / "results" / "context_conditions.parquet")  # fmt: skip
    pl.DataFrame(
        {"condition": ["VIX"], "bucket": ["calm"], "months_ahead": [3],
         "samples": [30], "mean_return": [0.02], "share_positive": [0.7],
         "mean_volatility": [0.15], "mean_drawdown": [-0.05]}
    ).write_parquet(tmp_path / "results" / "context_dial.parquet")  # fmt: skip
    page = client.get("/context").text
    for expected in ("Conditions now", "$80.00", "+11%", "calm ◂ now", "below bonds",
                     "Which traits paid off", "+0.040", "S&amp;P 500 did next"):  # fmt: skip
        assert expected in page, expected


def test_compare_page(client, tmp_path):
    assert "No comparison yet" in client.get("/compare").text
    folder = tmp_path / "results" / "make_vs_buy"
    folder.mkdir(parents=True)
    days = [date(1999, 1, 4) + timedelta(days=7 * k) for k in range(52 * 12)]
    growth = {"SPY": 1.08, "MTUM": 1.12, "ours: h": 1.18}  # annual growth rates
    rows, curves = [], []
    for key, rate in growth.items():
        category = {"SPY": "passive", "MTUM": "factor"}.get(key, "ours")
        rows.append({"key": key, "name": key.removeprefix("ours: "), "category": category,
                     "period": "full", "start": days[0]})  # fmt: skip
        curves += [{"date": d, "key": key, "growth": rate ** ((d - days[0]).days / 365.25)}
                   for d in days]  # fmt: skip
    pl.DataFrame(rows).write_parquet(folder / "summary.parquet")
    pl.DataFrame(curves).write_parquet(folder / "growth.parquet")
    page = client.get("/compare").text
    for expected in ("Returns", "Passive factor funds", "18.0%", "8.0%",
                     "since 1999-01-04", ">10Y<", ">20Y<"):  # fmt: skip
        assert expected in page, expected
    assert page.count('class="chart tall"') == 1
    assert "Not a fair fight" not in page


def test_tax_page(client, tmp_path):
    assert "Nothing published yet" in client.get("/taxes").text
    days = [date(2020, 1, 2), date(2021, 6, 1)]
    ids = {}
    for key, strategy in (("band0", "meanvar"), ("band0-defer", "meanvar"), ("spy", "buy_hold")):
        run = _run(strategy, days)
        run.daily = run.daily.with_columns(pl.Series("nav", [1.0, 2.0]))
        run.trades = pl.DataFrame(
            [(0, days[0], "AAA", 0.0, 1.0, 0.0), (0, days[0], "_cash", 1.0, 0.0, 0.0),
             (1, days[1], "AAA", 2.0, 2.0, 0.0), (1, days[1], "_cash", 0.0, 0.0, 0.0)],
            schema=["seq", "date", "key", "before", "after", "income"], orient="row",
        )  # fmt: skip
        ids[key] = save_run(run, tmp_path)
    folder = publish_runs(tmp_path, ids, tmp_path)
    assert "AAA" not in pl.read_parquet(folder / "band0.trades.parquet")["key"].to_list()
    page = client.get("/taxes").text
    for expected in ("Is the strategy worth sheltering", "Rough rates by income", "SPY, taxable",
                     "Taxable, holding gains a year", '"unrealized": [0.0, 1.0]'):  # fmt: skip
        assert expected in page, expected


def _write(folder, name, columns, **schema):
    pl.DataFrame(columns, schema_overrides=schema).write_parquet(folder / f"{name}.parquet")


def test_paper_page_empty_then_with_records(client, tmp_path):
    page = client.get("/paper")
    assert page.status_code == 200 and "No paper account records yet" in page.text
    folder = tmp_path / "trading" / "paper"
    folder.mkdir(parents=True)
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    _write(folder, "snapshots", {"date": days, "equity": [100_000.0, 101_000.0],
                                 "cash": [1_000.0] * 2, "positions": [2, 2]})  # fmt: skip
    _write(folder, "positions", {"date": [days[1]] * 2, "symbol": ["AAA", "BBB"],
                                 "qty": [10.0, 5.0],
                                 "market_value": [60_000.0, 40_000.0]})  # fmt: skip
    _write(folder, "targets", {"date": [days[0]] * 2, "symbol": ["AAA", "CCC"],
                               "weight": [0.6, 0.4]})  # fmt: skip
    _write(folder, "rebalances", {"date": [days[0]], "targets": [2], "closes": [0],
                                  "orders": [2], "equity": [100_000.0]})  # fmt: skip
    _write(folder, "orders", {
        "id": ["1"], "client_order_id": ["pl-20240102-AAA-buy"], "symbol": ["AAA"],
        "side": ["buy"], "notional": [59_400.0], "qty": [None], "status": ["filled"],
        "filled_qty": [10.0], "filled_avg_price": [5940.0],
    }, qty=pl.Float64)  # fmt: skip
    page = client.get("/paper")
    assert page.status_code == 200
    for expected in ("$101,000", "1.0%", "CCC", "Paper vs SPY", "Last rebalance: 2024-01-02"):
        assert expected in page.text
    assert "5940.00" not in page.text  # the orders table is gone; the orders live on Alpaca


def test_paper_page_reads_a_separate_trading_dir(tmp_path):
    trading = tmp_path / "ops"
    folder = trading / "paper"
    folder.mkdir(parents=True)
    _write(folder, "snapshots", {"date": [date(2024, 1, 2)], "equity": [100_000.0],
                                 "cash": [1_000.0], "positions": [0]})  # fmt: skip
    page = TestClient(create_app(tmp_path / "data", trading)).get("/paper")
    assert page.status_code == 200 and "$100,000" in page.text


def test_forecasts_page(client, tmp_path):
    assert "No forecasts yet" in client.get("/forecasts").text
    folder = tmp_path / "results" / "forecaster"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"pieces": {}, "bins": []}))
    assert "being rebuilt" in client.get("/forecasts").text
    sizes = {"small": 0.044, "mid": 0.035, "large": 0.026}
    piece = {"ic": 0.037, "ic_t": 4.5, "months_right": 127, "years_right": 16, "years": 18,
             "slope": 0.63, "r2": 0.001, "tenth_yr": 0.105, "ic_by_size": sizes,
             "liquid_ic": 0.034, "top_yr": 0.278}  # fmt: skip
    summary = {
        "start": "2009-01-30", "end": "2026-08-31", "months": 212, "yearly_noise": 0.025,
        "pieces": {k: {"label": f"{k} label", **piece}
                   for k in ("forecast", "production")},
        "yearly": [{"year": 2009 + k, "ic": 0.03, "margin": 0.02, "old": 0.02}
                   for k in range(18)],
        "bins": [{"bin": k, "forecast": k / 1000, "actual": k / 1600, "margin": 0.002,
                  "q25": -0.05, "q75": 0.05} for k in range(-10, 10)],
        "grinold_bins": [{"bin": k, "forecast": k / 1000, "actual": k / 1100, "margin": 0.002,
                          "q25": -0.05, "q75": 0.05} for k in range(-10, 10)],
        "grinold_slope": 0.125,
        "trailing": [{"date": f"20{10 + k}-01-31", "ic": 0.03, "old": 0.02}
                     for k in range(10)],
        "tenths": [{"date": f"20{10 + k}-01-31", "top": 1 + k / 10, "bottom": 1 - k / 20,
                    "p_top": 1.0, "p_bottom": 1.0} for k in range(10)],
    }  # fmt: skip
    (folder / "summary.json").write_text(json.dumps(summary))
    page = client.get("/forecasts").text
    for expected in ("27.8%", "0.034", "16 of 18", "60%", "production label",
                     "0.044 / 0.035 / 0.026", "±0.025", "slope 0.12", 'id="fit"',
                     'id="grinold"', 'id="trailing"', 'id="tenths"'):  # fmt: skip
        assert expected in page, expected
    assert "revious" not in page


def test_pages_link_assets_by_path_whatever_host_asked_first(client):
    """A cached page once kept its first visitor's host in every later visitor's links."""
    client.get("/compare", headers={"host": "localhost:8100"})
    page = client.get("/compare", headers={"host": "portfolio.example.com"}).text
    assert 'href="/static/style.css"' in page and "localhost" not in page
    cached = [value for _, value in series._CACHE.values() if isinstance(value, dict)]
    assert cached and all("request" not in value for value in cached)
