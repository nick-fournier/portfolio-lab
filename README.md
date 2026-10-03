# portfolio-lab

A research lab for portfolio strategies:
- **point-in-time data:** daily prices, the listed-stock universe and fundamentals;
- **one honest walk-forward backtest harness:** no look-ahead, trading costs, baselines;
- **a read-only dashboard** at `portfolio.nicholasfournier.com`.

This is research only. Strategies output target weights, so the same code can drive paper
trading later.

## Architecture

```
src/portfolio_lab/
  core/        settings, parquet storage, trading calendar, HTTP client, logging
  data/        sources/ (NASDAQ Trader, Alpaca, SEC EDGAR, FRED) and ingest/ jobs
  research/    point-in-time panel and DataView, derived factors (e.g. Piotroski)
  strategies/  Strategy protocol, portfolio constructors, baselines, meanvar/, ...
  backtest/    engine (timing, drift, costs), metrics, immutable run results
  web/         FastAPI + HTMX + Plotly dashboard (reads results only)
  jobs/        scheduler: runs ingest jobs when due, catching up missed sessions
  cli.py       `plab` command
```

**Dependency direction:** `core` ← `data` ← `research` ← `strategies` ← `backtest` ←
`web`/`jobs`/`cli`. Nothing imports upward. `web` never imports strategy code; it renders
the generic run artifacts each backtest writes.

**No look-ahead by construction:**
- Strategies only ever see a `DataView` bounded at the decision date.
- A "poison" test corrupts all future data and checks that every registered strategy still
  produces identical weights.

## Data

Datasets are Parquet files under `PORTFOLIO_DATA_DIR` (default `./data`). On orange the data
root `/home/nick/portfolio-data` holds two data directories, split by license so the licensed
one can be deleted on its own, `free/` (free sources; production runs on it) and `sharadar/`
(Sharadar's licensed history and its raw bulk zips in `sharadar/raw/`), plus `trading/`, the
paper account's records (`PORTFOLIO_TRADING_DIR`): operational history that can't be
re-downloaded, kept apart from market data. Every write is atomic. Backtest runs are immutable
directories, so the web app can read while jobs write.

| Data | Source |
|---|---|
| Daily prices | Alpaca Market Data (free plan; `feed=sip`, from 2016) |
| Universe | NASDAQ Trader symbol directory (NASDAQ, NYSE, AMEX; ETFs and derivatives excluded) |
| Fundamentals | SEC EDGAR `companyfacts` (point-in-time via each filing's date) |
| Risk-free rate | FRED 3-month T-bill (DTB3) |

**Known limitation:** free sources mostly lack delisted stocks, which introduces
survivorship bias. Every backtest records this caveat.

Put the Alpaca keys in a `.env` file at the repo root (gitignored):
```
ALPACA_API_KEY_ID=...
ALPACA_API_SECRET_KEY=...
```

## Running

```bash
uv run plab ingest all                            # universe, prices, benchmarks, rates
uv run plab backtest equal_weight --start 2017-01-03 --param top_n=500
uv run plab serve                                 # dashboard on http://127.0.0.1:8100
uv run plab schedule                              # long-running: jobs when due
```

The scheduler runs the daily ingest after 20:00 New York time on each session, catching up
any missed sessions. Weekly, it spot-checks stored prices and refreshes the baseline
backtests. Its state and every job's last result are shown on the dashboard's status page.

## Deployment

CI (`.github/workflows/ci.yml`) runs pre-commit and the tests on every push and pull
request. On `main` it also builds the arm64 image `nichfournier/portfolio-lab` on a native
arm64 runner and pushes it to Docker Hub. The repo needs the secrets `DOCKER_USERNAME` and
`DOCKER_PASSWORD`.

On orange, the image runs twice from `clubhouse-server/orange/compose.yaml`:
- `portfolio-web` runs `plab serve` with the data mounted read-only.
- `portfolio-jobs` runs `plab schedule` with the Alpaca keys.

Deploy with `docker compose pull && docker compose up -d`.

## Development

```bash
uv sync                      # create .venv with runtime + dev dependencies
uv run pre-commit install    # run the checks on every commit
uv run pytest
uv run plab --help
```

**Code standards, enforced by pre-commit and CI:**
- `ruff` lint and format with a line length of 100.
- Google-style docstrings on every module, public class and public function.
- At most 500 lines per Python file. When a module outgrows that, split it.
