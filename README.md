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
  data/        sources/ (HTTP clients), ingest/ (fetch jobs), the hive: ids, schemas,
               conform/ (one per source), reader, derived/ (daily, fundamentals, monthly)
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

Everything lives under one data directory (`PORTFOLIO_DATA_DIR`), the **data hive**:

```
ids/                 securities (one integer sid each), their tickers by date range
<source>/raw/        what the source delivered, kept as is (sharadar/raw/ = the licensed zips)
<source>/conformed/  the source rewritten into the shared tables: prices, listings,
                     actions, filings, series (data/schemas.py), keyed on sid
derived/             daily (liquidity, history, listing flags), fundamentals (prior year,
                     F-scores), monthly (every stock input), environment
results/             backtest runs, the scoreboard, comparisons
ingest/              the fetchers' own store (Alpaca, NASDAQ, Tiingo, FRED, EDGAR), which
                     the conformers read
```

The reader (`data/reader.py`) unions every source's copy of a table and keeps one row per
key: the most complete, then the source highest in its priority list (Sharadar first).
Nothing downstream knows a source exists; models and backtests see one continuous history
from 1998. Deleting `sharadar/` is the whole licence clean-up.

| Table | Sources |
|---|---|
| prices, listings, actions | Sharadar (1998 on), Alpaca (2016 on), NASDAQ Trader, Tiingo |
| filings | Sharadar, SEC EDGAR `companyfacts` (point-in-time via each filing's date) |
| series | FRED (3-month T-bill and the market-context series) |

Tickers are only names for a security over a date range (`data/ids.py`); a source's
tickers are resolved by date. The paper account's records (`PORTFOLIO_TRADING_DIR`) stay
apart from market data. Every write is atomic, and backtest runs are immutable directories,
so the web app can read while jobs write.

Put the Alpaca keys in a `.env` file at the repo root (gitignored):
```
ALPACA_API_KEY_ID=...
ALPACA_API_SECRET_KEY=...
```

## Running

```bash
uv run plab hive sharadar                         # once per bulk download: ids + tables
uv run plab ingest all                            # fetch: universe, prices, benchmarks, rates
uv run plab hive conform && uv run plab hive derive   # into the hive, then the derived tables
uv run plab backtest equal_weight --param top_n=500   # from the first date with a year of data
uv run plab serve                                 # dashboard on http://127.0.0.1:8100
uv run plab schedule                              # long-running: jobs when due
```

The next-month stock forecaster (research, on the Sharadar data directory; the trees can
be fitted on a GPU with `--device cuda`). Its model libraries are optional extras, kept out
of the production image: `forecaster` (XGBoost) and `nets` (PyTorch):

```bash
uv sync --extra forecaster --extra nets
export PORTFOLIO_DATA_DIR=/home/nick/portfolio-data/sharadar
uv run plab forecast inputs                       # extra stock inputs from the bulk zips
uv run plab forecast dataset                      # stock and market tables
uv run plab forecast run                          # walk forward, refit monthly; resumes
uv run plab forecast grade                        # grades on unseen months, 2009 on
```

The scheduler runs fetch, conform and derive after 20:00 New York time on each session,
catching up any missed sessions, then the paper account. Weekly, it spot-checks stored
prices, refreshes EDGAR and FRED, and reruns the baseline backtests. Its state and every job's last result are shown on the dashboard's status page.

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
