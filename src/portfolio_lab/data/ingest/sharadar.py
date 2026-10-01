"""Sharadar history: a self-contained data directory from Sharadar's bulk files (1998 on).

Converts Sharadar's full-history bulk zips (``tickers``, ``actions``, ``stocks``,
``funds``, ``fundamentals``, downloaded from api.sharadar.com) into the same datasets the
Alpaca/EDGAR ingest writes, in a **separate** data directory, so the whole research
pipeline runs unchanged on about 28 years of survivorship-free history:

- ``universe/symbols.parquet``: every US common stock (live and dead) on a major exchange.
- ``universe/delisted.parquet``: dead stocks; those not acquired or merged (bankruptcy,
  regulatory or voluntary delisting) are marked ``fell_to_otc`` and exit with the
  backtest's delisting return.
- ``prices/daily`` and ``prices/benchmarks``: raw (as traded) OHLCV plus returns from
  split- and dividend-adjusted closes, as ``data.ingest.prices`` stores them.
- ``fundamentals/states.parquet``, ``tickers.parquet``, ``companies.parquet``: one state
  per filing from Sharadar's as-reported trailing-twelve-month rows (``ART``), dated by
  the SEC filing date, with the same period a year earlier (``_py``). Sharadar's
  permaticker stands in for the SEC CIK.
- ``fundamentals/fscores.parquet``: Piotroski F-scores from the annual states.
- ``rates`` and ``macro`` are copied from the main data directory (FRED).

The license requires deleting Sharadar data within 30 days of cancelling: everything
this module writes lives under the one output directory for that reason.
"""

import logging
import shutil
import zipfile
from pathlib import Path

import polars as pl

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.research.piotroski import fscores_by_symbol, fscores_from_states

log = logging.getLogger(__name__)

EXCHANGES = ("NYSE", "NASDAQ", "NYSEMKT", "NYSEARCA", "BATS")
#: Actions marking a dead stock as bought out (exit at the last price, not a delisting loss).
ACQUIRED = (
    "acquisitionby", "mergerfrom", "acquisitioncash", "acquisitionstock",
    "acquisitionelectcash", "acquisitionelectstock", "spacmerger",
)  # fmt: skip
#: A stock is kept only if it ever closed above this price with this much dollar volume
#: (loose versions of the eligibility rules, to bound the panel's size).
EVER_PRICE, EVER_DOLLAR_VOLUME = 5.0, 1_000_000
BENCHMARKS = ("SPY", "QQQ", "IWM")
#: Sharadar fundamentals column -> our concept (sign flips for cash outflows below).
FUNDAMENTALS = {
    "netinc": "net_income", "ncfo": "cfo", "revenue": "revenue", "gp": "gross_profit",
    "cor": "cost_of_revenue", "opinc": "operating_income", "capex": "capex",
    "ncfdiv": "dividends", "shareswa": "shares_weighted", "assets": "assets",
    "assetsc": "assets_cur", "liabilitiesc": "liab_cur", "liabilities": "liabilities",
    "debtnc": "lt_debt", "debtc": "debt_cur", "equity": "equity", "cashneq": "cash",
}  # fmt: skip
OUTFLOWS = ("capex", "dividends")


def _read(raw: Path, table: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read one bulk zip's CSV (all columns as strings unless selected and cast later)."""
    with zipfile.ZipFile(raw / f"{table}.zip") as z:
        return pl.read_csv(z.open(z.namelist()[0]).read(), columns=columns, infer_schema_length=0)


def _scan(raw: Path, table: str) -> pl.LazyFrame:
    """Lazily scan a large bulk table, extracting its CSV next to the zip once."""
    csv = raw / f"{table}.csv"
    if not csv.exists():
        with zipfile.ZipFile(raw / f"{table}.zip") as z, open(csv, "wb") as f:
            shutil.copyfileobj(z.open(z.namelist()[0]), f)
    return pl.scan_csv(csv, infer_schema_length=0)


def universe(raw: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(symbols, delisted) tables for US common stocks on major exchanges."""
    tickers = _read(raw, "tickers").filter(
        (pl.col("table") == "SEP")
        & pl.col("category").str.starts_with("Domestic Common")
        & pl.col("exchange").is_in(EXCHANGES)
    )
    actions = _read(raw, "actions", ["ticker", "action"])
    acquired = set(actions.filter(pl.col("action").is_in(ACQUIRED))["ticker"])
    symbols = tickers.select(
        pl.col("ticker").alias("symbol"), "name", "exchange",
        pl.col("permaticker").cast(pl.Int64).alias("cik"),
        pl.col("siccode").cast(pl.Int64, strict=False).alias("sic"),
        pl.col("sicindustry").alias("sic_description"),
        (pl.col("isdelisted") == "Y").alias("dead"),
        pl.lit(True).alias("included"),
    ).unique("symbol")  # fmt: skip
    delisted = symbols.filter("dead").select(
        "symbol", "exchange",
        (~pl.col("symbol").is_in(list(acquired))).alias("fell_to_otc"),
        pl.lit(True).alias("included"), pl.lit(True).alias("has_prices"),
    )  # fmt: skip
    return symbols, delisted


def price_rows(raw: Path, table: str, symbols: set[str]) -> pl.DataFrame:
    """Raw OHLCV and adjusted returns (``ret_cc``, ``ret_co``) for ``symbols``.

    Sharadar's OHLC and volume are split-adjusted; ``closeunadj`` is the traded close and
    ``closeadj`` adjusts for dividends too. Raw levels undo the split factor
    (``closeunadj / close``), keeping dollar volume unchanged.
    """
    cols = ["ticker", "date", "open", "high", "low", "close", "volume", "closeadj", "closeunadj"]
    frame = (
        _scan(raw, table)
        .select(cols)
        .filter(pl.col("ticker").is_in(list(symbols)))
        .with_columns(pl.col("date").str.to_date(), *[pl.col(c).cast(pl.Float64) for c in cols[2:]])
        .collect()
        .sort("ticker", "date")
    )
    split = pl.col("closeunadj") / pl.col("close")
    prev_adj = pl.col("closeadj").shift(1).over("ticker")
    return frame.select(
        pl.col("ticker").alias("symbol"), "date",
        *[(pl.col(c) * split).alias(c) for c in ("open", "high", "low")],
        pl.col("closeunadj").alias("close"),
        (pl.col("volume") / split).alias("volume"),
        (pl.col("closeadj") / prev_adj - 1).alias("ret_cc"),
        (pl.col("open") * pl.col("closeadj") / pl.col("close") / prev_adj - 1).alias("ret_co"),
        split.alias("split"),
    ).filter(pl.col("close") > 0)  # fmt: skip


def _ever_tradeable(prices: pl.DataFrame) -> set[str]:
    """Symbols that at some point met loose price and dollar-volume floors."""
    ok = (pl.col("close") > EVER_PRICE) & (pl.col("close") * pl.col("volume") > EVER_DOLLAR_VOLUME)
    return set(prices.filter(ok)["symbol"].unique())


def filing_states(raw: Path, symbols: pl.DataFrame, splits: pl.DataFrame) -> pl.DataFrame:
    """One state per filing (see module docs), in ``research.fundamentals``' layout.

    Sharadar's share counts are adjusted for every later split; dividing by the split
    factor on the filing date (``splits``: symbol, date, split) restores the count as
    reported then, which ``research.features`` multiplies by the traded price.
    """
    cols = ["ticker", "dimension", "date", "reportperiod", "fiscalperiod", "sharesbas",
            *FUNDAMENTALS]  # fmt: skip
    f = _read(raw, "fundamentals", cols).filter(pl.col("dimension") == "ART")
    f = f.with_columns(
        pl.col("date").str.to_date().alias("filed"),
        pl.col("reportperiod").str.to_date().alias("period_end"),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in ("sharesbas", *FUNDAMENTALS)],
    ).rename(FUNDAMENTALS | {"sharesbas": "shares_out"})
    f = f.with_columns(-pl.col(c) for c in OUTFLOWS)
    f = f.join(symbols.select(pl.col("symbol").alias("ticker"), "cik"), on="ticker")
    f = (
        f.sort("filed")
        .join_asof(
            splits.select(pl.col("symbol").alias("ticker"), pl.col("date").alias("filed"), "split")
            .sort("filed"),
            on="filed", by="ticker", strategy="backward", check_sortedness=False,
        )
        .with_columns(pl.col("shares_out") / pl.col("split"))
        .drop("split")
    )  # fmt: skip
    concepts = list(FUNDAMENTALS.values())
    prior = f.select(
        "cik", "filed", (pl.col("period_end") + pl.duration(days=365)).alias("_match"),
        *[pl.col(c).alias(f"{c}_py") for c in concepts],
    ).sort("_match")  # fmt: skip
    states = f.sort("period_end").join_asof(
        prior, left_on="period_end", right_on="_match", by="cik", strategy="nearest",
        tolerance="20d", check_sortedness=False,
    )  # fmt: skip
    # The year-earlier state must have been filed by this filing (point in time).
    known = pl.col("filed_right") <= pl.col("filed")
    states = states.with_columns(
        pl.when(known).then(pl.col(f"{c}_py")).alias(f"{c}_py") for c in concepts
    )
    form = pl.when(pl.col("fiscalperiod").str.ends_with("Q4")).then(pl.lit("10-K"))
    return states.select(
        "cik",
        (pl.col("ticker") + "-" + pl.col("filed").cast(pl.String)).alias("accn"),
        form.otherwise(pl.lit("10-Q")).alias("form"),
        "filed", "period_end", "shares_out",
        *[c + s for s in ("", "_py") for c in concepts],
    ).sort("cik", "filed", "accn")  # fmt: skip


def fund_history(raw: Path, symbols: set[str]) -> pl.DataFrame:
    """Distribution-adjusted daily prices for exchange-traded funds and stocks in ``symbols``.

    From the fund table, with the stock table for anything not in it (e.g. BRK.B).

    Returns:
        symbol, date, adj_close, ret (the ``fund_prices`` layout).
    """
    frames = []
    for table in ("funds", "stocks"):
        found = {f["symbol"][0] for f in frames}
        wanted = [s for s in symbols if s not in found]
        if not wanted:
            break
        part = (
            _scan(raw, table)
            .select("ticker", "date", "closeadj")
            .filter(pl.col("ticker").is_in(wanted))
            .select(pl.col("ticker").alias("symbol"), pl.col("date").str.to_date(),
                    pl.col("closeadj").cast(pl.Float64).alias("adj_close"))
            .collect()
        )  # fmt: skip
        frames += part.partition_by("symbol")
    prices = pl.concat(frames).sort("symbol", "date")
    return prices.with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("ret")
    )


def build(raw: Path, out: Path, main: Path) -> dict:
    """Write the Sharadar data directory ``out`` (see module docs).

    Args:
        raw: Folder with the bulk zips.
        out: The new data directory (created; existing datasets are replaced).
        main: The main data directory, for FRED rates and macro series.
    """
    paths = DataPaths(out)
    symbols, delisted = universe(raw)
    prices = price_rows(raw, "stocks", set(symbols["symbol"]))
    keep = _ever_tradeable(prices)
    symbols = symbols.filter(pl.col("symbol").is_in(list(keep)))
    delisted = delisted.filter(pl.col("symbol").is_in(list(keep)))
    prices = prices.filter(pl.col("symbol").is_in(list(keep)))
    log.info("sharadar: %d stocks (%d dead), %d price rows", symbols.height, delisted.height,
             prices.height)  # fmt: skip
    for dataset, rows in (
        (paths.prices_daily, prices),
        (paths.prices_benchmarks, price_rows(raw, "funds", set(BENCHMARKS))),
    ):
        if dataset.exists():
            shutil.rmtree(dataset)
        for (year,), part in rows.group_by(pl.col("date").dt.year()):
            write_parquet_atomic(
                part.drop("split").sort("symbol", "date"), DataPaths.year_partition(dataset, year)
            )
    splits = prices.select("symbol", "date", "split")
    del prices
    write_parquet_atomic(symbols.drop("dead"), paths.universe_symbols)
    write_parquet_atomic(delisted, paths.universe_delisted)
    write_parquet_atomic(symbols.select("symbol", "cik"), paths.fundamentals_tickers)
    write_parquet_atomic(
        symbols.select(
            "cik", "name", "sic", "sic_description", pl.lit(None).alias("fiscal_year_end")
        ),
        paths.fundamentals_companies,
    )
    states = filing_states(raw, symbols, splits)
    write_parquet_atomic(states, paths.fundamentals_states)
    tickers = symbols.select("symbol", "cik")
    write_parquet_atomic(fscores_by_symbol(fscores_from_states(states), tickers), paths.fscores)
    main_paths = DataPaths(main)
    for src, dst in ((main_paths.rates, paths.rates), (main_paths.macro, paths.macro)):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return {"stocks": symbols.height, "dead": delisted.height, "filings": states.height}
