"""The daily table: per (sid, date), what the engine needs beyond the bar itself.

- ``adv``: trailing median dollar volume over :data:`ADV_WINDOW` sessions.
- ``bars_seen``: bars so far (excludes fresh listings).
- ``traded``: the bar had volume.
- ``common``: listed as common stock that day (``listings``), so ETFs, ADRs and
  preferreds never enter a universe.

Eligibility itself is a rule applied at load (:func:`panel`), so research can use a
wider universe than production without a second table. The table is rebuilt whole from
the reader's prices and listings (a minute on the full history); the engine then reads
prices and this table into a :class:`~portfolio_lab.research.panel.Panel` whose symbols
are sids as strings.
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from portfolio_lab.core.calendar import sessions
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.data import ids as ids_
from portfolio_lab.data import reader
from portfolio_lab.research.panel import FLOAT_FIELDS, EligibilityRules, Panel, _daily_rates

log = logging.getLogger(__name__)

TABLE = "daily"
ADV_WINDOW = 60
#: ``EligibilityRules.ever_tradeable``: the old ingest's admission floor on a single day.
EVER_PRICE, EVER_DOLLARS = 5.0, 1_000_000
#: The market index, in the panel under its own ticker.
MARKET = "SPY"
#: Actions after which a dead stock exits at its last price rather than as a delisting.
ACQUIRED = ("acquisitionby", "mergerfrom", "acquisitioncash", "acquisitionstock",
            "acquisitionelectcash", "acquisitionelectstock", "spacmerger")  # fmt: skip
CHUNKS = 6


def folder(root: Path) -> Path:
    """Where the table lives."""
    return DataPaths(root).root / "derived" / TABLE


def _common(listings: pl.DataFrame, prices: pl.DataFrame) -> pl.Series:
    """Whether each price row falls in one of its sid's common-stock listing episodes."""
    episodes = listings.filter(pl.col("category") == "common").with_columns(
        pl.col("to").fill_null(date(2999, 12, 31))
    )
    hit = (
        prices.select("sid", "date").with_row_index("_i")
        .join(episodes.select("sid", "from", "to"), on="sid", how="inner")
        .filter((pl.col("date") >= pl.col("from")) & (pl.col("date") <= pl.col("to")))
        .select("_i").unique()
    )  # fmt: skip
    flags = np.zeros(prices.height, dtype=bool)
    flags[hit["_i"].to_numpy()] = True
    return pl.Series("common", flags)


def build(root: Path) -> dict:
    """Rebuild the table from the reader's prices and listings."""
    listings = reader.read(root, "listings")
    out = folder(root)
    if out.exists():
        for old in out.glob("**/*.parquet"):
            old.unlink()
    total = 0
    for k in range(CHUNKS):
        sids = listings.filter(pl.col("sid") % CHUNKS == k)["sid"].unique().to_list()
        prices = reader.read(root, "prices", ["close", "volume"], sids=sids).sort("sid", "date")
        rows = prices.with_columns(
            (pl.col("close") * pl.col("volume"))
            .rolling_median(ADV_WINDOW, min_samples=ADV_WINDOW // 2).over("sid").alias("adv"),
            pl.int_range(1, pl.len() + 1).over("sid").cast(pl.Int32).alias("bars_seen"),
            (pl.col("volume") > 0).fill_null(False).alias("traded"),
            _common(listings, prices),
        ).select("sid", "date", "adv", "bars_seen", "traded", "common")  # fmt: skip
        for (year,), part in rows.group_by(pl.col("date").dt.year(), maintain_order=True):
            write_parquet_atomic(part, out / f"year={int(year)}" / f"part-{k}.parquet")
        total += rows.height
        log.info("daily: chunk %d of %d, %d rows", k + 1, CHUNKS, rows.height)
    return {"rows": total}


def _fell_to_otc(root: Path, securities: pl.DataFrame) -> list[str]:
    """Dead securities not bought out: they exit with the backtest's delisting return."""
    actions = reader.read(root, "actions")
    bought = set(actions.filter(pl.col("action").is_in(ACQUIRED))["sid"])
    dead = securities.filter(pl.col("last").is_not_null())["sid"]
    return [str(s) for s in dead if s not in bought]


def _names(ids: ids_.Ids, sids: list[int]) -> dict[str, str]:
    """Each sid's current ticker (its last dated name), for orders and results."""
    latest = (
        ids.tickers.filter(pl.col("sid").is_in(sids))
        .sort("dated", "from", descending=[False, False])
        .group_by("sid").agg(pl.col("ticker").last())
    )  # fmt: skip
    return {str(s): t for s, t in latest.rows()}


def panel(
    root: Path,
    start: date | None = None,
    end: date | None = None,
    rules: EligibilityRules | None = None,
) -> Panel:
    """Prices and the daily table as a :class:`Panel` (symbols are sids as strings).

    Columns are every security with a common-stock listing plus the market index, so
    the arrays stay small; ``panel.market`` is the index's column. The derived
    fundamentals (F-scores), monthly and environment tables are attached when built.
    """
    rules = rules or EligibilityRules()
    ids = ids_.Ids.load(DataPaths(root).ids)
    listings = reader.read(root, "listings")
    universe = set(listings.filter(pl.col("category") == "common")["sid"])
    market = ids_.lookup(ids, pl.DataFrame({"ticker": [MARKET], "date": [date.today()]}))["sid"][0]
    sids = sorted(universe | {market})
    lo, hi = reader.span(root, "prices")
    dates = sessions(start or lo, end or hi)
    date_pos = pl.DataFrame({"date": dates, "_row": range(len(dates))})
    sid_pos = pl.DataFrame({"sid": sids, "_col": range(len(sids))})
    symbols = [str(s) for s in sids]
    shape = (len(dates), len(sids))
    fields = {n: np.full(shape, np.nan, dtype=np.float32) for n in FLOAT_FIELDS}
    flags = {n: np.zeros(shape, dtype=bool) for n in ("eligible", "traded")}
    eligible = (
        pl.col("common").fill_null(False)
        & pl.col("traded").fill_null(False)
        & (pl.col("close") > rules.min_price)
        & (pl.col("adv") > rules.min_dollar_volume)
        & (pl.col("bars_seen") >= rules.min_history)
    ).fill_null(False)
    table = pl.scan_parquet(folder(root) / "year=*" / "*.parquet")
    # One chunk of securities at a time: the whole long table is several GB.
    for k in range(CHUNKS):
        chunk = [s for s in sids if s % CHUNKS == k]
        if rules.ever_tradeable:  # see EligibilityRules: looks ahead, research only
            history = reader.read(root, "prices", ["close", "volume"], sids=chunk)
            ok = (pl.col("close") > EVER_PRICE) & (
                pl.col("close") * pl.col("volume") > EVER_DOLLARS
            )
            admitted = set(history.filter(ok)["sid"].unique())
            chunk = [s for s in chunk if s in admitted or s == market]
        prices = reader.read(root, "prices", ["close", "ret_cc", "ret_co"], start, end, chunk)
        daily = table.filter(pl.col("sid").is_in(chunk))
        if start:
            daily = daily.filter(pl.col("date") >= start)
        if end:
            daily = daily.filter(pl.col("date") <= end)
        rows = (
            prices.join(daily.collect(), on=["sid", "date"], how="left")
            .with_columns(eligible.alias("eligible"))
            .join(date_pos, on="date")
            .join(sid_pos, on="sid")
        )
        r, c = rows["_row"].to_numpy(), rows["_col"].to_numpy()
        for name, array in fields.items():
            array[r, c] = rows[name].cast(pl.Float32).fill_null(np.nan).to_numpy()
        for name, array in flags.items():
            array[r, c] = rows[name].fill_null(False).to_numpy()
    rates = reader.read(root, "series").filter(pl.col("series") == "DTB3")
    rates = rates.select("date", pl.col("value").alias("rate"))
    out = Panel(
        dates, symbols, fields, flags["eligible"], _daily_rates(dates, rates),
        [str(s) for s in universe], fell_to_otc=_fell_to_otc(root, ids.securities),
        traded=flags["traded"], market=symbols.index(str(market)), names=_names(ids, sids),
    )  # fmt: skip
    out.ids = ids
    derived = DataPaths(root).root / "derived"
    if (derived / "fundamentals.parquet").exists():
        scores = pl.read_parquet(
            derived / "fundamentals.parquet", columns=["sid", "filed", "fscore", "n_signals"]
        )
        out.fundamentals = (
            scores.drop_nulls("fscore")
            .select(pl.col("sid").cast(pl.String).alias("symbol"), "filed", "fscore", "n_signals")
            .sort("filed")
        )
    for name in ("features", "environment"):
        file = derived / f"{'monthly' if name == 'features' else name}.parquet"
        if file.exists():
            setattr(out, name, pl.read_parquet(file).sort("date"))
    return out
