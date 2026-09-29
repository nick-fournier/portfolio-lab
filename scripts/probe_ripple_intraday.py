r"""Probe: after a big stock's sudden move, is there an actionable delay before its peers follow?

The intraday follow-up to ``probe_ripple.py``, in three steps:

1. **Wave makers.** Each session, the ``--leaders`` most liquid stocks (as of the day before)
   are candidates. Their move since the previous close, net of beta times SPY's move, is
   checked minute by minute; a *splash* fires at the first minute ``tau`` it reaches
   ``--threshold`` daily standard deviations with at least twice the usual volume so far.
   Only the leader's own minute bars up to ``tau`` decide this, so it is point-in-time.
   (Minute bars are only fetched for days whose high or low could possibly qualify, a
   complete pre-filter that never decides anything by itself.)
2. **Secondary movers.** The leader's ``--peers`` most correlated stocks over the previous
   ``--window`` sessions (daily market-adjusted returns, known before the day), against the
   same number of random stocks as controls.
3. **Actionable delay.** Entering ``delay`` minutes after ``tau`` (1 and 5), each stock's
   market-adjusted return to 5, 15, 30, 60 and 120 minutes later and to the close, signed by
   the splash's direction. ``pre`` is what already happened between the previous close and
   entry. Reported per splash as the peer basket minus the control basket: mean (bp), hit
   rate, reliability ratio (mean / sd per splash), t-stat across days, and the share of
   months with a positive mean. Splashes in the first five minutes (overnight news) are
   reported separately.

Minute bars are cached under ``prices/minute/`` so re-runs only fetch what is missing.

Usage::

    uv run python scripts/probe_ripple_intraday.py --plan       # count candidates only
    uv run python scripts/probe_ripple_intraday.py              # fetch, measure, report
    uv run python scripts/probe_ripple_intraday.py --report     # report on saved rows
"""

import argparse
import logging
import time
import warnings
from datetime import date

import numpy as np
import polars as pl
from probe_ripple import quintiles

from portfolio_lab.core.config import get_settings
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import scan, write_parquet_atomic
from portfolio_lab.data.sources.alpaca import MINUTE_SCHEMA, fetch_minute_bars, make_client
from portfolio_lab.research.panel import Panel

log = logging.getLogger("probe_ripple_intraday")

MINUTES = 390
DELAYS = (1, 5)
HORIZONS = (5, 15, 30, 60, 120)
OPEN_MINUTES = 5
MIN_COVERAGE = 0.95
VOLUME_WINDOW = 60
#: A raw close-to-close move this far from the adjusted return means a split or similar.
SPLIT_TOLERANCE = 0.02


def daily_raw(panel: Panel, paths: DataPaths) -> dict[str, np.ndarray]:
    """Raw high, low, close and volume as (dates x symbols) arrays aligned with the panel."""
    frames = [
        scan(dataset, "year=*/data.parquet").select("symbol", "date", "high", "low", "close",
                                                    "volume").collect()
        for dataset in (paths.prices_daily, paths.prices_benchmarks)
    ]  # fmt: skip
    prices = pl.concat(frames)
    rows = pl.DataFrame({"date": panel.dates, "_r": range(len(panel.dates))})
    cols = pl.DataFrame({"symbol": panel.symbols, "_c": range(len(panel.symbols))})
    prices = prices.join(rows, on="date").join(cols, on="symbol")
    r, c = prices["_r"].to_numpy(), prices["_c"].to_numpy()
    out = {}
    for name in ("high", "low", "close", "volume"):
        array = np.full((len(panel.dates), len(panel.symbols)), np.nan)
        array[r, c] = prices[name].to_numpy()
        out[name] = array
    return out


def candidates(ctx: dict, t: int, args: argparse.Namespace, rng: np.random.Generator) -> list:
    """Leaders whose day could hold a splash, with their peers and controls (known at t-1)."""
    ret, spy_col, w = ctx["ret"], ctx["spy_col"], args.window
    universe = np.flatnonzero(ctx["eligible"][t - 1])
    past = ret[t - w : t][:, universe]
    keep = np.isfinite(past).mean(0) >= MIN_COVERAGE
    cols, past = universe[keep], np.nan_to_num(past[:, keep])
    m = np.nan_to_num(ret[t - w : t, spy_col])
    mc = m - m.mean()
    beta = mc @ (past - past.mean(0)) / (mc @ mc)
    resid = past - np.outer(m, beta)
    sd = resid.std(0)
    raw = ctx["raw"]
    prev = raw["close"][t - 1]
    reach = np.nanmax(np.abs([raw["high"][t] / prev - 1, raw["low"][t] / prev - 1]), axis=0)
    spy_reach = reach[spy_col]
    adv = ctx["adv"][t - 1, cols]
    leaders = np.argsort(-np.nan_to_num(adv))[: args.leaders]
    split = np.abs(raw["close"][t] / prev - 1 - ret[t]) > SPLIT_TOLERANCE
    possible = [
        k for k in leaders
        if sd[k] > 0 and not split[cols[k]]
        and reach[cols[k]] + abs(beta[k]) * spy_reach >= args.threshold * sd[k]
    ]  # fmt: skip
    if not possible:
        return []
    zr = (resid - resid.mean(0)) / np.where(sd > 0, sd, np.inf)
    corr = zr[:, possible].T @ zr / w
    liquidity = quintiles(adv)
    out = []
    for i, k in enumerate(possible):
        order = [j for j in np.argsort(-corr[i]) if j != k][: args.peers]
        pool = np.setdiff1d(np.arange(len(cols)), [k, *order])
        controls = rng.choice(pool, size=min(args.peers, len(pool)), replace=False)
        members = [("leader", k, 0), *(("peer", j, n + 1) for n, j in enumerate(order)),
                   *(("control", j, 0) for j in controls)]  # fmt: skip
        out.append(
            {
                "leader": ctx["symbols"][cols[k]],
                "beta": float(beta[k]),
                "sd": float(sd[k]),
                "usual_volume": float(np.nanmedian(raw["volume"][t - VOLUME_WINDOW : t, cols[k]])),
                "members": [
                    {"kind": kind, "symbol": ctx["symbols"][cols[j]], "rank": rank,
                     "beta": float(beta[j]), "liq": int(liquidity[j]),
                     "prev": float(prev[cols[j]]), "split": bool(split[cols[j]])}
                    for kind, j, rank in members
                ],
            }
        )  # fmt: skip
    return out


def minute_bars(client, paths: DataPaths, day: date, symbols: set[str]) -> pl.DataFrame:
    """Minute bars for ``symbols`` on ``day``, fetching only what the cache lacks."""
    folder = paths.prices_minute / f"date={day.isoformat()}"
    bars_path, asked_path = folder / "bars.parquet", folder / "requested.parquet"
    bars = pl.read_parquet(bars_path) if bars_path.exists() else pl.DataFrame(schema=MINUTE_SCHEMA)
    asked = set(pl.read_parquet(asked_path)["symbol"]) if asked_path.exists() else set()
    missing = sorted(symbols - asked)
    if missing:
        bars = pl.concat([bars, fetch_minute_bars(client, missing, day)])
        write_parquet_atomic(bars, bars_path)
        write_parquet_atomic(pl.DataFrame({"symbol": sorted(asked | set(missing))}), asked_path)
    return bars.filter(pl.col("symbol").is_in(list(symbols)))


def price_grid(bars: pl.DataFrame, symbols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Last traded price and cumulative volume per minute (symbols x 390).

    Prices are NaN before a symbol's first trade of the day.
    """
    index = {s: i for i, s in enumerate(symbols)}
    price = np.full((len(symbols), MINUTES), np.nan)
    volume = np.zeros((len(symbols), MINUTES))
    if bars.height:
        minute = bars["ts"].dt.hour().cast(pl.Int32) * 60 + bars["ts"].dt.minute() - 570
        ok = ((minute >= 0) & (minute < MINUTES)).to_numpy()
        rows = np.array([index[s] for s in bars["symbol"]])[ok]
        cols = minute.to_numpy()[ok]
        price[rows, cols] = bars["close"].to_numpy()[ok]
        volume[rows, cols] = bars["volume"].to_numpy()[ok]
    for i in range(len(symbols)):  # forward-fill the last trade
        row = price[i]
        seen = np.isfinite(row)
        if seen.any():
            last = np.maximum.accumulate(np.where(seen, np.arange(MINUTES), -1))
            price[i] = np.where(last >= 0, row[np.maximum(last, 0)], np.nan)
    return price, np.cumsum(volume, axis=1)


def detect(cands: list, bars: pl.DataFrame, threshold: float) -> list:
    """Candidates that splash, with ``tau`` (first qualifying minute) and ``sign`` set.

    Uses only the leaders' and SPY's minute bars up to each minute (point-in-time).
    """
    symbols = sorted({c["members"][0]["symbol"] for c in cands} | {"SPY"})
    price, cumvol = price_grid(bars, symbols)
    index = {s: i for i, s in enumerate(symbols)}
    spy = price[index["SPY"]]
    splashed = []
    for c in cands:
        lead = c["members"][0]
        li = index[lead["symbol"]]
        move = price[li] / lead["prev"] - 1 - c["beta"] * (spy / c["spy_prev"] - 1)
        enough = cumvol[li] >= 2 * c["usual_volume"] * (np.arange(MINUTES) + 1) / MINUTES
        hit = np.flatnonzero((np.abs(move) >= threshold * c["sd"]) & enough)
        if len(hit):
            splashed.append({**c, "tau": int(hit[0]), "sign": float(np.sign(move[hit[0]]))})
    return splashed


def measure_day(day: date, splashes: list, bars: pl.DataFrame) -> list:
    """Ripple measurement for one session's splashes (see module docs, step 3)."""
    symbols = sorted({m["symbol"] for c in splashes for m in c["members"]} | {"SPY"})
    price, _ = price_grid(bars, symbols)
    index = {s: i for i, s in enumerate(symbols)}
    spy = price[index["SPY"]]
    rows = []
    for c in splashes:
        lead, tau, sign = c["members"][0], c["tau"], c["sign"]
        for delay in DELAYS:
            entry = tau + delay
            if entry >= MINUTES - 1:
                continue
            spy_e = spy[entry]
            for mem in c["members"]:
                if mem["split"]:
                    continue
                p = price[index[mem["symbol"]]]
                if not np.isfinite(p[entry]):
                    continue

                def adj(a: float, b: float, s_a: float, s_b: float, beta=mem["beta"]) -> float:
                    return a / b - 1 - beta * (s_a / s_b - 1)

                row = {
                    "date": day, "leader": lead["symbol"], "tau": tau, "delay": delay,
                    "open": tau < OPEN_MINUTES, "up": sign > 0, "kind": mem["kind"],
                    "rank": mem["rank"], "liq": mem["liq"],
                    "pre": sign * adj(p[entry], mem["prev"], spy_e, c["spy_prev"]),
                    "close": sign * adj(p[-1], p[entry], spy[-1], spy_e),
                    "full": sign * adj(p[-1], mem["prev"], spy[-1], c["spy_prev"]),
                }  # fmt: skip
                for h in HORIZONS:
                    x = entry + h
                    row[f"m{h}"] = (
                        sign * adj(p[x], p[entry], spy[x], spy_e) if x < MINUTES else None
                    )
                rows.append(row)
    return rows


def report(rows: pl.DataFrame) -> None:
    """Print reliability tables per delay: peer basket minus control basket per splash."""
    cols = ["pre", *(f"m{h}" for h in HORIZONS), "close"]
    keys = ["date", "leader", "delay"]
    splashes = rows.filter(pl.col("kind") == "leader").unique(keys)
    print(f"\n{splashes.height // len(DELAYS)} splashes on {splashes['date'].n_unique()} days "
          f"({splashes.filter('open').height // len(DELAYS)} in the first {OPEN_MINUTES} "
          "minutes). Each cell: mean bp, hit rate, reliability ratio, t across days; "
          "last column: share of months positive (to close).")  # fmt: skip
    controls = rows.filter(pl.col("kind") == "control").group_by(keys).agg(pl.col(cols).mean())

    def line(label: str, peers: pl.DataFrame) -> str:
        basket = peers.group_by(keys).agg(pl.col(cols).mean())
        ex = basket.join(controls, on=keys, suffix="_c").select(
            *keys, *[(pl.col(c) - pl.col(f"{c}_c")).alias(c) for c in cols]
        )
        cells = []
        for c in cols:
            x = ex[c].drop_nulls().to_numpy()
            per_day = ex.group_by("date").agg(pl.col(c).mean())[c].drop_nulls().to_numpy()
            if len(x) < 10 or len(per_day) < 3:
                cells.append(f"{'n/a':>23}")
                continue
            t = per_day.mean() / per_day.std(ddof=1) * np.sqrt(len(per_day))
            ratio = x.mean() / x.std()
            cells.append(f"{x.mean() * 1e4:+6.1f} {np.mean(x > 0):4.0%} {ratio:+.2f} {t:+5.1f}")
        months = ex.group_by(pl.col("date").dt.truncate("1mo")).agg(pl.col("close").mean())["close"]
        share = f"{(months > 0).mean():.0%}" if len(months) else "n/a"
        return f"{label:<24}{ex.height:>6} " + " | ".join(cells) + f" | {share}"  # fmt: skip

    header = f"{'':<24}{'n':>6} " + " | ".join(f"{c:^23}" for c in cols) + " | months+"
    for delay in DELAYS:
        d = rows.filter(pl.col("delay") == delay)
        peers = d.filter(pl.col("kind") == "peer")
        intra = peers.filter(~pl.col("open"))
        print(f"\nEntry {delay} minute(s) after the splash is visible")
        print(header)
        print(line("leader itself", d.filter((pl.col("kind") == "leader") & ~pl.col("open"))))
        print(line("peers, intraday splash", intra))
        print(line("  top 5 links", intra.filter(pl.col("rank") <= 5)))
        print(line("  links 6-20", intra.filter(pl.col("rank") > 5)))
        for name, lo, hi in (("  low liquidity", 0, 1), ("  mid liquidity", 2, 2),
                             ("  high liquidity", 3, 4)):  # fmt: skip
            print(line(name, intra.filter(pl.col("liq").is_between(lo, hi))))
        print(line("  up splashes", intra.filter("up")))
        print(line("  down splashes", intra.filter(~pl.col("up"))))
        print(line("peers, opening splash", peers.filter("open")))
        full = intra.select(pl.col("full").mean(), pl.col("close").mean())
        print(f"  share of the peers' day move still ahead at entry: "
              f"{full['close'][0] / full['full'][0]:.0%}")  # fmt: skip


def main() -> None:
    """Plan, fetch and measure (or just report), per the command-line flags."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--leaders", type=int, default=500)
    parser.add_argument("--peers", type=int, default=20)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan", action="store_true", help="count candidates, fetch nothing")
    parser.add_argument("--report", action="store_true", help="report on saved rows")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    warnings.simplefilter("ignore", RuntimeWarning)

    settings = get_settings()
    paths = DataPaths(settings.data_dir)
    saved = settings.data_dir / "results" / "probes" / "ripple_intraday.parquet"
    if args.report:
        report(pl.read_parquet(saved))
        return

    panel = Panel.load(settings.data_dir)
    ctx = {
        "ret": panel.field("ret_cc"),
        "spy_col": panel.symbol_index["SPY"],
        "eligible": panel.eligible,
        "adv": panel.field("adv"),
        "raw": daily_raw(panel, paths),
        "symbols": panel.symbols,
    }
    rng = np.random.default_rng(args.seed)
    days = [i for i, d in enumerate(panel.dates) if args.start <= d <= args.end]
    plan = {}
    for t in days:
        cands = candidates(ctx, t, args, rng)
        for c in cands:
            c["spy_prev"] = float(ctx["raw"]["close"][t - 1, ctx["spy_col"]])
        if cands:
            plan[panel.dates[t]] = cands
    n_cands = sum(len(c) for c in plan.values())
    # First pass: the candidate leaders' minute bars (pages of 10,000 bars).
    requests = sum(len(cs) * MINUTES // 10_000 + 1 for cs in plan.values())
    log.info("%d candidate leader-days on %d sessions; first pass ~%d requests",
             n_cands, len(plan), requests)  # fmt: skip
    if args.plan:
        return

    rows, t0 = [], time.time()
    with make_client(settings) as client:
        for n, (day, cands) in enumerate(sorted(plan.items()), 1):
            leaders = {c["members"][0]["symbol"] for c in cands} | {"SPY"}
            splashes = detect(cands, minute_bars(client, paths, day, leaders), args.threshold)
            if splashes:
                wanted = {m["symbol"] for c in splashes for m in c["members"]} | {"SPY"}
                rows += measure_day(day, splashes, minute_bars(client, paths, day, wanted))
            if n % 25 == 0:
                log.info("%d/%d sessions, %d rows, %.0f min", n, len(plan), len(rows),
                         (time.time() - t0) / 60)  # fmt: skip
    frame = pl.DataFrame(rows, infer_schema_length=None)
    write_parquet_atomic(frame, saved)
    report(frame)


if __name__ == "__main__":
    main()
