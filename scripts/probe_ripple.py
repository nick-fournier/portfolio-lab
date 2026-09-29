r"""Probe: when one stock makes a big move, do related stocks follow over the next days?

A diagnostic, not a trading signal. A *splash* is a day when a liquid stock (one of the
``--leaders`` most liquid) moves at least ``--threshold`` standard deviations more than the
market explains, on at least twice its usual volume. We don't ask why yet. The *waves* are
what its peers do next: its ``--peers`` most correlated stocks over the previous
``--window`` sessions (correlation of market-adjusted returns), compared with the same
number of random stocks as a control.

All returns are market-adjusted (return minus beta times SPY, beta from the window before
the splash) and signed by the splash's direction, so "follows the splash" is positive:

- ``day0``: the peers' move on the splash day (how much of it was priced at once).
- ``d1``..``d10``: cumulative moves over the next days (the delayed part).

Peers that splash themselves that day (their own news) are left out. Results are averaged
per splash day first and t-stats use the variation across days, so a busy day with many
splashes doesn't count many times. Peer drift is split into quintiles of trade size for the
stock's price (small trades for the price suggest retail trading), share price and
liquidity; q1 is the smallest.

``--reliability`` (on the saved rows, no recomputation) asks how *dependable* each wave is,
not just how big. The unit is one splash: the average of its peers (a basket you could
trade) minus the average of its controls. Per wave day and holding period it reports the
mean, the hit rate (share of splashes where the basket beat its controls), the reliability
ratio (mean / standard deviation per splash, a per-bet Sharpe ratio), the t-stat across
days and the share of calendar years with a positive mean, overall and by link strength,
peer liquidity, splash direction and splash size.

Usage::

    uv run python scripts/probe_ripple.py [--threshold 3] [--peers 20]
    uv run python scripts/probe_ripple.py --reliability
"""

import argparse
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from portfolio_lab.core.config import get_settings
from portfolio_lab.core.paths import DataPaths
from portfolio_lab.core.store import scan, write_parquet_atomic
from portfolio_lab.research.panel import Panel

log = logging.getLogger("probe_ripple")

HORIZON = 10
VOLUME_WINDOW = 60
MIN_COVERAGE = 0.95
START = 2017
REPORT_DAYS = (1, 2, 3, 5, 10)


def trade_fields(panel: Panel, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Volume and trade count as (dates x symbols) arrays aligned with the panel."""
    prices = (
        scan(DataPaths(data_dir).prices_daily, "year=*/data.parquet")
        .select("symbol", "date", "volume", "trade_count")
        .collect()
    )
    rows = pl.DataFrame({"date": panel.dates, "_r": range(len(panel.dates))})
    cols = pl.DataFrame({"symbol": panel.symbols, "_c": range(len(panel.symbols))})
    prices = prices.join(rows, on="date").join(cols, on="symbol")
    r, c = prices["_r"].to_numpy(), prices["_c"].to_numpy()
    volume = np.full((len(panel.dates), len(panel.symbols)), np.nan)
    trades = np.full_like(volume, np.nan)
    volume[r, c] = prices["volume"].to_numpy()
    trades[r, c] = prices["trade_count"].cast(pl.Float64).to_numpy()
    return volume, trades


def quintiles(values: np.ndarray) -> np.ndarray:
    """Cross-sectional quintile (0-4) of each value; NaN values get -1."""
    out = np.full(len(values), -1)
    ok = np.isfinite(values)
    ranks = np.argsort(np.argsort(values[ok]))
    out[ok] = np.minimum(ranks * 5 // max(ok.sum(), 1), 4)
    return out


def trade_size_proxy(usd_per_trade: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Log dollars per trade relative to what is typical for the stock's price.

    Dollars per trade mostly tracks share price (orders are sliced into lots of about 100
    shares), so the cross-sectional fit on log price is removed; negative means unusually
    small trades for the price.
    """
    x, y = np.log(price), np.log(usd_per_trade)
    ok = np.isfinite(x) & np.isfinite(y)
    out = np.full(len(x), np.nan)
    if ok.sum() > 10:
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        out[ok] = y[ok] - (intercept + slope * x[ok])
    return out


def probe_day(ctx: dict, t: int, args: argparse.Namespace, rng: np.random.Generator) -> list:
    """Splashes on session ``t`` and the ripple rows for their peers and controls."""
    ret, spy, w = ctx["ret"], ctx["spy"], args.window
    if t + HORIZON >= len(ret):
        return []
    universe = np.flatnonzero(ctx["eligible"][t - 1])
    past = ret[t - w : t][:, universe]
    ok = (np.isfinite(past).mean(0) >= MIN_COVERAGE) & np.isfinite(ret[t, universe])
    cols = universe[ok]
    past = np.nan_to_num(past[:, ok])
    m = spy[t - w : t]
    mc = m - m.mean()
    beta = mc @ (past - past.mean(0)) / (mc @ mc)
    resid = past - np.outer(m, beta)
    sd = resid.std(0)
    sd = np.where(sd > 0, sd, np.inf)
    z0 = (ret[t, cols] - beta * spy[t]) / sd

    volume = ctx["volume"]
    usual = np.nanmedian(volume[t - VOLUME_WINDOW : t, cols], axis=0)
    loud = volume[t, cols] >= 2 * usual
    adv = ctx["adv"][t - 1, cols]
    leaders = np.zeros(len(cols), dtype=bool)
    leaders[np.argsort(-np.nan_to_num(adv))[: args.leaders]] = True
    splashes = np.flatnonzero(leaders & loud & (np.abs(z0) >= args.threshold))
    if not len(splashes):
        return []

    # Market-adjusted forward returns, days 0..HORIZON, cumulative from day 1.
    fwd = np.nan_to_num(ret[t : t + HORIZON + 1, cols]) - np.outer(spy[t : t + HORIZON + 1], beta)
    cum = np.cumsum(fwd[1:], axis=0)
    zr = (resid - resid.mean(0)) / sd
    corr = zr[:, splashes].T @ zr / w  # splash leaders x all stocks
    own_news = np.abs(z0) >= args.threshold
    price = ctx["close"][t - 1, cols]
    usd_trade = np.nanmedian(
        (volume * ctx["close"] / ctx["trades"])[t - VOLUME_WINDOW : t, cols], axis=0
    )
    groups = {
        "trade_size": quintiles(trade_size_proxy(usd_trade, price)),
        "price": quintiles(price),
        "liquidity": quintiles(adv),
    }

    rows = []
    for k, a in enumerate(splashes):
        sign = np.sign(z0[a])
        order = np.argsort(-corr[k])
        peers = [j for j in order if j != a and not own_news[j]][: args.peers]
        pool = np.setdiff1d(np.flatnonzero(~own_news), [a, *peers])
        controls = rng.choice(pool, size=min(args.peers, len(pool)), replace=False)
        base = {"date": ctx["dates"][t], "leader": ctx["symbols"][cols[a]], "up": bool(sign > 0)}
        rows.append({**base, "kind": "leader", "day0": sign * fwd[0, a],
                     **{f"d{h}": sign * cum[h - 1, a] for h in range(1, HORIZON + 1)},
                     **{f"q_{g}": int(q[a]) for g, q in groups.items()}})  # fmt: skip
        for kind, members in (("peer", peers), ("control", controls)):
            for j in members:
                rows.append({**base, "kind": kind, "corr": float(corr[k, j]),
                             "day0": sign * fwd[0, j],
                             **{f"d{h}": sign * cum[h - 1, j] for h in range(1, HORIZON + 1)},
                             **{f"q_{g}": int(q[j]) for g, q in groups.items()}})  # fmt: skip
    return rows


def day_t(frame: pl.DataFrame, value: str) -> tuple[float, float, int]:
    """Mean over splash days of the per-day mean ``value``, its t-stat, and the day count."""
    per_day = frame.group_by("date").agg(pl.col(value).mean())[value].to_numpy()
    n = len(per_day)
    if n < 3:
        return float("nan"), float("nan"), n
    return float(per_day.mean()), float(per_day.mean() / per_day.std(ddof=1) * np.sqrt(n)), n


def report(rows: pl.DataFrame) -> None:
    """Print the ripple curves (in basis points) and their splits."""
    cols = ["day0", *(f"d{h}" for h in REPORT_DAYS)]
    bp = 1e4

    def line(label: str, frame: pl.DataFrame) -> str:
        cells = []
        for c in cols:
            mean, t, _ = day_t(frame, c)
            cells.append(f"{mean * bp:+7.1f} ({t:+5.1f})")
        return f"{label:<28}" + "".join(f"{c:>17}" for c in cells)

    header = f"{'':<28}" + "".join(f"{c:>17}" for c in cols)
    events = rows.filter(pl.col("kind") == "leader")
    print(f"\n{events.height} splashes on {events['date'].n_unique()} days; "
          "signed market-adjusted return in bp (t-stat across days)")  # fmt: skip
    print(header)
    # Peers minus controls, paired within each splash day.
    peer = rows.filter(pl.col("kind") == "peer")
    ctrl = rows.filter(pl.col("kind") == "control")
    diff = (
        peer.group_by("date").agg(pl.col(cols).mean())
        .join(ctrl.group_by("date").agg(pl.col(cols).mean()), on="date", suffix="_c")
        .with_columns([(pl.col(c) - pl.col(f"{c}_c")).alias(c) for c in cols])
    )  # fmt: skip
    for label, frame in (("leader (splashing stock)", events), ("peers", peer),
                         ("controls", ctrl), ("peers - controls", diff)):  # fmt: skip
        print(line(label, frame))
    for direction, up in (("up splashes", True), ("down splashes", False)):
        print(line(f"peers, {direction}", peer.filter(pl.col("up") == up)))
    for group in ("trade_size", "price", "liquidity"):
        print(f"\npeers by {group} quintile (q1 = smallest)")
        print(header)
        for q in range(5):
            print(line(f"  q{q + 1}", peer.filter(pl.col(f"q_{group}") == q)))
    print("\nleader's own drift by its trade_size quintile")
    print(header)
    for q in range(5):
        print(line(f"  q{q + 1}", events.filter(pl.col("q_trade_size") == q)))


def splash_baskets(rows: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Per splash and peer group: mean of the peers in the group minus mean of the controls.

    Adds ``link`` (top5 / 6-20 by correlation rank), ``liq`` (low / mid / high liquidity
    quintiles 1-2 / 3 / 4-5) and ``size`` (splash size tercile) to the peers first.
    """
    keys = ["date", "leader"]
    leaders = rows.filter(pl.col("kind") == "leader").select(
        *keys, pl.col("day0").abs().qcut(3, labels=["small", "medium", "large"]).alias("size")
    )
    peers = rows.filter(pl.col("kind") == "peer").with_columns(
        pl.when(pl.col("corr").rank("ordinal", descending=True).over(keys) <= 5)
        .then(pl.lit("top5"))
        .otherwise(pl.lit("6-20"))
        .alias("link"),
        pl.when(pl.col("q_liquidity") <= 1)
        .then(pl.lit("low"))
        .when(pl.col("q_liquidity") == 2)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
        .alias("liq"),
    )
    controls = rows.filter(pl.col("kind") == "control").group_by(keys).agg(pl.col(cols).mean())
    return peers.join(leaders, on=keys), controls


def reliability_line(label: str, peers: pl.DataFrame, controls: pl.DataFrame, cols: list[str]):
    """One report line: per column, mean bp / hit % / reliability ratio / t, plus years."""
    keys = ["date", "leader"]
    basket = peers.group_by(keys).agg(pl.col(cols).mean())
    excess = basket.join(controls, on=keys, suffix="_c").select(
        *keys, *[(pl.col(c) - pl.col(f"{c}_c")).alias(c) for c in cols]
    )
    cells = []
    for c in cols:
        x = excess[c].to_numpy()
        per_day = excess.group_by("date").agg(pl.col(c).mean())[c].to_numpy()
        t = per_day.mean() / per_day.std(ddof=1) * np.sqrt(len(per_day))
        cells.append(
            f"{x.mean() * 1e4:+6.1f} {np.mean(x > 0):4.0%} {x.mean() / x.std():+.2f} {t:+5.1f}"
        )
    years = excess.group_by(pl.col("date").dt.year()).agg(pl.col(cols[-1]).mean())[cols[-1]]
    return (
        f"{label:<26}{excess.height:>6} " + " | ".join(cells)
        + f" | {int((years > 0).sum())}/{len(years)}"
    )  # fmt: skip


def reliability(rows: pl.DataFrame) -> None:
    """Print how big and how dependable each wave is (see module docs)."""
    waves = [f"w{h}" for h in range(1, HORIZON + 1)]
    rows = rows.with_columns(
        pl.col("d1").alias("w1"),
        *[(pl.col(f"d{h}") - pl.col(f"d{h - 1}")).alias(f"w{h}") for h in range(2, HORIZON + 1)],
    )
    cell = "  mean  hit  ratio     t"
    print("\nPer splash: peer basket minus control basket. Each cell: mean bp, hit rate,")
    print("reliability ratio (mean/sd per splash), t across days. Last column: years positive.")

    peers, controls = splash_baskets(rows, [*waves, "d1", "d2", "d3", "d5", "d10"])
    print("\nWaves: return on each single day after the splash (all peers)")
    for chunk in (waves[:5], waves[5:]):
        print(f"{'':<26}{'n':>6} " + " | ".join(f"{c:^{len(cell)}}" for c in chunk) + " | yrs+")
        print(reliability_line("all peers", peers, controls, chunk))

    holds = ["d1", "d2", "d3", "d5"]
    print("\nHolding from the splash-day close through day h")
    print(f"{'':<26}{'n':>6} " + " | ".join(f"{c:^{len(cell)}}" for c in holds) + " | yrs+")
    print(reliability_line("all peers", peers, controls, holds))
    for name, col, values in (
        ("link", "link", ["top5", "6-20"]),
        ("liquidity", "liq", ["low", "mid", "high"]),
        ("splash size", "size", ["small", "medium", "large"]),
    ):
        for v in values:
            print(reliability_line(f"{name}: {v}", peers.filter(pl.col(col) == v), controls, holds))
    for up, direction in ((True, "up"), (False, "down")):
        subset = peers.filter(pl.col("up") == up)
        print(reliability_line(f"{direction} splashes", subset, controls, holds))
        for liq in ("low", "high"):
            for link in ("top5", "6-20"):
                cell_peers = subset.filter((pl.col("liq") == liq) & (pl.col("link") == link))
                print(reliability_line(f"  {liq} liq, {link}", cell_peers, controls, holds))


def main() -> None:
    """Find splashes, measure the ripples, save the rows and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--leaders", type=int, default=500)
    parser.add_argument("--peers", type=int, default=20)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reliability", action="store_true", help="report on the saved rows, don't recompute"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    settings = get_settings()
    saved = settings.data_dir / "results" / "probes" / "ripple.parquet"
    if args.reliability:
        reliability(pl.read_parquet(saved))
        return
    panel = Panel.load(settings.data_dir)
    volume, trades = trade_fields(panel, settings.data_dir)
    ret = panel.field("ret_cc")
    ctx = {
        "ret": ret,
        "spy": np.nan_to_num(ret[:, panel.symbol_index["SPY"]]),
        "eligible": panel.eligible,
        "adv": panel.field("adv"),
        "close": panel.field("close"),
        "volume": volume,
        "trades": np.where(trades > 0, trades, np.nan),
        "dates": panel.dates,
        "symbols": panel.symbols,
    }
    rng = np.random.default_rng(args.seed)
    warnings.simplefilter("ignore", RuntimeWarning)  # medians of all-missing columns
    first = next(i for i, d in enumerate(panel.dates) if d.year >= START)
    t0, rows = time.time(), []
    for t in range(max(first, args.window, VOLUME_WINDOW), len(panel.dates)):
        rows += probe_day(ctx, t, args, rng)
    log.info("%d rows in %.0fs", len(rows), time.time() - t0)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    write_parquet_atomic(frame, saved)
    report(frame)


if __name__ == "__main__":
    main()
