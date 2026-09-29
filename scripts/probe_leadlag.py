r"""Probe: do lagged links between stocks exist, and at which lags and time scales?

A diagnostic, not a trading signal. For back-to-back windows A then B of ``W`` sessions, it
estimates every pair's lagged correlation on market-adjusted daily returns,

    C_L[i, j] = corr(r_i(t - L), r_j(t))        (stock i leads stock j by L days)

in each window, and asks whether what window A shows is still there in window B. The null
is window B with its days shuffled, which keeps same-day co-movement but destroys timing;
every read-out is reported next to its null.

Read-outs, per window length W and lag L (averaged over all window pairs):

- ``sign_agree``: of the strongest 0.1% of pairs in A, the share with the same sign in B.
- ``oos_link``: their average link in B, signed by the direction seen in A (0 = no
  persistence), in units of the noise level 1/sqrt(W).
- ``lag_agree``: of the strongest pairs by best lag (lags 1-10), the share with the same
  best lag in B. ``lag_chance`` is the agreement expected if each pair's lags in A and B were
  drawn independently from how often each lag wins overall (10% if all lags win equally);
  agreement above it means pairs have their own characteristic lag.
- ``q1``..``q5``: average signed out-of-sample link over *all* pairs, by follower liquidity
  quintile (q1 = least liquid), showing whether some stocks are slower than others.

Lags ``c5`` and ``c10`` use the leader's cumulative return over the past 5 or 10 days.

Usage::

    uv run python scripts/probe_leadlag.py [--windows 20 60 120 252]
"""

import argparse
import logging
import time

import numpy as np
import polars as pl

from portfolio_lab.core.config import get_settings
from portfolio_lab.core.store import write_parquet_atomic
from portfolio_lab.research.panel import Panel

log = logging.getLogger("probe_leadlag")

LAGS = list(range(1, 11))
CUMULATIVE = (5, 10)
TOP_SHARE = 0.001
MIN_COVERAGE = 0.95
START = 2017


def adjusted_returns(ret: np.ndarray, market: np.ndarray) -> np.ndarray:
    """Remove each column's market-explained move (beta fitted within the window)."""
    m = market - market.mean()
    x = ret - ret.mean(axis=0)
    beta = m @ x / (m @ m)
    return x - np.outer(m, beta)


def standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance columns (constant columns become zero)."""
    sd = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(sd > 0, sd, np.inf)


def lagged(z: np.ndarray, lag: int | str) -> np.ndarray:
    """``C[i, j] = corr(leader_i(t), z_j(t))``, the leader being ``z`` lagged or cumulated."""
    if isinstance(lag, str):  # cumulative: sum of the past K days
        k = int(lag[1:])
        csum = np.cumsum(np.vstack([np.zeros((1, z.shape[1]), z.dtype), z]), axis=0)
        leader = standardize(csum[k:-1] - csum[: -k - 1])  # days t-k..t-1, for t = k..n-1
        follower = standardize(z[k:])
    else:
        leader, follower = standardize(z[:-lag]), standardize(z[lag:])
    c = leader.T @ follower / len(follower)
    np.fill_diagonal(c, 0.0)
    return c


def read_outs(ca: np.ndarray, cb: np.ndarray, quintile: np.ndarray, w: int) -> dict:
    """Persistence of window A's links in window B (see module docs)."""
    n_top = max(1, int(TOP_SHARE * ca.size))
    flat_a, flat_b = ca.ravel(), cb.ravel()
    top = np.argpartition(-np.abs(flat_a), n_top)[:n_top]
    sign = np.sign(flat_a[top])
    signed = np.sign(ca) * cb  # every pair's out-of-sample link in A's direction
    per_follower = signed.sum(axis=0) / (len(ca) - 1)
    out = {
        "sign_agree": float((sign == np.sign(flat_b[top])).mean()),
        "oos_link": float((sign * flat_b[top]).mean() * np.sqrt(w)),
    }
    for q in range(5):
        out[f"q{q + 1}"] = float(per_follower[quintile == q].mean() * np.sqrt(w))
    return out


def window_pairs(panel: Panel, w: int) -> list[tuple[int, int, int]]:
    """(start of A, start of B, end of B exclusive) for back-to-back windows from START."""
    first = next(i for i, d in enumerate(panel.dates) if d.year >= START)
    return [(a, a + w, a + 2 * w) for a in range(first, len(panel.dates) - 2 * w + 1, w)]


def probe_pair(panel: Panel, a: int, b: int, end: int, rng: np.random.Generator) -> list[dict]:
    """All read-outs for one (A, B) window pair, real and shuffled."""
    ret = panel.field("ret_cc")
    spy = np.nan_to_num(ret[a:end, panel.symbol_index["SPY"]])
    universe = np.flatnonzero(panel.eligible[b - 1])
    block = ret[a:end][:, universe]
    keep = (np.isfinite(block[: b - a]).mean(0) >= MIN_COVERAGE) & (
        np.isfinite(block[b - a :]).mean(0) >= MIN_COVERAGE
    )
    cols = universe[keep]
    block = np.nan_to_num(block[:, keep])
    za = standardize(adjusted_returns(block[: b - a], spy[: b - a])).astype(np.float32)
    zb = standardize(adjusted_returns(block[b - a :], spy[b - a :])).astype(np.float32)
    zb_null = zb[rng.permutation(len(zb))]
    adv = np.nan_to_num(panel.field("adv")[b - 1, cols])
    quintile = np.minimum((np.argsort(np.argsort(adv)) * 5) // len(cols), 4)
    w = b - a

    rows = []
    best = {}  # per variant: (max |C| so far, its lag) for windows A, B and shuffled B
    for lag in [*LAGS, *(f"c{k}" for k in CUMULATIVE)]:
        if isinstance(lag, int) and lag >= w // 2:
            continue
        ca = lagged(za, lag)
        for variant, zb_ in (("real", zb), ("null", zb_null)):
            cb = lagged(zb_, lag)
            rows.append({"lag": str(lag), "variant": variant, **read_outs(ca, cb, quintile, w)})
            if isinstance(lag, int):
                for key, c in ((("A", variant), ca), (("B", variant), cb)):
                    m, arg = best.get(key, (np.zeros_like(c), np.zeros(c.shape, np.int8)))
                    better = np.abs(c) > m
                    best[key] = (
                        np.where(better, np.abs(c), m),
                        np.where(better, lag, arg).astype(np.int8),
                    )
    for variant in ("real", "null"):
        max_a, lag_a = best[("A", variant)]
        _, lag_b = best[("B", variant)]
        n_top = max(1, int(TOP_SHARE * max_a.size))
        top = np.argpartition(-max_a.ravel(), n_top)[:n_top]
        la, lb = lag_a.ravel()[top], lag_b.ravel()[top]
        share_a = np.bincount(la, minlength=max(LAGS) + 1) / len(la)
        share_b = np.bincount(lb, minlength=max(LAGS) + 1) / len(lb)
        rows.append(
            {
                "lag": "best",
                "variant": variant,
                "lag_agree": float((la == lb).mean()),
                "lag_chance": float(share_a @ share_b),
            }
        )
    for row in rows:
        row.update(window=w, start=panel.dates[a], stocks=len(cols))
    return rows


def summarize(results: pl.DataFrame) -> pl.DataFrame:
    """Mean of each read-out per (window, lag), real and null side by side, with a t-stat."""
    metrics = ["sign_agree", "oos_link", "lag_agree", "lag_chance", "q1", "q2", "q3", "q4", "q5"]
    wide = results.pivot(on="variant", index=["window", "lag", "start"], values=metrics)
    exprs = []
    for m in metrics:
        real, null = pl.col(f"{m}_real"), pl.col(f"{m}_null")
        diff = real - null
        exprs += [
            real.mean().alias(m),
            null.mean().alias(f"{m}_null"),
            (diff.mean() / diff.std() * pl.len().sqrt()).alias(f"{m}_t"),
        ]
    order = {str(k): i for i, k in enumerate([*LAGS, *(f"c{k}" for k in CUMULATIVE), "best"])}
    return (
        wide.group_by("window", "lag")
        .agg(pl.len().alias("pairs"), *exprs)
        .with_columns(pl.col("lag").replace_strict(order).alias("_o"))
        .sort("window", "_o")
        .drop("_o")
    )


def main() -> None:
    """Run the probe for each window length, save raw read-outs and print summaries."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--windows", type=int, nargs="+", default=[20, 60, 120, 252])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    settings = get_settings()
    panel = Panel.load(settings.data_dir)
    rng = np.random.default_rng(args.seed)
    rows = []
    for w in args.windows:
        t0 = time.time()
        pairs = window_pairs(panel, w)
        for a, b, end in pairs:
            rows += probe_pair(panel, a, b, end, rng)
        log.info("W=%d: %d window pairs in %.0fs", w, len(pairs), time.time() - t0)
    results = pl.DataFrame(rows)
    out = settings.data_dir / "results" / "probes" / "leadlag.parquet"
    write_parquet_atomic(results, out)
    table = summarize(results)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(3)
    print(table.select("window", "lag", "pairs", "sign_agree", "sign_agree_null", "sign_agree_t",
                       "oos_link", "oos_link_null", "oos_link_t"))  # fmt: skip
    print(table.filter(pl.col("lag") == "best").select(
        "window", "pairs", "lag_agree", "lag_chance", "lag_agree_null", "lag_agree_t"))  # fmt: skip
    print(table.filter(pl.col("lag") != "best").select(
        "window", "lag", "q1", "q2", "q3", "q4", "q5", "q1_t", "q5_t"))  # fmt: skip
    log.info("raw read-outs saved to %s", out)


if __name__ == "__main__":
    main()
