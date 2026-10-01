"""Scorecard: judge candidate strategies the same way across full market cycles.

For a set of backtest runs over the same span, :func:`scorecard` reports per run:

- **Full span**: return per year, Sharpe (vs zero), worst drop, annual turnover.
- **Eras**: return per year in each of :data:`ERAS` (bubble and bust, the 2003-07 bull,
  the financial crisis, the 2010s, the 2020s).
- **Halves**: return per year in the first and second half of the span, to see whether
  a ranking holds out of the period it was chosen on.
- **Against the reference** (e.g. plain momentum): calendar years won, and the worst
  rolling three-year return relative to it (how long and how badly a run can lag).
"""

from datetime import date

import numpy as np
import polars as pl

ERAS = {
    "1999-2002 bubble & bust": (date(1999, 1, 1), date(2002, 12, 31)),
    "2003-07 bull": (date(2003, 1, 1), date(2007, 12, 31)),
    "2008-09 crisis": (date(2008, 1, 1), date(2009, 12, 31)),
    "2010-19": (date(2010, 1, 1), date(2019, 12, 31)),
    "2020-26": (date(2020, 1, 1), date(2026, 12, 31)),
}
YEAR = 252


def _cagr(ret: np.ndarray) -> float:
    return float(np.prod(1 + ret) ** (YEAR / len(ret)) - 1) if len(ret) else float("nan")


def _worst_drop(ret: np.ndarray) -> float:
    nav = np.cumprod(1 + ret)
    return float((nav / np.maximum.accumulate(nav) - 1).min())


def scorecard(
    daily: dict[str, pl.DataFrame], turnover: dict[str, float], reference: str
) -> pl.DataFrame:
    """One row per run (see module docs).

    Args:
        daily: Run name -> daily returns (date, ret), over a common span.
        turnover: Run name -> annual turnover.
        reference: The run others are compared with (years won, rolling 3-year).
    """
    start = max(f["date"].min() for f in daily.values())
    end = min(f["date"].max() for f in daily.values())
    frames = {
        n: f.filter(pl.col("date").is_between(start, end)).sort("date") for n, f in daily.items()
    }
    middle = start + (end - start) / 2
    ref = frames[reference]
    rows = []
    for name, frame in frames.items():
        ret = frame["ret"].to_numpy()
        sharpe = float(ret.mean() / ret.std() * np.sqrt(YEAR))
        row = {"run": name, "cagr": _cagr(ret), "sharpe": sharpe,
               "worst_drop": _worst_drop(ret), "turnover": turnover.get(name)}  # fmt: skip
        for era, (a, b) in ERAS.items():
            part = frame.filter(pl.col("date").is_between(a, b))["ret"].to_numpy()
            row[era] = _cagr(part) if len(part) > YEAR // 2 else None
        row["first_half"] = _cagr(frame.filter(pl.col("date") < middle)["ret"].to_numpy())
        row["second_half"] = _cagr(frame.filter(pl.col("date") >= middle)["ret"].to_numpy())
        both = frame.join(ref, on="date", suffix="_ref")
        yearly = both.group_by(pl.col("date").dt.year()).agg(
            ((1 + pl.col("ret")).product() - (1 + pl.col("ret_ref")).product()).alias("diff")
        )
        row["years_won_vs_ref"] = f"{int((yearly['diff'] > 0).sum())}/{yearly.height}"
        growth = np.cumprod(1 + both["ret"].to_numpy()) / np.cumprod(1 + both["ret_ref"].to_numpy())
        window = 3 * YEAR
        rolling = growth[window:] / growth[:-window] - 1 if len(growth) > window else np.array([])
        row["worst_3y_vs_ref"] = float(rolling.min()) if rolling.size else None
        rows.append(row)
    return pl.DataFrame(rows)
