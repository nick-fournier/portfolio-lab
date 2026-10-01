"""Market history since 1926: regime signals and a momentum proxy over ~25 bear markets.

Our stock data starts in 1998, so it holds three bear markets. Kenneth French's library
(daily since July 1926: the market, the momentum factor, ten portfolios sorted on prior
12-month return, 49 industries) and FRED (Moody's BAA yield since 1919, the 10-year
Treasury since 1953, the 3-month bill since 1934) carry the same questions back a century:

- :func:`load_french` reads the daily files; :func:`load_fred` the monthly rates.
- :func:`signals` evaluates at each month end the signals of ``research.regimes`` at the
  market level: bear and rebound (the VIX only exists from 1990, so a rebound uses the
  market's own 1-month volatility easing 20% from its three-month peak), and the fragile
  candidates (industry absorption rising, credit spread widening near the high, curve
  inverted, complacent volatility), with what followed.
- :func:`momentum_proxy` runs a stand-in for our strategy's momentum tilt, holding the
  top tenth of past winners, with and without the bear defense (the market instead of
  winners in a bear) and rebound switch (the market in a rebound), by era.
"""

import io
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from portfolio_lab.research.stress import absorption

FRENCH = {
    "factors": "F-F_Research_Data_Factors_daily_CSV.zip",
    "momentum": "F-F_Momentum_Factor_daily_CSV.zip",
    "deciles": "10_Portfolios_Prior_12_2_Daily_CSV.zip",
    "industries": "49_Industry_Portfolios_daily_CSV.zip",
}
FRED_SERIES = ("BAA", "GS10", "TB3MS")
YEAR = 252
BEAR_DRAWDOWN, REBOUND_DRAWDOWN, EASING, CRASH = 0.15, 0.20, 0.20, -0.15
ERAS = {"1927-1945": (1927, 1945), "1946-1972": (1946, 1972), "1973-1998": (1973, 1998),
        "1999-2026": (1999, 2026)}  # fmt: skip


def _first_table(raw: bytes) -> pd.DataFrame:
    """The first (value-weighted) table of a French CSV: dated rows until the first gap."""
    lines = raw.decode("latin1").splitlines()
    start = next(k for k, line in enumerate(lines) if line.startswith(","))
    rows = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.strip() or not line.strip()[0].isdigit():
            break
        rows.append(line)
    table = pd.read_csv(io.StringIO("\n".join(rows)), index_col=0)
    table.index = pd.to_datetime(table.index.astype(str), format="%Y%m%d")
    table.columns = [c.strip() for c in table.columns]
    return table.replace([-99.99, -999], np.nan) / 100


def load_french(folder: Path) -> pd.DataFrame:
    """Daily returns: market, rf, mom (factor), winners/losers (top/bottom tenth), industries."""
    tables = {}
    for name, file in FRENCH.items():
        with zipfile.ZipFile(folder / file) as z:
            tables[name] = _first_table(z.open(z.namelist()[0]).read())
    f = tables["factors"]
    out = pd.DataFrame({"market": f["Mkt-RF"] + f["RF"], "rf": f["RF"]})
    out["mom"] = tables["momentum"].iloc[:, 0]
    deciles = tables["deciles"]
    out["winners"], out["losers"] = deciles.iloc[:, -1], deciles.iloc[:, 0]
    industries = tables["industries"].add_prefix("ind_")
    return out.join(industries, how="left")


def load_fred(text_by_series: dict[str, str]) -> pd.DataFrame:
    """Monthly rates (percent) from FRED's CSV download, indexed by month start."""
    frames = []
    for series, text in text_by_series.items():
        table = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=True, na_values=".")
        frames.append(table.iloc[:, 0].rename(series))
    return pd.concat(frames, axis=1)


def _month_ends(index: pd.DatetimeIndex) -> list[int]:
    s = pd.Series(range(len(index)), index=index)
    return s.groupby(index.to_period("M")).last().tolist()


def signals(daily: pd.DataFrame, rates: pd.DataFrame) -> pl.DataFrame:
    """Signals and outcomes at each month end since enough history exists (module docs)."""
    level = (1 + daily["market"].fillna(0)).cumprod().to_numpy()
    ret = daily["market"].fillna(0).to_numpy()
    ind = daily.filter(like="ind_")
    winners = (1 + daily["winners"].fillna(0)).cumprod().to_numpy()
    losers = (1 + daily["losers"].fillna(0)).cumprod().to_numpy()
    rows = []
    for i in _month_ends(daily.index):
        if i < 2 * YEAR:
            continue
        when = daily.index[i]
        # Monthly averages are complete only after the month; use last month's.
        known = rates.loc[: when - pd.offsets.MonthBegin(2)]
        rate = known.iloc[-1] if len(known) else None
        row = {
            "date": when.date(),
            "drawdown": 1 - level[i] / level[i - 2 * YEAR : i + 1].max(),
            "below_ma200": level[i] < level[i - 199 : i + 1].mean(),
            "vol1m": ret[i - 20 : i + 1].std() * np.sqrt(YEAR),
            "vol3m": ret[i - 62 : i + 1].std() * np.sqrt(YEAR),
            "absorption": absorption(ind.iloc[i - YEAR + 1 : i + 1]),
            "credit": rate["BAA"] - rate["GS10"] if rate is not None else None,
            "curve": rate["GS10"] - rate["TB3MS"] if rate is not None else None,
        }  # fmt: skip
        for h, name in ((21, "fwd_1m"), (63, "fwd_3m"), (126, "fwd_6m")):
            row[name] = level[i + h] / level[i] - 1 if i + h < len(level) else None
        row["fwd_max_dd_6m"] = None
        if i + 126 < len(level):
            path = level[i : i + 127] / level[i]
            row["fwd_max_dd_6m"] = float((path / np.maximum.accumulate(path) - 1).min())
        row["winners_minus_losers_3m"] = None
        if i + 63 < len(level):
            row["winners_minus_losers_3m"] = (
                winners[i + 63] / winners[i] - losers[i + 63] / losers[i]
            )
        rows.append(row)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    vol_pct = []
    for k, v in enumerate(frame["vol3m"].to_list()):
        vol_pct.append(float(np.mean(np.asarray(frame["vol3m"][: k + 1]) <= v)))
    near_high = pl.col("drawdown") <= 0.05
    past = pl.col("absorption").shift(1)
    return frame.with_columns(pl.Series("vol_pct", vol_pct)).with_columns(
        ((pl.col("below_ma200").cast(pl.Int8).rolling_sum(3) == 3)
         & (pl.col("drawdown") >= BEAR_DRAWDOWN)).alias("bear"),
        ((pl.col("drawdown") >= REBOUND_DRAWDOWN)
         & (pl.col("vol1m") <= (1 - EASING) * pl.col("vol1m").rolling_max(3))).alias("rebound"),
        ((pl.col("absorption") - past.rolling_mean(12)) / past.rolling_std(12) > 1)
        .alias("absorption_rising"),
        (near_high & (pl.col("credit") - pl.col("credit").shift(3) > 0.25))
        .alias("credit_divergence"),
        (pl.col("curve") < 0).alias("curve_inverted"),
        (pl.col("vol_pct") <= 0.2).alias("complacent"),
    )  # fmt: skip


def summarize(frame: pl.DataFrame, by_era: bool = False) -> pl.DataFrame:
    """Per signal (and era): months, crash rate after, forward returns, winners vs losers."""
    names = ("all months", "bear", "rebound", "absorption_rising", "credit_divergence",
             "curve_inverted", "complacent")  # fmt: skip
    eras = ERAS if by_era else {"all": (1900, 2100)}
    rows = []
    for era, (a, b) in eras.items():
        span = frame.filter(pl.col("date").dt.year().is_between(a, b))
        for name in names:
            part = span if name == "all months" else span.filter(pl.col(name).fill_null(False))
            known = part.drop_nulls("fwd_max_dd_6m")
            rows.append({
                "era": era, "signal": name, "months": part.height,
                "crash_next_6m": float((known["fwd_max_dd_6m"] <= CRASH).mean())
                if known.height else None,
                "fwd_3m": part["fwd_3m"].mean(), "fwd_6m": part["fwd_6m"].mean(),
                "winners_minus_losers_3m": part["winners_minus_losers_3m"].mean(),
            })  # fmt: skip
    return pl.DataFrame(rows)


def bear_episodes(frame: pl.DataFrame) -> list[tuple[date, date]]:
    """Spans of consecutive bear months."""
    spans, start, last = [], None, None
    for d, bear in frame.select("date", pl.col("bear").fill_null(False)).iter_rows():
        if bear and start is None:
            start = d
        if not bear and start is not None:
            spans.append((start, last))
            start = None
        last = d
    return spans


def momentum_proxy(daily: pd.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """Past winners (top tenth), and with the bear and rebound switches, by era.

    Each month's state (from :func:`signals` at the month end) sets the next month's
    holding: winners normally; the market in a bear (defense) or a rebound (no momentum
    tilt). Annual return, worst drop and Sharpe per era.
    """
    flags = frame.select(
        "date", pl.col("bear").fill_null(False), pl.col("rebound").fill_null(False)
    ).iter_rows()
    states = {pd.Timestamp(d): "bear" if b else "rebound" if r else "normal" for d, b, r in flags}
    ends = pd.Series(daily.index, index=daily.index).groupby(daily.index.to_period("M")).last()
    next_month = {p + 1: states.get(ts, "normal") for p, ts in ends.items()}
    state = pd.Series([next_month.get(p, "normal") for p in daily.index.to_period("M")],
                      index=daily.index)  # fmt: skip
    variants = {
        "market": daily["market"],
        "winners (momentum)": daily["winners"],
        "winners + bear defense": daily["winners"].where(state != "bear", daily["market"]),
        "winners + bear + rebound": daily["winners"].where(state == "normal", daily["market"]),
    }
    rows = []
    for era, (a, b) in {"all": (1927, 2026), **ERAS}.items():
        span = (daily.index.year >= a) & (daily.index.year <= b)
        for name, r in variants.items():
            x = r[span].fillna(0).to_numpy()
            nav = np.cumprod(1 + x)
            excess = x - daily["rf"][span].fillna(0).to_numpy()
            sharpe = float(excess.mean() / excess.std() * np.sqrt(YEAR))
            worst = float((nav / np.maximum.accumulate(nav) - 1).min())
            rows.append({"era": era, "portfolio": name, "cagr": nav[-1] ** (YEAR / len(x)) - 1,
                         "worst_drop": worst, "sharpe": sharpe})  # fmt: skip
    return pl.DataFrame(rows)
