"""Modeling dataset: monthly features and environment, with forward-looking labels.

One row per stock per month end, from the point-in-time feature panel (``features``)
and environment (``context``). Labels look forward and are for training and evaluation
only: ``fwd_{h}`` is the stock's return over the next ``h`` sessions (acquisitions exit at
the last price, stocks that fell to OTC take the delisting return, as in the backtest), and
``y_{h}`` is 1 if it beat the median of all labeled stocks that month. ``label_end_{h}`` is
the session index the label reaches, used to keep training labels out of test periods.
"""

import polars as pl

from portfolio_lab.research.context import STOCK_FEATURES
from portfolio_lab.research.features import FEATURES
from portfolio_lab.research.panel import Panel
from portfolio_lab.research.scoreboard import forward_returns

HORIZONS = (21,)
#: Per-stock model inputs.
STOCK_COLUMNS = (*FEATURES, *STOCK_FEATURES)
#: Liquidity pool reported separately (the 500 most liquid stocks each month).
TOP_POOL = 500


def build_dataset(
    panel: Panel, features: pl.DataFrame, env: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Features (and environment) per stock-month with labels for each horizon.

    Args:
        panel: Prices, for forward returns.
        features: The monthly feature panel.
        env: The monthly environment; its numeric columns are added as ``env_*``.

    Returns:
        date, symbol, ``session`` (index into the panel's dates), ``top500``, the stock
        columns present, ``env_*`` columns, and per horizon ``fwd_{h}``, ``y_{h}`` and
        ``label_end_{h}`` (null where the future is not yet known).
    """
    columns = [c for c in STOCK_COLUMNS if c in features.columns]
    data = features.select("date", "symbol", *columns).with_columns(
        (pl.col("log_adv").rank("ordinal", descending=True).over("date") <= TOP_POOL)
        .fill_null(False)
        .alias("top500")
    )
    labels = []
    for day in data["date"].unique().sort():
        i = panel.date_index[day]
        row = {"date": day}
        for h in HORIZONS:
            if i + h < len(panel.dates):
                row[f"fwd_{h}"] = forward_returns(panel, i, h)
                row[f"label_end_{h}"] = i + h
        labels.append(row)
    data = data.join(_label_frame(panel, labels), on=["date", "symbol"], how="left")
    sessions = pl.DataFrame(
        {"date": [r["date"] for r in labels],
         "session": [panel.date_index[r["date"]] for r in labels]}
    )  # fmt: skip
    data = data.join(sessions, on="date", how="left").with_columns(
        # Null where the forward return is unknown (the comparison with null is null).
        (pl.col(f"fwd_{h}") > pl.col(f"fwd_{h}").median().over("date"))
        .cast(pl.Int8)
        .alias(f"y_{h}")
        for h in HORIZONS
    )
    if env is not None:
        numeric = [c for c, t in env.schema.items() if c != "date" and t.is_numeric()]
        data = data.join(
            env.select("date", *[pl.col(c).alias(f"env_{c}") for c in numeric]),
            on="date", how="left",
        )  # fmt: skip
    return data.sort("date", "symbol")


def _label_frame(panel: Panel, labels: list[dict]) -> pl.DataFrame:
    """Long (date, symbol, fwd_h, label_end_h) rows from per-date forward-return vectors."""
    frames = []
    symbols = pl.Series("symbol", panel.symbols)
    for row in labels:
        cols = {"date": [row["date"]] * len(panel.symbols), "symbol": symbols}
        for h in HORIZONS:
            fwd = row.get(f"fwd_{h}")
            cols[f"fwd_{h}"] = (
                pl.Series(fwd).fill_nan(None) if fwd is not None
                else pl.Series([None] * len(panel.symbols), dtype=pl.Float64)
            )  # fmt: skip
            cols[f"label_end_{h}"] = pl.Series(
                [row.get(f"label_end_{h}")] * len(panel.symbols), dtype=pl.Int64
            )
        frames.append(pl.DataFrame(cols))
    return pl.concat(frames)


def rank_features(data: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Each column as a percentile (0 to 1) within its month; missing stays missing.

    Ranks make inputs comparable across months (a 5% earnings yield means different things
    in 2017 and 2023) and robust to extreme values.
    """
    return data.with_columns(
        ((pl.col(c).rank("average").over("date") - 0.5) / pl.col(c).count().over("date")).alias(c)
        for c in columns
    )
