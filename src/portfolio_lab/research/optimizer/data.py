"""The optimizer's data: one row per stock-month, then per-month arrays.

:func:`build` makes the stock-month table. Its rows are the paper's universe (Kelly,
Kuznetsov, Malamud & Xu 2026, sec. 4.1): every common stock, with no price, liquidity or
history floor (:data:`PAPER_RULES`). Its columns are the characteristics of the derived
monthly table (rebuilt here for those rows, the same way the nightly derive builds them),
plus what the models need at each month end: the stock's market value and whether it is
listed on the NYSE (for the paper's nano-cap cut), production's eligibility and its trailing
dollar volume, and next month's return over the T-bill (the
Forecaster's definition: missing bars earn zero, a stock that fell to OTC takes the
delisting return).

:func:`months` turns that table into what one training step sees: for each month end, the
candidates, their characteristics ranked across the candidates into [-0.5, 0.5] (missing = 0,
the middle), and their next-month excess returns. Candidates, as in the paper: stocks
missing at most a third of the characteristics and at or above the 1st percentile of NYSE
market values that month.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from portfolio_lab.core.paths import DataPaths
from portfolio_lab.data import reader
from portfolio_lab.research.context import environment, sensitivities, tailwinds
from portfolio_lab.research.dataview import DataView
from portfolio_lab.research.features import build_features
from portfolio_lab.research.optimizer.jkp_monthly import month_key
from portfolio_lab.research.panel import EligibilityRules, Panel
from portfolio_lab.research.scoreboard import forward_returns

#: Table columns that are not characteristics.
NOT_INPUTS = ("date", "symbol", "sic2", "nyse", "eligible", "adv", "actual")
#: The JKP price-based characteristics (``research.optimizer.jkp``).
JKP_COLUMNS = frozenset((
    "prc", "age", "beta_60m", "resff3_12_1", "resff3_6_1",
    *(f"ret_{j}_{i}" for j, i in ((1, 0), (3, 1), (6, 1), (9, 1), (12, 1), (12, 7), (60, 12))),
    *(f"seas_{s}_{e}{k}" for s, e in ((1, 1), (2, 5), (6, 10), (11, 15), (16, 20))
      for k in ("an", "na")),
    "rvol_21d", "rmax1_21d", "rmax5_21d", "rmax5_rvol_21d", "rskew_21d", "ivol_capm_21d",
    "iskew_capm_21d", "coskew_21d", "ivol_ff3_21d", "iskew_ff3_21d", "beta_dimson_21d",
    "ivol_capm_252d", "betadown_252d", "corr_1260d", "betabab_1260d", "prc_highprc_252d",
    "ami_126d", "dolvol_126d", "dolvol_var_126d", "turnover_126d", "turnover_var_126d",
    "zero_trades_21d", "zero_trades_126d", "zero_trades_252d", "bidaskhl_21d",
    # accounting (research.optimizer.jkp_fund), issuance (jkp), composites (jkp_scores)
    "aliq_at", "aliq_mat", "at_be", "at_gr1", "at_me", "at_turnover", "be_gr1a", "be_me",
    "bev_mev", "capex_abn", "capx_gr1", "capx_gr2", "capx_gr3", "cash_at", "coa_gr1a",
    "col_gr1a", "cop_at", "cop_atl1", "cowc_gr1a", "dbnetis_at", "debt_gr3", "debt_me",
    "dgp_dsale", "div12m_me", "dsale_dinv", "dsale_drec", "dsale_dsga", "earnings_variability",
    "ebit_bev", "ebit_sale", "ebitda_mev", "eq_dur", "f_score", "fcf_me", "fnl_gr1a", "gp_at",
    "gp_atl1", "inv_gr1", "inv_gr1a", "ival_me", "kz_index", "lnoa_gr1a", "ncoa_gr1a",
    "ncol_gr1a", "netdebt_me", "nfna_gr1a", "ni_ar1", "ni_be", "ni_inc8q", "ni_ivol", "ni_me",
    "niq_at", "niq_at_chg1", "niq_be", "niq_be_chg1", "niq_su", "nncoa_gr1a", "noa_at",
    "noa_gr1a", "o_score", "oaccruals_at", "oaccruals_ni", "ocf_at", "ocf_at_chg1", "ocf_me",
    "ocfq_saleq_std", "op_at", "op_atl1", "ope_be", "ope_bel1", "opex_at", "pi_nix",
    "ppeinv_gr1a", "rd5_at", "rd_me", "rd_sale", "sale_bev", "sale_gr1", "sale_gr3", "sale_me",
    "saleq_gr1", "saleq_su", "taccruals_at", "taccruals_ni", "tangibility", "tax_gr1a",
    "chcsho_12m", "eqnpo_12m",
    "mispricing_mgmt", "mispricing_perf", "qmj", "qmj_prof", "qmj_growth", "qmj_safety",
))  # fmt: skip
#: The paper keeps characteristics missing for less than this share of stock-months (sec. 4.1).
PAPER_MAX_MISSING = 0.30
#: The paper's universe: every common stock, no price, liquidity or history floor.
PAPER_RULES = EligibilityRules(min_price=0.0, min_dollar_volume=0.0, min_history=1)
#: Nano caps (below this percentile of NYSE market values) are left out (paper, sec. 4.1).
NANO_PERCENTILE = 0.01


@dataclass(frozen=True)
class Month:
    """One month end's candidates.

    Attributes:
        date: The month end (decision date).
        symbols: The candidates, in row order.
        x: Characteristics ranked across the candidates, (candidates x inputs), in [-0.5, 0.5].
        r: Next month's return over the T-bill, (candidates,); NaN in the latest month.
    """

    date: date
    symbols: list[str]
    x: np.ndarray
    r: np.ndarray


def inputs(table: pl.DataFrame, which: str = "all") -> list[str]:
    """The characteristic columns of a :func:`build` table, in table order.

    Args:
        table: From :func:`build`.
        which: ``all``; ``ours`` (the derived monthly table's characteristics); ``jkp`` (the
            JKP characteristics, ``research.optimizer.jkp``); ``paper`` (the JKP ones, with our
            market value as JKP's market equity, missing for under 30% of the table's
            stock-months, as the paper's 132 of 153).
    """
    every = [c for c in table.columns if c not in NOT_INPUTS]
    if which == "all":
        return every
    if which == "paper":
        cand = [c for c in every if c in JKP_COLUMNS or c == "market_value"]
        missing = table.select(pl.col(c).cast(pl.Float64).is_nan().fill_null(True).mean()
                               for c in cand).row(0)  # fmt: skip
        return [c for c, m in zip(cand, missing, strict=True) if m < PAPER_MAX_MISSING]
    jkp_cols = [c for c in every if c in JKP_COLUMNS]
    if which == "jkp":
        return jkp_cols
    if which == "ours":
        return [c for c in every if c not in JKP_COLUMNS]
    raise ValueError(f"unknown input set {which!r}: all, ours or jkp")


def characteristics(root: Path, panel: Panel) -> pl.DataFrame:
    """The derived monthly table's columns for the panel's eligible stocks (not written).

    The same steps as ``data.derived.monthly.build``, on a panel with other rules.
    """
    states = pl.read_parquet(DataPaths(root).fundamentals).with_columns(
        pl.col("sid").cast(pl.String).alias("symbol")
    )
    scores = states.filter(pl.col("fscore").is_not_null()).select(
        "symbol", "filed", "fscore", "n_signals"
    )
    industry = panel.ids.securities.select(pl.col("sid").cast(pl.String).alias("symbol"), "sic")
    features = build_features(panel, states.drop("fscore", "n_signals"), industry, scores)
    observations = reader.read(root, "series").filter(pl.col("series") != "DTB3")
    dates = features["date"].unique().sort().to_list()
    context = tailwinds(
        sensitivities(panel, observations, dates), environment(observations, dates, features)
    )
    return features.join(context, on=["date", "symbol"], how="left")


def build(
    root: Path, rules: EligibilityRules = PAPER_RULES, jkp_chars: bool = True
) -> pl.DataFrame:
    """The stock-month table (module docs).

    Args:
        root: The data hive.
        rules: Which stocks get a row (default: the paper's universe).
        jkp_chars: Add the JKP price-based characteristics (``research.optimizer.jkp``).

    Returns:
        The characteristics plus ``nyse`` (listed on the NYSE; current listing, not point in
        time), ``eligible`` (production's rules), ``adv`` (trailing median dollar volume) and
        ``actual`` (next month's return minus the T-bill; null in the latest month), sorted
        by date and symbol.
    """
    monthly = characteristics(root, Panel.load(root, rules=rules))
    if jkp_chars:
        from portfolio_lab.research.optimizer import (  # noqa: PLC0415
            jkp,
            jkp_fund,
            jkp_scores,
        )

        mv = monthly.select("date", "symbol", "market_value")
        extra = jkp.build(root, mv)
        fund = (
            jkp_fund.build(
                root,
                mv.select(
                    pl.col("symbol").cast(pl.Int64).alias("sid"),
                    month_key(pl.col("date")).alias("ym"),
                    pl.col("market_value").alias("me"),
                ),
            )
            .with_columns(pl.col("sid").cast(pl.String).alias("symbol"))
            .drop("sid", "ppegt")
        )
        monthly = (
            monthly.with_columns(month_key(pl.col("date")).alias("ym"))
            .join(extra, on=["symbol", "ym"], how="left")
            .join(fund, on=["symbol", "ym"], how="left")
            .drop("ym")
        )
        monthly = jkp_scores.scores(monthly)
    panel = Panel.load(root)  # production's rules: eligibility, and the same prices
    nyse = panel.ids.securities.select(
        pl.col("sid").cast(pl.String).alias("symbol"), (pl.col("exchange") == "NYSE").alias("nyse")
    )
    ends = sorted(d for d in monthly["date"].unique().to_list() if d in panel.date_index)
    # The table's last row may be a month in progress (today's derive): no full month after
    # the one before it, so that month's target stays unknown too.
    known = len(ends) - 1 - (ends[-1] == panel.dates[-1])
    frames = []
    for k, day in enumerate(ends):
        i = panel.date_index[day]
        actual = np.full(len(panel.symbols), np.nan)
        if k < known:
            rf = DataView(panel, i).risk_free() / 12
            actual = forward_returns(panel, i, panel.date_index[ends[k + 1]] - i) - rf
        frames.append(pl.DataFrame({
            "date": [day] * len(panel.symbols), "symbol": panel.symbols,
            "eligible": panel.eligible[i], "adv": panel.field("adv")[i].astype(np.float64),
            "actual": actual.astype(np.float64),
        }))  # fmt: skip
    priced = pl.concat(frames).with_columns(pl.col("adv", "actual").fill_nan(None))
    table = monthly.join(priced, on=["date", "symbol"], how="inner")
    return table.join(nyse, on="symbol", how="left").with_columns(
        pl.col("nyse").fill_null(False)
    ).sort("date", "symbol")  # fmt: skip


def _ranked(values: np.ndarray) -> np.ndarray:
    """Each column ranked across rows into [-0.5, 0.5]; missing values go to 0."""
    out = np.zeros_like(values, dtype=np.float32)
    for j in range(values.shape[1]):
        ok = np.isfinite(values[:, j])
        n = int(ok.sum())
        if n > 1:  # ranks 1..n, ties averaged, mapped to (-0.5, 0.5)
            out[ok, j] = (rankdata(values[ok, j]) - 0.5) / n - 0.5
    return out


def months(
    table: pl.DataFrame, columns: list[str] | None = None, max_missing: float = 1 / 3
) -> list[Month]:
    """Per-month candidates and their ranked characteristics (module docs).

    Args:
        table: From :func:`build`.
        columns: Characteristics to use (default: all of :func:`inputs`).
        max_missing: Stocks missing more than this share of the characteristics are left out
            (the paper's filter, sec. 4.1).

    Returns:
        One :class:`Month` per month end, ascending.
    """
    columns = columns or inputs(table)
    missing = pl.sum_horizontal(
        pl.col(c).cast(pl.Float64).is_nan().fill_null(True) for c in columns
    )
    nano = (
        pl.col("market_value")
        .filter(pl.col("nyse"))
        .quantile(NANO_PERCENTILE, interpolation="linear")
        .over("date")
    )
    usable = table.filter(
        (missing <= max_missing * len(columns)) & (pl.col("market_value") >= nano)
    )
    out = []
    for (day,), g in usable.group_by("date", maintain_order=True):
        x = g.select(pl.col(c).cast(pl.Float64) for c in columns).to_numpy()
        r = g["actual"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        out.append(Month(day, g["symbol"].to_list(), _ranked(x), r.astype(np.float32)))
    return sorted(out, key=lambda m: m.date)
