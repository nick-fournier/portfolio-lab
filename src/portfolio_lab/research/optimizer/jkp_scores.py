"""JKP's cross-sectional composites: mispricing and quality-minus-junk (``bkelly-lab/jkp-data``).

Both rank stocks against each other every month, so they run on the finished stock-month table
(every stock with a market value), after the price and accounting characteristics:

- ``mispricing_mgmt`` / ``mispricing_perf`` (Stambaugh-Yuan; aux 7738-7841): each input ranked
  within the month (ties averaged) over the count of stocks with it, at least 10 stocks; the
  score is the mean of the available ranks, kept if no more than 3 inputs are missing.
- ``qmj``, ``qmj_prof``, ``qmj_growth``, ``qmj_safety`` (Asness-Frazzini-Pedersen; aux
  8479-8617): each input z-scored as a rank within the month; each component the mean of its
  available z-ranks, z-ranked again; ``qmj`` the z-rank of the three components' mean. Altman Z
  is missing in our data, so safety uses its other four inputs (JKP's mean skips missing ones).
"""

import polars as pl

C = pl.col
#: Mispricing inputs: (column, high value gets the low rank).
MGMT = (("chcsho_12m", True), ("eqnpo_12m", False), ("oaccruals_at", True), ("noa_at", True),
        ("at_gr1", True), ("ppeinv_gr1a", True))  # fmt: skip
PERF = (("o_score", True), ("ret_12_1", False), ("gp_at", False), ("niq_at", False))
#: Quality-minus-junk components: (column, sign).
PROF = (("gp_at", 1), ("ni_be", 1), ("_ni_at", 1), ("ocf_at", 1), ("_gp_sale", 1),
        ("oaccruals_at", -1))  # fmt: skip
GROWTH = (("_gpoa_ch5", 1), ("_roe_ch5", 1), ("_roa_ch5", 1), ("_cfoa_ch5", 1), ("_gmar_ch5", 1))
SAFETY = (("betabab_1260d", -1), ("_debt_at", -1), ("o_score", -1), ("_evol", -1))
MIN_STOCKS = 10


def _rank(e: pl.Expr, descending: bool = False) -> pl.Expr:
    """Rank over the count within the month; null if fewer than :data:`MIN_STOCKS` have it."""
    r = e.rank(method="average", descending=descending).over("date") / e.count().over("date")
    return pl.when(e.count().over("date") >= MIN_STOCKS).then(r)


def _z(e: pl.Expr) -> pl.Expr:
    """JKP ``z_ranks``: the rank within the month, standardized across stocks."""
    r = e.rank(method="average").over("date")
    z = (r - r.mean().over("date")) / r.std().over("date")
    return pl.when(e.count().over("date") >= MIN_STOCKS).then(z)


def _misp(inputs: tuple, min_present: int) -> pl.Expr:
    ranks = [_rank(C(c), d) for c, d in inputs]
    n = pl.sum_horizontal([r.is_not_null().cast(pl.Int32) for r in ranks])
    return pl.when(n >= min_present).then(pl.mean_horizontal(ranks))


def scores(table: pl.DataFrame) -> pl.DataFrame:
    """The table with the six composites added and the ``_`` helper columns dropped."""
    t = table.with_columns(
        pl.coalesce(2 * C("_roeq_be_std"), C("_roe_be_std")).alias("_evol"),
        _misp(MGMT, 3).alias("mispricing_mgmt"),
        _misp(PERF, 1).alias("mispricing_perf"),
    )
    comps = {}
    for name, inputs in (("qmj_prof", PROF), ("qmj_growth", GROWTH), ("qmj_safety", SAFETY)):
        comps[f"_{name}"] = pl.mean_horizontal([_z(sign * C(c)) for c, sign in inputs])
    t = t.with_columns(**comps)
    t = t.with_columns(_z(C(f"_{n}")).alias(n) for n in ("qmj_prof", "qmj_growth", "qmj_safety"))
    mean = (C("qmj_prof") + C("qmj_growth") + C("qmj_safety")) / 3  # any missing -> missing
    t = t.with_columns(_z(mean).alias("qmj"))
    return t.drop([c for c in t.columns if c.startswith("_")])
