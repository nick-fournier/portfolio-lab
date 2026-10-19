"""Tests for the growth series and return horizons behind the Overview and Compare pages."""

from datetime import date, timedelta

import polars as pl
import pytest

from portfolio_lab.web import series


def _curve(rate: float, years: int = 12, dip: float = 1.0) -> pl.DataFrame:
    """Weekly growth of $1 at ``rate`` a year, multiplied by ``dip`` in the middle third."""
    days = [date(2000, 1, 3) + timedelta(days=7 * k) for k in range(52 * years)]
    third = len(days) // 3
    growth = [(1 + rate) ** ((d - days[0]).days / 365.25) for d in days]
    growth = [g * (dip if third <= k < 2 * third else 1.0) for k, g in enumerate(growth)]
    return pl.DataFrame({"date": days, "growth": growth})


def test_horizons_annualize_the_curve():
    row = series.horizons(series.Series("X", "X", "passive", _curve(0.08)))
    for horizon in ("1Y", "5Y", "10Y", "Max"):
        assert row[horizon] == pytest.approx(0.08, abs=1e-3), horizon
    assert row["20Y"] is None  # twelve years of history
    assert row["worst_drop"] == pytest.approx(0.0)


def test_worst_drop_is_the_deepest_fall_from_a_high():
    row = series.horizons(series.Series("X", "X", "passive", _curve(0.0, dip=0.7)))
    assert row["worst_drop"] == pytest.approx(-0.3)


def test_the_comparison_supplies_funds_but_never_our_strategies():
    curves = {"SPY": ("passive", 0.08), "ours: old name": ("ours", 0.18)}
    summary = pl.DataFrame([
        {"key": k, "name": k, "category": c, "period": "full", "start": date(2000, 1, 3)}
        for k, (c, _) in curves.items()
    ])  # fmt: skip
    growth = pl.concat(
        _curve(r).with_columns(pl.lit(k).alias("key")) for k, (_, r) in curves.items()
    )
    assert set(series._from_growth(summary, growth)) == {"SPY"}
