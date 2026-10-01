from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio_lab.backtest.execution import Execution, adjust
from portfolio_lab.backtest.tax import CASH, Lots, TaxRates, _Carry, _Year, after_tax, year_tax

RATES = TaxRates(short=0.4, long=0.2, dividends=0.2, loss_offset=0.0)


def _log(rows):
    """Trade log from (seq, date, key, before, after, income) tuples."""
    schema = ["seq", "date", "key", "before", "after", "income"]
    return pl.DataFrame(rows, schema=schema, orient="row")


def _nav(points):
    return pl.DataFrame(points, schema=["date", "nav"], orient="row")


def test_lots_sell_in_tax_order_and_fifo():
    lots = Lots(RATES, "tax")
    day0 = date(2020, 1, 1).toordinal()
    lots.buy("A", day0, 100.0)  # will be an old gain
    lots.buy("A", day0 + 400, 100.0)  # a recent gain
    lots.revalue("A", 300.0)  # both lots worth 150
    # Tax order sells the long-term lot first (gain taxed at 20% vs 40%).
    assert lots.sell("A", day0 + 500, 150.0) == pytest.approx((0.0, 50.0))
    fifo = Lots(RATES, "fifo")
    fifo.buy("A", day0, 100.0)
    fifo.buy("A", day0 + 400, 200.0)
    fifo.revalue("A", 450.0)  # old lot 150 (gain 50), new 300 (gain 100)
    short, long = fifo.sell("A", day0 + 500, 300.0)  # the old lot, then half the new
    assert (short, long) == pytest.approx((50.0, 50.0))


def test_free_of_short_gains_counts_losses_and_long_lots():
    lots = Lots(RATES)
    day0 = date(2020, 1, 1).toordinal()
    lots.buy("A", day0, 100.0)
    lots.buy("B", day0, 100.0)
    lots.revalue("A", 120.0)
    lots.revalue("B", 80.0)
    assert lots.free_of_short_gains("A", day0 + 30) == 0.0
    assert lots.free_of_short_gains("B", day0 + 30) == pytest.approx(80.0)
    assert lots.free_of_short_gains("A", day0 + 400) == pytest.approx(120.0)


def test_year_tax_nets_and_carries_losses():
    tax, carry = year_tax(_Year(short=100.0, long=-30.0), _Carry(), RATES)
    assert tax == pytest.approx(70 * 0.4) and carry == _Carry()
    tax, carry = year_tax(_Year(short=-100.0, long=30.0), _Carry(), RATES)
    assert tax == 0.0 and carry == _Carry(short=-70.0, long=0.0)
    offset = TaxRates(short=0.4, long=0.2, dividends=0.2, loss_offset=50.0)
    tax, carry = year_tax(_Year(short=-100.0, dividends=10.0), _Carry(), offset)
    assert tax == pytest.approx(10 * 0.2 - 50 * 0.4) and carry.short == pytest.approx(-50.0)
    tax, _ = year_tax(_Year(long=100.0), carry, offset)
    assert tax == pytest.approx(50 * 0.2)


def test_after_tax_short_gain_paid_in_april():
    # Buy A on day 1, it doubles, sell it all in June 2020 (short-term), hold cash.
    rows = [
        (0, date(2020, 1, 2), "A", 0.0, 1.0, 0.0), (0, date(2020, 1, 2), CASH, 1.0, 0.0, 0.0),
        (1, date(2020, 6, 1), "A", 2.0, 0.0, 0.0), (1, date(2020, 6, 1), CASH, 0.0, 2.0, 0.0),
        (2, date(2021, 4, 30), CASH, 2.0, 2.0, 0.0),
    ]  # fmt: skip
    nav = _nav([(date(2020, 1, 2), 1.0), (date(2020, 6, 1), 2.0), (date(2021, 4, 30), 2.0)])
    out = after_tax(_log(rows), nav, RATES, start=100.0)
    assert out.short_gains == pytest.approx(100.0) and out.long_gains == 0.0
    assert out.taxes == pytest.approx(40.0)
    assert out.growth["after"].to_list() == pytest.approx([100.0, 200.0, 160.0])
    assert out.growth["before"].to_list() == pytest.approx([100.0, 200.0, 200.0])


def test_after_tax_dividends_taxed_and_unrealized_until_sold():
    rows = [
        (0, date(2020, 1, 2), "A", 0.0, 1.0, 0.0), (0, date(2020, 1, 2), CASH, 1.0, 0.0, 0.0),
        (1, date(2020, 12, 31), "A", 1.5, 1.5, 0.1), (1, date(2020, 12, 31), CASH, 0.0, 0.0, 0.0),
        (2, date(2022, 6, 1), "A", 2.0, 2.0, 0.0), (2, date(2022, 6, 1), CASH, 0.0, 0.0, 0.0),
    ]  # fmt: skip
    nav = _nav([(date(2020, 1, 2), 1.0), (date(2020, 12, 31), 1.5), (date(2022, 6, 1), 2.0)])
    held = after_tax(_log(rows), nav, RATES, start=100.0)
    assert held.dividends == pytest.approx(10.0)
    # 10 of dividends taxed (2); paying it in 2021 sells a little stock, a long-term gain.
    assert held.unrealized > 0 and held.short_gains == 0.0
    sold = after_tax(_log(rows), nav, RATES, start=100.0, sell_at_end=True)
    assert sold.unrealized == 0.0
    assert sold.growth["after"][-1] < held.growth["after"][-1]
    # Cost basis 110 (100 + reinvested 10); value after the 2021 payment is a bit less than 200.
    assert sold.long_gains == pytest.approx(held.unrealized + held.long_gains, rel=1e-9)


def test_after_tax_is_untaxed_with_zero_rates():
    rows = [
        (0, date(2020, 1, 2), "A", 0.0, 1.0, 0.0), (0, date(2020, 1, 2), CASH, 1.0, 0.0, 0.0),
        (1, date(2021, 6, 1), "A", 3.0, 0.0, 0.0), (1, date(2021, 6, 1), CASH, 0.0, 3.0, 0.0),
    ]  # fmt: skip
    nav = _nav([(date(2020, 1, 2), 1.0), (date(2021, 6, 1), 3.0)])
    out = after_tax(_log(rows), nav, TaxRates(0.0, 0.0, 0.0), start=1.0, sell_at_end=True)
    assert out.growth["after"].to_list() == pytest.approx(out.growth["before"].to_list())


def test_adjust_skips_small_changes_and_rescales_the_rest():
    held = np.array([0.30, 0.29, 0.40, 0.01, 0.0])
    target = np.array([0.31, 0.20, 0.40, 0.0, 0.09])
    new = adjust(target, held, Execution(band=0.02))
    # A (+1 pt) and C (no change) stay; D is dropped despite its small size; B and the new E
    # share the remaining 0.30 as 0.20 : 0.09.
    np.testing.assert_allclose(new, [0.30, 0.30 * 20 / 29, 0.40, 0.0, 0.30 * 9 / 29])
    assert new.sum() == pytest.approx(target.sum())


def test_adjust_defers_short_term_gains():
    day0 = date(2020, 1, 1).toordinal()
    lots = Lots()
    lots.buy(0, day0, 0.25)
    lots.revalue(0, 0.5)  # all short-term gain
    lots.buy(1, day0, 0.5)
    held = np.array([0.5, 0.5])
    target = np.array([0.0, 1.0])
    new = adjust(target, held, Execution(defer_short_gains=True), lots, day0 + 30)
    np.testing.assert_allclose(new, [0.5, 0.5])
    later = adjust(target, held, Execution(defer_short_gains=True), lots, day0 + 400)
    np.testing.assert_allclose(later, [0.0, 1.0])
