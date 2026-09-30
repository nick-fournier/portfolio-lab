from datetime import date

import polars as pl

from portfolio_lab.research.health import blowups


def test_blowups_flag_crashes_and_the_worst_tenth():
    returns = [-0.5, -0.2, *[0.01 * k for k in range(18)]]
    data = pl.DataFrame(
        {"date": [date(2020, 1, 31)] * 20, "symbol": [f"S{k}" for k in range(20)],
         "fwd_126": returns}
    )  # fmt: skip
    flags = blowups(data)["blowup"].to_list()
    assert flags[:2] == [1, 1]  # a 50% loss, and the second-worst (within the worst tenth)
    assert sum(flags) == 2
