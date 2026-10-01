import numpy as np
import pandas as pd

from portfolio_lab.research.regimes import market_state
from portfolio_lab.strategies.meanvar.strategy import contrarian_returns


def _market(*segments):
    return pd.Series(np.concatenate([np.full(n, r) for n, r in segments]))


def test_market_state_bear_rebound_and_normal():
    rising = _market((600, 0.001))
    assert market_state(rising, [15.0, 15.0, 15.0]) == "normal"
    falling = _market((500, 0.001), (120, -0.003))  # ~30% down, below the average for months
    assert market_state(falling, [40.0, 45.0, 44.0]) == "bear"
    assert market_state(falling, [60.0, 50.0, 40.0]) == "rebound"  # panic easing
    shallow = _market((500, 0.001), (80, -0.001))  # below average but only ~8% down
    assert market_state(shallow, [30.0, 30.0, 30.0]) == "normal"


def test_market_state_thresholds_move_the_bear_line():
    falling = _market((500, 0.001), (120, -0.0012))  # about 13% down
    assert market_state(falling, [40.0] * 3, bear_drawdown=0.10) == "bear"
    assert market_state(falling, [40.0] * 3, bear_drawdown=0.15) == "normal"


def test_contrarian_returns_favor_the_beaten_down():
    trailing = pd.Series({"UP": 0.6, "FLAT": 0.0, "DOWN": -0.5})
    vol = pd.Series({"UP": 0.3, "FLAT": 0.3, "DOWN": 0.3})
    mu = contrarian_returns(trailing, vol, 0.02)
    assert mu["DOWN"] > mu["FLAT"] > mu["UP"]
