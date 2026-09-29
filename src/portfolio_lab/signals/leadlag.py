"""Lead-lag signals: predict a stock from the latest moves of the stocks that lead it."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from portfolio_lab.research import leadlag
from portfolio_lab.research.dataview import DataView
from portfolio_lab.signals.base import Scores, register


def month_start_view(view: DataView) -> DataView:
    """The view at the last session of the previous calendar month.

    Networks are re-estimated monthly at that date, so every prediction within a month
    uses the same model regardless of where an evaluation starts.
    """
    recent = view.dates(25)
    this_month = (view.asof.year, view.asof.month)
    back = next(
        (n for n, d in enumerate(reversed(recent)) if (d.year, d.month) != this_month),
        len(recent) - 1,
    )
    return view.earlier(back)


@register("leadlag")
@dataclass
class LeadLag:
    """Lead-lag network: a stock's score is its leaders' latest returns, weighted by link.

    Args:
        horizon: Sessions per period, 1 (daily) or 5 (weekly); also the prediction horizon.
        window: Periods of history for the network (default: a year of days, or two years
            of weeks).
        k: Leaders kept per stock.
        mode: ``pairs`` (stock-to-stock network) or ``market`` (the most liquid stocks,
            averaged, as the only leader).
    """

    horizon: int = 1
    window: int | None = None
    k: int = 10
    mode: str = "pairs"
    name: str = "leadlag"
    _cache: dict = field(default_factory=dict, repr=False, metadata={"param": False})

    def __post_init__(self) -> None:
        self.horizon, self.k = int(self.horizon), int(self.k)
        if self.window is None:
            self.window = 252 if self.horizon == 1 else 104
        self.window = int(self.window)

    def score(self, view: DataView, symbols: Sequence[str]) -> Scores:
        """Scores from the network estimated at the previous month end."""
        at = month_start_view(view)
        if at.asof not in self._cache:
            self._cache.clear()  # evaluation moves forward in time; keep one model
            self._cache[at.asof] = leadlag.estimate(
                at, self.horizon, self.window, self.k, self.mode
            )
        network = self._cache[at.asof]
        if network is None:
            return {}
        wanted = set(symbols)
        return {
            s: v for s, v in leadlag.predict(network, view, self.horizon).items() if s in wanted
        }
