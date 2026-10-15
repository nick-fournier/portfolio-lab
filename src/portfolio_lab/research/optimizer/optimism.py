"""How much better the model looks on the months it trained on than on the month it trades.

The cost-aware loss weighs each trade's cost against the return it buys, but on training
months that return is in-sample: the weights were fit to those very outcomes, so it runs well
above what the same weights earn on an unseen month (about 2x, run19), and costs look smaller
than they are. This measures that optimism by bucket, so a cost multiplier can be set from data:
after each month's fit, the model's gross return per dollar held on its training months
(in-sample) and on the test month (out of sample), split by the stock's spread bucket (all
holdings) and by its borrow fee bucket (shorts only). Bucket edges are fixed quantiles of the
whole cost table, so buckets mean the same stocks in every month.

:func:`edges` sets the buckets; :func:`measure` returns, for one set of months, the mean over
months of each bucket's sum of w r and sum of |w| (spread buckets, then fee buckets on shorts).
Optimism of bucket b over a run = (sum in-sample w r / sum in-sample |w|) / (the same out of
sample), summed over months before dividing.
"""

import numpy as np
import torch

from portfolio_lab.research.optimizer import holdings

BUCKETS = 5


def edges(stock: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Inner quantile edges of every stock-month's spread and borrow fee (``BUCKETS`` buckets)."""
    q = np.linspace(0, 1, BUCKETS + 1)[1:-1]
    spread = np.concatenate([s.cpu().numpy() for s, _ in stock])
    fee = np.concatenate([f.cpu().numpy() for _, f in stock])
    device = stock[0][0].device
    return (torch.tensor(np.quantile(spread, q), dtype=torch.float32, device=device),
            torch.tensor(np.quantile(fee, q), dtype=torch.float32, device=device))  # fmt: skip


def _sums(w: torch.Tensor, r: torch.Tensor, spread: torch.Tensor, fee: torch.Tensor,
          cuts: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:  # fmt: skip
    """(4 x BUCKETS): w r and |w| by spread bucket, then w r and |w| of shorts by fee bucket."""
    out = w.new_zeros(4, BUCKETS)
    bs = torch.bucketize(spread, cuts[0])
    out[0].index_add_(0, bs, w * r)
    out[1].index_add_(0, bs, w.abs())
    short = w < 0
    bf = torch.bucketize(fee[short], cuts[1])
    out[2].index_add_(0, bf, (w * r)[short])
    out[3].index_add_(0, bf, -w[short])
    return out


def measure(model, data: list, pricing, months: list[int], cuts) -> list[float]:
    """Mean over ``months`` of :func:`_sums` for the model's weights (no gradient), flattened.

    Training months' inputs come from the pricing's book, as in training.
    """
    with torch.no_grad():
        total = data[0][1].new_zeros(4, BUCKETS)
        for k in months:
            w = model(holdings.inputs(data, pricing, pricing.book, k))
            total += _sums(w, data[k][1], *pricing.stock[k], cuts)
    return (total / max(len(months), 1)).flatten().tolist()


def test_month(w: np.ndarray, data: list, pricing, t: int, cuts) -> list[float]:
    """:func:`_sums` for the walk's holdings ``w`` on test month ``t``, flattened."""
    wt = torch.from_numpy(w).to(data[t][1])
    return _sums(wt, data[t][1], *pricing.stock[t], cuts).flatten().tolist()


def summary(log, column_in: str = "opt_in", column_out: str = "opt_out") -> dict[str, list]:
    """Optimism per bucket over a walk's log: in-sample over out-of-sample return per dollar."""
    a = np.array(log[column_in].to_list()).reshape(-1, 4, BUCKETS).sum(axis=0)
    b = np.array(log[column_out].to_list()).reshape(-1, 4, BUCKETS).sum(axis=0)
    per_in = a[[0, 2]] / a[[1, 3]]
    per_out = b[[0, 2]] / b[[1, 3]]
    return {"spread_in": per_in[0].tolist(), "spread_out": per_out[0].tolist(),
            "spread_ratio": (per_in[0] / per_out[0]).tolist(),
            "short_fee_in": per_in[1].tolist(), "short_fee_out": per_out[1].tolist(),
            "short_fee_ratio": (per_in[1] / per_out[1]).tolist()}  # fmt: skip
