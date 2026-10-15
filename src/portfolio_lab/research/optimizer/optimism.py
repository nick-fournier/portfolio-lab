"""How much better the model looks on the months it trained on than on the month it trades.

The cost-aware loss weighs each trade's cost against the return it buys, but on training
months that return is in-sample: the weights were fit to those very outcomes, so it runs well
above what the same weights earn on an unseen month (about 2x, run19), and costs look smaller
than they are. This measures that optimism by bucket, so a cost multiplier can be set from data:
after each month's fit, the model's gross return per dollar held on its training months
(in-sample) and on the test month (out of sample), split four ways: by the stock's spread
bucket (all holdings), by its borrow fee bucket (shorts only), by the position's size within
its month (quintiles of |w|) and by the stock's volatility (quintiles of a ranked volatility
trait, ``vol``: its column in the inputs). Spread and fee edges are fixed quantiles of the
whole cost table, so those buckets mean the same stocks in every month.

:func:`edges` sets the cost buckets; :func:`measure` returns, for one set of months, the mean
over months of each bucket's sum of w r and sum of |w|: rows (w r, |w|) for spread, fee on
shorts, size and volatility, in that order.
Optimism of bucket b over a run = (sum in-sample w r / sum in-sample |w|) / (the same out of
sample), summed over months before dividing.
"""

import numpy as np
import torch

from portfolio_lab.research.optimizer import holdings

BUCKETS = 5
GROUPS = ("spread", "short_fee", "size", "vol")
#: Edges of a ranked trait in [-0.5, 0.5] into five equal buckets.
RANK_EDGES = (-0.3, -0.1, 0.1, 0.3)


def edges(stock: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Inner quantile edges of every stock-month's spread and borrow fee (``BUCKETS`` buckets)."""
    q = np.linspace(0, 1, BUCKETS + 1)[1:-1]
    spread = np.concatenate([s.cpu().numpy() for s, _ in stock])
    fee = np.concatenate([f.cpu().numpy() for _, f in stock])
    device = stock[0][0].device
    return (torch.tensor(np.quantile(spread, q), dtype=torch.float32, device=device),
            torch.tensor(np.quantile(fee, q), dtype=torch.float32, device=device))  # fmt: skip


def _sums(w: torch.Tensor, r: torch.Tensor, spread: torch.Tensor, fee: torch.Tensor,
          cuts: tuple[torch.Tensor, torch.Tensor],
          vol: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """(8 x BUCKETS): rows of w r and |w| by spread, fee (shorts), size, volatility bucket.

    ``vol``: the stocks' ranked volatility (``None``: those rows stay 0).
    """
    out = w.new_zeros(2 * len(GROUPS), BUCKETS)
    gain, size = w * r, w.abs()
    short = w < 0
    rank = torch.argsort(torch.argsort(size)).float() / max(len(w), 1)
    groups = [(torch.bucketize(spread, cuts[0]), None),
              (torch.bucketize(fee, cuts[1]), short),
              ((rank * BUCKETS).long().clamp(max=BUCKETS - 1), None)]  # fmt: skip
    if vol is not None:
        groups.append((torch.bucketize(vol, vol.new_tensor(RANK_EDGES)), None))
    for g, (bucket, keep) in enumerate(groups):
        b, x, s = (bucket, gain, size) if keep is None else (bucket[keep], gain[keep], size[keep])
        out[2 * g].index_add_(0, b, x)
        out[2 * g + 1].index_add_(0, b, s)
    return out


def measure(model, data: list, pricing, months: list[int], cuts,
            vol: int | None = None) -> list[float]:  # fmt: skip
    """Mean over ``months`` of :func:`_sums` for the model's weights (no gradient), flattened.

    Training months' inputs come from the pricing's book, as in training.
    """
    with torch.no_grad():
        total = data[0][1].new_zeros(2 * len(GROUPS), BUCKETS)
        for k in months:
            w = model(holdings.inputs(data, pricing, pricing.book, k))
            v = None if vol is None else data[k][0][:, vol]
            total += _sums(w, data[k][1], *pricing.stock[k], cuts, v)
    return (total / max(len(months), 1)).flatten().tolist()


def test_month(w: np.ndarray, data: list, pricing, t: int, cuts,
               vol: int | None = None) -> list[float]:  # fmt: skip
    """:func:`_sums` for the walk's holdings ``w`` on test month ``t``, flattened."""
    wt = torch.from_numpy(w).to(data[t][1])
    v = None if vol is None else data[t][0][:, vol]
    return _sums(wt, data[t][1], *pricing.stock[t], cuts, v).flatten().tolist()


def summary(log, column_in: str = "opt_in", column_out: str = "opt_out") -> dict[str, list]:
    """Per group and bucket over a walk's log: return per dollar in and out of sample, the gap.

    Logs from before the size and volatility groups have only the first two.
    """
    raw_in, raw_out = np.array(log[column_in].to_list()), np.array(log[column_out].to_list())
    rows = raw_in.shape[1] // BUCKETS
    a = raw_in.reshape(-1, rows, BUCKETS).sum(axis=0)
    b = raw_out.reshape(-1, rows, BUCKETS).sum(axis=0)
    out = {}
    for g, name in enumerate(GROUPS[: rows // 2]):
        with np.errstate(invalid="ignore", divide="ignore"):
            pin, pout = a[2 * g] / a[2 * g + 1], b[2 * g] / b[2 * g + 1]
        out |= {f"{name}_in": pin.tolist(), f"{name}_out": pout.tolist(),
                f"{name}_gap": (pin - pout).tolist(),
                f"{name}_share": (a[2 * g + 1] / a[2 * g + 1].sum()).tolist()}  # fmt: skip
    return out
