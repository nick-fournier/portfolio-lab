"""Second model: neural nets whose stock weights are set by the market each month.

For stock ``i`` in month ``t``::

    e_i      = h(x_i)                                    8 numbers: a learned encoding of
                                                         the 103 stock inputs
    forecast = sum_j e_ij * (w_j + sum_k W_jk * m_k)     minus the month's average

``h`` is a small network (103 → 32 → 16 → 8, batch normalization and ReLU, then 8
outputs). ``m`` is every market input (:func:`market_columns`), each scaled by its mean
and spread over earlier months. ``W`` (8 by 116) says how each market input tilts each
weight this month; it starts at 0 and carries a ridge penalty (:data:`MARKET_PENALTY`),
the stock network a small L1 penalty. Forecasts are centered within each month (training
batches are whole months), so market inputs can only reorder stocks.

Training: early stopping on a fixed random 20% of rows; fresh weights in the first month,
then each month continues from the previous month's nets for up to :data:`WARM_PASSES`
passes over all earlier months (warm start). :data:`NETS` nets with different seeds are
averaged. Needs PyTorch (the ``nets`` extra).
"""

from pathlib import Path

import numpy as np
import polars as pl

LAYERS = (32, 16, 8)
ENCODING = 8
LEARNING_RATE = 1e-3
#: Months per training batch (about 10,000 stock-months).
BATCH_MONTHS = 4
STOCK_L1 = 1e-5
MARKET_PENALTY = 1.0
NETS = 5
HELD_SHARE = 0.2
FIRST_PASSES = 100
WARM_PASSES = 5
PATIENCE = 5


def market_columns(market: pl.DataFrame) -> list[str]:
    """The market inputs the nets read: environment, market state and dispersion."""
    return [c for c in market.columns if c.startswith(("env_", "mkt_"))] + ["dispersion"]


class Nets:
    """The averaged nets, carried from month to month (module docs).

    Args:
        device: ``cpu`` or ``cuda``.
        state: File to keep the nets in between runs, so a later run (e.g. next month's
            refit) continues from them; ``None`` keeps them in memory only.
        seed: Seed for weights, batches and the held-out rows.
    """

    def __init__(self, device: str = "cpu", state: Path | None = None, seed: int = 0):
        import torch  # noqa: PLC0415 - optional dependency (the ``nets`` extra)

        self.torch, self.device, self.state, self.seed = torch, device, state, seed
        self.nets: list = []
        self.scale: tuple | None = None
        if state is not None and state.exists():
            saved = torch.load(state, map_location=device, weights_only=False)
            self.scale = saved["scale"]
            self.nets = [_net(torch, *saved["shape"], seed=k).to(device) for k in range(NETS)]
            for net, weights in zip(self.nets, saved["nets"], strict=True):
                net.load_state_dict(weights)

    def _tensor(self, a: np.ndarray):
        return self.torch.as_tensor(np.nan_to_num(np.asarray(a, dtype=np.float32)),
                                    device=self.device)  # fmt: skip

    def _market(self, m: np.ndarray):
        mean, sd = self.scale
        return self._tensor((np.asarray(m, dtype=np.float64) - mean) / sd)

    def fit(self, stock: np.ndarray, market: np.ndarray, y: np.ndarray, month: np.ndarray) -> None:
        """Train (or keep training) on every earlier stock-month.

        Args:
            stock: Stock inputs, one row per stock-month, sorted by month.
            market: Market inputs for the same rows (:func:`market_columns`).
            y: Capped target.
            month: Month index of each row (rows of a month are contiguous).
        """
        first_rows = np.r_[0, np.flatnonzero(np.diff(month)) + 1]
        readings = np.asarray(market[first_rows], dtype=np.float64)
        mean = np.nan_to_num(np.nanmean(readings, axis=0))
        sd = np.nanstd(readings, axis=0)
        self.scale = (mean, np.where(np.isfinite(sd) & (sd > 0), sd, 1.0))
        s, m, target = self._tensor(stock), self._market(market), self._tensor(y)
        held = np.random.default_rng(self.seed).random(len(y)) < HELD_SHARE
        groups = _month_rows(month, ~held, self.device, self.torch)
        held_groups = _month_rows(month, held, self.device, self.torch)
        first = not self.nets
        if first:
            shape = (stock.shape[1], market.shape[1])
            self.nets = [_net(self.torch, *shape, seed=self.seed * 1000 + k).to(self.device)
                         for k in range(NETS)]  # fmt: skip
        passes = FIRST_PASSES if first else WARM_PASSES
        for k, net in enumerate(self.nets):
            self._train(net, s, m, target, groups, held_groups, passes, k + len(groups))
        if self.state is not None:
            self.torch.save({"scale": self.scale, "shape": (stock.shape[1], market.shape[1]),
                             "nets": [n.state_dict() for n in self.nets]}, self.state)  # fmt: skip

    def _train(self, net, s, m, target, groups, held_groups, passes: int, seed: int) -> None:
        torch = self.torch
        opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        order_rng = np.random.default_rng(self.seed * 1000 + seed)

        def held_error() -> float:
            net.eval()
            with torch.no_grad():
                sq = sum(((_centered(net(s[r], m[r])) - target[r]) ** 2).sum() for r in held_groups)
                return float(sq / sum(len(r) for r in held_groups))

        best, best_state, stale = held_error(), _copy(net), 0
        for _ in range(passes):
            net.train()
            order = order_rng.permutation(len(groups))
            for b in range(0, len(order), BATCH_MONTHS):
                batch = [groups[i] for i in order[b : b + BATCH_MONTHS] if len(groups[i]) > 1]
                if not batch:
                    continue
                rows = torch.cat(batch)
                parts = torch.split(net(s[rows], m[rows]), [len(r) for r in batch])
                f = torch.cat([_centered(p) for p in parts])
                loss = ((f - target[rows]) ** 2).mean() + net.penalty()
                opt.zero_grad()
                loss.backward()
                opt.step()
            err = held_error()
            if err < best:
                best, best_state, stale = err, _copy(net), 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        net.load_state_dict(best_state)

    def predict(self, stock: np.ndarray, market: np.ndarray) -> np.ndarray:
        """The nets' average forecast for one month's stocks, centered."""
        s, m = self._tensor(stock), self._market(market)
        with self.torch.no_grad():
            outs = []
            for net in self.nets:
                net.eval()
                outs.append(_centered(net(s, m)).cpu().numpy())
        return np.mean(outs, axis=0).astype(np.float64)

    def release(self) -> None:
        """Hand this month's scratch GPU memory back (the nets themselves stay)."""
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def _centered(f):
    return f - f.mean()


def _month_rows(month: np.ndarray, keep: np.ndarray, device, torch) -> list:
    """Row indices (on the device) of each month, keeping only rows where ``keep``."""
    idx = np.flatnonzero(keep)
    cuts = np.flatnonzero(np.diff(month[idx])) + 1
    return [torch.as_tensor(g, device=device) for g in np.split(idx, cuts) if len(g)]


def _copy(net) -> dict:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _net(torch, n_stock: int, n_market: int, seed: int):
    """One net (module docs) with a ``penalty`` method."""
    torch.manual_seed(seed)
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            widths = (n_stock, *LAYERS[:-1])
            self.lin = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, LAYERS, strict=True))
            self.bn = nn.ModuleList(nn.BatchNorm1d(b) for b in LAYERS)
            self.head = nn.Linear(LAYERS[-1], ENCODING)
            self.w = nn.Linear(ENCODING, 1, bias=False)
            self.W = nn.Linear(n_market, ENCODING, bias=False)
            nn.init.zeros_(self.w.weight)  # start at "no forecast"
            nn.init.zeros_(self.W.weight)  # and "no market effect"

        def forward(self, s, m):
            h = s
            for lin, bn in zip(self.lin, self.bn, strict=True):
                h = torch.relu(bn(lin(h)))
            return (self.head(h) * (self.w.weight + self.W(m))).sum(1)

        def penalty(self):
            stock = sum(lin.weight.abs().sum() for lin in self.lin) + self.head.weight.abs().sum()
            return STOCK_L1 * stock + MARKET_PENALTY * (self.W.weight**2).sum()

    return Net()
