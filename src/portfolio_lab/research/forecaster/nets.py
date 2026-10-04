"""Part 2 as neural nets: Gu, Kelly & Xiu's NN3, warm-started month to month.

Each net: inputs → 32 → 16 → 8 → 1, batch normalization then ReLU after every hidden layer,
squared error on the target (or part 1's residual) plus an L1 penalty on the weights, Adam.
:data:`NETS` nets with different seeds are averaged.

Inputs: the stock inputs (percentiles, missing as 0) and the market inputs, each market input
scaled by its mean and spread over the month's training rows (past only), missing as 0.

Training stops early on held-out rows: each stock-month is held out with probability
:data:`HELD_SHARE` (drawn once per row, so a held-out row is never trained on later), and
the weights with the lowest held-out error are kept. The output layer starts at zero, so a
net that has learned nothing forecasts 0 rather than noise. The first month trains from
fresh weights for up to :data:`FIRST_PASSES` passes; every later month continues from the previous
month's nets for up to :data:`WARM_PASSES` passes over all past months (warm start).
"""

import numpy as np

from portfolio_lab.research.forecaster.dataset import INPUTS

LAYERS = (32, 16, 8)
LEARNING_RATE = 1e-3
BATCH = 10_000
L1 = 1e-5
NETS = 5
HELD_SHARE = 0.2
FIRST_PASSES = 100
WARM_PASSES = 5
#: Passes without a better held-out error before training stops.
PATIENCE = 5


class NetsPart:
    """Part 2 as averaged neural nets (``walk.run``'s ``part2``; module docs).

    Args:
        device: ``cpu`` or ``cuda``.
        seed: Seed for the nets' weights, batches and the held-out years.
        dispersion: Add last month's return dispersion to the market inputs.
        market: False leaves the market inputs out (stock inputs only).
        yearly_fresh: Start from fresh weights once a year (when a new calendar year
            enters the training data) instead of warm-starting throughout.
    """

    def __init__(self, device: str = "cpu", seed: int = 0, dispersion: bool = False,
                 market: bool = True, yearly_fresh: bool = False):  # fmt: skip
        import torch  # noqa: PLC0415 - research only; not a dependency of production

        self.torch, self.device, self.seed, self.dispersion = torch, device, seed, dispersion
        self.market, self.yearly_fresh, self.last_year = market, yearly_fresh, None
        self.nets: list = []
        self.scale: tuple | None = None
        self.passes: list[int] = []

    def arrays(self, x: np.ndarray):
        """``x`` as it is (each month's rows go to the device in :meth:`fit`)."""
        return x, np.asarray

    def _inputs(self, x: np.ndarray):
        """Scaled inputs as a float tensor on the device (missing as 0)."""
        mean, sd = self.scale
        z = np.array(x, dtype=np.float32)
        n = len(INPUTS)
        z[:, n:] = (z[:, n:] - mean) / sd
        if not self.market:
            z = z[:, :n]
        return self.torch.as_tensor(np.nan_to_num(z), device=self.device)

    def _new_net(self, width: int, k: int):
        torch = self.torch
        torch.manual_seed(self.seed * 1000 + k)
        layers, last = [], width
        for size in LAYERS:
            layers += [torch.nn.Linear(last, size), torch.nn.BatchNorm1d(size), torch.nn.ReLU()]
            last = size
        out = torch.nn.Linear(last, 1)
        torch.nn.init.zeros_(out.weight)
        torch.nn.init.zeros_(out.bias)
        layers.append(out)
        return torch.nn.Sequential(*layers).to(self.device)

    def fit(self, x: np.ndarray, residual: np.ndarray, years: np.ndarray) -> None:
        """Train (or keep training) the nets on these rows (module docs).

        Rows are every month before the forecast month, in a fixed order, so row ``i`` is the
        same stock-month every month and keeps its held-out draw.
        """
        torch = self.torch
        market = np.asarray(x[:, len(INPUTS) :], dtype=np.float64)
        mean = np.nanmean(market, axis=0)
        sd = np.nanstd(market, axis=0)
        self.scale = (np.nan_to_num(mean).astype(np.float32),
                      np.where(np.isfinite(sd) & (sd > 0), sd, 1.0).astype(np.float32))  # fmt: skip
        held = np.random.default_rng(self.seed).random(len(x)) < HELD_SHARE
        z = self._inputs(x)
        target = torch.as_tensor(np.asarray(residual, dtype=np.float32), device=self.device)
        fit_rows = torch.as_tensor(np.flatnonzero(~held), device=self.device)
        held_rows = torch.as_tensor(np.flatnonzero(held), device=self.device)
        latest = int(years.max())
        if self.yearly_fresh and latest != self.last_year:
            self.nets = []
        self.last_year = latest
        first = not self.nets
        if first:
            self.nets = [self._new_net(z.shape[1], k) for k in range(NETS)]
        passes = FIRST_PASSES if first else WARM_PASSES
        self.passes = [self._train(net, z, target, fit_rows, held_rows, passes, k)
                       for k, net in enumerate(self.nets)]  # fmt: skip

    def _train(self, net, z, target, fit_rows, held_rows, passes: int, k: int) -> int:
        """Train one net with early stopping; returns the passes used for the kept weights."""
        torch = self.torch
        opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        gen = torch.Generator(device=self.device).manual_seed(self.seed * 1000 + k)

        def held_error() -> float:
            net.eval()
            with torch.no_grad():
                pred = net(z[held_rows]).squeeze(1)
                return float(((pred - target[held_rows]) ** 2).mean())

        best, best_state, best_pass, stale = held_error(), _copy(net), 0, 0
        for p in range(1, passes + 1):
            net.train()
            order = fit_rows[torch.randperm(len(fit_rows), generator=gen, device=self.device)]
            for b in range(0, len(order), BATCH):
                rows = order[b : b + BATCH]
                if len(rows) < 2:  # batch normalization needs two rows
                    continue
                loss = ((net(z[rows]).squeeze(1) - target[rows]) ** 2).mean()
                loss = loss + L1 * sum(w.abs().sum() for n, w in net.named_parameters()
                                       if n.endswith("weight") and w.dim() > 1)  # fmt: skip
                opt.zero_grad()
                loss.backward()
                opt.step()
            err = held_error()
            if err < best:
                best, best_state, best_pass, stale = err, _copy(net), p, 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        net.load_state_dict(best_state)
        return best_pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The nets' average output for each row of ``x``."""
        z = self._inputs(x)
        with self.torch.no_grad():
            outs = []
            for net in self.nets:
                net.eval()
                outs.append(net(z).squeeze(1).cpu().numpy())
        return np.mean(outs, axis=0).astype(np.float64)

    def release(self) -> None:
        """Nothing to free: the nets carry over to next month."""


def _copy(net) -> dict:
    """A detached copy of ``net``'s weights."""
    return {k: v.detach().clone() for k, v in net.state_dict().items()}
