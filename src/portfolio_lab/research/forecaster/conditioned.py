"""Part 2 as market-conditioned neural nets (experiment): stock-only, bilinear and FiLM.

All three share one stock network: the 103 stock inputs (missing as 0) → 32 → 16 → 8
(batch normalization, ReLU) → ``exposures`` numbers per stock. They differ only in how the
market inputs (``COLUMNS``: VIX and last month's return dispersion, as centered
percentiles) enter:

- ``stock``: forecast = exposures · w (no market input; the control).
- ``bilinear``: forecast = exposures · (w0 + W m): the market sets what each exposure pays
  this month. ``W`` starts at 0.
- ``film``: every hidden layer h becomes (1 + G m) ⊙ h + B m before its ReLU, so the
  market reshapes how stock inputs combine; then exposures · w. ``G`` and ``B`` start at 0.

The market-side weights (``W``, ``G``, ``B``) carry an L2 penalty ``market_penalty``; the
stock-side weights an L1 penalty. Forecasts are centered within each month (training
batches are whole months), so the market inputs can only reorder stocks, never fit a
month's level. :data:`NETS` nets with different seeds are averaged.

Training follows ``nets``: early stopping on a fixed random 20% of rows, fresh weights in
the first month, then warm-started each month for up to :data:`WARM_PASSES` passes. The
best held-out error of the month is kept as ``held_err``, so a grid of settings can be
chosen month by month from information available then (:func:`pick`).
"""

import numpy as np
import polars as pl

from portfolio_lab.research.forecaster.dataset import INPUTS

COLUMNS = ["vix", "dispersion"]
LAYERS = (32, 16, 8)
LEARNING_RATE = 1e-3
#: Months per training batch (about 10,000 stock-months).
BATCH_MONTHS = 4
L1 = 1e-5
NETS = 5
HELD_SHARE = 0.2
FIRST_PASSES = 100
WARM_PASSES = 5
PATIENCE = 5


class CondNetsPart:
    """Market-conditioned nets as part 2 (``walk.run``'s ``part2``; module docs).

    Args:
        device: ``cpu`` or ``cuda``.
        arch: ``stock``, ``bilinear`` or ``film``.
        exposures: Numbers the stock network outputs per stock.
        market_penalty: L2 penalty on the market-side weights.
        seed: Seed for weights, batches and the held-out rows.
    """

    def __init__(self, device: str = "cpu", arch: str = "bilinear", exposures: int = 4,
                 market_penalty: float = 1e-3, seed: int = 0, factors: int | None = None):  # fmt: skip
        import torch  # noqa: PLC0415 - research only; not a dependency of production

        self.torch, self.device, self.arch, self.seed = torch, device, arch, seed
        self.exposures, self.market_penalty = exposures, market_penalty
        # factors: every market input (walk's default list, plus dispersion), compressed each
        # month to this many principal components of the past months' market readings
        self.factors = factors
        self.columns = None if factors else COLUMNS
        self.dispersion = bool(factors)
        self.pca: tuple | None = None
        self.row_months: np.ndarray | None = None  # set by walk.run: month index per row
        self.nets: list = []
        self.held_err: float | None = None

    def arrays(self, x: np.ndarray):
        """``x`` as it is (each month's rows go to the device in :meth:`fit`)."""
        return x, np.asarray

    def _tensor(self, a: np.ndarray):
        return self.torch.as_tensor(np.nan_to_num(np.asarray(a, dtype=np.float32)),
                                    device=self.device)  # fmt: skip

    def fit(self, x: np.ndarray, residual: np.ndarray, years: np.ndarray) -> None:
        """Train (or keep training) the nets on these rows (module docs)."""
        torch = self.torch
        n = len(INPUTS)
        month = self.row_months[: len(x)]
        if self.factors:
            self._fit_pca(np.asarray(x[:, n:], dtype=np.float64), month)
        s, m = self._tensor(x[:, :n]), self._tensor(self._market(x[:, n:]))
        target = self._tensor(residual)
        held = np.random.default_rng(self.seed).random(len(x)) < HELD_SHARE
        groups = _month_rows(month, ~held, self.device, torch)
        held_groups = _month_rows(month, held, self.device, torch)
        first = not self.nets
        if first:
            width = len(self.columns) if not self.factors else (m.shape[1])
            self.nets = [_Net(torch, n, width, self.arch, self.exposures,
                              self.seed * 1000 + k).to(self.device) for k in range(NETS)]  # fmt: skip
        passes = FIRST_PASSES if first else WARM_PASSES
        errs = [self._train(net, s, m, target, groups, held_groups, passes, k)
                for k, net in enumerate(self.nets)]  # fmt: skip
        self.held_err = float(np.mean(errs))

    def _forward(self, net, s, m, groups):
        """Forecasts for each group of rows (one month each), centered within the group."""
        out = []
        for rows in groups:
            f = net(s[rows], m[rows])
            out.append(f - f.mean())
        return out

    def _train(self, net, s, m, target, groups, held_groups, passes: int, k: int) -> float:
        torch = self.torch
        opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        gen = np.random.default_rng(self.seed * 1000 + k + len(groups))

        def held_error() -> float:
            net.eval()
            with torch.no_grad():
                preds = self._forward(net, s, m, held_groups)
                sq = sum(
                    ((p - target[r]) ** 2).sum() for p, r in zip(preds, held_groups, strict=True)
                )
                return float(sq / sum(len(r) for r in held_groups))

        best, best_state, stale = held_error(), _copy(net), 0
        for _ in range(passes):
            net.train()
            order = gen.permutation(len(groups))
            for b in range(0, len(order), BATCH_MONTHS):
                batch = [groups[i] for i in order[b : b + BATCH_MONTHS] if len(groups[i]) > 1]
                if not batch:
                    continue
                rows = torch.cat(batch)
                f = net(s[rows], m[rows])
                # center within each month of the batch
                sizes = [len(r) for r in batch]
                parts = torch.split(f, sizes)
                f = torch.cat([p - p.mean() for p in parts])
                loss = ((f - target[rows]) ** 2).mean() + net.penalty(L1, self.market_penalty)
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
        return best

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The nets' average forecast for the rows of one month, centered."""
        n = len(INPUTS)
        s, m = self._tensor(x[:, :n]), self._tensor(self._market(x[:, n:]))
        with self.torch.no_grad():
            outs = []
            for net in self.nets:
                net.eval()
                f = net(s, m)
                outs.append((f - f.mean()).cpu().numpy())
        return np.mean(outs, axis=0).astype(np.float64)

    def _fit_pca(self, market: np.ndarray, month: np.ndarray) -> None:
        """Principal components of the past months' market readings (one row per month).

        Each component's sign is aligned with last month's, so warm-started nets keep reading
        the same direction; factor values are scaled to unit spread.
        """
        first = np.r_[0, np.flatnonzero(np.diff(month)) + 1]
        rows = market[first]
        mean = np.nanmean(rows, axis=0)
        sd = np.nanstd(rows, axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        z = np.nan_to_num((rows - np.nan_to_num(mean)) / sd)
        if self.factors < 0:  # every input as it is, scaled: no compression
            self.pca = (np.nan_to_num(mean), sd, np.eye(rows.shape[1]), np.ones(rows.shape[1]))
            return
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        v = vt[: self.factors].T
        if self.pca is not None:
            v = v * np.where(np.sum(v * self.pca[2], axis=0) < 0, -1.0, 1.0)
        spread = (z @ v).std(axis=0)
        self.pca = (np.nan_to_num(mean), sd, v, np.where(spread > 0, spread, 1.0))

    def _market(self, market: np.ndarray) -> np.ndarray:
        """The market inputs as the nets read them (factors, or the columns as they are)."""
        if not self.factors:
            return market
        mean, sd, v, spread = self.pca
        z = np.nan_to_num((np.asarray(market, dtype=np.float64) - mean) / sd)
        return (z @ v) / spread

    def release(self) -> None:
        """Hand this month's scratch GPU memory back; the nets carry over."""
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def _month_rows(month: np.ndarray, keep: np.ndarray, device, torch) -> list:
    """Row indices (on the device) of each month, keeping only rows where ``keep``."""
    idx = np.flatnonzero(keep)
    cuts = np.flatnonzero(np.diff(month[idx])) + 1
    return [torch.as_tensor(g, device=device) for g in np.split(idx, cuts) if len(g)]


def _copy(net) -> dict:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _Net(torch, n_stock: int, n_market: int, arch: str, exposures: int, seed: int):  # noqa: N802
    """The network for ``arch`` (module docs), with a ``penalty`` method."""
    torch.manual_seed(seed)
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            widths = (n_stock, *LAYERS)
            self.lin = nn.ModuleList(
                nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:], strict=True)
            )
            self.bn = nn.ModuleList(nn.BatchNorm1d(b) for b in LAYERS)
            self.head = nn.Linear(LAYERS[-1], exposures)
            self.w = nn.Linear(exposures, 1, bias=False)
            nn.init.zeros_(self.w.weight)  # start at "no forecast"
            if arch == "bilinear":
                self.W = nn.Linear(n_market, exposures, bias=False)
                nn.init.zeros_(self.W.weight)
            if arch == "film":
                self.G = nn.ModuleList(nn.Linear(n_market, b, bias=False) for b in LAYERS)
                self.B = nn.ModuleList(nn.Linear(n_market, b, bias=False) for b in LAYERS)
                for p in [*self.G.parameters(), *self.B.parameters()]:
                    nn.init.zeros_(p)

        def forward(self, s, m):
            h = s
            for i, (lin, bn) in enumerate(zip(self.lin, self.bn, strict=True)):
                h = bn(lin(h))
                if arch == "film":
                    h = (1 + self.G[i](m)) * h + self.B[i](m)
                h = torch.relu(h)
            e = self.head(h)
            if arch == "bilinear":
                return (e * (self.w.weight + self.W(m))).sum(1)
            return self.w(e).squeeze(1)

        def market_params(self):
            if arch == "bilinear":
                return [self.W.weight]
            if arch == "film":
                return [*self.G.parameters(), *self.B.parameters()]
            return []

        def penalty(self, l1: float, l2_market: float):
            stock = sum(lin.weight.abs().sum() for lin in self.lin) + self.head.weight.abs().sum()
            market = (
                sum((p**2).sum() for p in self.market_params()) if self.market_params() else 0.0
            )
            return l1 * stock + l2_market * market

    return Net()


def pick(forecasts: list[pl.DataFrame]) -> pl.DataFrame:
    """Per month, the run whose nets had the lowest held-out error at that month's fit.

    ``held_err`` comes from training rows only, so the choice uses nothing from the month
    being forecast. Adds ``picked`` (the index of the chosen run).
    """
    errs = pl.concat([f.group_by("date").agg(pl.col("held_err").first()).with_columns(pl.lit(i).alias("run"))
                      for i, f in enumerate(forecasts)])  # fmt: skip
    best = errs.sort("held_err").group_by("date").agg(pl.col("run").first().alias("picked"))
    out = [f.join(best.filter(pl.col("picked") == i), on="date") for i, f in enumerate(forecasts)]
    return pl.concat(out).sort("date", "symbol")


def pick_by_past(forecasts: list[pl.DataFrame]) -> pl.DataFrame:
    """Per month, the run whose earlier forecasts had the lowest squared error.

    Errors use each month's forecasts centered within the month against ``actual``, summed
    over all earlier months (the same rule ``walk.combine`` uses for the trees' strength).
    Adds ``picked`` (the index of the chosen run); the first month picks run 0.
    """
    errs = []
    for i, f in enumerate(forecasts):
        c = f.drop_nulls("actual").with_columns(
            (pl.col("correction") - pl.col("correction").mean().over("date")).alias("_f")
        )
        e = c.group_by("date").agg(((pl.col("actual") - pl.col("_f")) ** 2).sum().alias("e"))
        errs.append(e.sort("date").with_columns(pl.col("e").cum_sum().shift(1).fill_null(0.0)
                                                .alias("past"), pl.lit(i).alias("run")))  # fmt: skip
    table = pl.concat(errs)
    best = table.sort(["past", "run"]).group_by("date").agg(pl.col("run").first().alias("picked"))
    months = forecasts[0].select("date").unique()
    best = months.join(best, on="date", how="left").with_columns(pl.col("picked").fill_null(0))
    out = [f.join(best.filter(pl.col("picked") == i), on="date") for i, f in enumerate(forecasts)]
    return pl.concat(out).sort("date", "symbol")
