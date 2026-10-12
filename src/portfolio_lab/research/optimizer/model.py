"""The three models compared, as in the paper (Kelly, Kuznetsov, Malamud & Xu 2026, sec. 3).

Every model maps one month's candidates, ``x`` (candidates x inputs), to a score per stock.
As in the paper, the scores are the portfolio weights (w = Y lam): long or short, any total
size. ``long_only`` instead takes the softmax of the scores over the month's candidates
(positive, summing to 1), a departure from the paper kept for later.

- ``linear`` (O1): score = x @ lam. Each stock judged on its own inputs, no depth.
- ``mlp`` (O2): ``blocks`` feed-forward blocks, then x @ lam. Depth and nonlinearity, each stock
  still judged alone (the paper's "same network with attention removed").
- ``transformer`` (O3): ``blocks`` blocks of attention across the month's stocks then the
  feed-forward layer, then x @ lam.

A block, on the (candidates x inputs) matrix Y, with residual connections and no layer norm
(paper eqs. 13-17; one head):

    Y <- Y + softmax((Y Q)(Y K)^T) (Y V)       attention: every stock against every stock
    Y <- Y + relu(Y W1 + b1) W2 + b2           feed-forward: each stock alone

Initialization as in the paper (sec. 4.3): Q, K, V ~ N(0, 1/D); W1 ~ N(0, 1/hidden),
W2 ~ N(0, 1/D), biases 0; lam ~ N(0, 1/D).

Loss for one month (paper eq. 9 without the penalty): (1 - w . r)^2, w the weights and r next
month's returns over the T-bill. Over many months, with the weights free in size, minimizing it
gives the maximum-Sharpe portfolio. With long-only weights summing to 1 it instead equals
1 - 2 (w.r - (w.r)^2 / 2): Kelly's growth approximation with no cap, which puts everything on
one stock (stage 1, 2026-10-11). ``cap`` bounds every weight (:func:`capped_softmax`), so the
portfolio holds at least 1 / cap stocks. Alone, the cap fails: training grows the scores
until ten names sit at the cap and the gradient is exactly zero (run13, 2026-10-12).
``names`` fixes the concentration instead (:func:`spread_softmax`): each month the softmax
temperature is set so the portfolio holds that many effective names (1 / sum of squared
weights), so the model learns only the ranking.
"""

import torch
from torch import nn

MODELS = ("linear", "mlp", "transformer")
#: Temperatures tried on standardized scores, from nearly one stock to nearly equal weights.
TEMPERATURES = torch.logspace(-2, 1, 128)


def spread_softmax(scores: torch.Tensor, names: float) -> torch.Tensor:
    """Standardized scores divided by the temperature that holds ``names`` effective names.

    Effective names = 1 / sum of squared softmax weights. The temperature is picked from
    :data:`TEMPERATURES` (no gradient through the choice), so the result depends only on the
    scores' pattern across stocks, never on their size: they cannot run away.
    """
    z = (scores - scores.mean()) / scores.std()
    with torch.no_grad():
        t = TEMPERATURES.to(z.device)
        eff = 1 / (torch.softmax(z[None, :] / t[:, None], dim=1) ** 2).sum(dim=1)
        best = t[(eff.log() - torch.log(torch.tensor(float(names)))).abs().argmin()]
    return z / best


def capped_softmax(scores: torch.Tensor, cap: float) -> torch.Tensor:
    """Softmax weights with none above ``cap`` (needs at least 1 / cap names).

    Capped names get exactly ``cap`` and the rest share what is left in their softmax
    proportions (water-filling). Each round fixes the names over the cap at the cap and
    re-spreads the remainder over the others; at most 1 / cap rounds. Gradients flow through
    the uncapped names' scores.
    """
    capped = torch.zeros_like(scores, dtype=torch.bool)
    while True:
        free = torch.softmax(scores.masked_fill(capped, float("-inf")), dim=0)
        w = torch.where(capped, torch.full_like(scores, cap), (1 - cap * capped.sum()) * free)
        over = (w > cap) & ~capped
        if not bool(over.any()):
            return w
        capped = capped | over


def _normal(*shape: int, var: float) -> nn.Parameter:
    return nn.Parameter(torch.randn(*shape) * var**0.5)


class Block(nn.Module):
    """One block: optional attention across stocks, then the per-stock feed-forward layer.

    Args:
        width: Inputs per stock (D); the block keeps this width.
        hidden: Feed-forward hidden width.
        attention: Include the attention sublayer.
        fast: Compute attention in bfloat16 with torch's fused kernel (about 5x faster on the
            GPU for ~4,000 stocks; inputs rounded to 8-bit mantissas). Default: float32, the
            formula written out.
    """

    def __init__(self, width: int, hidden: int, attention: bool, fast: bool = False):
        super().__init__()
        self.attention = attention
        self.fast = fast
        if attention:
            self.q = _normal(width, width, var=1 / width)
            self.k = _normal(width, width, var=1 / width)
            self.v = _normal(width, width, var=1 / width)
        self.w1 = _normal(width, hidden, var=1 / hidden)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = _normal(hidden, width, var=1 / width)
        self.b2 = nn.Parameter(torch.zeros(width))

    def peers(self, y: torch.Tensor) -> torch.Tensor:
        """The attention weights: row i = how much stock i draws on each stock (rows sum to 1)."""
        return torch.softmax((y @ self.q) @ (y @ self.k).T, dim=-1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """(candidates x width) -> (candidates x width)."""
        if self.attention and self.fast:  # same formula, paper's scale (none: scale=1)
            q, k, v = (y @ m for m in (self.q, self.k, self.v))
            mixed = nn.functional.scaled_dot_product_attention(
                *(t.bfloat16()[None, None] for t in (q, k, v)), scale=1.0
            )
            y = y + mixed[0, 0].float()
        elif self.attention:
            y = y + self.peers(y) @ (y @ self.v)
        return y + torch.relu(y @ self.w1 + self.b1) @ self.w2 + self.b2


class Optimizer(nn.Module):
    """Scores a month's candidates; :meth:`weights` turns the scores into the portfolio.

    Args:
        kind: One of :data:`MODELS`.
        width: Inputs per stock (D).
        blocks: Blocks for ``mlp`` and ``transformer`` (ignored for ``linear``).
        hidden: Feed-forward hidden width.
        long_only: Softmax the scores (module docs); default: the scores are the weights.
        fast_attention: bfloat16 fused attention (:class:`Block`).
        scale: With ``long_only``, multiply the softmax by one learned positive number m
            (``exp`` of a parameter, starting at 1), so the loss can size the portfolio freely
            and aims at the best long-only Sharpe rather than the best return at full size.
        cap: With ``long_only``, the largest weight in any stock (:func:`capped_softmax`).
        names: With ``long_only``, the effective names held (:func:`spread_softmax`).
    """

    def __init__(  # noqa: PLR0913, PLR0917 - one argument per setting
        self,
        kind: str,
        width: int,
        blocks: int = 1,
        hidden: int = 256,
        long_only: bool = False,
        fast_attention: bool = False,
        scale: bool = False,
        cap: float | None = None,
        names: float | None = None,
    ):
        super().__init__()
        if kind not in MODELS:
            raise ValueError(f"unknown model {kind!r}; known: {MODELS}")
        n = 0 if kind == "linear" else blocks
        self.blocks = nn.ModuleList(
            Block(width, hidden, kind == "transformer", fast_attention) for _ in range(n)
        )
        self.lam = _normal(width, var=1 / width)
        self.long_only = long_only
        self.cap = cap
        self.names = names
        self.log_m = nn.Parameter(torch.zeros(())) if (long_only and scale) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(candidates x width) -> a score per candidate."""
        for block in self.blocks:
            x = block(x)
        return x @ self.lam

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """Portfolio weights: the scores, their softmax if long-only, times m if scaled."""
        scores = self(x)
        if not self.long_only:
            return scores
        if self.names:
            scores = spread_softmax(scores, self.names)
        w = capped_softmax(scores, self.cap) if self.cap else torch.softmax(scores, dim=0)
        return w * self.log_m.exp() if self.log_m is not None else w


def loss(model: Optimizer, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """(1 - portfolio return)^2 for one month (module docs)."""
    return (1 - model.weights(x) @ r) ** 2
