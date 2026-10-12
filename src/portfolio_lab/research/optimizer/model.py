"""The three models compared, as in the paper (Kelly, Kuznetsov, Malamud & Xu 2026, sec. 3).

Every model maps one month's candidates, ``x`` (candidates x inputs), to a score per stock;
the weights are the softmax of the scores over the month's candidates (positive, summing to 1:
long-only, fully invested).

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
month's returns over the T-bill. With the weights summing to 1 this equals
1 - 2 (w.r - (w.r)^2 / 2): minimizing it maximizes Kelly's growth approximation.
"""

import torch
from torch import nn

MODELS = ("linear", "mlp", "transformer")


def _normal(*shape: int, var: float) -> nn.Parameter:
    return nn.Parameter(torch.randn(*shape) * var**0.5)


class Block(nn.Module):
    """One block: optional attention across stocks, then the per-stock feed-forward layer.

    Args:
        width: Inputs per stock (D); the block keeps this width.
        hidden: Feed-forward hidden width.
        attention: Include the attention sublayer.
    """

    def __init__(self, width: int, hidden: int, attention: bool):
        super().__init__()
        self.attention = attention
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
        if self.attention:
            y = y + self.peers(y) @ (y @ self.v)
        return y + torch.relu(y @ self.w1 + self.b1) @ self.w2 + self.b2


class Optimizer(nn.Module):
    """Scores a month's candidates; :meth:`weights` turns the scores into the portfolio.

    Args:
        kind: One of :data:`MODELS`.
        width: Inputs per stock (D).
        blocks: Blocks for ``mlp`` and ``transformer`` (ignored for ``linear``).
        hidden: Feed-forward hidden width.
    """

    def __init__(self, kind: str, width: int, blocks: int = 1, hidden: int = 256):
        super().__init__()
        if kind not in MODELS:
            raise ValueError(f"unknown model {kind!r}; known: {MODELS}")
        n = 0 if kind == "linear" else blocks
        self.blocks = nn.ModuleList(Block(width, hidden, kind == "transformer") for _ in range(n))
        self.lam = _normal(width, var=1 / width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(candidates x width) -> a score per candidate."""
        for block in self.blocks:
            x = block(x)
        return x @ self.lam

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """Portfolio weights over the candidates: softmax of the scores."""
        return torch.softmax(self(x), dim=0)


def loss(model: Optimizer, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """(1 - portfolio return)^2 for one month (module docs)."""
    return (1 - model.weights(x) @ r) ** 2
