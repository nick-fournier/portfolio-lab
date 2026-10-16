"""The three models compared, as in the paper (Kelly, Kuznetsov, Malamud & Xu 2026, sec. 3).

Every model maps one month's candidates, ``x`` (candidates x inputs), to a score per stock,
and the scores are the portfolio weights (w = Y lam): long or short, any total size.

- ``linear`` (O1): score = x @ lam. Each stock judged on its own inputs, no depth.
- ``mlp`` (O2): ``blocks`` feed-forward blocks, then x @ lam. Depth and nonlinearity, each stock
  still judged alone (the paper's "same network with attention removed").
- ``transformer`` (O3): ``blocks`` blocks of attention across the month's stocks then the
  feed-forward layer, then x @ lam.

A block, on the (candidates x inputs) matrix Y, with residual connections and no layer norm
(paper eqs. 13-17; one head):

    Y <- Y + softmax((Y Q)(Y K)^T) (Y V)       attention: every stock against every stock
    Y <- Y + relu(Y W1 + b1) W2 + b2           feed-forward: each stock alone

Initialization as in the paper (sec. 4.3), its N(0, a) read as standard deviations: Q, K, V,
W2 and lam ~ N(0, sd 1/D), W1 ~ N(0, sd 1/hidden), biases 0. Each stock's row then keeps its
size through 10 blocks and the starting loss is ~1; read as variances, rows grow ~1.7x per
block and deep models diverge (2026-10-13).

Dropout (2026-10-15, not in the paper): with ``dropout`` > 0 each block zeroes that share of
its attention and feed-forward updates at random while training (``model.train()``), scaled up
to keep their size; scoring and trading run the whole model (``model.eval()``).

Trade speed (2026-10-16, not in the paper; Garleanu & Pedersen 2013): with ``speed_column`` the
input carries one more column after the D the blocks see, the book held going in (h, model
units), and the model trades part of the way from it toward the blocks' output (the target):
w = h + tau (target - h), tau = sigmoid(a + b * s) per stock, s its log-spread input column,
a and b learned. From an all-cash book tau is 1: the first purchase is made in full.

Loss for one month (paper eq. 9 without the penalty): (1 - w . r)^2, w the weights and r next
month's returns over the T-bill. Over many months, with the weights free in size, minimizing it
gives the maximum-Sharpe portfolio.
"""

import torch
from torch import nn

MODELS = ("linear", "mlp", "transformer")


def _normal(*shape: int, sd: float) -> nn.Parameter:
    return nn.Parameter(torch.randn(*shape) * sd)


class Block(nn.Module):
    """One block: optional attention across stocks, then the per-stock feed-forward layer.

    Args:
        width: Inputs per stock (D); the block keeps this width.
        hidden: Feed-forward hidden width.
        attention: Include the attention sublayer.
        fast: Compute attention in bfloat16 with torch's fused kernel (about 5x faster on the
            GPU for ~4,000 stocks; inputs rounded to 8-bit mantissas). Default: float32, the
            formula written out.
        dropout: Share of each update zeroed while training (module docs).
    """

    def __init__(self, width: int, hidden: int, attention: bool, fast: bool = False,
                 dropout: float = 0.0):  # fmt: skip
        super().__init__()
        self.attention = attention
        self.fast = fast
        self.drop = nn.Dropout(dropout)
        if attention:
            self.q = _normal(width, width, sd=1 / width)
            self.k = _normal(width, width, sd=1 / width)
            self.v = _normal(width, width, sd=1 / width)
        self.w1 = _normal(width, hidden, sd=1 / hidden)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = _normal(hidden, width, sd=1 / width)
        self.b2 = nn.Parameter(torch.zeros(width))

    def peers(self, y: torch.Tensor) -> torch.Tensor:
        """The attention weights: row i = how much stock i draws on each stock (rows sum to 1)."""
        return torch.softmax((y @ self.q) @ (y @ self.k).T, dim=-1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """(candidates x width) -> (candidates x width)."""
        if self.attention and self.fast:  # same formula, unscaled like the paper
            q, k, v = (y @ m for m in (self.q, self.k, self.v))
            mixed = nn.functional.scaled_dot_product_attention(
                *(t.bfloat16()[None, None] for t in (q, k, v)), scale=1.0
            )
            y = y + self.drop(mixed[0, 0].float())
        elif self.attention:
            y = y + self.drop(self.peers(y) @ (y @ self.v))
        return y + self.drop(torch.relu(y @ self.w1 + self.b1) @ self.w2 + self.b2)


class Optimizer(nn.Module):
    """A month's candidates -> their portfolio weights (module docs).

    Args:
        kind: One of :data:`MODELS`.
        width: Inputs per stock (D).
        blocks: Blocks for ``mlp`` and ``transformer`` (ignored for ``linear``).
        hidden: Feed-forward hidden width.
        fast_attention: bfloat16 fused attention (:class:`Block`).
        dropout: Dropout share in every block while training (:class:`Block`).
        speed_column: With trade speed (module docs), the index of the log-spread column; the
            input then has ``width`` + 1 columns, the last the book held going in.

    Built in ``eval`` mode: training switches dropout on with ``train()``.
    """

    def __init__(
        self, kind: str, width: int, blocks: int = 1, hidden: int = 256,
        fast_attention: bool = False, dropout: float = 0.0, speed_column: int | None = None,
    ):  # fmt: skip
        super().__init__()
        if kind not in MODELS:
            raise ValueError(f"unknown model {kind!r}; known: {MODELS}")
        n = 0 if kind == "linear" else blocks
        self.blocks = nn.ModuleList(
            Block(width, hidden, kind == "transformer", fast_attention, dropout) for _ in range(n)
        )
        self.lam = _normal(width, sd=1 / width)
        self.speed_column = speed_column
        if speed_column is not None:  # tau starts at 1/2 for every stock
            self.speed = nn.Parameter(torch.zeros(2))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(candidates x width) -> a weight per candidate."""
        held = None
        if self.speed_column is not None:
            x, held = x[:, :-1], x[:, -1]
            spread = x[:, self.speed_column]
        for block in self.blocks:
            x = block(x)
        target = x @ self.lam
        if held is None:
            return target
        tau = torch.sigmoid(self.speed[0] + self.speed[1] * spread)
        tau = torch.where(held.abs().sum() > 0, tau, torch.ones_like(tau))  # all cash: buy in full
        return held + tau * (target - held)

    def trade_speed(self, log_spread: torch.Tensor) -> torch.Tensor:
        """Tau for stocks with these log-spread inputs (trade speed only)."""
        return torch.sigmoid(self.speed[0] + self.speed[1] * log_spread)


def loss(model: Optimizer, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """(1 - portfolio return)^2 for one month (module docs)."""
    return (1 - model(x) @ r) ** 2
