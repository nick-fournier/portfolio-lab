"""Class 3, optimizer track: a transformer that outputs portfolio weights (docs/class3-plan.md).

After Kelly, Kuznetsov, Malamud & Xu (2026), "Artificial Intelligence Asset Pricing Models":
each month every candidate stock's characteristics go in, attention lets each stock blend in
the stocks it learns to treat as peers, and the model outputs one weight per stock, trained on
the portfolio's realized returns (loss: mean of (1 - portfolio return)^2) rather than on a
forecast. Long or short, any total size; per-stock trading and borrowing costs can enter the
loss.

- :mod:`.data`: the monthly table (characteristics, eligibility, liquidity, next month's
  return) and the per-month arrays the models train on.
- :mod:`.model`: the three models compared (linear, the network without attention, the
  transformer).
- :mod:`.walk`: walk-forward training (a fresh fit each month, or a warm start from the last).
- :mod:`.costs`, :mod:`.priced`, :mod:`.realism`, :mod:`.holdings`: costs and the account's
  limits in the loss; holdings and spreads as inputs.
- :mod:`.stopping`, :mod:`.averaging`, :mod:`.clipping`: held-out months, the parameter
  average, gradient clipping.
- :mod:`.ridge`: the paper's linear benchmark.

Research code: torch is needed (``uv sync --extra nets``), and runs go on the GPU host.
"""
