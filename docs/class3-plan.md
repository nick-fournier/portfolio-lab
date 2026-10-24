# Class 3 research plan: transformers

Status: draft for discussion (2026-10-11). Nothing built. Issue #53.

## Goal

Beat the Forecaster (class 2, `forecasters.linear` + Grinold + Kelly) with transformers, along two
tracks that are built to be combined:

- **Optimizer track:** one model reads every stock's characteristics and outputs the portfolio
  weights directly, trained to maximize Sharpe. It replaces forecast + Grinold + Kelly.
- **History track (AR):** attention over each stock's own past months, so the model conditions on
  what just happened (including recent surprises) every time it runs, without a refit.

**The bar:** the Forecaster's backtest (28.7% a year 2004-2026, Sharpe 0.91), beaten in both halves
(2004-15, 2016-26), net of costs. Long-only: no shorting.

## The paper this starts from

Kelly, Kuznetsov, Malamud & Xu, "Artificial Intelligence Asset Pricing Models" (NBER w33351, rev.
June 2026).

- **Inputs:** each month, every stock's 132 characteristics (value, momentum, size, volatility,
  profitability, ...), ranked across stocks to [-0.5, 0.5], missing = 0. Stock data only, no
  market data. No sequences: history enters only through characteristics such as 12-month momentum.
- **Model:** K blocks. Each block: attention across stocks (each stock blends in the stocks it
  learns to treat as peers), then a per-stock feed-forward layer; residual connections. 1 head
  (more helped little); width = the 132 inputs; feed-forward 256 wide.
- **Output:** a weight for every stock (long or short). No forecast.
- **Loss:** average over training months of (1 - portfolio return)^2. Minimizing it gives the
  maximum-Sharpe portfolio (Britten-Jones 1999).
- **Training:** 60-month rolling window, retrained every month, 10 seeds averaged.
- **Results (Sharpe 1968-2022 / post-2002):** linear weights 3.60 / 2.03; linear attention 3.89 /
  2.25; deep own-stock network, no attention 4.31 / 3.07; transformer (10 blocks) 4.57 / 3.37.
  1 block: 3.8. Gains grow with depth.
- **Lessons:** most of the gain is depth and nonlinearity; attention adds on top, and only when
  deep. The fair test of attention is the same network with attention removed. Breadth of inputs
  matters (linear on 132 characteristics: 3.6 vs 1.0 for the six-factor Fama-French model).
  Attention mostly learned "similar stocks": shared size and liquidity group, industry, patents,
  return correlation.
- **Caveats for us:** long-short, all stocks including micro caps, no costs; Sharpe highest in micro
  caps (mega caps 1.84). The transformer's relative edge is largest in large and mega caps.

## What we already know (don't relearn)

- Earlier own-stock neural nets on our inputs landed level with the linear model (IC 0.015-0.030);
  market inputs fed in directly overfit. Averaging nets with linear gave the best IC (0.036).
- A stock's own return history adds little: AR(1) was dead. So the history track uses the full
  characteristic history, not just past returns.

## Shared design

- **Universe:** the Forecaster's (price >= $1, >= $100k a day).
- **Inputs:** every stock characteristic in the hive (about 100; counted in stage 0), ranked across
  stocks each month to [-0.5, 0.5], missing = 0. Momentum and reversal enter as characteristics.
- **Weights:** long-only, fully invested (a softmax over stocks: positive, summing to 1). The only
  hard constraint is no shorting. The 10% cap and a cash option are tests, kept only if they help.
- **Costs:** charged in the backtest; added to the loss if turnover turns out to matter.
- **Training:** walk-forward. First fit, then a warm start each month (continue from last month's
  weights for a few epochs). Window: 60-month rolling (paper) vs expanding, tested once.
  Production can afford full retrains if the model is promoted.
- **Inference:** one fast pass each rebalance; new data changes the output without retraining.
- **Start small, scale on evidence:** 1-2 blocks, one seed while screening; deeper, more seeds only
  after the small version beats its baseline. First run measures the time per refit.
- **Compute:** cubevm's RTX 5060 Ti (16 GB), shared, so runs are bounded and one at a time.
  Scratch copies of data only.

## Optimizer track (first)

The paper's model on our data, long-only.

| Step | Model (same inputs) | Tests |
|---|---|---|
| O1 | Linear weights: weights from a linear score of each stock (no attention, no depth) | baseline |
| O2 | Same network as O3 with attention removed (per-stock only) | does depth / nonlinearity help over O1? |
| O3 | Transformer, 1-2 blocks | does attention help over O2? |
| O4 | O3 deeper (more blocks), more seeds | does it scale? |
| O5 | O3/O4 with a cash option, with the 10% cap | do the constraints help? |

Graded by backtest (CAGR, Sharpe, max drawdown, both halves, years won) against each other and the
Forecaster. About 260 test months, so seeds matter: a gain must beat seed-to-seed spread.

**Deliverable after O3:** a deep, plain-language explanation of how the trained model works (what
attention learned, which stocks it treats as peers, how inputs map to weights), on a page.

## History track (second)

| Step | Model | Tests |
|---|---|---|
| H1 | AR forecaster: time attention over each stock's past months of characteristics, predicting next month's return (cross-sectionally ranked) | IC vs the Forecaster and ridge |
| H2 | H1's forecast fed to the optimizer as one more input | does it add to the optimizer? |

## Combining the tracks (third)

From loosest to tightest; each compared with the previous:

1. **Feed in** (H2): AR forecast as an optimizer input. Each part graded alone.
2. **End to end:** per-stock time attention -> a summary per stock -> cross-stock attention ->
   weights, all trained together on the Sharpe loss.
3. **Fully unified:** every stock-month a token; layers alternate attention across time (within a
   stock) and across stocks (within a month) ("axial attention"), so history and peers inform each
   other at every layer.

For 2 and 3: train on two losses at once, Sharpe plus a forecast loss on each stock's next-month
return. The Sharpe loss gets one number per month; the forecast loss gets ~1M stock-months, which
keeps the history part from overfitting and gives a forecast to grade by IC.

## Stages and stop rules

| Stage | Work | Continue only if |
|---|---|---|
| 0 | Data: monthly arrays (ranked characteristics, history sequences), point-in-time checks | arrays reproduce the Forecaster's inputs on overlapping terms |
| 1 | Optimizer O1-O3 | O3 beats O2 and O1 in both halves |
| 2 | Optimizer O4-O5; explanation page | gains hold beyond seed noise |
| 3 | History H1-H2 | H1 IC beats the Forecaster's, or H2 adds to the optimizer |
| 4 | Combination 1 -> 2 -> 3 | each beats the previous |
| 5 | Weekly / daily | only after a monthly model beats the Forecaster |

Every stage's settings are agreed before running; results go on a page.

## Open questions

- Window: 60-month rolling vs expanding.
- Whether to add daily-derived characteristics (volatility, max daily return).
- Market data: the paper uses none; ours could enter later as interactions (an ablation).
