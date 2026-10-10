# Class 3 research plan: a transformer forecaster

Status: draft for discussion (2026-10-10). Nothing built. Issue #53.

## Goal

Forecast every stock's next-month return over the T-bill, like class 2, but let the model learn
(a) each stock's own history and (b) how stocks inform each other, instead of hand-built terms.
It plugs into the same strategy (health pool, Kelly, 10% cap, bear switch) through Grinold's rule,
so it is judged exactly like the Forecaster.

**The bar:** the Forecaster's (class 2, `forecasters.linear`) out-of-sample rank IC (about 0.05 in the full pool, 0.034 in the
400 most liquid) and its backtest (28.7% 2004-2026, Sharpe 0.91). Class 3 is promoted only if it
beats both, in both halves (2004-15, 2016-26), net of costs.

## What we already know (don't relearn)

- Earlier neural nets on these inputs reached IC 0.015-0.030, about level with the linear model;
  market inputs fed in directly were the main source of overfitting. Market data helped only as
  interactions with stock data. Averaging nets with the linear model gave the best IC (0.036).
- The Forecaster's value over plain momentum is about 1.5 points a year, and only inside the
  volatility-weighted Kelly pipeline; as a plain top-10 ranking it is worse than momentum.
- Kelly, Kuznetsov, Malamud & Xu (2025, "Artificial Intelligence Asset Pricing Models"): attention
  **across stocks** at each month on 132 standard characteristics. Out-of-sample Sharpe 3.6 for a
  linear model, 3.9 with attention added (long-short, all US stocks incl. micro caps, no costs). No
  attention over time; history enters through the characteristics. So cross-stock attention is the
  documented gain; attention over time is untested.

## Design

### Frequency

Monthly first (decisions, targets and inputs at month ends). Weekly and daily only if monthly
shows signal beyond the Forecaster.

### Target

Next month's return over the T-bill, ranked or standardized across stocks each month (a
cross-sectional target, so additive market moves cancel and only relative performance is learned).
Same universe as the Forecaster's fit: price >= $1, >= $100k a day.

### Inputs (every one normalized; never raw prices)

| Block | What | Transform |
|---|---|---|
| Own history (sequence) | monthly log returns, all available months (length is an ablation, below) | each divided by the stock's trailing volatility; missing months masked |
| Stock characteristics (this month) | the Forecaster's fundamentals (cash flow / assets, FCF yield, sales yield, earnings yield, R&D / market value) plus size, liquidity, volatility | ranked across stocks each month, mapped to [-1, 1] |
| Market / macro (this month) | SPY 12-month return and the context set | standardized over time; enters only through interactions (a gate or a market token the stocks attend to), never added directly |

Redundancy rule: the hand-built momentum terms (12m, 6m) and the SPY interactions are what
attention should learn. They are kept only in the baseline and in one ablation (below), so we can
see whether attention replaces or adds to them.

### Architecture options (tested in this order)

1. **Cross-stock only** (as Kelly et al.): each stock's token = its characteristics plus a summary
   of its own history (a small per-stock encoder); stocks attend to each other within the month.
2. **Own history only**: attention over each stock's past months (an autoregressive head), no
   cross-stock attention. Compare with the momentum terms it should replace.
3. **Both**: per-stock history encoder, then cross-stock attention.

Small models only (1-2 layers, a few heads, width 32-64), heavy dropout and weight decay: about
3,500 stocks x ~300 months is little data for a transformer.

### Training protocol

- **Start small, scale on evidence.** Every stage begins with the cheapest version that can show
  signal (tiny model, one seed, IC only); bigger models, more seeds, the full backtest come only
  after the small version beats its baseline. No long runs on a guess.
- Walk-forward over expanding windows, predicting the next month. First forecast after 10 years
  of training data. **Warm start:** train once on the first window, then each month continue from
  last month's weights for a few epochs on the window with the new month added (the class 2 nets
  did this; their full 2009-26 run took about 18 minutes on cubevm).
- If class 3 earns promotion, production can afford a full retrain (from scratch, more seeds)
  each month; research doesn't need it.
- Compute: cubevm's RTX 5060 Ti (16 GB), shared with another use, so runs are bounded and one at
  a time. The data is small (about 300 months x 3,500 stocks, ~1M rows); one-month attention
  across ~3,500 stocks is ~12M pairs per head. First run measures the actual time per refit.
- Validation: the last 2 years of each training window, for early stopping and the few settings.
  Settings chosen once on 1998-2010 windows and then frozen; never tuned on the test years.
- Seeds: one while screening; several, averaged, once a model shows signal (transformers are
  noisy, so a gain must beat seed-to-seed spread before it counts).
- GPU on cubevm; nothing touches the live data (scratch copies only).

### Grading (same as the Forecaster)

1. Forecast quality, unseen months: rank IC (full pool and the 400 most liquid), its t-statistic,
   years with positive IC, forecast-vs-outcome slope, top-minus-bottom tenth.
2. Backtest through the same strategy (Grinold's rule, Kelly risk aversion 1/16, 10% cap, bear
   switch), 2004-2026: CAGR, Sharpe, max drawdown, both halves, years won vs the Forecaster.

### Ablations (one change each)

- Cross-stock attention off (each stock alone): is the cross-sectional gain there for us?
- Own-history attention off: does it add to the momentum terms?
- Momentum terms added back as inputs: does attention replace them or add to them?
- Market block removed: does conditioning help or overfit, as it did for the nets?
- Average with the Forecaster's forecast (50/50): as with the nets, the blend may be the win.
- History length: 24 months vs 60 months vs all available. At monthly frequency length costs
  nothing to compute (attention grows with length squared, and even 300 months is tiny); the
  question is statistical: older history may add noise and shifting regimes, and stocks with short
  histories see less. Compute only matters at daily frequency (~5,000 days per stock).

## Stages and stop rules

| Stage | Work | Continue only if |
|---|---|---|
| 0 | Data: monthly arrays (history sequences, ranked characteristics, market block), point-in-time checks | arrays reproduce the Forecaster's inputs on overlapping terms |
| 1 | Ridge on the same inputs (seconds), then a tiny option 1 (cross-stock), one seed, warm start, IC only | ridge reproduces the Forecaster's IC; the tiny model beats ridge in both halves |
| 1b | Scale option 1 up (width, layers, seeds) | gains hold or grow beyond seed noise |
| 2 | Options 2 and 3 | adds to stage 1 beyond seed-to-seed noise |
| 3 | Best model through the strategy | beats the Forecaster's backtest in both halves |
| 4 | Weekly / daily | only after stage 3 passes |

Every stage's settings are agreed before running; results go on a page.

## Open questions

- Whether to include daily-derived features (volatility, max daily return) monthly.
- Ranked target vs raw excess return (ranked is more robust; raw keeps size information for Kelly).
- Whether to add the 132-characteristic style breadth later (we have about 100 stock inputs in the
  hive; most were dead ends for the linear model, but attention may use them differently).
