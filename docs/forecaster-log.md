# Forecaster log

What was tried for the next-month stock forecaster (`forecasters`), and why each
variant is or is not the front runner. Newest decisions first within each section.

## How variants are judged

These rules judged the previous forecaster's variants. The nine-term model was judged on the
portfolio yardstick described in its section.


- **Target:** each stock's return from one month end to the next, minus that month's
  average across stocks. All stocks. Capped at the month's 0.1%/99.9% for fitting.
- **Graded:** unseen months only, 2009-01 to 2026-08 (212 months), every model refit each
  month on all earlier months. Unless noted.
- **Measures:** IC (rank correlation of forecast and outcome, averaged over months) and
  its t-statistic; years with a positive IC (of 18); slope of outcome on forecast across
  20 forecast groups (1 = right-sized forecasts); R² against forecasting the average;
  best-minus-worst forecast tenth, per year.
- **Noise:** the same model rerun with 5 different tree seeds moves IC by ±0.0022 (sd),
  slope ±0.04, best−worst tenth ±0.9%/yr. A difference under about **0.004 in IC** or
  **2%/yr in the tenth gap** is not evidence of a better model.

## Front runner (since 2026-10-07): least squares on nine terms

`forecasters.nine`, run with `plab forecast nine`. Rebuilt from scratch after the
previous forecaster (below) lost to plain momentum inside the portfolio: it gave momentum one
small vote among 103 inputs, so its top picks among the liquid stocks earned less than the
trailing-return ranking it was meant to beat.

- **Target:** next month's return minus that month's T-bill, uncapped (what the optimizer
  needs as an expected return). All stocks with price ≥ $1 and ≥ $100k a day (about 3,400 a
  month).
- **Terms:** one-year and six-month momentum, cash flow / assets, FCF, sales and earnings
  yields, R&D / market value, and each momentum × SPY's one-year return. Each stock input is
  `asinh(x / median|x|)`, standardized per month (R&D / market also clipped at 1/99%).
- **Fit:** ordinary least squares with an intercept, every earlier month, refit monthly.
- **Yardstick (how terms were admitted):** equal-weight top 30 among the 400 most liquid,
  2004-01 to 2026-08 (272 months), against momentum's top 30 (13.7%/yr). A term stayed only
  if it raised the top 30 or the top 100/spread and lost nothing in either half
  (2004–15, 2016–26); R&D / market and momentum × SPY were admitted as reviewed exceptions
  (each gained overall but lost a little in one half). Result: 20.9%/yr (+8.1 and +6.0 points in the two halves).

Graded on the page (2009-01 to 2026-08, stock-months all three cover):

| Model | Top 30 of 400 /yr | IC, 400 most liquid | IC, all | Slope | Tenth /yr |
|---|---|---|---|---|---|
| **Nine terms** | 27.8% | 0.034 | 0.048 | 0.66 | 30.0% |
| Previous forecaster | 21.9% | 0.029 | 0.058 | 0.93 | 38.3% |
| Production: trailing one-year return | 19.3% | 0.006 | 0.027 | 0.06 | 5.5% |

The previous forecaster ranks all stocks better, almost entirely in the smallest third (IC
0.091 vs 0.063), which the strategy never buys.

### Rebuild, 2026-10-05 to 10-07: what didn't make it

- **Extra stock inputs added one at a time** (the 30 reviewed plus 77 more, each transformed
  by its distribution): only R&D / market passed. Additive inputs added nothing beyond the
  core.
- **Interactions:** trees on the residual (three times) found no strong stock × stock
  interactions (the strongest were flag artifacts). Stock × market interactions were real only
  for momentum (credit spread, SPY one-year, market volatility); SPY one-year subsumes the rest.
- **Macro and market-state terms** (credit spread, VIX, bear and rebound flags, PCA axes of
  the environment): nothing beyond momentum × SPY.
- **Momentum windows** of 3 and 24 months, a symmetric six-month × six-month SPY term; size,
  liquidity and volatility as inputs; an asinh or volatility-scaled target; elastic net (its
  squared-error selection keeps nothing); residual PLS; sector terms.

### The previous forecaster (kept for comparison on the page)

Two models, each centered within the month, averaged 50/50:

1. **Linear + trees:** PLS (1 component, chosen by k-fold over past years) on the 103 inputs
   and each input × VIX and × last month's return dispersion, plus XGBoost trees on its
   residual (103 stock + 115 market inputs; 31 leaves, ≥2,000 stock-months per leaf, learning
   rate 0.05, 400 rounds) at a strength chosen walk-forward (0.25 every month).
2. **Bilinear nets:** a network turns each stock's 103 inputs into 8 numbers; all 116 market
   inputs (scaled by past months) tilt the weights on those numbers each month, with a ridge
   penalty of 1 on the 8 × 116 market weights; output centered within the month; 5 nets,
   warm-started monthly.

| Model | IC | t | Years | Slope | R² | Tenth /yr |
|---|---|---|---|---|---|---|
| **Forecast (average)** | 0.037 | 4.5 | 16 | 0.63 | +0.10% | 10.5% |
| Linear + trees alone | 0.030 | 3.3 | 15 | 0.57 | +0.06% | 10.5% |
| Nets alone | 0.033 | 4.9 | 17 | 0.43 | −0.03% | 10.0% |
| Production today: AR(1) on trailing returns | 0.004 | 0.4 | 13 | 0.02 | −8.09% | −0.5% |

The average's gain over linear + trees held across 4 net seeds (±0.0013 IC) and penalties
1–100. A year's IC moves by about ±0.035 by chance at this level of skill (monthly IC sd
0.12), which matches the actual spread of yearly ICs: good and bad years are mostly luck.

Grinold's rule (rescale by the outcome-on-forecast slope of earlier months) was tested and not
applied: every walk-forward version lowered R² (all earlier months +0.094%, last 120 months
+0.096%, last 60 +0.056%, last 36 +0.040%, vs +0.099% unscaled); scales anchored to the badly
sized early years (~0.4). Forecasts are too big in weak years and about right in strong ones
(slope 0.25–0.46 in 2009–2020, about 1.1 in 2021–2026), and their size does not widen with
market conditions.

## Tried and not adopted

### 2026-10-03/04: run list T1–T4, T12 (code on branch `forecast-experiments`; plots: https://claude.ai/artifact/RUdTseT9iZrpnpD12TeTfC)

| Variant | IC | t | Years | Slope | R² | Tenth /yr | Why not |
|---|---|---|---|---|---|---|---|
| T1: 5 seeds, trees draw 80% rows + 80% columns, each run | 0.029–0.035 | 3.2–3.7 | 15–16 | 0.57–0.67 | +0.05–0.08% | 10.3–12.3% | Measures noise (see above) |
| T1: average of the 5 seeds' trees | 0.033 | 3.5 | 16 | 0.65 | +0.07% | 11.3% | Gain within noise; 5× the research runtime. Revisit for production, where it is cheap |
| T2: PLS components fixed at 3 | 0.023 | 2.9 | 13 | 0.31 | −0.24% | 11.1% | Lower IC, forecasts ~2× too big |
| T2: PLS components fixed at 5 | 0.026 | 3.9 | 16 | 0.26 | −0.38% | 11.3% | Same; a richer linear base does not help under the trees |
| T3: trees alone, no linear part (+ dispersion input) | 0.021 | 2.4 | 15 | 0.24 | −0.66% | 10.9% | IC 0.009 lower (beyond noise); large-stock IC 0.008. The linear base does real work |
| T4: neural nets alone (NN3 32-16-8, 5 nets, warm start monthly), first attempt | 0.008 | 0.8 | 12 | 0.01 | −1.24% | −1.5% | Invalid: nets never trained (early stopping on 2 noisy held-out years stopped at pass 0–1; untrained nets output noise) |
| T4, training fixed (output layer starts at 0; held-out = random 20% of rows) | 0.015 | 1.6 | 13 | 0.07 | −2.50% | 1.5% | Overfits: ranks a little, forecasts ~14× too big. Untested suspects: warm start piling passes onto old months; month-constant market inputs |
| T4 diagnostic: nets on stock inputs only (no market inputs), warm start | 0.024 | 4.8 | 15 | 0.25 | −0.30% | 7.6% | The month-constant market inputs drove the overfitting; still below the front runner on IC, size and spread |
| T4 diagnostic: stock + market inputs, fresh nets once a year | 0.020 | 2.0 | 11 | 0.16 | −1.69% | 9.1% | Worse than warm start: warm start was not the problem |
| T12: XGBoost trees refit each January (linear part monthly) | 0.028 | 2.9 | 15 | 0.36 | −0.09% | 3.9% | Worse than monthly refits; the walk-forward strength often dropped the trees (0–0.25). The yearly scikit-learn run's 0.033 is within noise, so neither the library nor yearly refits is worth adopting |
| Diagnostic: front runner with the trees given stock inputs only (no market inputs) | 0.027 | 3.3 | 16 | 0.47 | −0.02% | 7.4% | Worse: the tenth gap drops 3.1%/yr (beyond noise) and forecasts are worse-sized. The trees' market inputs do real work; in fits each January 2010–2026 they account for 52–63% of the trees' improvement (top: 10-year yield, credit spread, 2-year yield, absorption, oil) |
| Conditioned nets, stock inputs only, month-wise batches, output centered per month (4 or 8 encoding numbers, picked monthly by held-out error) | 0.026 | 5.4 | 15 | 0.30 | −0.16% | 7.6% | Steadiest IC of any model (t 5.4) but below the front runner on IC, size and spread |
| Conditioned nets, bilinear on VIX + dispersion (4 settings, picked monthly) | 0.024 | 4.3 | 15 | 0.25 | −0.42% | 9.1% | No better than stock-only; the pick favored the light market penalty, which graded worse |
| Conditioned nets, FiLM on VIX + dispersion (4 settings, picked monthly) | 0.026 | 5.2 | 15 | 0.31 | −0.17% | 8.4% | No better than stock-only |
| Conditioned nets, bilinear on market PCA factors (4/8/15 factors × penalty 0.1/1.0; best: 8 factors, 1.0) | 0.029 | 5.2 | 15 | 0.30 | −0.15% | 9.2% | Strong penalty wins at every size; picked monthly by earlier accuracy: IC 0.027 |
| Conditioned nets, bilinear on all 116 market inputs uncompressed, penalty 1.0 | 0.030 | 5.0 | 17 | 0.43 | −0.05% | 9.8% | Ties the front runner on IC, steadier (17/18 years), but forecasts too big (slope 0.43) and a slightly smaller spread; best of 7 tries. Candidate to build on, not adopted |
| Average: front runner + all-116 bilinear nets, each centered per month, 50/50 (weight chosen walk-forward from 0–1 picked 0.5 every month) | 0.036 | 4.5 | 16 | 0.63 | +0.09% | 10.7% | Best so far: +0.006 IC over either alone (beyond noise), best slope and R²; the two forecasts correlate 0.51. Caveat: the nets are the best of 7 tries |
| Same average with the nets at penalty 3 / 10 / the average of the 3 nets | 0.035 / 0.038 / 0.037 | 4.6 / 4.9 / 4.7 | 17 / 17 / 17 | 0.59 / 0.54 / 0.60 | +0.06% / +0.08% / +0.08% | 10.4% / 11.3% / 10.5% | The average holds whichever nets setting is used: not a lucky pick |
| Same average, nets at penalty 1 with seeds 1 / 2 / 3; and at penalty 30 / 100 | 0.038 / 0.035 / 0.037; 0.035 / 0.034 | 4.7 / 4.5 / 4.7; 4.7 / 4.8 | 17 / 15 / 17; 16 / 17 | 0.58 / 0.58 / 0.56; 0.56 / 0.58 | +0.06% / +0.05% / +0.06%; +0.06% / +0.07% | 10.7% / 11.0% / 10.1%; 10.9% / 11.1% | Seed noise of the average: ±0.0013 IC. Nets alone: ±0.0024 between seeds, as big as the differences between penalties 1–10, so the penalty is fixed at 1; 30–100 drift toward the stock-only nets |
| All-116 bilinear nets alone, penalty 3 / 10 | 0.027 / 0.032 | 4.2 / 5.0 | 16 / 15 | 0.36 / 0.32 | −0.13% / −0.11% | 9.2% / 9.9% | Penalty above 1 makes no clear difference |
| Linear part + all-116 nets on its residual (in place of the trees) | 0.029 | 3.4 | 17 | 0.54 | −0.02% | 7.2% | Narrower spread than linear + trees: the trees are the better partner |
| Linear + all-116 nets (50/50, no trees) | 0.035 | 4.4 | 17 | 0.59 | +0.02% | 8.6% | Same ranking as with the trees, but 2%/yr less spread and lower R²: the trees add the extremes, not ranking (averaging more net runs matches their IC gain, not their spread). Their gain is not concentrated in stocks with missing inputs |
| Stack: (linear + all-116 nets) as main model, trees on its residual, strength walk-forward | 0.030 | 3.5 | 16 | 0.50 | +0.06% | 8.7% | Worse than the main model alone (0.035): after the nets, the residual is mostly noise. The trees work as an independent model next to the nets, not as a correction on top |

Linear part alone with 1–10 components (no trees): more components give a steadier but
lower IC (0.029 → 0.023; t 3.1 → 4.0) and a wider tenth gap (5% → 9%), but forecasts too
big (slope 0.49 → 0.18). The k-fold picks 1 component in every month because it scores
squared error.

### 2026-10-02/03: building the front runner (2009–2026 unless noted)

| Variant | IC | t | Years | Slope | R² | Tenth /yr | Why not |
|---|---|---|---|---|---|---|---|
| PLS only (103 inputs, no interactions) | 0.031 | 3.7 | 14 | 0.27 | −0.11% | 3.0% | Forecasts ~4× too big; the old baseline |
| + interactions with VIX and dispersion (model B) | 0.029 | 3.0 | 15 | 0.49 | −0.06% | 5.2% | Became the front runner's linear part |
| B with the interactions scaled instead of PLS'd | 0.024 | 2.8 | 14 | 0.48 | −0.04% | 0.5% | Lower IC and spread |
| Interactions with 6 market inputs | 0.008 | 0.8 | 11 | 0.08 | −0.56% | 0.9% | Overfits |
| 6 market inputs, scaled | 0.012 | 1.4 | 11 | 0.32 | −0.04% | 2.7% | Overfits |
| Recency weighting, half-life 36 months | 0.010 | 1.2 | 13 | −0.04 | −0.23% | −3.9% | Discards history the model needs |
| Recency weighting, half-life 60 months | 0.018 | 2.2 | 15 | 0.06 | −0.15% | −2.1% | Same |
| Recency weighting, half-life 120 months | 0.026 | 3.1 | 14 | 0.19 | −0.12% | 0.2% | Same |
| Trees (scikit-learn) refit each January on B, strength chosen walk-forward | 0.033 | 3.4 | 15 | 0.50 | −0.03% | 8.2% | Within noise of the front runner, and the model must refit monthly. (At a fixed 0.25, picked with hindsight: IC 0.038, slope 0.58, tenth 11.6%.) |
| Same, strength 0.5 / 1.0 | 0.036 / 0.030 | 4.1 / 3.4 | 15 / 15 | 0.42 / 0.22 | −0.05% / −0.72% | 13.0% / 12.1% | Strengths above 0.25 make forecasts too big |
| Trees (scikit-learn) refit each January on PLS only, strength 0.25 | 0.037 | 4.5 | 15 | 0.41 | −0.01% | 7.9% | Superseded by trees on B |
| B + trees refit monthly, strength 0.5 / 1.0 | 0.028 / 0.022 | 3.3 / 2.7 | 16 / 15 | 0.40 / 0.22 | −0.04% / −0.79% | 11.2% / 11.7% | Same; 0.25 is chosen walk-forward |
| Forecaster inside production's 100-stock health pool | 0.013 | | | | | | Below trailing returns (AR(1), 0.018) in that pool: the forecaster is for picking from the whole market |

### 2026-10-02: linear methods and tuning (graded 2004–2026, 23 years)

| Variant | IC | t | Years | Slope | Tenth /yr | Why not |
|---|---|---|---|---|---|---|
| 103 inputs, least squares | 0.024 | 4.5 | 18/23 | 0.23 | 8.6% | Forecasts far too big (R² −0.40%) |
| 103 inputs, true Huber | 0.026 | 4.6 | 18/23 | 0.23 | 8.6% | Same as capping the target; capping kept |
| 103 inputs, shrink each weight by its certainty | 0.026 | 4.6 | 18/23 | 0.25 | 6.5% | Below PLS |
| 103 inputs, LASSO | 0.016 | 2.7 | 15/23 | 0.41 | 3.7% | Drops inputs that matter |
| 103 inputs, PCA regression | 0.028 | 3.6 | 18/23 | 0.28 | 1.2% | Below PLS on the tenth gap |
| 103 inputs, PLS | 0.029 | 4.2 | 19/23 | 0.27 | 4.7% | Adopted |

Tuning the component count: choosing on the latest 12 months was too noisy; forward
windows (last 60/120 months) picked sparse models; k-fold over all past calendar years
was kept. Training on all past months beat the latest 60 months.
