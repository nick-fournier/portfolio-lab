"""Next-month stock forecaster: each stock's return from this month end to the next.

The front-runner model (still being tested; docs/forecaster-log.md has what else was tried),
for all stocks, refit every month on all earlier months:

1. ``dataset``: 103 stock inputs as within-month percentiles; the target is the return
   minus the month's average across stocks, capped at the month's 0.1%/99.9% for fitting.
2. ``linear`` (part 1): PLS on the inputs and each input · VIX and · last month's return
   dispersion; components chosen by k-fold over past calendar years.
3. ``trees`` (part 2): boosted trees on part 1's residual, from the stock inputs and the
   market's (environment and market state).
4. ``walk``: forecast = part 1 + k · part 2, k chosen each month from earlier months' error.
5. ``grade``: graded on unseen months.
"""
