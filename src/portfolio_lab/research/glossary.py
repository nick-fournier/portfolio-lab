"""Plain-language glossary of every scoreboard signal: what it measures and why it might work.

Each entry has ``what`` (the calculation in words), ``why`` (the intuition for why it might
predict returns) and ``expect`` (the direction research leads us to expect: ``higher`` or
``lower`` is better, or ``unclear``). A negative IC on a ``lower is better`` signal is the
expected result, not a failure. Industry-relative versions (``*_ind``) are described by
:func:`describe` from their base entry.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    """One glossary entry (see module docs)."""

    what: str
    why: str
    expect: str = "higher"


_MARKET_VALUE = "market value (shares outstanding times price)"

GLOSSARY: dict[str, Entry] = {
    # Valuation: fundamentals relative to price. Higher yield = cheaper.
    "earnings_yield": Entry(
        f"Net income over the last twelve months divided by {_MARKET_VALUE}; the inverse "
        "of the price/earnings ratio.",
        "Cheap relative to profits. Investors tend to overpay for glamorous stocks and "
        "underpay for dull ones, so cheap stocks have historically earned more (the value "
        "premium).",
    ),
    "book_to_market": Entry(
        f"Shareholders' equity (the balance-sheet book value) divided by {_MARKET_VALUE}.",
        "The classic value measure of academic research: cheap relative to net assets. It "
        "has worked poorly since about 2007, as more value sits in intangibles that books "
        "don't record.",
    ),
    "cf_yield": Entry(
        f"Operating cash flow over the last twelve months divided by {_MARKET_VALUE}.",
        "Value measured with cash rather than accounting profit, which is harder to dress up.",
    ),
    "fcf_yield": Entry(
        "Operating cash flow minus capital spending (the cash left after investing to keep "
        f"the business going), over the last twelve months, divided by {_MARKET_VALUE}.",
        "Cheap relative to the cash the business actually frees up for its owners.",
    ),
    "sales_yield": Entry(
        f"Revenue over the last twelve months divided by {_MARKET_VALUE}; the inverse of "
        "price/sales.",
        "Value that also works for companies with no profits yet; weaker because it ignores "
        "margins.",
    ),
    "dividend_yield": Entry(
        f"Dividends paid over the last twelve months divided by {_MARKET_VALUE}.",
        "Cash returned to shareholders; high yields can signal value, or a price that fell "
        "because the dividend is at risk.",
        "unclear",
    ),
    # Quality and the raw Piotroski inputs.
    "roa": Entry(
        "Return on assets: net income over the last twelve months divided by total assets.",
        "Profitable companies keep outperforming because investors underestimate how "
        "persistent profitability is (Piotroski signal 1).",
    ),
    "cfo_to_assets": Entry(
        "Operating cash flow over the last twelve months divided by total assets.",
        "Profitability measured in cash, which is harder to manipulate than earnings "
        "(Piotroski signal 2).",
    ),
    "accruals": Entry(
        "Net income minus operating cash flow, divided by total assets: the part of profit "
        "not yet received in cash.",
        "Profits not backed by cash tend to reverse; companies with high accruals have "
        "disappointed (Sloan, 1996; Piotroski signal 4).",
        "lower",
    ),
    "gross_profitability": Entry(
        "Gross profit (revenue minus cost of goods) divided by total assets.",
        "The cleanest profitability measure, before accounting choices lower down the income "
        "statement (Novy-Marx, 2013).",
    ),
    "operating_margin": Entry(
        "Operating income divided by revenue.",
        "Pricing power and efficiency: how much of each sales dollar survives operating costs.",
    ),
    "leverage": Entry(
        "Total liabilities divided by total assets.",
        "Debt adds risk; highly indebted firms can do worse in downturns, but the evidence "
        "is mixed.",
        "unclear",
    ),
    "current_ratio": Entry(
        "Current assets divided by current liabilities: the cushion to pay the next year's bills.",
        "Liquidity lowers the risk of distress, but a very high ratio can mean idle cash.",
        "unclear",
    ),
    "d_roa": Entry(
        "Change in return on assets from a year earlier.",
        "Improving profitability; the market is slow to price in a turnaround (Piotroski "
        "signal 3).",
    ),
    "d_lt_debt": Entry(
        "Change in long-term debt as a share of assets from a year earlier.",
        "Rising debt can signal a company that can't fund itself from operations (Piotroski "
        "signal 5).",
        "lower",
    ),
    "d_current_ratio": Entry(
        "Change in the current ratio from a year earlier.",
        "Improving liquidity (Piotroski signal 6).",
    ),
    "d_gross_margin": Entry(
        "Change in gross margin (gross profit over revenue) from a year earlier.",
        "Rising margins signal pricing power or falling costs (Piotroski signal 8).",
    ),
    "d_asset_turnover": Entry(
        "Change in revenue divided by total assets from a year earlier.",
        "Assets being used more productively (Piotroski signal 9).",
    ),
    "share_issuance": Entry(
        "Percentage change in the average number of shares outstanding from a year earlier "
        "(positive = new shares issued, negative = buybacks).",
        "Companies tend to sell new shares when they think their stock is expensive and buy "
        "back when it's cheap, and issuance dilutes existing owners (Piotroski signal 7).",
        "lower",
    ),
    "asset_growth": Entry(
        "Percentage growth in total assets from a year earlier.",
        "Fast-growing, acquisitive companies have tended to underperform as growth and "
        "empire-building disappoint (Cooper, Gulen and Schill, 2008).",
        "lower",
    ),
    "sales_growth": Entry(
        "Percentage growth in revenue over the last twelve months from a year earlier.",
        "Growth is good for a company but often already priced in; the evidence for it "
        "predicting returns is weak.",
        "unclear",
    ),
    "fscore": Entry(
        "Piotroski F-score, 0 to 9: one point each for positive profit, positive cash flow, "
        "rising return on assets, cash flow above profit, falling debt, rising liquidity, "
        "no new shares, rising gross margin and rising asset turnover, from the latest "
        "annual report.",
        "A checklist of financial health and improvement, originally used to separate the "
        "winners from the losers among cheap stocks (Piotroski, 2000).",
    ),
    # Size, price and trading.
    "log_size": Entry(
        f"Logarithm of {_MARKET_VALUE}.",
        "Historically small companies earned more (a premium for risk and neglect), though "
        "the effect has been weak for decades.",
        "unclear",
    ),
    "mom_12_1": Entry(
        "Return from 12 months ago to 1 month ago (skipping the latest month).",
        "Momentum: investors under-react to news, so past winners keep winning for a while.",
    ),
    "momentum": Entry(
        "Return from 12 months ago to 1 month ago (skipping the latest month); the same as "
        "mom_12_1.",
        "Momentum: investors under-react to news, so past winners keep winning for a while.",
    ),
    "ret_1m": Entry(
        "Return over the past month.",
        "Short-term reversal: very recent winners give some back as temporary buying or "
        "selling pressure fades.",
        "lower",
    ),
    "reversal": Entry(
        "Minus the return over the past month (so last month's losers score highest).",
        "Short-term reversal: very recent losers bounce as temporary selling pressure fades.",
    ),
    "volatility": Entry(
        "Standard deviation of daily returns over the past year.",
        "The low-volatility anomaly: calm stocks have earned about as much as wild ones, so "
        "per unit of risk they did better; investors overpay for lottery-like stocks.",
        "lower",
    ),
    "low_vol": Entry(
        "Minus the standard deviation of daily returns over the past year (calmest scores "
        "highest).",
        "The low-volatility anomaly: investors overpay for lottery-like stocks, so calm "
        "ones have done better per unit of risk.",
    ),
    "beta": Entry(
        "Sensitivity of the stock's daily returns to the market's (SPY) over the past year.",
        "The low-beta anomaly: high-beta stocks haven't earned the extra return their "
        "market risk should demand (Frazzini and Pedersen, 2014).",
        "lower",
    ),
    "log_adv": Entry(
        "Logarithm of the median daily dollar volume over the past 60 trading days.",
        "Liquidity. Illiquid stocks should pay a premium for being costly to trade, though "
        "liquid ones have done better recently.",
        "unclear",
    ),
    # Return forecasts behind mean-variance.
    "forecast (model=historical_mean)": Entry(
        "Each stock's average annual return over the past year.",
        "Pure extrapolation: assumes last year's return continues. Close to momentum, but "
        "including the latest month.",
    ),
    "forecast (model=ar1_logret)": Entry(
        "An AR(1) model on each stock's past year of daily returns: the next return predicted "
        "from the latest one. Daily returns have almost no memory, so it collapses to "
        "roughly the trailing average.",
        "Extrapolation, like the trailing average.",
    ),
    "forecast (model=arima320_price)": Entry(
        "The original optimizer's ARIMA(3,2,0) on price levels: extrapolates the recent price "
        "trend and its curvature.",
        "Trend-following on prices; the legacy model kept for comparison.",
    ),
    # Market context (per stock).
    "macro_tailwind": Entry(
        "Sum of the four tailwinds (oil, 10-year yield, dollar, credit spread).",
        "Whether the overall environment has recently moved in the stock's favor; tests "
        "whether such moves carry over or reverse.",
        "unclear",
    ),
}

_FACTORS = {
    "oil": "oil prices",
    "yield_10y": "the 10-year Treasury yield",
    "dollar": "the dollar",
    "baa_spread": "the Baa credit spread",
}
for _key, _label in _FACTORS.items():
    GLOSSARY[f"{_key}_beta"] = Entry(
        f"Sensitivity of the stock's weekly returns to weekly moves in {_label} over the past "
        "two years, net of the market's move.",
        f"Which stocks benefit or suffer when {_label} moves; mainly useful combined with "
        "what that factor is doing now (see the tailwind).",
        "unclear",
    )
    GLOSSARY[f"{_key}_tailwind"] = Entry(
        f"Sensitivity to {_label} times its move over the past three months: the boost (or "
        "drag) the recent move implies for this stock.",
        "Tests whether stocks favored by recent moves keep benefiting, or give it back.",
        "unclear",
    )

EXPECT_TEXT = {
    "higher": "higher is better",
    "lower": "lower is better",
    "unclear": "no clear prior",
}


def describe(label: str) -> Entry | None:
    """The glossary entry for a scoreboard signal label (``None`` if there is none)."""
    if label in GLOSSARY:
        return GLOSSARY[label]
    if label.endswith("_ind") and label.removesuffix("_ind") in GLOSSARY:
        base = GLOSSARY[label.removesuffix("_ind")]
        return Entry(
            f"{label.removesuffix('_ind')} as a percentile (0 to 1) among stocks in the same "
            "industry (SIC major group) that month.",
            "Compares a company with its peers, removing industry-wide differences (a "
            f"'cheap' bank and a 'cheap' software firm mean different things). {base.why}",
            base.expect,
        )
    return None
