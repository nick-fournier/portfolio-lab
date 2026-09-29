from pathlib import Path

import polars as pl
import pytest

from portfolio_lab.data.sources.universe import (
    classify,
    exclude_reason,
    normalize_symbol,
    parse_nasdaq_listed,
    parse_other_listed,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def universe() -> dict[str, str | None]:
    nasdaq = parse_nasdaq_listed((FIXTURES / "nasdaqlisted.txt").read_text())
    other = parse_other_listed((FIXTURES / "otherlisted.txt").read_text())
    df = classify(pl.concat([nasdaq, other]))
    return dict(zip(df["symbol"], df["exclude_reason"], strict=True))


def test_footer_is_dropped(universe):
    assert not any(s.startswith("File Creation") for s in universe)


@pytest.mark.parametrize(
    ("symbol", "reason"),
    [
        # Kept: common stocks, share classes, ADRs, and names that merely look risky.
        ("AAPL", None),
        ("A", None),
        ("BRK.B", None),
        ("BF.B", None),
        ("AACG", None),  # ADR
        ("RLX", None),  # ADR "representing the right to receive" is not a right
        ("AMX", None),  # same
        ("PFBC", None),  # "Preferred Bank" common stock
        ("OBAI", None),  # "Our Bond, Inc." common stock
        # Excluded by directory flags and exchange.
        ("AAAP", "etf"),
        ("ZVZZT", "test_issue"),
        ("SPY", "etf"),
        # Excluded by symbol suffix or NASDAQ fifth letter.
        ("ABR$D", "preferred"),
        ("AAC.U", "unit"),
        ("AAC.W", "warrant"),
        ("AIIA.R", "right"),
        ("QUMSR", "right"),
        ("SCATU", "unit"),
        # Excluded by security name.
        ("ACGLN", "fixed_income"),  # 4.550% ... Preferred Share
        ("CHSCN", "preferred"),  # "- Preferred Class B"
        ("AMPGZ", "right"),  # "Series B Right"
        ("RWAYL", "fixed_income"),  # 7.50% Notes due 2027
        ("ABXL", "fixed_income"),  # 9.875% Senior Notes due 2028
        ("EAI", "fixed_income"),  # First Mortgage Bonds, 4.875%
        ("OXLC", "fund"),  # Closed End Fund
        ("BHK", "fund"),  # Core Bond Trust
    ],
)
def test_classification(universe, symbol, reason):
    assert universe[symbol] == reason


def test_other_exchanges_are_excluded():
    assert exclude_reason("XYZ", "Some Corp Common Stock", "Z", False, False) == "exchange"


def test_debt_without_coupon_in_name():
    assert exclude_reason("XYZ", "Some Corp Senior Notes due 2030", "N", False, False) == "debt"


def test_normalize_symbol():
    assert normalize_symbol(" brk.b ") == "BRK.B"


def test_classify_dedupes_and_sorts():
    rows = pl.DataFrame(
        {
            "symbol": ["MSFT", "aapl", "AAPL"],
            "name": ["Microsoft Common Stock", "Apple Common Stock", "Apple Common Stock"],
            "exchange": ["Q", "Q", "Q"],
            "etf": [False, False, False],
            "test_issue": [False, False, False],
        }
    )
    out = classify(rows)
    assert out["symbol"].to_list() == ["AAPL", "MSFT"]
    assert out["included"].all()
