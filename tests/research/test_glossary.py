import pytest

from portfolio_lab.jobs.tasks import SCOREBOARD_SIGNALS, signal_label
from portfolio_lab.research.glossary import EXPECT_TEXT, describe


@pytest.mark.parametrize("label", sorted({signal_label(n, p) for n, p in SCOREBOARD_SIGNALS}))
def test_every_scoreboard_signal_is_explained(label):
    entry = describe(label)
    assert entry is not None, f"add {label!r} to research/glossary.py"
    assert len(entry.what) > 20 and len(entry.why) > 20
    assert entry.expect in EXPECT_TEXT


def test_industry_relative_entries_build_on_the_base():
    entry = describe("roa_ind")
    assert "percentile" in entry.what and "same industry" in entry.what
    assert entry.expect == describe("roa").expect
    assert describe("no_such_signal") is None
