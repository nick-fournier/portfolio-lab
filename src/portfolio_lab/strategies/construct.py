"""Portfolio construction rules: turn a set of symbols or scores into weights."""

from collections.abc import Sequence

from portfolio_lab.strategies.base import Weights


def equal_weight(symbols: Sequence[str]) -> Weights:
    """Weight each symbol equally, fully invested; empty input gives an all-cash portfolio."""
    unique = list(dict.fromkeys(symbols))
    return {s: 1.0 / len(unique) for s in unique} if unique else {}


def top_k_equal(scores: dict[str, float], k: int) -> Weights:
    """Equal-weight the ``k`` highest-scoring symbols (ties broken by symbol name)."""
    ranked = sorted(scores, key=lambda s: (-scores[s], s))
    return equal_weight(ranked[:k])
