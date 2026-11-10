"""The small statistics the reports here use; nothing else computes a test or a summary."""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Callable, Iterable


def exact_binomial_p(a: int, b: int) -> float:
    """Two-sided exact binomial test of a : b against 1 : 1.

    On discordant pairs (parent right & trap wrong vs the reverse) this is the exact McNemar
    test; on items leaning one way or the other it is the sign test.
    """
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def majority(votes: Iterable[bool], ties: bool = True) -> bool:
    """Majority of the available votes. `ties` decides a tie: repetitions of a verdict count a
    tie as a pass (a configuration with two reps split 1-1 is not scored a failure)."""
    votes = list(votes)
    return sum(votes) * 2 > len(votes) or (ties and sum(votes) * 2 == len(votes) and bool(votes))


def chi_square_2x2(w: int, x: int, y: int, z: int) -> float:
    """Pearson chi-square (1 df, no continuity correction) of the table [[w, x], [y, z]] — used
    to contrast two arms' discordant counts (w:x against y:z). NaN when a margin is empty."""
    n = w + x + y + z
    denominator = (w + x) * (y + z) * (w + y) * (x + z)
    return n * (w * z - x * y) ** 2 / denominator if denominator else float("nan")


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for one value; NaN for none)."""
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def _binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def sign_test_greater_p(a: int, b: int) -> float:
    """One-sided exact sign test: P(at least a of a + b leaning one way | no preference)."""
    return _binomial_tail(a, a + b, 0.5) if a + b else 1.0


def fisher_greater_p(a: int, n1: int, c: int, n2: int) -> float:
    """One-sided Fisher exact test that group 1's share a/n1 exceeds group 2's c/n2."""
    k, n = a + c, n1 + n2
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / math.comb(n, k)


def rank_sum_greater_p(x: list[float], y: list[float]) -> float:
    """One-sided exact permutation test that x tends to exceed y (Wilcoxon–Mann–Whitney rank sum,
    average ranks for ties): the share of all relabellings whose x rank sum is at least the observed."""
    values = sorted(x + y)
    rank = {v: (values.index(v) + len(values) - values[::-1].index(v) + 1) / 2 for v in values}
    ranks = [rank[v] for v in x + y]
    observed = sum(rank[v] for v in x)
    total = at_least = 0
    for chosen in itertools.combinations(range(len(ranks)), len(x)):
        total += 1
        at_least += sum(ranks[i] for i in chosen) >= observed - 1e-9
    return at_least / total


def clopper_pearson(k: int, n: int, confidence: float = 0.90) -> tuple[float, float]:
    """Exact two-sided binomial confidence interval for k successes in n."""
    if n == 0:
        return 0.0, 1.0
    alpha = 1 - confidence

    def solve(target: Callable[[float], float]) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if target(mid) < 0 else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: _binomial_tail(k, n, p) - alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: alpha / 2 - (1 - _binomial_tail(k + 1, n, p)))
    return lower, upper
