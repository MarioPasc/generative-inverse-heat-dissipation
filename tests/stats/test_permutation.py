"""Tests of ``ihdm.stats.permutation``: the exact enumeration and the sampled fallback."""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

from ihdm.stats.errors import StatsError
from ihdm.stats.permutation import permutation_test


def test_three_against_three_enumerates_twenty_assignments():
    """The design's cell size makes the test exact with ``C(6, 3) = 20`` assignments."""
    result = permutation_test([0.10, 0.12, 0.11], [0.20, 0.22, 0.19])
    assert result.exact is True
    assert result.n_assignments == comb(6, 3) == 20
    assert result.p_min == pytest.approx(0.1)


def test_exact_p_value_equals_the_hand_enumeration():
    """The p-value is the proportion of assignments at least as extreme, computed by hand."""
    cell_a = [0.10, 0.12, 0.11]
    cell_b = [0.20, 0.22, 0.19]
    observed = np.mean(cell_a) - np.mean(cell_b)

    pooled = np.array(cell_a + cell_b)
    statistics = []
    for choice in combinations(range(6), 3):
        mask = np.zeros(6, dtype=bool)
        mask[list(choice)] = True
        statistics.append(pooled[mask].mean() - pooled[~mask].mean())
    expected = np.mean(np.abs(statistics) >= abs(observed) - 1e-12)

    result = permutation_test(cell_a, cell_b)
    assert result.statistic == pytest.approx(observed)
    assert result.p_value == pytest.approx(float(expected))
    assert result.p_value == pytest.approx(2.0 / 20.0)


def test_the_most_extreme_separation_reaches_the_floor_p_value():
    """Completely separated cells give the smallest attainable exact p-value, 2/20."""
    result = permutation_test([1.0, 1.1, 1.2], [5.0, 5.1, 5.2])
    assert result.p_value == pytest.approx(0.1)
    assert result.p_value == pytest.approx(result.p_min)


def test_identical_cells_give_p_equal_to_one():
    """With a zero statistic every assignment is at least as extreme."""
    result = permutation_test([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert result.statistic == 0.0
    assert result.p_value == pytest.approx(1.0)


def test_sampled_branch_is_used_above_the_exact_threshold():
    """A pooled size of 8 exceeds ``exact_below=7`` and is sampled with the add-one estimator."""
    rng = np.random.default_rng(0)
    cell_a = rng.normal(0.0, 1.0, size=4)
    cell_b = rng.normal(0.0, 1.0, size=4)
    result = permutation_test(cell_a, cell_b, n_perm=500, rng_seed=0)
    assert result.exact is False
    assert result.n_assignments == 500
    assert 0.0 < result.p_value <= 1.0
    assert result.p_value >= 1.0 / 501.0


def test_sampled_branch_detects_a_large_separation():
    """Well separated cells of size five give a small sampled p-value."""
    result = permutation_test(
        [0.0, 0.05, -0.05, 0.02, -0.02], [3.0, 3.05, 2.95, 3.02, 2.98], n_perm=2000, rng_seed=0
    )
    assert result.exact is False
    assert result.p_value < 0.01


def test_sampled_branch_is_reproducible():
    """The same seed gives the same p-value."""
    kwargs = {"n_perm": 300, "rng_seed": 11}
    a, b = [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]
    assert permutation_test(a, b, **kwargs).p_value == permutation_test(a, b, **kwargs).p_value


def test_exact_threshold_is_configurable():
    """``exact_below`` moves the enumeration boundary; 4 + 4 enumerates ``C(8, 4) = 70``."""
    result = permutation_test([1.0, 2, 3, 4], [5.0, 6, 7, 8], exact_below=9)
    assert result.exact is True
    assert result.n_assignments == comb(8, 4) == 70


@pytest.mark.parametrize(
    ("cell_a", "cell_b", "kwargs"),
    [
        ([], [1.0], {}),
        ([1.0], [], {}),
        ([1.0, np.inf], [1.0, 2.0], {}),
        ([1.0, 2.0], [1.0, 2.0], {"n_perm": 0}),
    ],
)
def test_permutation_test_rejects_bad_input(cell_a, cell_b, kwargs):
    """Empty, non-finite and out-of-domain arguments raise :class:`StatsError`."""
    with pytest.raises(StatsError):
        permutation_test(cell_a, cell_b, **kwargs)
