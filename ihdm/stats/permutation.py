"""The permutation test of ``05-metrics.md`` §8.

Two cells of the design (one dataset, two arms) hold ``s`` seeds each. Under the null the arm
label carries no information, so every way of splitting the pooled seeds into a group of
``len(cell_a)`` and a group of ``len(cell_b)`` is equally likely. With ``s = 3`` per cell the
pooled size is 6 and there are exactly ``C(6, 3) = 20`` assignments, so the test is enumerated
rather than sampled and the smallest attainable two-sided p-value is ``2/20 = 0.1``: the test is
reported for completeness and the bootstrap interval is what the design is powered on.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from ihdm.stats.errors import StatsError

__all__ = ["PermutationResult", "permutation_test"]

#: Default number of random assignments when the pooled sample is too large to enumerate.
N_PERM: int = 10_000

#: Pooled sizes strictly below this are enumerated exactly (``05-metrics.md`` §8: ``n <= 6``).
EXACT_BELOW: int = 7


@dataclass(frozen=True)
class PermutationResult:
    """Result of :func:`permutation_test`.

    Parameters
    ----------
    statistic : float
        The observed difference of means, ``mean(cell_a) - mean(cell_b)``.
    p_value : float
        Two-sided p-value. Exact enumeration returns the proportion of assignments whose
        statistic is at least as extreme in absolute value as the observed one (the observed
        assignment is one of them, so the value is never zero). The sampled branch returns
        ``(1 + count) / (1 + n_perm)``, the add-one estimator, which is also never zero.
    n_assignments : int
        How many assignments entered the p-value: ``C(n_a + n_b, n_a)`` when exact, ``n_perm``
        otherwise.
    exact : bool
        Whether the enumeration was exhaustive.
    n_a, n_b : int
        The two cell sizes.
    p_min : float
        The smallest p-value the design can produce, ``2 / n_assignments`` when exact. Reported
        so that a non-significant result at ``s = 3`` is not read as evidence of no effect.
    """

    statistic: float
    p_value: float
    n_assignments: int
    exact: bool
    n_a: int
    n_b: int
    p_min: float

    def to_json(self) -> dict[str, Any]:
        """Return the record written into the analysis tables."""
        return {
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "n_assignments": int(self.n_assignments),
            "exact": bool(self.exact),
            "n_a": int(self.n_a),
            "n_b": int(self.n_b),
            "p_min": float(self.p_min),
        }


def _cell(values: Any, name: str) -> np.ndarray:
    """Return a cell as a finite 1-D ``float64`` array."""
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise StatsError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise StatsError(f"{name} holds a non-finite value")
    return array


def _exact_statistics(pooled: np.ndarray, n_a: int) -> np.ndarray:
    """Difference of means for every way of choosing ``n_a`` of the pooled values."""
    total = pooled.sum()
    size = pooled.size
    assignments = np.array(
        [list(choice) for choice in combinations(range(size), n_a)], dtype=np.int64
    )
    sums_a = pooled[assignments].sum(axis=1)
    return sums_a / n_a - (total - sums_a) / (size - n_a)


def _sampled_statistics(
    pooled: np.ndarray, n_a: int, n_perm: int, rng_seed: int
) -> np.ndarray:
    """Difference of means for ``n_perm`` random assignments of the pooled values."""
    rng = np.random.default_rng(rng_seed)
    total = pooled.sum()
    size = pooled.size
    order = np.argsort(rng.random((n_perm, size)), axis=1)
    sums_a = np.take_along_axis(np.broadcast_to(pooled, (n_perm, size)), order[:, :n_a], axis=1)
    sums_a = sums_a.sum(axis=1)
    return sums_a / n_a - (total - sums_a) / (size - n_a)


def permutation_test(
    cell_a: Any,
    cell_b: Any,
    n_perm: int = N_PERM,
    exact_below: int = EXACT_BELOW,
    rng_seed: int = 0,
) -> PermutationResult:
    """Two-sided permutation test of the difference of means of two cells.

    Parameters
    ----------
    cell_a, cell_b : array-like
        The per-seed values of the two cells; they need not be the same size, but with the
        design's ``s = 3`` they are.
    n_perm : int
        Random assignments drawn when the pooled sample is not enumerated.
    exact_below : int
        Pooled sizes strictly below this are enumerated exhaustively. The default 7 makes the
        design's ``3 + 3 = 6`` exact, with ``C(6, 3) = 20`` assignments.
    rng_seed : int
        Seed of the sampling branch's RNG.

    Returns
    -------
    PermutationResult
        The observed statistic, the two-sided p-value and how it was obtained.

    Raises
    ------
    StatsError
        If either cell is empty or non-finite, or ``n_perm`` is not positive.
    """
    values_a = _cell(cell_a, "cell_a")
    values_b = _cell(cell_b, "cell_b")
    if n_perm < 1:
        raise StatsError(f"n_perm must be positive, got {n_perm}")

    pooled = np.concatenate([values_a, values_b])
    n_a, n_b = int(values_a.size), int(values_b.size)
    observed = float(values_a.mean() - values_b.mean())
    exact = pooled.size < int(exact_below)

    if exact:
        statistics = _exact_statistics(pooled, n_a)
        n_assignments = int(statistics.size)
        extreme = int(np.count_nonzero(np.abs(statistics) >= abs(observed) - 1e-12))
        p_value = extreme / n_assignments
        p_min = 2.0 / n_assignments
    else:
        statistics = _sampled_statistics(pooled, n_a, int(n_perm), rng_seed)
        n_assignments = int(n_perm)
        extreme = int(np.count_nonzero(np.abs(statistics) >= abs(observed) - 1e-12))
        p_value = (1.0 + extreme) / (1.0 + n_assignments)
        p_min = 1.0 / (1.0 + n_assignments)

    return PermutationResult(
        statistic=observed,
        p_value=float(p_value),
        n_assignments=n_assignments,
        exact=bool(exact),
        n_a=n_a,
        n_b=n_b,
        p_min=float(p_min),
    )
