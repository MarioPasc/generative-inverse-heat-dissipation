"""Tests of ``ihdm.stats.bootstrap``: the percentile interval, the paired contrasts, the gate."""

from __future__ import annotations

import numpy as np
import pytest

from ihdm.metrics.spectral import lsd
from ihdm.stats.bootstrap import (
    bootstrap_ci,
    interaction,
    paired_delta,
    paired_lsd_gate,
)
from ihdm.stats.errors import StatsError

# --------------------------------------------------------------------------------------------
# bootstrap_ci
# --------------------------------------------------------------------------------------------


def test_bootstrap_ci_matches_the_hand_computation_on_a_fixed_sample():
    """The interval equals an independently written resampler with the same RNG (H-METRICS §1)."""
    values = np.array([0.31, 0.44, 0.27, 0.52, 0.39, 0.41, 0.33, 0.48])
    result = bootstrap_ci(values, n_boot=2000, alpha=0.05, rng_seed=7)

    rng = np.random.default_rng(7)
    replicates = np.empty(2000)
    draws = rng.integers(0, values.size, size=(2000, values.size))
    for index in range(2000):
        replicates[index] = values[draws[index]].mean()
    low, high = np.percentile(replicates, [2.5, 97.5])

    assert result.point == pytest.approx(float(values.mean()), abs=1e-12)
    assert result.low == pytest.approx(float(low), abs=1e-9)
    assert result.high == pytest.approx(float(high), abs=1e-9)
    assert result.n == 8
    assert result.n_boot == 2000


def test_bootstrap_ci_of_a_constant_sample_is_degenerate():
    """Every resample of a constant sample has the same mean, so the interval is a point."""
    result = bootstrap_ci(np.full(6, 2.5), n_boot=500, rng_seed=0)
    assert result.low == pytest.approx(2.5)
    assert result.high == pytest.approx(2.5)
    assert not result.excludes_zero or result.low > 0.0


def test_bootstrap_ci_brackets_the_normal_standard_error():
    """On a large normal sample the percentile interval is within 5% of mean +- 1.96 SE."""
    rng = np.random.default_rng(3)
    sample = rng.normal(loc=1.0, scale=0.5, size=4000)
    result = bootstrap_ci(sample, n_boot=4000, rng_seed=1)
    standard_error = sample.std(ddof=1) / np.sqrt(sample.size)
    assert result.low == pytest.approx(sample.mean() - 1.96 * standard_error, rel=0.05)
    assert result.high == pytest.approx(sample.mean() + 1.96 * standard_error, rel=0.05)


def test_bootstrap_ci_median_statistic():
    """The median branch resamples the median, not the mean."""
    values = np.array([1.0, 2.0, 3.0, 100.0])
    assert bootstrap_ci(values, n_boot=200, rng_seed=0, statistic="median").point == 2.5


@pytest.mark.parametrize(
    ("kwargs", "values"),
    [
        ({}, []),
        ({}, [1.0, np.nan]),
        ({"n_boot": 0}, [1.0, 2.0]),
        ({"alpha": 0.0}, [1.0, 2.0]),
        ({"statistic": "mode"}, [1.0, 2.0]),
    ],
)
def test_bootstrap_ci_rejects_bad_input(kwargs, values):
    """Empty, non-finite and out-of-domain arguments raise :class:`StatsError`."""
    with pytest.raises(StatsError):
        bootstrap_ci(values, **kwargs)


# --------------------------------------------------------------------------------------------
# paired_delta and interaction
# --------------------------------------------------------------------------------------------


def test_paired_delta_pairs_by_seed_not_by_position():
    """The difference uses the seed keys, whatever order the mappings are written in."""
    arm = {3: 0.5, 1: 0.2, 2: 0.4}
    a0 = {1: 0.3, 2: 0.4, 3: 0.9}
    result = paired_delta(arm, a0, n_boot=500, rng_seed=0)
    assert result.seeds == (1, 2, 3)
    np.testing.assert_allclose(result.deltas, [-0.1, 0.0, -0.4])
    assert result.mean == pytest.approx(-0.5 / 3.0)


def test_paired_delta_rejects_unpaired_cells():
    """A seed present in only one cell is a design error, not a missing value."""
    with pytest.raises(StatsError, match="not paired"):
        paired_delta({1: 0.1, 2: 0.2}, {1: 0.1, 3: 0.2})


def test_interaction_sign_follows_the_larger_mri_effect():
    """A more negative MRI delta than photograph delta gives a negative interaction."""
    result = interaction([-0.30, -0.28, -0.32], [-0.05, -0.04, -0.06], n_boot=2000, rng_seed=0)
    assert result.point == pytest.approx(-0.30 + 0.05, abs=1e-9)
    assert result.point < 0.0
    assert result.interval.high < 0.0
    assert result.mean_mri == pytest.approx(-0.30)
    assert result.mean_photo == pytest.approx(-0.05)
    assert result.n_mri == 3 and result.n_photo == 3

    flipped = interaction([-0.05, -0.04, -0.06], [-0.30, -0.28, -0.32], n_boot=2000, rng_seed=0)
    assert flipped.point == pytest.approx(-result.point, abs=1e-9)


def test_interaction_interval_contains_zero_when_the_two_sides_agree():
    """No interaction: the two data types move by the same amount."""
    result = interaction([-0.10, -0.12, -0.08], [-0.10, -0.11, -0.09], n_boot=4000, rng_seed=0)
    assert result.interval.low < 0.0 < result.interval.high
    assert not result.interval.excludes_zero


# --------------------------------------------------------------------------------------------
# paired_lsd_gate
# --------------------------------------------------------------------------------------------


def _pink_stack(rng: np.random.Generator, n: int, width: int, alpha: float) -> np.ndarray:
    """Return an ``(n, width, width)`` float stack whose spectrum falls like ``n^-alpha``."""
    from scipy.fft import idctn

    index = np.arange(width)
    radial = np.sqrt(index[:, None] ** 2 + index[None, :] ** 2)
    amplitude = np.where(radial > 0, np.power(np.maximum(radial, 1.0), -alpha / 2.0), 0.0)
    coefficients = rng.normal(size=(n, width, width)) * amplitude[None]
    return idctn(coefficients, axes=(1, 2), norm="ortho")


def test_gate_point_estimate_equals_the_frozen_lsd():
    """The fast path reproduces ``ihdm.metrics.spectral.lsd`` exactly on the identity resample."""
    rng = np.random.default_rng(0)
    reference = _pink_stack(rng, 40, 96, alpha=3.0)
    stack_a = _pink_stack(rng, 24, 96, alpha=2.0)
    stack_b = _pink_stack(rng, 24, 96, alpha=2.6)

    result = paired_lsd_gate(stack_a, stack_b, reference, n_boot=50, rng_seed=0)
    assert result.lsd_a == pytest.approx(lsd(stack_a, reference).lsd, abs=1e-12)
    assert result.lsd_b == pytest.approx(lsd(stack_b, reference).lsd, abs=1e-12)
    assert result.difference.point == pytest.approx(result.lsd_a - result.lsd_b, abs=1e-12)


def test_gate_extends_when_the_later_checkpoint_is_closer_to_the_reference():
    """``b`` closer to the reference than ``a`` gives a positive difference and ``extend``."""
    rng = np.random.default_rng(1)
    reference = _pink_stack(rng, 60, 96, alpha=3.0)
    stack_a = _pink_stack(rng, 48, 96, alpha=1.6)
    stack_b = _pink_stack(rng, 48, 96, alpha=2.8)

    result = paired_lsd_gate(stack_a, stack_b, reference, n_boot=400, rng_seed=0)
    assert result.lsd_a > result.lsd_b
    assert result.difference.point > 0.0
    assert result.difference.low > 0.0
    assert result.extend is True
    assert result.relative_change > 0.0
    assert result.n_seeds == 48


def test_gate_interval_contains_zero_for_identical_stacks():
    """Two identical checkpoints cannot be distinguished: the difference is exactly zero."""
    rng = np.random.default_rng(2)
    reference = _pink_stack(rng, 40, 96, alpha=3.0)
    stack = _pink_stack(rng, 32, 96, alpha=2.2)

    result = paired_lsd_gate(stack, stack.copy(), reference, n_boot=300, rng_seed=0)
    assert result.difference.point == pytest.approx(0.0, abs=1e-12)
    assert result.difference.low <= 0.0 <= result.difference.high
    assert result.extend is False


def test_gate_is_paired_so_a_common_shift_cancels():
    """Shifting both stacks with the same resample leaves the difference tight around zero."""
    rng = np.random.default_rng(4)
    reference = _pink_stack(rng, 40, 96, alpha=3.0)
    stack = _pink_stack(rng, 32, 96, alpha=2.2)
    perturbed = stack * 1.0001

    result = paired_lsd_gate(stack, perturbed, reference, n_boot=300, rng_seed=0)
    assert abs(result.difference.high - result.difference.low) < 5e-3


def test_gate_rejects_unpaired_stacks():
    """The two checkpoints must expose the same seeds in the same order."""
    rng = np.random.default_rng(5)
    reference = _pink_stack(rng, 20, 96, alpha=3.0)
    with pytest.raises(StatsError, match="not paired"):
        paired_lsd_gate(
            _pink_stack(rng, 8, 96, alpha=2.0),
            _pink_stack(rng, 9, 96, alpha=2.0),
            reference,
        )


def test_gate_accepts_uint8_stacks():
    """``samples.npy`` is ``uint8``; the fast path converts it like ``_DcRemoved`` does."""
    rng = np.random.default_rng(6)
    reference = (rng.random((24, 96, 96)) * 255).astype(np.uint8)
    stack_a = (rng.random((16, 96, 96)) * 255).astype(np.uint8)
    stack_b = (rng.random((16, 96, 96)) * 255).astype(np.uint8)

    result = paired_lsd_gate(stack_a, stack_b, reference, n_boot=50, rng_seed=0)
    assert result.lsd_a == pytest.approx(lsd(stack_a, reference).lsd, abs=1e-12)
    assert result.lsd_b == pytest.approx(lsd(stack_b, reference).lsd, abs=1e-12)
