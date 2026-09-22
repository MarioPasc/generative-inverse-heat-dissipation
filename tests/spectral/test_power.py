"""Unit tests of ``ihdm.spectral.power``."""

from __future__ import annotations

import numpy as np
import pytest

from ihdm.spectral.errors import SpectralError
from ihdm.spectral.power import (
    MAX_CYCLES_PER_IMAGE,
    OCTAVE_LABELS,
    eigenvalues,
    fit_alpha,
    inherited_share,
    mode_power,
    octave_bins,
    octave_shares,
    power_law,
    radial_index,
    radial_profile,
    radial_spectrum,
)


def test_fit_alpha_recovers_the_synthetic_exponent(alpha2_field: np.ndarray) -> None:
    """512 fields at 64^2 with alpha = 2 are recovered within the ticket's 0.15."""
    assert fit_alpha(mode_power(alpha2_field)) == pytest.approx(2.0, abs=0.15)


@pytest.mark.parametrize("alpha", [1.5, 2.0, 3.0, 3.5])
def test_fit_alpha_recovers_an_analytic_power_law(alpha: float) -> None:
    """On an exact ``n^-alpha`` spectrum the fit returns ``alpha``.

    Not exactly: the radial bands average ``n^-alpha`` over an annulus of unit width, which is
    a convex function of ``n``, so the band means sit a little above the curve. The bias is
    below 0.01 over the whole window.
    """
    assert fit_alpha(power_law(192, alpha, 1.0)) == pytest.approx(alpha, abs=0.01)


def test_mode_power_matches_the_one_shot_variance() -> None:
    """The two-pass form equals ``dctn(...).var(axis=0)`` up to rounding."""
    from scipy.fft import dctn

    rng = np.random.default_rng(3)
    images = rng.normal(size=(64, 16, 16))
    reference = dctn(images, axes=(1, 2), norm="ortho").var(axis=0)
    reference[0, 0] = 0.0
    np.testing.assert_allclose(mode_power(images), reference, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("chunk", [1, 7, 64, 1000])
def test_mode_power_is_independent_of_the_chunk_size(chunk: int) -> None:
    """Chunking only changes the summation order."""
    rng = np.random.default_rng(4)
    images = rng.normal(size=(33, 8, 8))
    np.testing.assert_allclose(mode_power(images, chunk=chunk), mode_power(images), rtol=1e-12)


def test_mode_power_zeroes_the_dc_mode() -> None:
    """The DC entry is excluded, whatever the stack's mean."""
    images = np.full((5, 8, 8), 0.7)
    images[0] += 0.1
    assert mode_power(images)[0, 0] == 0.0


def test_mode_power_of_identical_images_is_zero() -> None:
    """A stack with no between-image variation carries no per-mode variance."""
    images = np.tile(np.linspace(0, 1, 64).reshape(8, 8), (6, 1, 1))
    np.testing.assert_allclose(mode_power(images), 0.0, atol=1e-24)


@pytest.mark.parametrize(
    "images", [np.zeros((4, 8, 9)), np.zeros((4, 8)), np.zeros((0, 8, 8))]
)
def test_mode_power_rejects_bad_stacks(images: np.ndarray) -> None:
    """Non-square, non-3-D and empty stacks raise."""
    with pytest.raises(SpectralError):
        mode_power(images)


def test_mode_power_rejects_a_non_positive_chunk() -> None:
    """A chunk size below one raises."""
    with pytest.raises(SpectralError):
        mode_power(np.zeros((2, 4, 4)), chunk=0)


def test_octave_bins_are_the_frozen_eight() -> None:
    """The bins are those of ``05-metrics.md`` §1, closed at 96 cycles per image."""
    bins = octave_bins()
    assert [label for label, _, _ in bins] == list(OCTAVE_LABELS)
    assert [lo for _, lo, _ in bins] == [0.5, 1, 2, 4, 8, 16, 32, 64]
    assert bins[-1][2] == MAX_CYCLES_PER_IMAGE == 96.0


def test_octave_shares_sum_to_one_and_exclude_the_corner_modes() -> None:
    """The eight shares sum to 1; modes above 96 c/img are excluded from the total."""
    power = power_law(192, 2.0, 1.0)
    shares = octave_shares(power)
    assert set(shares) == set(OCTAVE_LABELS)
    assert sum(shares.values()) == pytest.approx(1.0)
    corner = power[radial_index(192) / 2.0 > MAX_CYCLES_PER_IMAGE].sum()
    assert corner > 0.0  # the corner modes exist and are genuinely left out


def test_octave_shares_are_flat_for_alpha_two() -> None:
    """A scale-invariant 2-D spectrum holds roughly equal variance in every full octave.

    Checked on the octaves 2–4 … 16–32 cycles per image: the coarsest bins hold too few
    discrete modes for the continuum argument, and the 32–64 and 64–96 bins are clipped by the
    corner of the square grid.
    """
    shares = octave_shares(power_law(192, 2.0, 1.0))
    full = [shares[b] for b in OCTAVE_LABELS[2:6]]
    assert max(full) / min(full) < 1.25


def test_octave_shares_reject_an_empty_spectrum() -> None:
    """An all-zero spectrum raises rather than dividing by zero."""
    with pytest.raises(SpectralError):
        octave_shares(np.zeros((32, 32)))


def test_radial_profile_bins_and_edge_cases() -> None:
    """The profile averages inside each bin; bad edges raise."""
    power = np.ones((16, 16))
    power[0, 0] = 0.0
    profile = radial_profile(power, np.array([0.5, 1.0, 2.0, 4.0]))
    np.testing.assert_allclose(profile, 1.0)
    with pytest.raises(SpectralError):
        radial_profile(power, np.array([1.0]))
    with pytest.raises(SpectralError):
        radial_profile(power, np.array([2.0, 1.0]))


def test_radial_spectrum_bands_are_half_integer_cycles() -> None:
    """Band ``b`` is returned at ``b / 2`` cycles per image, for ``b = 1 … W - 1``."""
    freq, binned = radial_spectrum(power_law(64, 2.0, 1.0))
    assert freq.shape == binned.shape == (63,)
    np.testing.assert_allclose(freq[:3], [0.5, 1.0, 1.5])


def test_eigenvalues_are_the_dct_laplacian() -> None:
    """``lambda_{i,j} = pi^2 (i^2 + j^2) / W^2``, zero at DC."""
    lam = eigenvalues(8)
    assert lam[0, 0] == 0.0
    assert lam[1, 0] == pytest.approx(np.pi**2 / 64)
    assert lam[3, 4] == pytest.approx(np.pi**2 * 25 / 64)


def test_inherited_share_limits() -> None:
    """Everything survives a zero blur; almost nothing survives a very large one."""
    power = power_law(64, 2.0, 1.0)
    assert inherited_share(power, 0.0) == pytest.approx(1.0)
    assert inherited_share(power, 64.0) < 1e-3
    assert 0.0 < inherited_share(power, 8.0) < 1.0


def test_inherited_share_rejects_an_empty_spectrum() -> None:
    """An all-zero spectrum raises."""
    with pytest.raises(SpectralError):
        inherited_share(np.zeros((16, 16)), 4.0)


def test_fit_alpha_rejects_a_degenerate_window() -> None:
    """Fewer than three usable bands raises."""
    with pytest.raises(SpectralError):
        fit_alpha(power_law(8, 2.0, 1.0), lo_frac=0.9, hi_frac=0.95)
