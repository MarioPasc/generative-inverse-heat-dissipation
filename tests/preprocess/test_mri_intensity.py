"""Tests of the intensity rule: foreground p99 scaling, no low clip, uint8 quantisation."""

from __future__ import annotations

import numpy as np
import pytest

from ihdm.preprocess.errors import PreprocessError
from ihdm.preprocess.mri import scale_intensity, slices_to_uint8


def phantom_volume(rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return a small volume with a bright core, a dim rim and a noisy background."""
    rng = np.random.default_rng(rng_seed)
    volume = rng.uniform(1.0, 6.0, size=(20, 20, 20)).astype(np.float32)
    volume[5:15, 5:15, 5:15] = 400.0
    volume[8:12, 8:12, 8:12] = 900.0
    foreground = np.zeros_like(volume, dtype=bool)
    foreground[4:16, 4:16, 4:16] = True
    return volume, foreground


def test_scale_intensity_maps_the_foreground_p99_to_one() -> None:
    """The foreground 99th percentile becomes 1.0 and nothing exceeds it."""
    volume, foreground = phantom_volume()
    scaled, info = scale_intensity(volume, foreground)

    expected_p99 = float(np.percentile(volume[foreground], 99))
    assert info["p99"] == pytest.approx(expected_p99)
    assert scaled.max() == pytest.approx(1.0)
    assert scaled.dtype == np.float32
    # A voxel at exactly the percentile maps to 1.0.
    np.testing.assert_allclose(
        np.clip(volume / expected_p99, 0.0, 1.0).astype(np.float32), scaled, rtol=1e-6
    )


def test_scale_intensity_keeps_the_background_noise() -> None:
    """There is no low clip, so a noisy background survives as small positive values."""
    volume, foreground = phantom_volume()
    scaled, _ = scale_intensity(volume, foreground)
    background = scaled[0, 0, :]
    assert np.all(background > 0.0), "the background must not be clipped to zero"
    assert np.all(background < 0.05)
    assert float((scaled == 0.0).mean()) == 0.0


def test_scale_intensity_maps_zero_to_zero() -> None:
    """Padding written by the resampler stays exactly zero (a pure division, no offset)."""
    volume, foreground = phantom_volume()
    volume[0, :, :] = 0.0
    scaled, _ = scale_intensity(volume, foreground)
    assert np.all(scaled[0, :, :] == 0.0)


def test_scale_intensity_uses_only_the_foreground_for_the_percentile() -> None:
    """Enlarging the background does not move the percentile, so the scale is stable."""
    volume, foreground = phantom_volume()
    _, info_small = scale_intensity(volume, foreground)

    padded = np.zeros((40, 40, 40), dtype=np.float32)
    padded[10:30, 10:30, 10:30] = volume
    padded_foreground = np.zeros_like(padded, dtype=bool)
    padded_foreground[10:30, 10:30, 10:30] = foreground
    _, info_large = scale_intensity(padded, padded_foreground)

    assert info_large["p99"] == pytest.approx(info_small["p99"])


@pytest.mark.parametrize("percentile", [50.0, 90.0, 99.0, 99.9])
def test_scale_intensity_clip_fraction_is_consistent(percentile: float) -> None:
    """The reported high-clip fraction matches the voxels above the percentile."""
    volume, foreground = phantom_volume()
    _, info = scale_intensity(volume, foreground, percentile=percentile)
    assert info["clip_high"] == pytest.approx(float((volume > info["p99"]).mean()))


def test_scale_intensity_rejects_bad_inputs() -> None:
    """Shape mismatch, an empty foreground and a non-positive percentile all raise."""
    volume, foreground = phantom_volume()
    with pytest.raises(PreprocessError):
        scale_intensity(volume, np.zeros((4, 4, 4), dtype=bool))
    with pytest.raises(PreprocessError):
        scale_intensity(volume, np.zeros_like(volume, dtype=bool))
    with pytest.raises(PreprocessError):
        scale_intensity(np.zeros_like(volume), foreground)


def test_slices_to_uint8_rounds_to_nearest() -> None:
    """Quantisation is ``round(x * 255)`` on the float64 promotion of the input."""
    values = np.array([[0.0, 1.0, 0.5, 1.0 / 255.0, 0.25]], dtype=np.float32)
    quantised = slices_to_uint8(values)
    assert quantised.dtype == np.uint8
    np.testing.assert_array_equal(quantised, np.array([[0, 255, 128, 1, 64]], dtype=np.uint8))


def test_slices_to_uint8_uses_numpy_half_to_even_on_exact_midpoints() -> None:
    """An exact ``k + 0.5`` level rounds to the even neighbour (``np.round`` semantics)."""
    values = np.array([0.5 / 255.0, 1.5 / 255.0, 2.5 / 255.0], dtype=np.float64)
    np.testing.assert_array_equal(slices_to_uint8(values), np.array([0, 2, 2], dtype=np.uint8))


def test_slices_to_uint8_is_within_half_a_level_of_the_input() -> None:
    """The quantisation error never exceeds half a level, over the whole range."""
    values = np.linspace(0.0, 1.0, 4001, dtype=np.float32)
    error = np.abs(slices_to_uint8(values).astype(np.float64) / 255.0 - values)
    assert error.max() <= 0.5 / 255.0 + 1e-6


@pytest.mark.parametrize(
    "bad", [np.array([-1e-3, 0.5]), np.array([0.5, 1.001]), np.array([0.5, np.nan])]
)
def test_slices_to_uint8_rejects_values_outside_the_unit_interval(bad: np.ndarray) -> None:
    """A value outside ``[0, 1]`` or a NaN is a contract violation, not something to clip."""
    with pytest.raises(PreprocessError):
        slices_to_uint8(bad)
