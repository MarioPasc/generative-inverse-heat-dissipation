"""Identities of the numpy heat kernel (H-METRICS §1, row "low-pass").

The reference is the released ``model_code.utils.DCTBlur``, instantiated on the CPU, which is the
operator the forward process of the model applies; ``ihdm.metrics.lowpass.dct_lowpass`` must be
the same map, because ``M_lp`` and ``D_lp`` are defined as "the metric at the scale the prior
works on" and would mean nothing if the two blurs disagreed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ihdm.metrics.errors import MetricError
from ihdm.metrics.lowpass import dct_lowpass, remove_dc, to_float_stack
from model_code.utils import DCTBlur


def released_blur(images: np.ndarray, sigma_b: float) -> np.ndarray:
    """Blur a stack with the released ``DCTBlur`` module on the CPU.

    Parameters
    ----------
    images : numpy.ndarray
        ``float32`` stack of shape ``(N, H, W)``.
    sigma_b : float
        Blur length-scale in pixels.

    Returns
    -------
    numpy.ndarray
        The blurred stack.
    """
    module = DCTBlur(np.array([0.0, sigma_b], dtype=np.float32), images.shape[1], "cpu")
    with torch.no_grad():
        return module(torch.from_numpy(images), [1] * images.shape[0]).numpy()


@pytest.fixture
def field() -> np.ndarray:
    """A small random stack in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``float32`` stack of shape ``(4, 32, 32)``.
    """
    return np.random.default_rng(20260923).random((4, 32, 32)).astype(np.float32)


@pytest.mark.parametrize("side", [32, 64])
@pytest.mark.parametrize("sigma_b", [1.0, 4.0, 16.0])
def test_matches_released_dctblur(side: int, sigma_b: float) -> None:
    """``dct_lowpass`` reproduces the released heat kernel to 1e-5 (H-METRICS §1)."""
    images = np.random.default_rng(side).random((3, side, side)).astype(np.float32)
    np.testing.assert_allclose(
        dct_lowpass(images, sigma_b), released_blur(images, sigma_b), atol=1e-5, rtol=0.0
    )


def test_sigma_zero_is_the_identity(field: np.ndarray) -> None:
    """``sigma_b = 0`` returns the converted stack unchanged."""
    np.testing.assert_array_equal(dct_lowpass(field, 0.0), field)


def test_semigroup(field: np.ndarray) -> None:
    """Two blurs of ``sigma`` equal one blur of ``sqrt(2) sigma`` and differ from one of them.

    The kernel is ``exp(-lambda sigma^2 / 2)``, so the operator family is a semigroup in
    ``sigma^2``. This is the exact form of the harness's "blurring twice is not blurring once".
    """
    once = dct_lowpass(field, 3.0)
    twice = dct_lowpass(once, 3.0)
    np.testing.assert_allclose(twice, dct_lowpass(field, 3.0 * np.sqrt(2.0)), atol=1e-5, rtol=0.0)
    assert np.abs(twice - once).max() > 1e-3


@pytest.mark.parametrize("sigma_b", [0.5, 1.0, 2.0, 4.0, 8.0])
def test_variance_decreases_with_sigma(field: np.ndarray, sigma_b: float) -> None:
    """Every blur is a contraction of the non-DC variance, monotonically in ``sigma_b``."""
    coarser = float(np.var(remove_dc(dct_lowpass(field, sigma_b))))
    finer = float(np.var(remove_dc(dct_lowpass(field, sigma_b / 2.0))))
    assert 0.0 < coarser < finer <= float(np.var(remove_dc(field)))


def test_dc_is_preserved(field: np.ndarray) -> None:
    """The per-image mean survives the blur (the DC mode has ``lambda = 0``)."""
    np.testing.assert_allclose(
        dct_lowpass(field, 6.0).mean(axis=(1, 2)), field.mean(axis=(1, 2)), atol=1e-6, rtol=0.0
    )


def test_uint8_and_float_inputs_agree(field: np.ndarray) -> None:
    """A ``uint8`` stack is divided by 255 and then blurred like its float twin."""
    quantised = np.rint(field * 255.0).astype(np.uint8)
    np.testing.assert_allclose(
        dct_lowpass(quantised, 4.0),
        dct_lowpass(quantised.astype(np.float32) / 255.0, 4.0),
        atol=1e-7,
        rtol=0.0,
    )


def test_chunking_is_invisible() -> None:
    """A stack longer than one chunk is blurred exactly like its pieces."""
    images = np.random.default_rng(7).random((300, 16, 16)).astype(np.float32)
    whole = dct_lowpass(images, 2.0)
    pieces = np.concatenate([dct_lowpass(images[:100], 2.0), dct_lowpass(images[100:], 2.0)])
    np.testing.assert_array_equal(whole, pieces)


def test_output_is_float32(field: np.ndarray) -> None:
    """The contract fixes the output dtype."""
    assert dct_lowpass(field.astype(np.float64), 2.0).dtype == np.float32


@pytest.mark.parametrize(
    ("images", "sigma_b"),
    [
        (np.zeros((4, 8), dtype=np.float32), 1.0),
        (np.zeros((2, 8, 6), dtype=np.float32), 1.0),
        (np.zeros((0, 8, 8), dtype=np.float32), 1.0),
        (np.full((2, 8, 8), np.nan, dtype=np.float32), 1.0),
        (np.zeros((2, 8, 8), dtype=np.float32), -1.0),
        (np.zeros((2, 8, 8), dtype=np.float32), np.nan),
    ],
)
def test_invalid_input_raises(images: np.ndarray, sigma_b: float) -> None:
    """Rank, squareness, emptiness, finiteness and ``sigma_b`` are checked at the boundary."""
    with pytest.raises(MetricError):
        dct_lowpass(images, sigma_b)


def test_to_float_stack_scales_integers() -> None:
    """``uint8`` in, ``float32`` in ``[0, 1]`` out."""
    stack = to_float_stack(np.array([[[0, 255], [128, 64]]], dtype=np.uint8), "x")
    assert stack.dtype == np.float32
    np.testing.assert_allclose(stack, np.array([[[0.0, 1.0], [128 / 255, 64 / 255]]]), atol=1e-7)


def test_remove_dc_centres_every_image(field: np.ndarray) -> None:
    """Every image of the result has zero mean, and the original is not modified."""
    before = field.copy()
    np.testing.assert_allclose(remove_dc(field).mean(axis=(1, 2)), 0.0, atol=1e-7)
    np.testing.assert_array_equal(field, before)
