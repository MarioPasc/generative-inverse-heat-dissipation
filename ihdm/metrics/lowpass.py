"""The released heat kernel as a pure numpy low-pass, and the image-stack conventions of M4.

``dct_lowpass`` is ``model_code.utils.DCTBlur`` rewritten on ``scipy.fft``: the orthonormal
2-D DCT-II of every image, multiplied by ``exp(-lambda_{ij} sigma_B^2 / 2)`` with ``lambda`` the
negated-Laplacian eigenvalues of ``ihdm.spectral.power.eigenvalues``, transformed back. It is the
same operator the forward process of the model applies, so the low-pass variants of the
memorisation ratio and of the within-seed diversity (``05-metrics.md`` §3, §4) measure the
sample sets *at the scale the model's own prior works on*, not at some unrelated smoothing.

The module also owns the two conventions every metric of T4.2 applies at its boundary:
:func:`to_float_stack` (uint8 or float in, ``float32`` in ``[0, 1]`` out, ``MetricError`` on a
malformed or non-finite stack) and :func:`remove_dc` (the per-image mean subtracted, as every
pixel-space quantity of `05` §3 and §4 requires). ``ihdm.metrics.spectral`` has its own private
chunked adapter for the same job; that one yields ``float64`` blocks for ``mode_power`` and is
indexed lazily, which is the wrong shape for the torch consumers here.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.fft import dctn, idctn

from ihdm.metrics.errors import MetricError
from ihdm.spectral.power import eigenvalues

__all__ = ["CHUNK", "dct_lowpass", "remove_dc", "to_float_stack"]

logger = logging.getLogger(__name__)

#: Images transformed at a time by :func:`dct_lowpass`. At 192 px a chunk of 256 images costs
#: 75 MB in the ``float64`` working dtype, which is the point of chunking at all: the low-passed
#: copies of the 3 200-image training split are what ``memorisation_ratio`` holds for ``M_lp``.
CHUNK: int = 256


def to_float_stack(images: Any, name: str, min_images: int = 1) -> np.ndarray:
    """Validate a stack of square images and return it as ``float32`` in ``[0, 1]``.

    Integer stacks are divided by 255 (the on-disk contract: ``images.npy`` and ``samples.npy``
    are both ``uint8``); floating stacks are cast to ``float32`` and left in their own range,
    because the synthetic fields of the tests are zero-mean and every metric of this milestone
    removes the DC anyway.

    Parameters
    ----------
    images : Any
        Candidate stack of shape ``(N, H, W)`` with ``H == W``.
    name : str
        Name used in error messages.
    min_images : int
        Smallest acceptable ``N``.

    Returns
    -------
    numpy.ndarray
        A ``float32`` array of shape ``(N, H, W)``; always a fresh array, never a view of the
        caller's buffer.

    Raises
    ------
    MetricError
        If the rank, the squareness or the image count fails, or the stack holds a non-finite
        value.
    """
    stack = np.asanyarray(images)
    if stack.ndim != 3:
        raise MetricError(f"{name}: expected a stack of shape (N, H, W), got {stack.shape}")
    if stack.shape[1] != stack.shape[2]:
        raise MetricError(f"{name}: expected square images, got {stack.shape[1:]}")
    if stack.shape[0] < min_images:
        raise MetricError(f"{name}: expected at least {min_images} images, got {stack.shape[0]}")
    if np.issubdtype(stack.dtype, np.integer):
        out = np.asarray(stack, dtype=np.float32) / np.float32(255.0)
    else:
        out = np.array(stack, dtype=np.float32, copy=True)
    if not np.isfinite(out).all():
        raise MetricError(f"{name}: non-finite values in the stack")
    return out


def remove_dc(images: np.ndarray) -> np.ndarray:
    """Subtract the per-image mean from every image of a ``float32`` stack.

    Every pixel-space quantity of ``05-metrics.md`` §3 and §4 is defined on the DC-removed
    images, so that a global intensity offset never counts as a distance or as diversity.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of shape ``(..., H, W)``.

    Returns
    -------
    numpy.ndarray
        The centred stack as ``float32``, same shape. The mean is accumulated in ``float64``:
        a ``float32`` sum over the 36 864 pixels of a 192 px image loses about three digits.
    """
    mean = np.asarray(images).mean(axis=(-2, -1), keepdims=True, dtype=np.float64)
    return np.asarray(images - mean, dtype=np.float32)


def dct_lowpass(images: np.ndarray, sigma_b: float) -> np.ndarray:
    """Blur a stack with the heat kernel of the released forward process.

    Implements ``model_code.utils.DCTBlur`` in numpy: ``idct(dct(x) * exp(-lambda t))`` with
    ``t = sigma_b^2 / 2`` and ``lambda_{ij} = pi^2 (i^2 + j^2) / H^2``, both transforms
    orthonormal DCT-II/III. The DC mode has ``lambda = 0`` and is left untouched, so the
    operation commutes with :func:`remove_dc`.

    The transform is accumulated in ``float64`` and returned as ``float32``: the
    ``dct -> multiply -> idct`` round trip costs about 1e-6 relative in ``float32``, which would
    consume most of the 1e-5 budget the harness allows against the released module, while the
    ``float32`` result keeps the low-passed copy of a 3 200-image training split at 472 MB.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of shape ``(N, H, W)``, ``uint8`` or float.
    sigma_b : float
        Blur length-scale in pixels. ``0`` is the identity.

    Returns
    -------
    numpy.ndarray
        ``float32`` stack of shape ``(N, H, W)``.

    Raises
    ------
    MetricError
        If the stack is malformed or non-finite, or ``sigma_b`` is negative or not finite.
    """
    if not np.isfinite(sigma_b) or sigma_b < 0.0:
        raise MetricError(f"sigma_b must be finite and non-negative, got {sigma_b!r}")
    stack = to_float_stack(images, "images")
    if sigma_b == 0.0:
        return stack

    kernel = np.exp(-eigenvalues(stack.shape[1]) * (sigma_b**2 / 2.0))
    out = np.empty_like(stack)
    for start in range(0, stack.shape[0], CHUNK):
        block = np.asarray(stack[start : start + CHUNK], dtype=np.float64)
        coefficients = dctn(block, axes=(1, 2), norm="ortho") * kernel
        out[start : start + CHUNK] = idctn(coefficients, axes=(1, 2), norm="ortho")
    return out
