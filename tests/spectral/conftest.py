"""Fixtures for the spectral tests: synthetic power-law fields and a white spectrum."""

from __future__ import annotations

import numpy as np
import pytest


def power_law_field(n_images: int, side: int, alpha: float, seed: int = 0) -> np.ndarray:
    """Isotropic Gaussian random fields whose power spectral density follows ``f^-alpha``.

    Built in Fourier space and returned in image space, without any per-image rescaling, so the
    measured DCT per-mode variance follows ``n^-alpha`` up to the sampling noise of the stack.

    Parameters
    ----------
    n_images : int
        Number of fields.
    side : int
        Side length in pixels.
    alpha : float
        Spectral exponent.
    seed : int
        Seed of the generator.

    Returns
    -------
    numpy.ndarray
        Stack of shape ``(n_images, side, side)``, ``float64``.
    """
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(side)[:, None]
    fy = np.fft.fftfreq(side)[None, :]
    freq = np.sqrt(fx**2 + fy**2)
    freq[0, 0] = 1.0
    amplitude = freq ** (-alpha / 2)
    amplitude[0, 0] = 0.0
    spectrum = rng.normal(size=(n_images, side, side)) + 1j * rng.normal(
        size=(n_images, side, side)
    )
    return np.real(np.fft.ifft2(spectrum * amplitude[None], axes=(1, 2)))


@pytest.fixture(scope="session")
def alpha2_field() -> np.ndarray:
    """512 Gaussian fields at 64^2 with ``alpha = 2``, the ticket's alpha-recovery case."""
    return power_law_field(512, 64, 2.0, seed=2026)


@pytest.fixture(scope="session")
def white_power() -> np.ndarray:
    """A flat per-mode variance on the 64-grid, with the DC mode zeroed."""
    power = np.ones((64, 64), dtype=np.float64)
    power[0, 0] = 0.0
    return power


@pytest.fixture(scope="session")
def random_power() -> np.ndarray:
    """The per-mode variance of a small random stack, for the comparison with the originals."""
    rng = np.random.default_rng(7)
    from scipy.fft import dctn

    images = rng.normal(size=(48, 32, 32))
    power = dctn(images, axes=(1, 2), norm="ortho").var(axis=0)
    power[0, 0] = 0.0
    return power
