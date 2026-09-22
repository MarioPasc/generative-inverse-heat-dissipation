"""Per-mode variance of an image stack and the radial quantities derived from it.

Frozen definitions: ``docs/SPECIFICATIONS/05-metrics.md`` §1. A "mode" is an orthonormal
2-D DCT-II coefficient ``(i, j)`` on the ``W x W`` grid; its radial index is
``n = sqrt(i^2 + j^2)`` and it carries ``n / 2`` cycles per image. The DC mode is excluded
from every quantity in this module.

Ported, with attribution, from the planning scripts of the TFM knowledge base
(``projects/GenAI/analysis/``, "written by an AI assistant during planning; evidence, not a
deliverable"):

* :func:`mode_power`, :func:`eigenvalues`, :func:`octave_shares`, :func:`power_law` from
  ``delta_star_from_psd.py``;
* :func:`radial_spectrum` from ``explainer_figures.py``;
* :func:`fit_alpha` from ``control_profile.py``;
* :func:`inherited_share` from ``knob_evidence.py``.

Those modules are never imported at runtime: this is a self-contained port, checked against
the originals by ``tests/spectral/test_against_originals.py``.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.fft import dctn

from ihdm.spectral.errors import SpectralError

__all__ = [
    "SpectralError",
    "OCTAVE_EDGES",
    "OCTAVE_LABELS",
    "MAX_CYCLES_PER_IMAGE",
    "mode_power",
    "eigenvalues",
    "radial_index",
    "octave_bins",
    "octave_shares",
    "radial_profile",
    "radial_spectrum",
    "fit_alpha",
    "inherited_share",
    "power_law",
]

logger = logging.getLogger(__name__)

# Octave bins in cycles per image (05-metrics.md §1). The last bin is closed at 96 c/img;
# modes above it (the corner modes, up to 135 c/img) are excluded from every share.
OCTAVE_EDGES: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 96.0)
OCTAVE_LABELS: tuple[str, ...] = tuple(
    f"{lo:g}-{hi:g}" for lo, hi in zip(OCTAVE_EDGES[:-1], OCTAVE_EDGES[1:], strict=True)
)
MAX_CYCLES_PER_IMAGE: float = OCTAVE_EDGES[-1]

# fit_alpha's window, as a fraction of the DCT radius range; the values of
# ``control_profile.fit_alpha``. At W = 192 they select the integer radii b in [20, 134],
# i.e. 10.0 - 67.0 cycles per image.
ALPHA_LO_FRAC: float = 0.10
ALPHA_HI_FRAC: float = 0.70


def mode_power(images: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Per-mode variance of a stack of images in the orthonormal DCT basis.

    The stack is mean-centred across images and the DCT-II of every centred image is taken;
    ``P(i, j)`` is the mean of the squared coefficients over the stack. The DCT is linear, so
    centring in pixel space and centring in coefficient space agree exactly; the two-pass form
    is used because it avoids both the cancellation of ``E[X^2] - E[X]^2`` on the coarse modes
    and holding the whole transformed stack in memory.

    Ported from ``delta_star_from_psd.mode_power`` (which computes
    ``dctn(images, axes=(1, 2), norm="ortho").var(axis=0)`` in one shot). The DC entry is set
    to zero here, as required by ``docs/SPECIFICATIONS/M1-data/T1.3-profile-and-schedules.md``
    §1; the original leaves it in and every consumer zeroes it.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of shape ``(n, W, W)`` with values in ``[0, 1]``. Any float or integer dtype is
        accepted; the computation is done in ``float64``.
    chunk : int
        Number of images transformed at a time.

    Returns
    -------
    numpy.ndarray
        Array ``P`` of shape ``(W, W)``, ``float64``, with ``P[0, 0] = 0``.

    Raises
    ------
    SpectralError
        If the stack is not a stack of square images, or is empty.
    """
    if images.ndim != 3 or images.shape[1] != images.shape[2]:
        raise SpectralError("expected a stack of square images of shape (n, W, W)")
    n_images = images.shape[0]
    if n_images == 0:
        raise SpectralError("expected at least one image")
    if chunk < 1:
        raise SpectralError(f"chunk must be positive, got {chunk}")

    mean = np.zeros(images.shape[1:], dtype=np.float64)
    for start in range(0, n_images, chunk):
        mean += np.asarray(images[start : start + chunk], dtype=np.float64).sum(axis=0)
    mean /= n_images

    power = np.zeros(images.shape[1:], dtype=np.float64)
    for start in range(0, n_images, chunk):
        block = np.asarray(images[start : start + chunk], dtype=np.float64) - mean
        power += np.square(dctn(block, axes=(1, 2), norm="ortho")).sum(axis=0)
    power /= n_images
    power[0, 0] = 0.0
    return power


def eigenvalues(n_pix: int) -> np.ndarray:
    """Negated Laplacian eigenvalues ``lambda_{i,j}`` on an ``n_pix`` square grid.

    Eq. (2) of Rissanen et al. (ICLR 2023). Ported verbatim from
    ``delta_star_from_psd.eigenvalues``.

    Parameters
    ----------
    n_pix : int
        Image side length in pixels.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_pix, n_pix)`` holding ``pi^2 (i^2 + j^2) / n_pix^2``.
    """
    idx = np.arange(n_pix)
    return np.pi**2 * (idx[:, None] ** 2 + idx[None, :] ** 2) / n_pix**2


def radial_index(n_pix: int) -> np.ndarray:
    """DCT radial index ``n = sqrt(i^2 + j^2)`` of every mode of an ``n_pix`` grid.

    Parameters
    ----------
    n_pix : int
        Image side length in pixels.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_pix, n_pix)``.
    """
    idx = np.arange(n_pix)
    return np.sqrt(idx[:, None] ** 2 + idx[None, :] ** 2)


def octave_bins() -> tuple[tuple[str, float, float], ...]:
    """The eight octave bins of ``05-metrics.md`` §1, in cycles per image.

    Returns
    -------
    tuple[tuple[str, float, float], ...]
        ``(label, low, high)`` per bin. Every bin is half-open except the last, which is closed
        at 96 cycles per image.
    """
    return tuple(
        (label, lo, hi)
        for label, lo, hi in zip(
            OCTAVE_LABELS, OCTAVE_EDGES[:-1], OCTAVE_EDGES[1:], strict=True
        )
    )


def _octave_masks(n_pix: int) -> list[np.ndarray]:
    """Boolean mask per octave bin on the ``n_pix`` grid (DC excluded by construction)."""
    cycles = radial_index(n_pix) / 2.0
    masks = []
    for _, lo, hi in octave_bins():
        inside = (cycles >= lo) & (cycles < hi)
        if hi == MAX_CYCLES_PER_IMAGE:
            inside |= cycles == hi
        masks.append(inside)
    return masks


def octave_shares(power: np.ndarray) -> dict[str, float]:
    """Share of the between-image variance held by each octave of radial frequency.

    A scale-invariant spectrum (``alpha = 2`` in 2-D) holds equal variance in every octave;
    departures identify a characteristic scale. Shares are normalised by the total over the
    eight bins, so the modes above 96 cycles per image are excluded from both the numerator
    and the denominator. Ported from ``delta_star_from_psd.octave_shares``, with the bins of
    ``05-metrics.md`` §1 (the original stops at ``N - 1``, i.e. 95.5 c/img).

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)``.

    Returns
    -------
    dict[str, float]
        One entry per octave label, in increasing frequency; the values sum to 1.

    Raises
    ------
    SpectralError
        If the eight bins hold no variance.
    """
    values = [float(power[mask].sum()) for mask in _octave_masks(power.shape[0])]
    total = float(sum(values))
    if not total > 0.0:
        raise SpectralError("the octave bins hold no variance")
    return dict(zip(OCTAVE_LABELS, [v / total for v in values], strict=True))


def radial_profile(power: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Mean per-mode variance over each radial bin.

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)``.
    bins : numpy.ndarray
        Increasing bin edges in **cycles per image**, length ``B + 1``. Bins are half-open
        except the last, which is closed.

    Returns
    -------
    numpy.ndarray
        Array of length ``B``; ``nan`` where a bin holds no mode.

    Raises
    ------
    SpectralError
        If fewer than two edges are given, or the edges are not increasing.
    """
    edges = np.asarray(bins, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise SpectralError("bins must be a 1-D array of at least two edges")
    if not np.all(np.diff(edges) > 0):
        raise SpectralError("bin edges must be strictly increasing")
    cycles = radial_index(power.shape[0]) / 2.0
    out = np.full(edges.size - 1, np.nan)
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        inside = (cycles >= lo) & (cycles < hi)
        if b == edges.size - 2:
            inside |= cycles == hi
        if inside.any():
            out[b] = float(power[inside].mean())
    return out


def radial_spectrum(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged per-mode variance on integer DCT-radius bands.

    Band ``b`` collects the modes with ``b - 0.5 <= n < b + 0.5``. Ported from
    ``explainer_figures.radial_spectrum``, which takes the image stack and computes the power
    itself; here the power is passed in so the stack is transformed once.

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Radial frequency in cycles per image (``b / 2`` for ``b = 1 … W - 1``) and the mean
        variance of each band.
    """
    n_pix = power.shape[0]
    radius = radial_index(n_pix)
    bands = np.arange(1, n_pix)
    binned = np.array(
        [power[(radius >= b - 0.5) & (radius < b + 0.5)].mean() for b in bands]
    )
    return bands / 2.0, binned


def fit_alpha(
    power: np.ndarray, lo_frac: float = ALPHA_LO_FRAC, hi_frac: float = ALPHA_HI_FRAC
) -> float:
    """Fit ``P(n) ~ n^-alpha`` by least squares of ``log P`` on ``log n``.

    The fit window is a fixed fraction of the DCT radius range, held constant across datasets:
    the brain spectrum is curved and a fitted exponent is meaningless without its window. With
    the defaults and ``W = 192`` the window is the integer radii ``b in [20, 134]``, i.e.
    **10.0 – 67.0 cycles per image**. Ported verbatim from ``control_profile.fit_alpha``.

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)``.
    lo_frac, hi_frac : float
        Window bounds as a fraction of ``W``, applied to the integer DCT radius.

    Returns
    -------
    float
        The fitted exponent (positive for a decreasing spectrum).

    Raises
    ------
    SpectralError
        If the window contains fewer than three usable bands.
    """
    n_pix = power.shape[0]
    bands = np.arange(1, n_pix)
    profile = radial_spectrum(power)[1]
    keep = (bands >= lo_frac * n_pix) & (bands <= hi_frac * n_pix) & (profile > 0)
    if keep.sum() < 3:
        raise SpectralError("fit window contains fewer than three usable bands")
    return float(-np.polyfit(np.log(bands[keep]), np.log(profile[keep]), 1)[0])


def inherited_share(power: np.ndarray, sigma_max: float) -> float:
    """Share of the non-DC between-image variance a sample inherits from its seed.

    Eq. (2) of the proposal: ``sum_i P_i exp(-2 lambda_i t) / sum_i P_i`` with
    ``t = sigma_max^2 / 2``. Every non-DC mode of the grid contributes, including those above
    96 cycles per image. Ported verbatim from ``knob_evidence.inherited_share``.

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)``.
    sigma_max : float
        Terminal blur length-scale in pixels.

    Returns
    -------
    float
        A number in ``[0, 1]``.

    Raises
    ------
    SpectralError
        If the spectrum holds no non-DC variance.
    """
    lam = eigenvalues(power.shape[0])
    p = power.copy()
    p[0, 0] = 0.0
    total = float(p.sum())
    if not total > 0.0:
        raise SpectralError("the spectrum holds no non-DC variance")
    return float((np.exp(-2.0 * lam * sigma_max**2 / 2.0) * p).sum() / total)


def power_law(n_pix: int, alpha: float, total_variance: float) -> np.ndarray:
    """Synthetic per-mode variance following ``P(n) ~ n^-alpha``.

    Ported from ``delta_star_from_psd.power_law``; used by the tests as a spectrum with a known
    exponent. The DC entry is zeroed here, where the original substitutes its neighbour's value
    before normalising; every consumer excludes the DC mode anyway.

    Parameters
    ----------
    n_pix : int
        Image side length in pixels.
    alpha : float
        Spectral exponent; ``2.0`` is the natural-image value assumed by the paper.
    total_variance : float
        Sum of ``P`` over all modes, used to match a measured dataset.

    Returns
    -------
    numpy.ndarray
        Per-mode variance of shape ``(n_pix, n_pix)`` with ``P[0, 0] = 0``.
    """
    radius = radial_index(n_pix)
    profile = np.where(radius > 0, np.maximum(radius, 1e-9) ** (-alpha), 0.0)
    profile = profile * total_variance / profile.sum()
    profile[0, 0] = 0.0
    return profile
