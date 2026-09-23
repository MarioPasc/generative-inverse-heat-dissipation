"""Spectral endpoints of the experiment: LSD, its octave profile, ``T_tau``, the inherited band.

Frozen definitions: ``docs/SPECIFICATIONS/05-metrics.md`` §1 (per-mode variance and the radial
profile), §2 (log-spectral distance and ``T_tau``) and §5 (the inherited-band mechanism check).
Everything here is a pure function over numpy arrays layered on the primitives of
``ihdm.spectral.power``; nothing in this module recomputes a quantity that module already owns.

Units, as everywhere in the project: a "mode" is an orthonormal 2-D DCT-II coefficient ``(i, j)``
with radial index ``n = sqrt(i^2 + j^2)`` carrying ``n / 2`` cycles per image; the DC mode is
excluded from every quantity; the binned quantities stop at 96 cycles per image.

Two places where the implementation departs from the ticket's restatement of the contract, both
derived in ``docs/AGENT-LOGS/M4-metrics/T4.1-spectral-metrics.md`` §2:

* the LSD is the RMS over the **populated** log bins (43 of the 48 at ``W = 192``); five of the
  48 log-spaced bins between 0.5 and 96 cycles per image contain no mode at all, and
  ``radial_profile`` returns ``nan`` for them by contract;
* the inherited-band residual is taken about the **prior state** ``d_K x_seed``, not about the raw
  seed, because the raw-seed residual has expectation ``2 (1 - d) P`` and cannot equal the
  prediction line ``1 - d^2`` it is plotted against.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.fft import dctn

from ihdm.metrics.errors import MetricError
from ihdm.spectral.power import (
    MAX_CYCLES_PER_IMAGE,
    OCTAVE_EDGES,
    OCTAVE_LABELS,
    eigenvalues,
    inherited_share,
    mode_power,
    radial_profile,
)

__all__ = [
    "InheritedResult",
    "LOW_BAND_SIGMA_PX",
    "LsdResult",
    "MIN_CYCLES_PER_IMAGE",
    "N_LOG_BINS",
    "inherited_band",
    "log_bin_centres",
    "log_bin_edges",
    "lsd",
    "radial_log_profile",
    "t_tau",
]

logger = logging.getLogger(__name__)

#: Low edge of the fine radial grid, in cycles per image (``05-metrics.md`` §1). The high edge is
#: ``ihdm.spectral.power.MAX_CYCLES_PER_IMAGE`` = 96.
MIN_CYCLES_PER_IMAGE: float = 0.5

#: Number of log-spaced bins of the fine profile the LSD is computed on.
N_LOG_BINS: int = 48

#: Default width, in pixels, of the "low band" of the inherited-share restriction: a mode belongs
#: to the low band when its characteristic blur scale ``sigma_n = sqrt(2 / lambda_n)`` is at least
#: this, i.e. when a blur of this sigma leaves it standing. At ``W = 192`` this is ``n <= 10.80``,
#: i.e. 5.40 cycles per image. Only used when a caller passes ``low_band_sigma_px``.
LOW_BAND_SIGMA_PX: float = 8.0

#: Images transformed at a time by the chunked helpers.
CHUNK: int = 256


# --------------------------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------------------------


class _DcRemoved:
    """Read-only view of an image stack: converted to float, per-image DC removed.

    ``mode_power`` reads its input in chunks; this adapter converts and centres one chunk at a
    time, so the 2 000-image final sample set of ``05-metrics.md`` §8a never needs a converted
    copy of itself in memory, and a ``numpy.memmap`` of ``images.npy`` can be passed straight in.

    Integer stacks are divided by 255 into ``float32`` (the contract: ``samples.npy`` and
    ``images.npy`` are both ``uint8``); floating stacks keep their dtype. Both are then centred
    in ``float64``, which is the accumulation dtype of ``mode_power``.
    """

    def __init__(self, images: np.ndarray) -> None:
        self._images = images
        self.ndim: int = images.ndim
        self.shape: tuple[int, ...] = tuple(images.shape)

    def __getitem__(self, item: Any) -> np.ndarray:
        block = np.asarray(self._images[item])
        if np.issubdtype(block.dtype, np.integer):
            block = block.astype(np.float32) / np.float32(255.0)
        block = np.asarray(block, dtype=np.float64)
        return block - block.mean(axis=(-2, -1), keepdims=True)


def _as_stack(images: Any, name: str, min_images: int = 2) -> np.ndarray:
    """Validate an image stack and return it without copying.

    Parameters
    ----------
    images : Any
        Candidate stack of shape ``(N, W, W)``.
    name : str
        Name used in error messages.
    min_images : int
        Smallest acceptable ``N``.

    Returns
    -------
    numpy.ndarray
        The stack itself (a ``memmap`` stays a ``memmap``).

    Raises
    ------
    MetricError
        If the rank, the squareness, the image count or the finiteness fails.
    """
    stack = np.asanyarray(images)
    if stack.ndim != 3:
        raise MetricError(f"{name}: expected a stack of shape (N, W, W), got {stack.shape}")
    if stack.shape[1] != stack.shape[2]:
        raise MetricError(f"{name}: expected square images, got {stack.shape[1:]}")
    if stack.shape[0] < min_images:
        raise MetricError(f"{name}: expected at least {min_images} images, got {stack.shape[0]}")
    _check_finite(stack, name)
    return stack


def _check_finite(stack: np.ndarray, name: str) -> None:
    """Raise :class:`MetricError` if a floating stack holds a non-finite value.

    Integer stacks cannot hold one and are not scanned. Floating stacks are scanned in chunks so
    that the temporary boolean array stays small.

    Parameters
    ----------
    stack : numpy.ndarray
        The stack to scan.
    name : str
        Name used in the error message.

    Raises
    ------
    MetricError
        If any value is ``nan`` or infinite.
    """
    if not np.issubdtype(stack.dtype, np.floating):
        return
    for start in range(0, stack.shape[0], CHUNK):
        if not np.isfinite(np.asarray(stack[start : start + CHUNK])).all():
            raise MetricError(f"{name}: non-finite values in images [{start}, {start + CHUNK})")


def _stack_power(images: Any, name: str, min_images: int = 2) -> tuple[np.ndarray, int]:
    """Per-mode variance of a validated, DC-removed stack.

    Parameters
    ----------
    images : Any
        Stack of shape ``(N, W, W)``, ``uint8`` or float.
    name : str
        Name used in error messages.
    min_images : int
        Smallest acceptable ``N``.

    Returns
    -------
    tuple[numpy.ndarray, int]
        ``(P, N)`` with ``P`` of shape ``(W, W)`` and ``P[0, 0] = 0``.

    Raises
    ------
    MetricError
        If the stack is malformed or non-finite.
    """
    stack = _as_stack(images, name, min_images=min_images)
    return mode_power(_DcRemoved(stack), chunk=CHUNK), int(stack.shape[0])


# --------------------------------------------------------------------------------------------
# The fine radial grid
# --------------------------------------------------------------------------------------------


def log_bin_edges(n_bins: int = N_LOG_BINS) -> np.ndarray:
    """Edges of the fine radial grid, in cycles per image.

    ``n_bins`` bins log-spaced between 0.5 and 96 cycles per image (``05-metrics.md`` §1). The
    grid does not depend on the image size, so profiles of different runs are directly
    comparable; bins that hold no mode of the grid in use are dropped by the consumers.

    Parameters
    ----------
    n_bins : int
        Number of bins.

    Returns
    -------
    numpy.ndarray
        ``n_bins + 1`` increasing edges.

    Raises
    ------
    MetricError
        If ``n_bins`` is smaller than two.
    """
    if n_bins < 2:
        raise MetricError(f"n_bins must be at least 2, got {n_bins}")
    return np.logspace(
        math.log10(MIN_CYCLES_PER_IMAGE), math.log10(MAX_CYCLES_PER_IMAGE), n_bins + 1
    )


def log_bin_centres(n_bins: int = N_LOG_BINS) -> np.ndarray:
    """Geometric centres of the bins of :func:`log_bin_edges`, in cycles per image.

    Parameters
    ----------
    n_bins : int
        Number of bins.

    Returns
    -------
    numpy.ndarray
        ``n_bins`` centres.
    """
    edges = log_bin_edges(n_bins)
    return np.sqrt(edges[:-1] * edges[1:])


def _log10_profile(power: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Base-10 logarithm of the radial profile, ``nan`` where it is empty or non-positive.

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)``.
    edges : numpy.ndarray
        Bin edges in cycles per image.

    Returns
    -------
    numpy.ndarray
        One value per bin.
    """
    profile = radial_profile(power, edges)
    out = np.full(profile.shape, np.nan)
    usable = np.isfinite(profile) & (profile > 0.0)
    out[usable] = np.log10(profile[usable])
    return out


def radial_log_profile(
    images: np.ndarray, n_bins: int = N_LOG_BINS
) -> tuple[np.ndarray, np.ndarray]:
    """Fine radial profile of an image stack, in log10 of the mean per-mode variance.

    The stack is converted to float (integers divided by 255), the per-image DC is removed, the
    per-mode variance ``P`` of ``05-metrics.md`` §1 is taken over the stack (which mean-centres
    across images and excludes the DC mode), and ``P`` is averaged over the modes falling in each
    of the ``n_bins`` log-spaced bins between 0.5 and 96 cycles per image. Modes above 96 cycles
    per image are outside the grid and do not enter.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of shape ``(N, W, W)``, ``uint8`` or float, ``N >= 2``.
    n_bins : int
        Number of log-spaced bins.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(centres, log10_profile)``; the profile is ``nan`` in bins that hold no mode of the
        ``W`` grid (five of the 48 at ``W = 192``) or whose mean variance is zero.

    Raises
    ------
    MetricError
        If the stack is malformed, holds fewer than two images, or holds a non-finite value.
    """
    power, _ = _stack_power(images, "images")
    return log_bin_centres(n_bins), _log10_profile(power, log_bin_edges(n_bins))


# --------------------------------------------------------------------------------------------
# Log-spectral distance
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LsdResult:
    """Result of :func:`lsd` (``05-metrics.md`` §2).

    Parameters
    ----------
    lsd : float
        RMS over the populated log bins of ``log10 P_samples - log10 P_reference``.
    octaves : dict[str, float]
        Signed ``log10`` difference of the mean per-mode variance on the eight octave bins, keyed
        by the labels of ``ihdm.spectral.power.octave_bins``; positive means the samples carry too
        much variance in that band.
    variance_ratio : float
        ``sum P_samples / sum P_reference`` over every non-DC mode.
    n_samples : int
        Number of sample images.
    n_reference : int
        Number of reference images.
    """

    lsd: float
    octaves: dict[str, float]
    variance_ratio: float
    n_samples: int
    n_reference: int


def _octave_difference(power_a: np.ndarray, power_b: np.ndarray) -> dict[str, float]:
    """Signed log10 difference of the octave-binned mean per-mode variance.

    Parameters
    ----------
    power_a, power_b : numpy.ndarray
        Per-mode variances of the same shape.

    Returns
    -------
    dict[str, float]
        One entry per octave label of ``ihdm.spectral.power``.

    Raises
    ------
    MetricError
        If an octave bin is empty or holds no variance in either spectrum, so that its logarithm
        is undefined.
    """
    edges = np.asarray(OCTAVE_EDGES, dtype=np.float64)
    profile_a = radial_profile(power_a, edges)
    profile_b = radial_profile(power_b, edges)
    out: dict[str, float] = {}
    for label, value_a, value_b in zip(OCTAVE_LABELS, profile_a, profile_b, strict=True):
        if not (np.isfinite(value_a) and np.isfinite(value_b) and value_a > 0 and value_b > 0):
            raise MetricError(
                f"octave bin {label} cycles/image holds no usable variance "
                f"(samples {value_a!r}, reference {value_b!r})"
            )
        out[label] = float(math.log10(value_a) - math.log10(value_b))
    return out


def lsd(samples: np.ndarray, reference: np.ndarray, n_bins: int = N_LOG_BINS) -> LsdResult:
    """Log-spectral distance between a sample stack and a reference stack.

    ``05-metrics.md`` §2: the RMS, over the bins of the fine radial grid, of the difference of the
    base-10 logarithms of the mean per-mode variance. Bins that hold no mode of the ``W`` grid, or
    whose variance is zero in either stack, are dropped from the RMS; at ``W = 192`` and
    ``n_bins = 48`` that leaves 43 bins, the same 43 for every stack, so the metric stays a fixed
    functional of the two spectra. The reference stack must be the ``ref`` split of the dataset,
    never the training split.

    Parameters
    ----------
    samples : numpy.ndarray
        Sample stack ``(N_s, W, W)``, ``uint8`` or float.
    reference : numpy.ndarray
        Reference stack ``(N_r, W, W)``, same ``W``.
    n_bins : int
        Number of log-spaced bins.

    Returns
    -------
    LsdResult
        The distance, the octave profile, the total variance ratio and the two stack sizes.

    Raises
    ------
    MetricError
        If either stack is malformed, holds fewer than two images or a non-finite value; if the
        two image sizes differ; if fewer than two bins are usable; if an octave bin is empty; or
        if the reference holds no non-DC variance.
    """
    power_s, n_samples = _stack_power(samples, "samples")
    power_r, n_reference = _stack_power(reference, "reference")
    if power_s.shape != power_r.shape:
        raise MetricError(
            f"image size mismatch: samples {power_s.shape}, reference {power_r.shape}"
        )

    edges = log_bin_edges(n_bins)
    log_s = _log10_profile(power_s, edges)
    log_r = _log10_profile(power_r, edges)
    usable = np.isfinite(log_s) & np.isfinite(log_r)
    if int(usable.sum()) < 2:
        raise MetricError(
            f"fewer than two usable radial bins ({int(usable.sum())} of {n_bins}); "
            "the stacks hold no resolvable spectrum"
        )
    difference = log_s[usable] - log_r[usable]
    value = float(np.sqrt(np.mean(np.square(difference))))

    total_r = float(power_r.sum())
    if not total_r > 0.0:
        raise MetricError("the reference stack holds no non-DC variance")

    return LsdResult(
        lsd=value,
        octaves=_octave_difference(power_s, power_r),
        variance_ratio=float(power_s.sum()) / total_r,
        n_samples=n_samples,
        n_reference=n_reference,
    )


def t_tau(lsd_by_step: dict[int, float], threshold: float) -> int | None:
    """First checkpoint step at which the LSD reaches a threshold (``05-metrics.md`` §2).

    ``T_tau(arm)`` is the smallest step ``s`` with ``LSD(s) <= threshold``, the threshold being
    the final LSD of the A0 run with the same seed. ``None`` is the "not reached" of the spec.

    Parameters
    ----------
    lsd_by_step : dict[int, float]
        LSD keyed by checkpoint step; the keys need not be sorted.
    threshold : float
        The target LSD.

    Returns
    -------
    int or None
        The smallest step meeting the threshold, or ``None`` if none does.

    Raises
    ------
    MetricError
        If the mapping is empty, a key is not an integer step, or a value or the threshold is not
        finite. An empty mapping means the caller found no checkpoints, which is a caller bug and
        not a "not reached".
    """
    if not lsd_by_step:
        raise MetricError("lsd_by_step is empty; there is no curve to threshold")
    if not math.isfinite(float(threshold)):
        raise MetricError(f"threshold must be finite, got {threshold!r}")
    items: list[tuple[int, float]] = []
    for key, value in lsd_by_step.items():
        step = int(key)
        number = float(value)
        if not math.isfinite(number):
            raise MetricError(f"non-finite LSD at step {step}: {value!r}")
        items.append((step, number))
    for step, number in sorted(items):
        if number <= float(threshold):
            return step
    return None


# --------------------------------------------------------------------------------------------
# Inherited band
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InheritedResult:
    """Result of :func:`inherited_band` (``05-metrics.md`` §5).

    Parameters
    ----------
    radial_measured : numpy.ndarray
        Radial profile of the measured per-mode ratio ``V / P_ref``, one value per fine bin,
        ``nan`` in bins holding no mode.
    radial_predicted : numpy.ndarray
        Radial profile of the linear-Gaussian prediction ``1 - exp(-2 lambda_n t_K)``.
    centres : numpy.ndarray
        Bin centres in cycles per image.
    share_measured : float
        ``1 - sum V / sum P_ref`` over the modes of the summation band.
    share_predicted : float
        ``sum P_ref exp(-2 lambda t_K) / sum P_ref`` over the same band, from
        ``ihdm.spectral.power.inherited_share``.
    sigma_max : float
        The terminal blur the two shares were computed at, in pixels.
    """

    radial_measured: np.ndarray
    radial_predicted: np.ndarray
    centres: np.ndarray
    share_measured: float
    share_predicted: float
    sigma_max: float


def _low_band_mask(n_pix: int, sigma_px: float) -> np.ndarray:
    """Modes whose characteristic blur scale is at least ``sigma_px`` pixels.

    A mode is damped by ``exp(-lambda_n sigma^2 / 2)``, so its characteristic blur scale is
    ``sigma_n = sqrt(2 / lambda_n)``: the sigma at which its amplitude falls by ``1 / e``. The
    low band is ``sigma_n >= sigma_px``, i.e. ``lambda_n <= 2 / sigma_px^2``, i.e.
    ``n <= sqrt(2) W / (pi sigma_px)`` — the modes a blur of ``sigma_px`` leaves standing. At
    ``W = 192`` and ``sigma_px = 8`` that is ``n <= 10.80``, i.e. 5.40 cycles per image.

    Parameters
    ----------
    n_pix : int
        Image side length in pixels.
    sigma_px : float
        The blur scale, in pixels.

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``(n_pix, n_pix)``; the DC mode is inside it and must be excluded
        separately.

    Raises
    ------
    MetricError
        If ``sigma_px`` is not positive and finite.
    """
    if not (math.isfinite(sigma_px) and sigma_px > 0.0):
        raise MetricError(f"low_band_sigma_px must be positive and finite, got {sigma_px!r}")
    return eigenvalues(n_pix) <= 2.0 / sigma_px**2


def _residual_power(
    samples_per_seed: np.ndarray, seeds: np.ndarray, damping: np.ndarray
) -> np.ndarray:
    """Per-mode variance of the samples about their prior state, averaged over seeds and samples.

    ``V(i, j) = mean_{s,m} (DCT(y_sm)(i, j) - d(i, j) DCT(x_s)(i, j))^2`` with the per-image DC
    removed from every image and ``d`` the terminal blur kernel. Under the linear-Gaussian model
    of ``05-metrics.md`` §5 this is exactly ``(1 - d^2) P``.

    Parameters
    ----------
    samples_per_seed : numpy.ndarray
        Validated stack ``(S, M, W, W)``.
    seeds : numpy.ndarray
        Validated stack ``(S, W, W)``.
    damping : numpy.ndarray
        ``d(i, j) = exp(-lambda_ij sigma_max^2 / 2)`` of shape ``(W, W)``.

    Returns
    -------
    numpy.ndarray
        ``V`` of shape ``(W, W)`` with ``V[0, 0] = 0``.
    """
    n_seeds, n_per_seed = samples_per_seed.shape[0], samples_per_seed.shape[1]
    view_samples = _DcRemoved(samples_per_seed)
    view_seeds = _DcRemoved(seeds)
    residual = np.zeros(samples_per_seed.shape[2:], dtype=np.float64)
    for s in range(n_seeds):
        prior = damping * dctn(view_seeds[s], norm="ortho")
        for start in range(0, n_per_seed, CHUNK):
            block = view_samples[s, start : start + CHUNK]
            coefficients = dctn(block, axes=(1, 2), norm="ortho")
            residual += np.square(coefficients - prior).sum(axis=0)
    residual /= float(n_seeds * n_per_seed)
    residual[0, 0] = 0.0
    return residual


def inherited_band(
    samples_per_seed: np.ndarray,
    seeds: np.ndarray,
    reference_power: np.ndarray,
    sigma_max: float,
    n_bins: int = N_LOG_BINS,
    *,
    low_band_sigma_px: float | None = None,
) -> InheritedResult:
    """Inherited-band mechanism check of ``05-metrics.md`` §5.

    For each held-out seed ``x_s`` and its ``M`` samples ``y_sm``, the per-mode variance of the
    samples about the prior state ``d_K x_s`` is measured and compared, mode by mode, with the
    linear-Gaussian prediction ``1 - d_K^2(n) = 1 - exp(-2 lambda_n t_K)`` with
    ``t_K = sigma_max^2 / 2``:
    the chain regenerates the part of each mode the forward blur removed, whose variance is
    ``(1 - d_K^2) P``, while the surviving part ``d_K x_s`` is the same in all ``M`` samples.
    The measured curve lying on the line means the model inherits exactly the modes the prior
    hands it; curvature is the finding.

    The residual is taken about ``d_K x_s`` and not about ``x_s``: the raw-seed residual has
    expectation ``(1 - d)^2 P + (1 - d^2) P = 2 (1 - d) P``, which is not the quantity the
    prediction line describes and which makes the share negative whenever the variance-weighted
    mean of ``d`` is below one half (every dataset at ``sigma_max = 96``). ``d_K x_s`` is the seed
    blurred to level ``K``, i.e. the prior state the chain starts from, up to the prior noise.

    Parameters
    ----------
    samples_per_seed : numpy.ndarray
        Samples of shape ``(S, M, W, W)``, ``uint8`` or float, as written by
        ``ihdm.cli.sample_ckpt`` (``samples.npy``).
    seeds : numpy.ndarray
        The seed images of shape ``(S, W, W)`` (``seeds.npy``).
    reference_power : numpy.ndarray
        Population per-mode variance ``P_ref`` of shape ``(W, W)``, from
        ``ihdm.spectral.power.mode_power`` on the dataset's ``ref`` split.
    sigma_max : float
        Terminal blur ``sigma_{B,max}`` of the run, in pixels.
    n_bins : int
        Number of log-spaced bins of the reported profiles.
    low_band_sigma_px : float or None
        When given, both shares are restricted to the modes whose characteristic blur scale is at
        least this many pixels (see :func:`_low_band_mask`). The default ``None`` sums over every
        non-DC mode, which is the set ``inherited_share`` uses and the only choice under which the
        measured and predicted shares are comparable.

    Returns
    -------
    InheritedResult
        The two radial curves, the bin centres, the two shares and ``sigma_max``.

    Raises
    ------
    MetricError
        If the shapes disagree, ``S < 1`` or ``M < 1``, an input is non-finite, ``sigma_max`` is
        not positive and finite, or the reference spectrum holds no variance in the band.
    """
    samples = np.asanyarray(samples_per_seed)
    if samples.ndim != 4:
        raise MetricError(f"samples_per_seed: expected (S, M, W, W), got {samples.shape}")
    n_seeds, n_per_seed, n_pix = samples.shape[0], samples.shape[1], samples.shape[2]
    if n_seeds < 1 or n_per_seed < 1:
        raise MetricError(f"samples_per_seed: expected S >= 1 and M >= 1, got {samples.shape}")
    if samples.shape[3] != n_pix:
        raise MetricError(f"samples_per_seed: expected square images, got {samples.shape[2:]}")
    for s in range(n_seeds):
        _check_finite(samples[s], f"samples_per_seed[{s}]")

    seed_stack = _as_stack(seeds, "seeds", min_images=1)
    if seed_stack.shape != (n_seeds, n_pix, n_pix):
        raise MetricError(
            f"seeds: expected {(n_seeds, n_pix, n_pix)}, got {tuple(seed_stack.shape)}"
        )
    power_ref = np.asarray(reference_power, dtype=np.float64)
    if power_ref.shape != (n_pix, n_pix):
        raise MetricError(
            f"reference_power: expected {(n_pix, n_pix)}, got {tuple(power_ref.shape)}"
        )
    if not np.isfinite(power_ref).all():
        raise MetricError("reference_power: non-finite values")
    if not (math.isfinite(sigma_max) and sigma_max > 0.0):
        raise MetricError(f"sigma_max must be positive and finite, got {sigma_max!r}")

    lam = eigenvalues(n_pix)
    t_k = sigma_max**2 / 2.0
    damping = np.exp(-lam * t_k)
    predicted_modes = 1.0 - np.exp(-2.0 * lam * t_k)
    residual = _residual_power(samples, seed_stack, damping)

    power_ref = power_ref.copy()
    power_ref[0, 0] = 0.0
    ratio = np.divide(
        residual, power_ref, out=np.zeros_like(residual), where=power_ref > 0.0
    )

    edges = log_bin_edges(n_bins)
    band = np.ones((n_pix, n_pix), dtype=bool)
    band[0, 0] = False
    if low_band_sigma_px is not None:
        band &= _low_band_mask(n_pix, low_band_sigma_px)
    banded_ref = np.where(band, power_ref, 0.0)
    total_ref = float(banded_ref.sum())
    if not total_ref > 0.0:
        raise MetricError("the reference spectrum holds no variance inside the summation band")

    return InheritedResult(
        radial_measured=radial_profile(ratio, edges),
        radial_predicted=radial_profile(predicted_modes, edges),
        centres=log_bin_centres(n_bins),
        share_measured=1.0 - float(residual[band].sum()) / total_ref,
        share_predicted=float(inherited_share(banded_ref, sigma_max)),
        sigma_max=float(sigma_max),
    )
